from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
import matplotlib

from botorch.optim import optimize_acqf

from ..acquisition import RampingCostAwareAcquisition, get_base_acq
from ..models import BoTorchGPWrapper, PriorMean
from ..utils.sampling import proximal_ordered_init_sampler
from .trust_region import TrustRegionState


class BOController:
    """High-level BO loop orchestrator.

    The oracle must be callable like: `oracle(x: np.ndarray | None) -> dict`
    and return a dict containing:
      - "x": current state
      - "x_rd": readback state (optional)
      - "y": scalar objective value (maximize)
    """

    def __init__(
        self,
        oracle_evaluator,
        bounds: torch.Tensor,
        *,
        use_readback: bool = True,
        prior_mean: Optional[PriorMean] = None,
        kernel_type: str = "matern",
        max_workers: int = 1,
    ):
        self.oracle = oracle_evaluator
        self.bounds = bounds.to(dtype=torch.float64)
        self.use_readback = bool(use_readback)

        self.gp = BoTorchGPWrapper(kernel_type=kernel_type, prior_mean=prior_mean)

        self.executor = ThreadPoolExecutor(max_workers=int(max_workers))
        self.current_future = None
        self.X_candidate: Optional[torch.Tensor] = None

        self.train_x: list[torch.Tensor] = []
        self.train_y: list[torch.Tensor] = []
        self.history: list[Dict[str, Any]] = []
        self.timing = {"train": [], "search": [], "oracle": []}

        # initial poll
        init_res = self.oracle(x=None)
        self.X_last = torch.tensor(init_res["x"], dtype=torch.float64)

        self.tr_state = TrustRegionState(bounds.shape[1], self.bounds)
        self.tr_state.center = self.X_last
        self.tr_state.best_value = float(init_res["y"])

        self.last_acq_object = None

    def _register_data(self, res: Dict[str, Any]) -> None:
        if "t_start" in res and "t_end" in res:
            dt = (res["t_end"] - res["t_start"]).total_seconds()
            self.timing["oracle"].append(dt)

        x_val = res.get("x_rd", res["x"]) if self.use_readback else res.get("x_set", res["x"])
        y_val = float(res["y"])

        self.train_x.append(torch.tensor(x_val, dtype=torch.float64))
        self.train_y.append(torch.tensor([y_val], dtype=torch.float64))
        self.history.append(res)

        self.X_last = torch.tensor(x_val, dtype=torch.float64)
        self.tr_state.update(y_val, self.X_last)

    def _submit_job(self, x: torch.Tensor) -> None:
        self.X_candidate = x

        def task():
            t0 = datetime.now()
            out = self.oracle(x.detach().cpu().numpy())
            t1 = datetime.now()
            out.update({"t_start": t0, "t_end": t1, "x_set": x.detach().cpu().numpy()})
            return out

        self.current_future = self.executor.submit(task)

    def _update_model(self, *, fresh_train: bool = True) -> None:
        if not self.train_x:
            return
        t0 = time.time()
        X = torch.stack(self.train_x)
        Y = torch.stack(self.train_y)
        self.gp.fit(X, Y, self.bounds, fresh_train=fresh_train)
        self.timing["train"].append(time.time() - t0)

    def initialize(
        self,
        *,
        budget: int,
        local_init: bool = False,
        ramping_rate: torch.Tensor | float | None = None,
        fixed_features: Optional[Dict[int, float]] = None,
        seed: int | None = None,
    ) -> None:
        """Queue an initial design and submit the last point asynchronously."""
        init_bounds = self.bounds.clone()
        X_current = self.X_candidate if self.X_candidate is not None else self.X_last

        if local_init:
            span = self.bounds[1] - self.bounds[0]
            w = 0.1 * span
            init_bounds[0] = torch.max(self.bounds[0], X_current - w / 2)
            init_bounds[1] = torch.min(self.bounds[1], X_current + w / 2)

        samples = proximal_ordered_init_sampler(
            budget,
            init_bounds,
            X_current,
            ramping_rate=ramping_rate,
            fixed_features=fixed_features,
            seed=seed,
        )

        if self.current_future is not None:
            self._register_data(self.current_future.result())

        for i in range(budget - 1):
            self._submit_job(samples[i])
            self._register_data(self.current_future.result())

        self._submit_job(samples[-1])

    def _auto_ramp_cost_config(self, base_acq, *, asynchro: bool, mode: str) -> Dict[str, Any]:
        mode = mode.lower()
        ramp_cost_config: Dict[str, Any] = {
            "penalize_pending": asynchro and mode != "finetune",
            "use_ramping_favor": mode == "global",
            "L_penal": None,
            "C_penal": None,
            "L_favor": None,
            "C_favor": None,
            "polarity_penalty": None,
        }

        X_current = self.X_candidate if asynchro else self.X_last
        if len(self.train_x) < 2:
            return ramp_cost_config

        X_train = torch.stack(self.train_x)  # (N, d)
        d = X_train.shape[1]
        nsample = min(2 * d, len(X_train) - 1)

        def estimate_L_C(target_center: torch.Tensor, scale_L: float, scale_C: float):
            target_center = target_center.reshape(1, -1)

            # independent per-dimension scaling
            dists = torch.abs(X_train - target_center)  # (N, d)
            k = min(nsample, dists.shape[0])
            vals, _ = torch.topk(dists, k, dim=0, largest=False)
            local_L = vals[-1, :]

            noise = torch.randn(8 * d, d, device=target_center.device, dtype=target_center.dtype)
            samples = target_center + noise * (local_L + 1e-6)
            samples = torch.max(torch.min(samples, self.bounds[1]), self.bounds[0])

            with torch.no_grad():
                acq_vals = base_acq(samples.unsqueeze(1))
            range_y = acq_vals.max() - acq_vals.min()

            return (scale_L * local_L), (scale_C * range_y).item()

        if (self.X_candidate is not None) and ramp_cost_config["penalize_pending"]:
            l_p, c_p = estimate_L_C(self.X_candidate, scale_L=0.5, scale_C=1.0)
            ramp_cost_config["L_penal"] = l_p
            ramp_cost_config["C_penal"] = c_p

        if ramp_cost_config["use_ramping_favor"]:
            l_f, c_f = estimate_L_C(X_current, scale_L=10.0, scale_C=0.5)
            ramp_cost_config["L_favor"] = l_f
            ramp_cost_config["C_favor"] = c_f

        if ramp_cost_config.get("polarity_penalty") is None:
            with torch.no_grad():
                _, range_at_curr = estimate_L_C(X_current, scale_L=1.0, scale_C=1.0)
                ramp_cost_config["polarity_penalty"] = 0.2 * range_at_curr

        return ramp_cost_config

    def step(
        self,
        *,
        asynchro: bool = True,
        mode: str = "global",
        acq_type: str = "qUCB",
        acq_config: Optional[Dict[str, Any]] = None,
        ramp_cost_config: Optional[Dict[str, Any]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        fresh_train: bool = False,
        plot_acq: bool = False,
        optimize_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """One BO step (optimize acquisition, submit point, optionally async)."""
        if (not asynchro) and (self.current_future is not None):
            self._register_data(self.current_future.result())

        self._update_model(fresh_train=fresh_train)

        t0 = time.time()
        mode_l = mode.lower()
        if mode_l in ("local", "finetune"):
            search_bounds = self.tr_state.get_bounds()
        else:
            search_bounds = self.bounds

        X_pen = (
            self.X_candidate.unsqueeze(0)
            if (self.X_candidate is not None and asynchro)
            else None
        )

        best_f = torch.max(torch.stack(self.train_y)).item() if self.train_y else 0.0
        a_conf = dict(acq_config or {})
        a_conf["best_f"] = best_f

        base_acq = get_base_acq(self.gp, acq_type, X_pending=X_pen, acq_config=a_conf)

        if ramp_cost_config is None:
            ramp_cost_config = self._auto_ramp_cost_config(base_acq, asynchro=asynchro, mode=mode_l)

        acq = RampingCostAwareAcquisition(
            self.gp,
            acq_type,
            self.X_last,
            self.bounds,
            X_pending=X_pen,
            acq_config=a_conf,
            ramp_cost_config=ramp_cost_config,
        )
        self.last_acq_object = acq

        ok = dict(num_restarts=5, raw_samples=50)
        if optimize_kwargs:
            ok.update(optimize_kwargs)

        candidate, _ = optimize_acqf(
            acq,
            search_bounds,
            q=1,
            fixed_features=fixed_features,
            **ok,
        )
        self.timing["search"].append(time.time() - t0)

        if plot_acq:
            self.plot_acq(
                X_pending=(X_pen.detach().squeeze(0) if X_pen is not None else None),
                X_candidate=candidate.detach().squeeze(0),
            )

        if asynchro and self.current_future is not None:
            self._register_data(self.current_future.result())

        self._submit_job(candidate.detach().squeeze(0))

    def finalize(self) -> None:
        if self.current_future is not None:
            self._register_data(self.current_future.result())
            self.current_future = None
            self.X_candidate = None
            self._update_model(fresh_train=False)

    def plot_acq(
        self,
        *,
        X_pending: Optional[torch.Tensor] = None,
        X_candidate: Optional[torch.Tensor] = None,
        n_each: int = 32,
        fig=None,
        axes=None,
    ):
        """2D helper plot (requires dim=2)."""
        if self.gp.model is None:
            return None
        if self.bounds.shape[1] != 2:
            raise ValueError("plot_acq only supports 2D bounds.")

        # Make matplotlib CI-friendly
        try:
            matplotlib.use("Agg", force=False)
        except Exception:
            pass

        import matplotlib.pyplot as plt  # local import

        bounds = self.bounds
        x = np.linspace(bounds[0, 0].item(), bounds[1, 0].item(), n_each)
        y = np.linspace(bounds[0, 1].item(), bounds[1, 1].item(), n_each)
        XX, YY = np.meshgrid(x, y)
        grid_x = torch.tensor(np.column_stack((XX.ravel(), YY.ravel())), dtype=torch.float64)

        with torch.no_grad():
            post = self.gp.posterior(grid_x)
            mean = post.mean.view(n_each, n_each).cpu().numpy()

        if fig is None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        c1 = axes[0].contourf(XX, YY, mean, levels=16)
        plt.colorbar(c1, ax=axes[0])

        train_x_np = torch.stack(self.train_x).cpu().numpy()
        axes[0].scatter(train_x_np[:, 0], train_x_np[:, 1], c="k", marker=".", s=20, label="Data")
        axes[0].set_title("Model Mean")

        if self.last_acq_object is not None:
            with torch.no_grad():
                grid_in = grid_x.unsqueeze(1)
                base_vals = (
                    self.last_acq_object.base_acq(grid_in).view(n_each, n_each).cpu().numpy()
                )
                c2 = axes[1].contourf(XX, YY, base_vals, levels=16)
                plt.colorbar(c2, ax=axes[1])
                axes[1].set_title("Base Acquisition")

                final_vals = self.last_acq_object(grid_in).view(n_each, n_each).cpu().numpy()
                c3 = axes[2].contourf(XX, YY, final_vals, levels=16)
                plt.colorbar(c3, ax=axes[2])
                axes[2].set_title("Acq + Ramping Costs")

                for i in range(1, 3):
                    axes[i].scatter(
                        train_x_np[:, 0], train_x_np[:, 1], c="k", marker=".", s=20, label="Data"
                    )
                    axes[i].legend()

        for i in range(3):
            if X_pending is not None:
                axes[i].scatter(X_pending[0], X_pending[1], c="r", marker="x", s=100, label="Pending")
            if X_candidate is not None:
                axes[i].scatter(
                    X_candidate[0], X_candidate[1], c="b", marker="*", s=100, label="Candidate"
                )

        axes[0].legend()
        plt.tight_layout()
        return fig, axes

    def plot_timecost(self, fig=None, ax=None):
        try:
            matplotlib.use("Agg", force=False)
        except Exception:
            pass
        import matplotlib.pyplot as plt  # local import

        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.timing["train"], label="Train Time")
        ax.plot(self.timing["search"], label="Acq Opt Time")
        ax.plot(self.timing["oracle"], label="Oracle Time")
        ax.legend()
        plt.tight_layout()
        return fig, ax
