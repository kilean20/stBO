from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

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
        adjust_trust_region: bool = True,
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
        self.timing = {"train": [], "search": [], "oracle": [], "total": []}

        # initial poll
        init_res = self.oracle(x=None)
        self.X_last = torch.tensor(init_res["x"], dtype=torch.float64)

        self.adjust_trust_region = bool(adjust_trust_region)
        self.tr_state = TrustRegionState(bounds.shape[1], self.bounds)
        self.tr_state.center = self.X_last
        self.tr_state.best_value = float(init_res["y"])

        self.last_acq_object = None
        self.t_submit = None

    def _register_data(self, res: Dict[str, Any]) -> None:
        if self.t_submit is None:
            self.t_submit = time.time()
        if "t_start" in res and "t_end" in res:
            dt = (res["t_end"] - res["t_start"]).total_seconds()
            self.timing["oracle"].append(dt)

        x_val = res.get("x_rd", res["x"]) if self.use_readback else res.get("x_set", res["x"])
        y_val = float(res["y"])

        self.train_x.append(torch.tensor(x_val, dtype=torch.float64))
        self.train_y.append(torch.tensor([y_val], dtype=torch.float64))
        self.history.append(res)

        self.X_last = torch.tensor(x_val, dtype=torch.float64)
        self.tr_state.update(y_val, self.X_last, adjust_trust_region=self.adjust_trust_region)
        self.timing["total"].append(time.time() - self.t_submit)

    def _submit_job(self, x: torch.Tensor) -> None:
        self.t_submit = time.time()
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
            self.timing["train"].append(None)
            self.timing["search"].append(None)

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

        # # Make matplotlib CI-friendly
        # try:
        #     matplotlib.use("Agg", force=False)
        # except Exception:
        #     pass


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
    
    def plot_history(
        self,
        fig=None,
        axes=None,
        *,
        maximize: bool = True,
        title: str | None = None,
        show: bool = True,
    ):
        """
        Plot a two-panel summary of the BO run:
        (1) Objective history + best-so-far
        (2) Timing breakdown per iteration

        Parameters
        ----------
        fig : matplotlib.figure.Figure | None
            Existing figure to draw on. If None, a new one is created.
        axes : tuple[matplotlib.axes.Axes, matplotlib.axes.Axes] | None
            Existing axes (ax_obj, ax_time). If None, new subplots are created.
        maximize : bool
            Whether larger objective values are better (True) or smaller are better (False).
        title : str | None
            Figure title (suptitle). If None, a default is used.
        show : bool
            If True, calls plt.show() at the end (useful in scripts).

        Returns
        -------
        (fig, (ax_obj, ax_time))
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import rc_context

        # --- Collect objective history ---
        ys = []
        for h in getattr(self, "history", []):
            y = h.get("y", None)
            if y is None:
                continue
            if hasattr(y, "detach"):  # torch
                y = y.detach().cpu().item()
            elif hasattr(y, "item"):  # numpy scalar
                y = y.item()
            ys.append(float(y))

        if len(ys) == 0:
            raise ValueError("No objective values found: expected bo.history[*]['y'].")

        y = np.asarray(ys, dtype=float)
        x_obj = np.arange(1, len(y) + 1)

        best_so_far = np.maximum.accumulate(y) if maximize else np.minimum.accumulate(y)
        best_idx = int(np.argmax(best_so_far) if maximize else np.argmin(best_so_far))
        best_x = x_obj[best_idx]
        best_y = best_so_far[best_idx]

        # --- Collect timing history ---
        timing = getattr(self, "timing", {}) or {}
        train_t = np.asarray(timing.get("train", []), dtype=float)
        search_t = np.asarray(timing.get("search", []), dtype=float)
        oracle_t = np.asarray(timing.get("oracle", []), dtype=float)
        total_t = np.asarray(timing.get("total", []), dtype=float)

        # x-axis for timing (may differ in length from objective)
        n_time = max(len(train_t), len(search_t), len(oracle_t), 0)
        x_time = np.arange(1, n_time + 1)

        def _pad(arr, n):
            if len(arr) == 0:
                return np.full(n, np.nan)
            if len(arr) >= n:
                return arr[:n]
            out = np.full(n, np.nan)
            out[: len(arr)] = arr
            return out

        train_t = _pad(train_t, n_time)
        search_t = _pad(search_t, n_time)
        oracle_t = _pad(oracle_t, n_time)
        total_t = _pad(total_t, n_time)


        # --- Publication-ish defaults (no explicit colors) ---
        rc = {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
        }

        with rc_context(rc):
            if fig is None or axes is None:
                fig, (ax_obj, ax_time) = plt.subplots(
                    2, 1, sharex=False, figsize=(7.2, 6.2), constrained_layout=True
                )
            else:
                ax_obj, ax_time = axes

            # ---- Top panel: objective ----
            ax_obj.plot(x_obj, y, marker="o", markersize=4, label="Objective")
            ax_obj.plot(x_obj, best_so_far, label="Best so far")
            ax_obj.scatter([best_x], [best_y], zorder=5, label=f"Best @ {best_x}")
            ax_obj.set_ylabel("Objective")
            ax_obj.set_title("Objective over evaluations")
            ax_obj.grid(True, alpha=0.3)
            ax_obj.legend(loc="best", frameon=False)

            # ---- Bottom panel: timing ----
            if n_time > 0:
                ax_time.plot(x_time, train_t, marker="o", markersize=3, label="Train")
                ax_time.plot(x_time, search_t, marker="o", markersize=3, label="Acq opt")
                ax_time.plot(x_time, oracle_t, marker="o", markersize=3, label="Oracle")
                ax_time.plot(x_time, total_t, marker="o", markersize=3, label="Total")
                ax_time.set_xlabel("Iteration")
                ax_time.set_ylabel("Time (s)")
                ax_time.set_title("Timing per iteration")
                ax_time.grid(True, alpha=0.3)
                ax_time.legend(loc="best", frameon=False)
            else:
                ax_time.text(
                    0.5,
                    0.5,
                    "No timing data available",
                    ha="center",
                    va="center",
                    transform=ax_time.transAxes,
                )
                ax_time.set_axis_off()

            # fig.suptitle(title or "BO run diagnostics", y=1.02)

            if show:
                plt.show()

            return fig, (ax_obj, ax_time)

