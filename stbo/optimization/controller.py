from __future__ import annotations

import keyword
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

    _DEFAULT_ADAPTIVE_ASYNC_CONFIG: Dict[str, Any] = {
        "window": 5,
        "min_observations": 3,
        "enable_ratio": (0.5, 2.0),
        "keep_ratio": (0.35, 2.5),
    }

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
        tr_length_init: float = 0.2,
        adaptive_async_config: Optional[Dict[str, Any]] = None,
    ):
        self.oracle = oracle_evaluator
        self.bounds = bounds.to(dtype=torch.float64)
        self.use_readback = bool(use_readback)
        self.tr_length_init = float(tr_length_init)

        self.gp = BoTorchGPWrapper(kernel_type=kernel_type, prior_mean=prior_mean)

        self.executor = ThreadPoolExecutor(max_workers=int(max_workers))
        self.current_future = None
        self.X_candidate: Optional[torch.Tensor] = None

        self.train_x: list[torch.Tensor] = []
        self.train_y: list[torch.Tensor] = []
        self.history: list[Dict[str, Any]] = []
        self.timing = {"train": [], "search": [], "oracle": [], "total": []}
        self._oracle_detail_timing_keys: set[str] = set()
        self.adaptive_async_config = self._normalize_adaptive_async_config(
            adaptive_async_config
        )
        self.async_history: list[Dict[str, Any]] = []

        # initial poll
        init_res = self.oracle(x=None)
        self.X_last = torch.tensor(init_res["x"], dtype=torch.float64)

        self.adjust_trust_region = bool(adjust_trust_region)
        
        # Initialize Trust Region with the custom length
        self.tr_state = TrustRegionState(
            bounds.shape[1], 
            self.bounds, 
            length=self.tr_length_init
        )
        self.tr_state.center = self.X_last
        self.tr_state.best_value = float(init_res["y"])
        self.local_init = False

        self.last_acq_object = None
        self.t_submit = None

    @classmethod
    def _normalize_adaptive_async_config(
        cls,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = dict(cls._DEFAULT_ADAPTIVE_ASYNC_CONFIG)
        if config:
            cfg.update(dict(config))

        cfg["window"] = max(1, int(cfg["window"]))
        cfg["min_observations"] = max(1, int(cfg["min_observations"]))

        for key in ("enable_ratio", "keep_ratio"):
            value = cfg[key]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"adaptive_async_config[{key!r}] must be a length-2 ratio.")
            lo, hi = float(value[0]), float(value[1])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo < 0 or hi < lo:
                raise ValueError(
                    f"adaptive_async_config[{key!r}] must satisfy 0 <= lower <= upper."
                )
            cfg[key] = (lo, hi)

        return cfg

    @staticmethod
    def _finite_float_or_none(value: Any) -> Optional[float]:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value_f):
            return None
        return value_f

    def _last_auto_async_decision(self) -> bool:
        for entry in reversed(getattr(self, "async_history", []) or []):
            if entry.get("requested") == "auto":
                return bool(entry.get("resolved", False))
        return False

    def _adaptive_async_metrics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        window = int(config["window"])
        min_observations = int(config["min_observations"])
        timing = getattr(self, "timing", {}) or {}

        compute_values: list[float] = []
        for train_t, search_t in zip(timing.get("train", []), timing.get("search", [])):
            train_f = self._finite_float_or_none(train_t)
            search_f = self._finite_float_or_none(search_t)
            if train_f is None or search_f is None:
                continue
            compute_values.append(train_f + search_f)

        oracle_values = [
            value_f
            for value in timing.get("oracle", [])
            if (value_f := self._finite_float_or_none(value)) is not None
        ]

        compute_recent = compute_values[-window:]
        oracle_recent = oracle_values[-window:]

        metrics: Dict[str, Any] = {
            "n_compute": len(compute_recent),
            "n_oracle": len(oracle_recent),
            "compute_time": None,
            "oracle_time": None,
            "ratio": None,
            "reason": "",
        }

        if len(compute_recent) < min_observations or len(oracle_recent) < min_observations:
            metrics["reason"] = "insufficient_timing_history"
            return metrics

        compute_time = float(np.median(np.asarray(compute_recent, dtype=float)))
        oracle_time = float(np.median(np.asarray(oracle_recent, dtype=float)))
        metrics["compute_time"] = compute_time
        metrics["oracle_time"] = oracle_time

        if oracle_time <= 0:
            metrics["reason"] = "nonpositive_oracle_time"
            return metrics

        metrics["ratio"] = compute_time / oracle_time
        return metrics

    def _resolve_asynchro(
        self,
        asynchro: Any,
        *,
        mode: str,
        adaptive_async_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not hasattr(self, "async_history"):
            self.async_history = []

        if isinstance(asynchro, str):
            if asynchro.lower() != "auto":
                raise ValueError("asynchro must be a bool or 'auto'.")

            base_config = getattr(self, "adaptive_async_config", None)
            config = self._normalize_adaptive_async_config(base_config)
            if adaptive_async_config:
                config = self._normalize_adaptive_async_config({**config, **adaptive_async_config})

            metrics = self._adaptive_async_metrics(config)
            ratio = metrics.get("ratio")
            previous_auto = self._last_auto_async_decision()
            resolved = False
            if ratio is not None:
                key = "keep_ratio" if previous_auto else "enable_ratio"
                lower, upper = config[key]
                resolved = bool(lower <= ratio <= upper)
                metrics["reason"] = f"{'within' if resolved else 'outside'}_{key}"

            record = {
                "step_index": len(getattr(self, "async_history", []) or []),
                "mode": mode,
                "requested": "auto",
                "resolved": resolved,
                **metrics,
            }
            self.async_history.append(record)
            return resolved

        resolved = bool(asynchro)
        self.async_history.append(
            {
                "step_index": len(getattr(self, "async_history", []) or []),
                "mode": mode,
                "requested": resolved,
                "resolved": resolved,
                "n_compute": None,
                "n_oracle": None,
                "compute_time": None,
                "oracle_time": None,
                "ratio": None,
                "reason": "fixed",
            }
        )
        return resolved

    def _append_oracle_detail_timing(self, res: Dict[str, Any]) -> None:
        detail = res.get("timing") or {}
        if not isinstance(detail, dict):
            detail = {}

        values: Dict[str, float] = {}
        for key, value in detail.items():
            key = str(key)
            if key in {"train", "search", "oracle", "total"}:
                key = f"oracle_detail_{key}"
            try:
                values[key] = float(value)
            except (TypeError, ValueError):
                continue

        n_oracle = len(self.timing["oracle"])
        all_keys = self._oracle_detail_timing_keys | set(values)
        for key in sorted(all_keys):
            if key not in self.timing:
                self.timing[key] = [None] * max(n_oracle - 1, 0)
            self.timing[key].append(values.get(key))

        self._oracle_detail_timing_keys.update(values)

    def _register_data(self, res: Dict[str, Any]) -> None:
        if self.t_submit is None:
            self.t_submit = time.time()
        if "t_start" in res and "t_end" in res:
            oracle_dt = (res["t_end"] - res["t_start"]).total_seconds()
            self.timing["oracle"].append(oracle_dt)
            self._append_oracle_detail_timing(res)

        x_val = res.get("x_rd", res["x"]) if self.use_readback else res.get("x_set", res["x"])
        y_val = float(res["y"])

        self.train_x.append(torch.tensor(x_val, dtype=torch.float64))
        self.train_y.append(torch.tensor([y_val], dtype=torch.float64))
        self.history.append(res)

        self.X_last = torch.tensor(x_val, dtype=torch.float64)
        
        # Update TR state (success/failure/shrink/expand)
        self.tr_state.update(y_val, self.X_last, adjust_trust_region=self.adjust_trust_region)

    def _append_total_wall_time(
        self,
        iteration_start: float,
        iteration_end: Optional[float] = None,
    ) -> None:
        if iteration_end is None:
            iteration_end = time.time()
        self.timing["total"].append(iteration_end - iteration_start)

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

    def _update_model(
        self,
        *,
        fresh_train: bool = True,
        record_timing: bool = True,
    ) -> None:
        if not self.train_x:
            return
        t0 = time.time()
        X = torch.stack(self.train_x)
        Y = torch.stack(self.train_y)
        self.gp.fit(X, Y, self.bounds, fresh_train=fresh_train)
        if record_timing:
            self.timing["train"].append(time.time() - t0)

    def _find_objective_function(self):
        for func in getattr(self.oracle, "df_manipulators", []) or []:
            owner = getattr(func, "__self__", None)
            if (
                owner is not None
                and getattr(func, "__name__", "") == "calculate_objectives_from_df"
                and callable(getattr(owner, "set_objective_weight", None))
                and callable(getattr(owner, "__call__", None))
            ):
                return owner
        raise ValueError(
            "Could not find a SingleTaskObjectiveFunction attached to "
            "oracle.df_manipulators."
        )

    def _reset_trust_region_to_best_history(self, y_values: List[float]) -> int:
        if not y_values:
            raise ValueError("Cannot reset trust region without objective values.")
        best_idx = int(np.argmax(np.asarray(y_values, dtype=float)))
        best_x = self.train_x[best_idx].clone()
        self.tr_state.length = float(getattr(self, "tr_length_init", 0.2))
        self.tr_state.success_counter = 0
        self.tr_state.failure_counter = 0
        self.tr_state.restart_triggered = False
        self.tr_state.best_value = float(y_values[best_idx])
        self.tr_state.center = best_x
        return best_idx

    @staticmethod
    def _history_mapping_keys(row: Any) -> set[str]:
        if isinstance(row, dict):
            return set(row)
        if hasattr(row, "index"):
            return set(getattr(row, "index"))
        return set()

    @staticmethod
    def _history_mapping_get(row: Any, key: str) -> Any:
        if isinstance(row, dict):
            return row[key]
        return row[key]

    @staticmethod
    def _as_float_array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=float)
        return value.reshape(-1)

    def _history_row_matches_train_x(self, row: Any, train_x: torch.Tensor) -> bool:
        keys = self._history_mapping_keys(row)
        names = getattr(self.oracle, "control_RDs" if self.use_readback else "control_CSETs", None)
        if not names or len(names) != int(train_x.numel()):
            return False
        if any(name not in keys for name in names):
            return False
        try:
            row_x = np.asarray(
                [self._history_mapping_get(row, name) for name in names],
                dtype=float,
            ).reshape(-1)
            train_x_np = train_x.detach().cpu().numpy().reshape(-1)
        except (TypeError, ValueError):
            return False
        return bool(np.allclose(row_x, train_x_np, rtol=1e-7, atol=1e-10))

    def _align_oracle_means_to_training_history(
        self,
        oracle_means: List[Any],
        objective_function: Any,
        old_y_values: List[float],
    ) -> tuple[List[Any], int]:
        n_train = len(old_y_values)
        n_oracle = len(oracle_means)
        if n_oracle < n_train:
            raise ValueError(
                "Cannot map historical raw objective PV values to BO training data. "
                f"len(train_x)={len(self.train_x)}, len(train_y)={len(self.train_y)}, "
                f"len(bo.history)={len(self.history)}, len(oracle.history['mean'])={n_oracle}."
            )
        if n_oracle == n_train:
            return list(oracle_means), 0

        composite_name = objective_function.composite_objective_name
        candidates: list[tuple[int, int]] = []
        for start in range(n_oracle - n_train + 1):
            rows = oracle_means[start : start + n_train]
            composite_values = []
            has_composite_values = True
            for row in rows:
                if composite_name not in self._history_mapping_keys(row):
                    has_composite_values = False
                    break
                try:
                    composite_values.append(float(self._history_mapping_get(row, composite_name)))
                except (TypeError, ValueError):
                    has_composite_values = False
                    break
            if not has_composite_values:
                continue
            if not np.allclose(
                np.asarray(composite_values, dtype=float),
                np.asarray(old_y_values, dtype=float),
                rtol=1e-7,
                atol=1e-10,
            ):
                continue
            x_matches = sum(
                self._history_row_matches_train_x(row, train_x)
                for row, train_x in zip(rows, self.train_x)
            )
            candidates.append((x_matches, start))

        if candidates:
            _, start = max(candidates, key=lambda item: (item[0], -item[1]))
            return list(oracle_means[start : start + n_train]), start

        x_candidates: list[tuple[int, int]] = []
        for start in range(n_oracle - n_train + 1):
            rows = oracle_means[start : start + n_train]
            x_matches = sum(
                self._history_row_matches_train_x(row, train_x)
                for row, train_x in zip(rows, self.train_x)
            )
            if x_matches:
                x_candidates.append((x_matches, start))
        if x_candidates:
            x_matches, start = max(x_candidates, key=lambda item: (item[0], -item[1]))
            if x_matches == n_train:
                return list(oracle_means[start : start + n_train]), start

        raise ValueError(
            "Cannot align oracle.history['mean'] rows to BO training data. "
            "The oracle has extra raw-history rows, but no contiguous block matches "
            "the stored BO objectives/readbacks. "
            f"len(train_x)={len(self.train_x)}, len(train_y)={len(self.train_y)}, "
            f"len(bo.history)={len(self.history)}, len(oracle.history['mean'])={n_oracle}."
        )

    def reweight_objectives(
        self,
        new_weight: Dict[str, float],
        *,
        refit_model: bool = True,
        reset_trust_region: bool = True,
    ) -> Dict[str, Any]:
        if self.current_future is not None:
            raise RuntimeError("Call finalize() before reweighting objectives.")
        if not self.train_x or not self.train_y or not self.history:
            raise ValueError("No BO training data are available to reweight.")

        objective_function = self._find_objective_function()
        old_weights = dict(getattr(objective_function, "objective_weight", {}) or {})
        old_y_values = [
            float(self._scalar_objective(value.detach().cpu().numpy().tolist()))
            if isinstance(value, torch.Tensor)
            else float(self._scalar_objective(value))
            for value in self.train_y
        ]
        old_best = max(old_y_values) if old_y_values else None

        oracle_history = getattr(self.oracle, "history", {}) or {}
        oracle_means = oracle_history.get("mean", [])
        n_train = len(self.train_y)
        if len(self.train_x) != n_train or len(self.history) != n_train:
            raise ValueError(
                "Cannot map historical raw objective PV values to BO training data. "
                f"len(train_x)={len(self.train_x)}, len(train_y)={len(self.train_y)}, "
                f"len(bo.history)={len(self.history)}, len(oracle.history['mean'])={len(oracle_means)}."
            )
        oracle_training_means, oracle_history_start = self._align_oracle_means_to_training_history(
            list(oracle_means),
            objective_function,
            old_y_values,
        )

        raw_history_values = []
        required_pvs = list(objective_function.objective_PVs)
        for idx, mean_series in enumerate(oracle_training_means):
            available_keys = self._history_mapping_keys(mean_series)
            if not available_keys:
                raise ValueError(
                    "Historical oracle mean rows must be pandas Series or dict-like "
                    f"objects for reweighting; got {type(mean_series).__name__} "
                    f"at oracle.history['mean'] index {oracle_history_start + idx}."
                )
            missing = [pv for pv in required_pvs if pv not in available_keys]
            if missing:
                raise ValueError(
                    "Historical oracle mean is missing objective PVs for reweighting "
                    f"at oracle.history['mean'] index {oracle_history_start + idx}: {missing}"
                )
            raw_history_values.append(
                np.asarray(
                    [self._history_mapping_get(mean_series, pv) for pv in required_pvs],
                    dtype=float,
                )
            )

        y_values: List[float] = []
        try:
            new_weights = objective_function.set_objective_weight(new_weight)
            for raw_values in raw_history_values:
                y = float(objective_function(raw_values).detach().cpu().reshape(-1)[0])
                y_values.append(y)
        except Exception:
            objective_function.set_objective_weight(old_weights)
            raise

        for entry, mean_series, y in zip(self.history, oracle_training_means, y_values):
            entry["y"] = y
            mean_series[objective_function.composite_objective_name] = y

        self.train_y = [
            torch.tensor([y], dtype=torch.float64)
            for y in y_values
        ]
        self.last_acq_object = None

        best_idx = None
        if reset_trust_region:
            best_idx = self._reset_trust_region_to_best_history(y_values)

        if refit_model:
            self._update_model(fresh_train=True, record_timing=False)

        return {
            "old_weights": old_weights,
            "new_weights": dict(new_weights),
            "old_best": old_best,
            "new_best": max(y_values),
            "best_index": best_idx,
            "n_recomputed": len(y_values),
            "oracle_history_start": oracle_history_start,
            "oracle_history_ignored": len(oracle_means) - len(y_values),
        }

    def initialize(
        self,
        *,
        budget: int,
        local_init: bool = False,
        ramping_rate: torch.Tensor | float | List[float] | None = None,
        fixed_features: Optional[Dict[int, float]] = None,
        seed: int | None = None,
    ) -> None:
        """Queue an initial design and submit the last point asynchronously."""
        init_bounds = self.bounds.clone()
        X_current = self.X_candidate if self.X_candidate is not None else self.X_last

        self.local_init = local_init or self.local_init

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
            iteration_start = self.t_submit or time.time()
            self._register_data(self.current_future.result())
            self.timing["train"].append(None)
            self.timing["search"].append(None)
            self._append_total_wall_time(iteration_start)

        for i in range(budget - 1):
            self._submit_job(samples[i])
            iteration_start = self.t_submit
            self._register_data(self.current_future.result())
            self.timing["train"].append(None)
            self.timing["search"].append(None)
            self._append_total_wall_time(iteration_start)

        self._submit_job(samples[-1])


    def _auto_ramp_cost_config(self, base_acq, *, effective_pending: bool, mode: str) -> Dict[str, Any]:
        # mode is already normalized to "global", "finetune", or "local"
        ramp_cost_config: Dict[str, Any] = {
            "penalize_pending": effective_pending and mode != "finetune",
            "use_ramping_favor": mode == "global",
            "L_penal": None,
            "C_penal": None,
            "L_favor": None,
            "C_favor": None,
            "polarity_penalty": None,
        }

        X_current = self.X_candidate if effective_pending else self.X_last
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
                if self.local_init:
                    ramp_cost_config["polarity_penalty"] = 0.5 * range_at_curr
                else:
                    ramp_cost_config["polarity_penalty"] = 0.2 * range_at_curr

        return ramp_cost_config

    def step(
        self,
        *,
        asynchro: Any = True,
        mode: str = "global",
        acq_type: str = "qUCB",
        acq_config: Optional[Dict[str, Any]] = None,
        ramp_cost_config: Optional[Dict[str, Any]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        fresh_train: bool = False,
        plot_acq: bool = False,
        optimize_kwargs: Optional[Dict[str, Any]] = None,
        adaptive_async_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """One BO step (optimize acquisition, submit point, optionally async)."""

        mode_l = mode.lower().replace("_", "")
        asynchro = self._resolve_asynchro(
            asynchro,
            mode=mode_l,
            adaptive_async_config=adaptive_async_config,
        )
        has_pending = self.current_future is not None and self.X_candidate is not None
        effective_pending = asynchro and has_pending
        iteration_start = (
            self.t_submit
            if has_pending and self.t_submit is not None
            else time.time()
        )
        if (not asynchro) and (self.current_future is not None):
            self._register_data(self.current_future.result())

        self._update_model(fresh_train=fresh_train)

        t0 = time.time()

        if mode_l in ("local", "finetune"):
            search_bounds = self.tr_state.get_bounds()
        else:
            search_bounds = self.bounds

        X_pen = (
            self.X_candidate.unsqueeze(0)
            if effective_pending
            else None
        )

        best_f = torch.max(torch.stack(self.train_y)).item() if self.train_y else 0.0
        a_conf = dict(acq_config or {})
        a_conf["best_f"] = best_f

        base_acq = get_base_acq(self.gp, acq_type, X_pending=X_pen, acq_config=a_conf)

        if ramp_cost_config is None:
            # pass normalized mode_l
            ramp_cost_config = self._auto_ramp_cost_config(
                base_acq,
                effective_pending=effective_pending,
                mode=mode_l,
            )

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
        search_dt = time.time() - t0
        self.timing["search"].append(search_dt)

        if plot_acq:
            self.plot_acq(
                X_pending=(X_pen.detach().squeeze(0) if X_pen is not None else None),
                X_candidate=candidate.detach().squeeze(0),
                search_bounds=search_bounds,
            )

        if asynchro and self.current_future is not None:
            self._register_data(self.current_future.result())

        self._submit_job(candidate.detach().squeeze(0))
        self._append_total_wall_time(iteration_start, self.t_submit)


    def finalize(self) -> None:
        if self.current_future is not None:
            iteration_start = self.t_submit or time.time()
            self._register_data(self.current_future.result())
            self.current_future = None
            self.X_candidate = None
            self._update_model(fresh_train=False)
            self.timing["search"].append(None)
            self._append_total_wall_time(iteration_start)

    def dump(
        self,
        path: str | Path,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Dump BO run data to a single HDF5 file.

        The file stores plain Python/numeric data derived from the controller,
        not live runtime objects such as the executor, pending future, or
        fitted GP. PyTorch tensors are converted before writing.
        """
        try:
            import tables
        except ImportError as exc:
            raise ImportError("BOController.dump(..., .h5) requires PyTables (`tables`).") from exc

        def _to_builtin(value):
            if isinstance(value, torch.Tensor):
                return _to_builtin(value.detach().cpu().numpy())
            if isinstance(value, np.ndarray):
                return _to_builtin(value.tolist())
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, dict):
                return {str(key): _to_builtin(val) for key, val in value.items()}
            if isinstance(value, (list, tuple)):
                return [_to_builtin(val) for val in value]

            to_dict = getattr(value, "to_dict", None)
            module = getattr(type(value), "__module__", "")
            if callable(to_dict) and module.startswith("pandas"):
                return _to_builtin(to_dict())

            if value is None or isinstance(value, (str, int, float, bool)):
                return value

            return repr(value)

        def _safe_name(name: str, used: set[str]) -> str:
            safe = re.sub(r"\W+", "_", str(name)).strip("_")
            if not safe:
                safe = "item"
            if keyword.iskeyword(safe):
                safe = f"{safe}_"
            if safe[0].isdigit():
                safe = f"n_{safe}"
            base = safe
            idx = 1
            while safe in used:
                safe = f"{base}_{idx}"
                idx += 1
            used.add(safe)
            return safe

        def _string_array(values):
            encoded = [str(value).encode("utf-8") for value in values]
            itemsize = max([len(value) for value in encoded] + [1])
            return np.asarray(encoded, dtype=f"S{itemsize}")

        def _try_create_array(h5, parent, name: str, value):
            if isinstance(value, bool):
                return h5.create_array(parent, name, np.asarray(value, dtype=np.bool_))
            if isinstance(value, int) and not isinstance(value, bool):
                return h5.create_array(parent, name, np.asarray(value, dtype=np.int64))
            if isinstance(value, float):
                return h5.create_array(parent, name, np.asarray(value, dtype=np.float64))
            if isinstance(value, str):
                node = h5.create_array(parent, name, _string_array([value]))
                node._v_attrs["python_type"] = "str"
                node._v_attrs["encoding"] = "utf-8"
                return node
            if isinstance(value, list):
                if not value:
                    return None
                if all(item is None or isinstance(item, (bool, int, float)) for item in value):
                    has_none = any(item is None for item in value)
                    arr = (
                        np.asarray(
                            [np.nan if item is None else item for item in value],
                            dtype=np.float64,
                        )
                        if has_none
                        else np.asarray(value)
                    )
                    node = h5.create_array(parent, name, arr)
                    if has_none:
                        node._v_attrs["none_as_nan"] = True
                    return node
                if all(isinstance(item, str) for item in value):
                    node = h5.create_array(parent, name, _string_array(value))
                    node._v_attrs["python_type"] = "list[str]"
                    node._v_attrs["encoding"] = "utf-8"
                    return node
                try:
                    arr = np.asarray(value)
                except (TypeError, ValueError):
                    return None
                if arr.dtype != object and arr.dtype.kind in "biufc":
                    return h5.create_array(parent, name, arr)
            return None

        def _write_value(h5, parent, name: str, value):
            value = _to_builtin(value)
            node = _try_create_array(h5, parent, name, value)
            if node is not None:
                return node

            group = h5.create_group(parent, name)
            if value is None:
                group._v_attrs["python_type"] = "None"
                return group

            if isinstance(value, dict):
                group._v_attrs["python_type"] = "dict"
                used: set[str] = set()
                for key, child_value in value.items():
                    child_name = _safe_name(key, used)
                    child = _write_value(h5, group, child_name, child_value)
                    child._v_attrs["original_key"] = str(key)
                return group

            if isinstance(value, list):
                group._v_attrs["python_type"] = "list"
                group._v_attrs["length"] = len(value)
                used: set[str] = set()
                width = max(6, len(str(max(len(value) - 1, 0))))
                for idx, child_value in enumerate(value):
                    child_name = _safe_name(f"item_{idx:0{width}d}", used)
                    child = _write_value(h5, group, child_name, child_value)
                    child._v_attrs["list_index"] = idx
                return group

            group._v_attrs["python_type"] = type(value).__name__
            group._v_attrs["repr"] = repr(value)
            return group

        path = Path(path)
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            path = path.with_suffix(".h5")
        path.parent.mkdir(parents=True, exist_ok=True)

        tr_state = {
            key: _to_builtin(value)
            for key, value in vars(self.tr_state).items()
        }

        payload = {
            "format": "stbo.BOController.dump",
            "format_version": 2,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "history": self.history,
            "timing": self.timing,
            "async_history": getattr(self, "async_history", []),
            "adaptive_async_config": getattr(
                self,
                "adaptive_async_config",
                self._DEFAULT_ADAPTIVE_ASYNC_CONFIG,
            ),
            "train_x": self.train_x,
            "train_y": self.train_y,
            "bounds": self.bounds,
            "use_readback": self.use_readback,
            "tr_length_init": getattr(self, "tr_length_init", 0.2),
            "X_last": self.X_last,
            "X_candidate": self.X_candidate,
            "trust_region": tr_state,
            "oracle_history": getattr(self.oracle, "history", None),
            "oracle_evaluator": (
                self.oracle.to_dump_dict()
                if callable(getattr(self.oracle, "to_dump_dict", None))
                else None
            ),
        }

        with tables.open_file(path, mode="w") as h5:
            h5.root._v_attrs["format"] = payload["format"]
            h5.root._v_attrs["format_version"] = payload["format_version"]
            h5.root._v_attrs["created_at"] = payload["created_at"]

            for key in [
                "metadata",
                "history",
                "timing",
                "async_history",
                "adaptive_async_config",
                "train_x",
                "train_y",
                "bounds",
                "use_readback",
                "tr_length_init",
                "X_last",
                "X_candidate",
                "trust_region",
                "oracle_history",
                "oracle_evaluator",
            ]:
                _write_value(h5, h5.root, key, payload[key])

        return path

    @staticmethod
    def read_dump(path: str | Path) -> Dict[str, Any]:
        """Read a BOController HDF5 dump into plain Python data."""
        try:
            import tables
        except ImportError as exc:
            raise ImportError("BOController.read_dump(..., .h5) requires PyTables (`tables`).") from exc

        def _attr(node, name: str, default=None):
            try:
                return getattr(node._v_attrs, name)
            except AttributeError:
                return default

        def _decode(value):
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if isinstance(value, np.bytes_):
                return bytes(value).decode("utf-8")
            return value

        def _nan_to_none(value):
            if isinstance(value, float) and np.isnan(value):
                return None
            if isinstance(value, list):
                return [_nan_to_none(item) for item in value]
            return value

        def _read_value(node):
            if isinstance(node, tables.Group):
                python_type = _attr(node, "python_type")
                if python_type == "None":
                    return None
                if python_type == "list":
                    children = list(node._v_children.values())
                    children.sort(key=lambda child: _attr(child, "list_index", 0))
                    return [_read_value(child) for child in children]

                children = {}
                for child_name, child in node._v_children.items():
                    key = _attr(child, "original_key", child_name)
                    children[str(key)] = _read_value(child)
                if python_type in {None, "dict"}:
                    return children
                if "repr" in node._v_attrs:
                    return _attr(node, "repr")
                return children

            python_type = _attr(node, "python_type")
            data = node.read()

            if python_type == "str":
                return _decode(data.tolist()[0])
            if python_type == "list[str]":
                return [_decode(item) for item in data.tolist()]

            if isinstance(data, np.ndarray) and data.dtype.kind == "S":
                decoded = [_decode(item) for item in data.tolist()]
                return decoded[0] if data.shape == (1,) else decoded

            value = data.tolist() if hasattr(data, "tolist") else data
            if _attr(node, "none_as_nan", False):
                value = _nan_to_none(value)
            return value

        path = Path(path)
        with tables.open_file(path, mode="r") as h5:
            payload: Dict[str, Any] = {
                "format": _attr(h5.root, "format"),
                "format_version": _attr(h5.root, "format_version"),
                "created_at": _attr(h5.root, "created_at"),
            }
            for child_name, child in h5.root._v_children.items():
                payload[child_name] = _read_value(child)

        return payload

    @staticmethod
    def _scalar_objective(value: Any) -> Any:
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    @staticmethod
    def _datetime_or_original(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return value
        if isinstance(value, dict) and value.get("__kind__") == "datetime.datetime":
            try:
                return datetime.fromisoformat(value["value"])
            except (KeyError, ValueError, TypeError):
                return value
        return value

    @classmethod
    def _restore_history_entry(cls, entry: Any) -> Any:
        if not isinstance(entry, dict):
            return entry
        entry = dict(entry)
        if "y" in entry:
            entry["y"] = cls._scalar_objective(entry["y"])
        for key in ("t_start", "t_end"):
            if key in entry:
                entry[key] = cls._datetime_or_original(entry[key])
        return entry

    @staticmethod
    def _tensor_or_none(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        return torch.tensor(value, dtype=torch.float64)

    @staticmethod
    def _tensor_list(values: Any, *, target_dim: Optional[int] = None) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        for value in values or []:
            tensor = torch.tensor(value, dtype=torch.float64)
            if target_dim is not None:
                tensor = tensor.reshape(target_dim)
            tensors.append(tensor)
        return tensors

    def _restore_dump_payload(
        self,
        payload: Dict[str, Any],
        *,
        restore_oracle_history: bool = True,
        fit_model: bool = False,
    ) -> None:
        bounds = payload.get("bounds", self.bounds)
        self.bounds = torch.tensor(bounds, dtype=torch.float64)
        self.use_readback = bool(payload.get("use_readback", self.use_readback))
        self.tr_length_init = float(payload.get("tr_length_init", getattr(self, "tr_length_init", 0.2)))

        history = []
        for entry in payload.get("history", []) or []:
            history.append(self._restore_history_entry(entry))
        self.history = history

        self.timing = {
            str(key): list(value or [])
            for key, value in (payload.get("timing", {}) or {}).items()
        }
        self.async_history = [
            dict(entry)
            for entry in (payload.get("async_history", []) or [])
            if isinstance(entry, dict)
        ]
        self.adaptive_async_config = self._normalize_adaptive_async_config(
            payload.get(
                "adaptive_async_config",
                getattr(self, "adaptive_async_config", None),
            )
        )
        self._oracle_detail_timing_keys = {
            key
            for key in self.timing
            if key not in {"train", "search", "oracle", "total"}
        }

        dim = int(self.bounds.shape[1])
        self.train_x = self._tensor_list(payload.get("train_x", []), target_dim=dim)
        self.train_y = [
            torch.tensor([float(self._scalar_objective(value))], dtype=torch.float64)
            for value in payload.get("train_y", []) or []
        ]

        self.X_last = self._tensor_or_none(payload.get("X_last"))
        self.X_candidate = self._tensor_or_none(payload.get("X_candidate"))
        self.current_future = None
        self.t_submit = None
        self.last_acq_object = None
        self.local_init = False

        tr_state = payload.get("trust_region", {}) or {}
        self.tr_state = TrustRegionState(
            dim=int(tr_state.get("dim", dim)),
            bounds=self.bounds,
            length=float(tr_state.get("length", 0.8)),
            length_min=float(tr_state.get("length_min", 0.5**7)),
            length_max=float(tr_state.get("length_max", 1.0)),
            failure_counter=int(tr_state.get("failure_counter", 0)),
            failure_tolerance=int(tr_state.get("failure_tolerance", 16)),
            success_counter=int(tr_state.get("success_counter", 0)),
            success_tolerance=int(tr_state.get("success_tolerance", 10)),
            best_value=float(tr_state.get("best_value", -float("inf"))),
            restart_triggered=bool(tr_state.get("restart_triggered", False)),
            center=self._tensor_or_none(tr_state.get("center")),
        )

        if restore_oracle_history and self.oracle is not None:
            oracle_bundle = payload.get("oracle_evaluator")
            if oracle_bundle is not None and callable(getattr(self.oracle, "_restore_state_from_dump_dict", None)):
                try:
                    self.oracle._restore_state_from_dump_dict(
                        oracle_bundle.get("state", {}) or {},
                        restore_history=True,
                    )
                except Exception:
                    pass
            elif payload.get("oracle_history") is not None:
                try:
                    self.oracle.history = payload.get("oracle_history")
                except Exception:
                    pass

        if fit_model and self.train_x and self.train_y:
            self._update_model(fresh_train=True, record_timing=False)

    def import_dump(
        self,
        path: str | Path,
        *,
        restore_oracle: bool = False,
        restore_oracle_history: bool = True,
        fit_model: bool = False,
    ) -> Dict[str, Any]:
        """Load a BOController HDF5 dump into this controller.

        By default, the live oracle/executor/GP wrapper owned by this object
        are retained. Historical data, timing, train tensors, bounds, and
        trust-region state are replaced by the dump contents. Set
        restore_oracle=True to replace the oracle from the dump bundle.
        """
        payload = self.read_dump(path)
        if restore_oracle and payload.get("oracle_evaluator") is not None:
            from machineIO.construct_machineIO import OracleEvaluator

            self.oracle = OracleEvaluator.from_dump_dict(
                payload["oracle_evaluator"],
                restore_history=restore_oracle_history,
            )
        self._restore_dump_payload(
            payload,
            restore_oracle_history=restore_oracle_history,
            fit_model=fit_model,
        )
        return payload

    @classmethod
    def from_dump(
        cls,
        path: str | Path,
        oracle_evaluator=None,
        *,
        use_readback: Optional[bool] = None,
        prior_mean: Optional[PriorMean] = None,
        kernel_type: str = "matern",
        adjust_trust_region: bool = True,
        max_workers: int = 1,
        restore_oracle_history: bool = True,
        fit_model: bool = False,
    ) -> "BOController":
        """Construct a BOController from a dump without running an oracle poll."""
        payload = cls.read_dump(path)
        bounds = torch.tensor(payload["bounds"], dtype=torch.float64)
        if oracle_evaluator is None and payload.get("oracle_evaluator") is not None:
            from machineIO.construct_machineIO import OracleEvaluator

            oracle_evaluator = OracleEvaluator.from_dump_dict(
                payload["oracle_evaluator"],
                restore_history=restore_oracle_history,
            )

        bo = cls.__new__(cls)
        bo.oracle = oracle_evaluator
        bo.bounds = bounds
        bo.use_readback = bool(payload.get("use_readback", True) if use_readback is None else use_readback)
        bo.tr_length_init = float(payload.get("tr_length_init", 0.2))
        bo.adjust_trust_region = bool(adjust_trust_region)
        bo.gp = BoTorchGPWrapper(kernel_type=kernel_type, prior_mean=prior_mean)
        bo.executor = ThreadPoolExecutor(max_workers=int(max_workers))
        bo.current_future = None
        bo.X_candidate = None
        bo.train_x = []
        bo.train_y = []
        bo.history = []
        bo.timing = {"train": [], "search": [], "oracle": [], "total": []}
        bo._oracle_detail_timing_keys = set()
        bo.adaptive_async_config = cls._normalize_adaptive_async_config(
            payload.get("adaptive_async_config")
        )
        bo.async_history = []
        bo.X_last = None
        bo.tr_state = TrustRegionState(bounds.shape[1], bounds)
        bo.local_init = False
        bo.last_acq_object = None
        bo.t_submit = None

        bo._restore_dump_payload(
            payload,
            restore_oracle_history=restore_oracle_history,
            fit_model=fit_model,
        )
        return bo

    def _compute_projected_grid(
        self,
        func,
        dim_xaxis: int,
        dim_yaxis: int,
        n_each: int,
        fixed_values: Optional[Dict[int, float]],
        project_mode: str
    ):
        """Helper to compute projected function surface on a 2D grid."""
        bounds_np = self.bounds.cpu().numpy()
        dim = bounds_np.shape[1]
        
        # 1. Define 2D plot grid
        x = np.linspace(bounds_np[0, dim_xaxis], bounds_np[1, dim_xaxis], n_each)
        y = np.linspace(bounds_np[0, dim_yaxis], bounds_np[1, dim_yaxis], n_each)
        XX, YY = np.meshgrid(x, y) # (n, n)

        # 2. Identify projection strategy
        project_mode = project_mode.lower()
        if project_mode in ['max', 'maximum']:
            aggregator = np.nanmax
        elif project_mode in ['min', 'minimum']:
            aggregator = np.nanmin
        elif project_mode in ['mean', 'average']:
            aggregator = np.nanmean
        else:
            raise ValueError(f"Unknown project_mode: {project_mode}")

        # 3. Setup hidden dimension grid
        # If dim=2, there are no hidden dimensions to project over.
        hidden_dims = [d for d in range(dim) if d != dim_xaxis and d != dim_yaxis]
        
        hidden_grids = []
        for d in hidden_dims:
            if fixed_values and d in fixed_values:
                hidden_grids.append(np.array([fixed_values[d]]))
            else:
                hidden_grids.append(np.linspace(bounds_np[0, d], bounds_np[1, d], n_each))
        
        # Pre-construct the hidden mesh (shape: N_hidden x n_hidden_dims)
        if hidden_grids:
            hidden_mesh = np.meshgrid(*hidden_grids, indexing='xy') # cartesian
            hidden_flat = [hm.ravel() for hm in hidden_mesh]
            hidden_tensor = torch.tensor(
                np.column_stack(hidden_flat), 
                dtype=torch.float64, 
                device=self.bounds.device
            )
        else:
            hidden_tensor = torch.zeros((1, 0), dtype=torch.float64, device=self.bounds.device)

        n_hidden_samples = hidden_tensor.shape[0]
        vals_grid = np.zeros((n_each, n_each))

        # 4. Iterate over the 2D plot pixels
        # For each pixel (x, y), we evaluate func on the cartesian product {(x, y)} x {hidden_grid}
        # This mirrors the user's snippet logic of "inner_grid"
        for i in range(n_each):     # rows (Y)
            for j in range(n_each): # cols (X)
                cur_x = XX[i, j]
                cur_y = YY[i, j]

                # Create batch: (n_hidden_samples, dim)
                # We start with empty or hidden tensor and insert x, y
                if dim == 2:
                     batch_x = torch.tensor([[cur_x, cur_y]], dtype=torch.float64, device=self.bounds.device)
                     # Swap if axes are reversed
                     if dim_xaxis == 1: 
                         batch_x = batch_x[:, [1, 0]]
                else:
                    # Construct full tensor
                    # hidden_tensor is (M, D-2). We need (M, D).
                    # We can clone it and insert columns, or build list of columns.
                    cols = [None] * dim
                    cols[dim_xaxis] = torch.full((n_hidden_samples,), cur_x, dtype=torch.float64, device=self.bounds.device)
                    cols[dim_yaxis] = torch.full((n_hidden_samples,), cur_y, dtype=torch.float64, device=self.bounds.device)
                    
                    hidden_idx = 0
                    for d in range(dim):
                        if d != dim_xaxis and d != dim_yaxis:
                            cols[d] = hidden_tensor[:, hidden_idx]
                            hidden_idx += 1
                    
                    batch_x = torch.stack(cols, dim=1)

                # Evaluate
                with torch.no_grad():
                    # model/acq usually expect (batch, q, d). q=1 implies unsqueeze(1).
                    val_batch = func(batch_x.unsqueeze(1))
                    
                    # If func is Posterior mean (from BoTorchGPWrapper), it might return a Tensor (batch, 1) or similar.
                    # BoTorch posterior.mean returns (batch, output_dim). Here output_dim=1.
                    if isinstance(val_batch, torch.Tensor):
                        val_batch = val_batch.detach().cpu().numpy()
                    
                # Aggregate
                vals_grid[i, j] = aggregator(val_batch)

        return XX, YY, vals_grid
        
        
    def plot_model(
        self,
        *,
        X_pending: Optional[torch.Tensor] = None,
        X_candidate: Optional[torch.Tensor] = None,
        search_bounds: Optional[torch.Tensor] = None,
        n_each: int = 16,
        dim_xaxis: int = 0,
        dim_yaxis: int = 1,
        project_mode: str = "max",
        fixed_values: Optional[Dict[int, float]] = None,
        fig=None,
        ax=None,
    ):
        """Plot only the GP model mean surface (1st subplot of plot_acq).

        Parameters
        ----------
        X_pending : torch.Tensor | None
            Pending point to mark on the plot (red ×).
        X_candidate : torch.Tensor | None
            Candidate/next point to mark on the plot (blue ★).
        search_bounds : torch.Tensor | None
            If provided, draws a dashed rectangle showing the search region.
        n_each : int
            Grid resolution along each axis (default 16).
        dim_xaxis : int
            Which parameter dimension to place on the x-axis (default 0).
        dim_yaxis : int
            Which parameter dimension to place on the y-axis (default 1).
        project_mode : str
            How to project hidden dimensions onto the 2D slice.
            One of ``"max"``, ``"min"``, or ``"mean"`` (default ``"max"``).
        fixed_values : dict[int, float] | None
            Pin specific hidden dimensions to a fixed value instead of
            projecting over them.
        fig : matplotlib.figure.Figure | None
            Existing figure to draw on. If None, a new one is created.
        ax : matplotlib.axes.Axes | None
            Existing axes to draw on. If None, a new single-panel figure is
            created.

        Returns
        -------
        (fig, ax)
        """
        if self.gp.model is None:
            return None

        dim = self.bounds.shape[1]
        if dim_xaxis >= dim or dim_yaxis >= dim:
            raise ValueError(f"Axes {dim_xaxis},{dim_yaxis} out of bounds for dim {dim}")

        # Make matplotlib CI-friendly
        try:
            plt.switch_backend('Agg') if not plt.get_backend() else None
        except Exception:
            pass

        # Compute model mean surface
        def func_mean(x):
            return self.gp.posterior(x).mean

        XX, YY, mean_vals = self._compute_projected_grid(
            func_mean, dim_xaxis, dim_yaxis, n_each, fixed_values, project_mode
        )

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        # ---- Plot ----
        c = ax.contourf(XX, YY, mean_vals, levels=16)
        plt.colorbar(c, ax=ax)

        # Scatter training data
        train_x_np = torch.stack(self.train_x).cpu().numpy()
        ax.scatter(
            train_x_np[:, dim_xaxis], train_x_np[:, dim_yaxis],
            c="k", marker=".", s=20, label="Data (Proj)"
        )

        # Scatter pending / candidate points
        if X_pending is not None:
            p = X_pending.cpu().numpy()
            ax.scatter(p[dim_xaxis], p[dim_yaxis], c="r", marker="x", s=100, label="Pending")
        if X_candidate is not None:
            c_ = X_candidate.cpu().numpy()
            ax.scatter(c_[dim_xaxis], c_[dim_yaxis], c="b", marker="*", s=100, label="Candidate")

        # Draw search bounds rectangle
        if search_bounds is not None:
            sb = search_bounds.cpu().numpy()
            x0 = sb[0, dim_xaxis]
            y0 = sb[0, dim_yaxis]
            w  = sb[1, dim_xaxis] - x0
            h  = sb[1, dim_yaxis] - y0
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (x0, y0), w, h,
                linewidth=2, edgecolor='red', facecolor='none',
                linestyle='--', label='Search Region', zorder=10
            )
            ax.add_patch(rect)

        ax.set_title(f"Model Mean ({project_mode}-proj)")
        ax.set_xlabel(f"Dim {dim_xaxis}")
        ax.set_ylabel(f"Dim {dim_yaxis}")
        ax.legend()

        plt.tight_layout()
        return fig, ax

    def plot_model_pairs(
        self,
        dim_names: Optional[List[str]] = None,
        *,
        n_each: int = 16,
        project_mode: str = "max",
        fixed_values: Optional[Dict[int, float]] = None,
        search_bounds: Optional[torch.Tensor] = None,
        X_pending: Optional[torch.Tensor] = None,
        X_candidate: Optional[torch.Tensor] = None,
        fig=None,
        axes=None,
    ):
        """Plot GP model mean for consecutive dimension pairs (x0,x1), (x2,x3), ...

        Parameters
        ----------
        dim_names : list[str] | None
            Names of each input dimension (e.g. control_CSETs).  When provided,
            axis labels are set to the corresponding name; otherwise the default
            ``plot_model`` labels ``Dim N`` are used.
        n_each : int
            Grid resolution per axis (default 16).
        project_mode : str
            Projection mode for hidden dims: ``"max"``, ``"min"``, or ``"mean"``.
        fixed_values : dict[int, float] | None
            Pin specific hidden dimensions to a fixed value.
        search_bounds : torch.Tensor | None
            If provided, draws a dashed search-region rectangle on every panel.
        X_pending : torch.Tensor | None
            Pending point marker (red ×).
        X_candidate : torch.Tensor | None
            Candidate/next point marker (blue ★).
        fig : matplotlib.figure.Figure | None
            Existing figure to draw on.  Created automatically if None.
        axes : array-like of matplotlib.axes.Axes | None
            Existing axes to draw on.  Created automatically if None.

        Returns
        -------
        (fig, axes_flat)  –  figure and flattened axes array.
        """
        import math

        dim = self.bounds.shape[1]
        n_pairs = dim // 2

        if n_pairs == 0:
            raise ValueError("Need at least 2 dimensions to plot pairs.")

        ncol = min(n_pairs, 3)
        nrow = math.ceil(n_pairs / ncol)
        fig_w = 3.5 * ncol
        fig_h = 3.0 * nrow

        if fig is None or axes is None:
            fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h))

        axes_flat = np.array(axes).flatten()

        for i in range(n_pairs):
            ax = axes_flat[i]
            self.plot_model(
                dim_xaxis=2 * i,
                dim_yaxis=2 * i + 1,
                n_each=n_each,
                project_mode=project_mode,
                fixed_values=fixed_values,
                search_bounds=search_bounds,
                X_pending=X_pending,
                X_candidate=X_candidate,
                fig=fig,
                ax=ax,
            )
            if dim_names is not None:
                ax.set_xlabel(dim_names[2 * i])
                ax.set_ylabel(dim_names[2 * i + 1])

        # Hide unused axes when n_pairs < nrow * ncol
        for j in range(n_pairs, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        return fig, axes_flat


    def plot_acq(
        self,
        *,
        X_pending: Optional[torch.Tensor] = None,
        X_candidate: Optional[torch.Tensor] = None,
        search_bounds: Optional[torch.Tensor] = None,
        n_each: int = 16,
        dim_xaxis: int = 0,
        dim_yaxis: int = 1,
        project_mode: str = "max",
        fixed_values: Optional[Dict[int, float]] = None,
        fig=None,
        axes=None,
    ):
        """2D helper plot with projection support for high dimensions.
        
        If problem dim > 2, projects the landscape onto dim_xaxis and dim_yaxis
        using the aggregation strategy defined by project_mode (max, min, mean).
        """
        if self.gp.model is None:
            return None
        
        dim = self.bounds.shape[1]
        if dim_xaxis >= dim or dim_yaxis >= dim:
            raise ValueError(f"Axes {dim_xaxis},{dim_yaxis} out of bounds for dim {dim}")

        # Make matplotlib CI-friendly
        try:
            plt.switch_backend('Agg') if not plt.get_backend() else None
        except Exception:
            pass

        # 1. Compute Grids for Plotting
        # We need three surfaces: Model Mean, Base Acq, Acq+Cost
        
        # A. Model Mean
        def func_mean(x):
            return self.gp.posterior(x).mean

        XX, YY, mean_vals = self._compute_projected_grid(
            func_mean, dim_xaxis, dim_yaxis, n_each, fixed_values, project_mode
        )

        # B. Base Acquisition
        if self.last_acq_object is not None:
             def func_base(x):
                 return self.last_acq_object.base_acq(x)
             
             _, _, base_vals = self._compute_projected_grid(
                func_base, dim_xaxis, dim_yaxis, n_each, fixed_values, project_mode
             )
             
             def func_final(x):
                 return self.last_acq_object(x)
             
             _, _, final_vals = self._compute_projected_grid(
                 func_final, dim_xaxis, dim_yaxis, n_each, fixed_values, project_mode
             )
        else:
            base_vals = None
            final_vals = None

        if fig is None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Helper to plot one panel
        def _plot_panel(ax, Z, title):
            c = ax.contourf(XX, YY, Z, levels=16)
            plt.colorbar(c, ax=ax)
            
            # Scatter Data (Projected)
            train_x_np = torch.stack(self.train_x).cpu().numpy()
            # We scatter the projection of the data onto these two axes
            ax.scatter(train_x_np[:, dim_xaxis], train_x_np[:, dim_yaxis], c="k", marker=".", s=20, label="Data (Proj)")
            
            # Scatter Pending/Candidate
            if X_pending is not None:
                p = X_pending.cpu().numpy()
                ax.scatter(p[dim_xaxis], p[dim_yaxis], c="r", marker="x", s=100, label="Pending")
            if X_candidate is not None:
                c_ = X_candidate.cpu().numpy()
                ax.scatter(c_[dim_xaxis], c_[dim_yaxis], c="b", marker="*", s=100, label="Candidate")
            
            # Draw Search Bounds
            if search_bounds is not None:
                sb = search_bounds.cpu().numpy()
                x0 = sb[0, dim_xaxis]
                y0 = sb[0, dim_yaxis]
                w = sb[1, dim_xaxis] - x0
                h = sb[1, dim_yaxis] - y0
                # Use zorder=10 to ensure it is drawn ON TOP of the contour plot
                rect = Rectangle((x0, y0), w, h, linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Search Region', zorder=10)
                ax.add_patch(rect)

            ax.set_title(f"{title} ({project_mode}-proj)")
            ax.set_xlabel(f"Dim {dim_xaxis}")
            ax.set_ylabel(f"Dim {dim_yaxis}")
            ax.legend()

        _plot_panel(axes[0], mean_vals, "Model Mean")
        
        if base_vals is not None:
            _plot_panel(axes[1], base_vals, "Base Acquisition")
            _plot_panel(axes[2], final_vals, "Acq + Ramping Costs")
        else:
             axes[1].axis('off')
             axes[2].axis('off')

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
