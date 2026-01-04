from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base import get_base_acq
from ..utils.costs import (
    get_sigmoid_polarity_penalty,
    get_knob_excursion_favor,
    get_pending_point_penalty,
)


class RampingCostAwareAcquisition:
    """Wrap a BoTorch acquisition function and add physical cost terms."""

    def __init__(
        self,
        model_wrapper,
        acq_type: str,
        X_last: torch.Tensor,
        bounds: torch.Tensor,
        *,
        X_pending: Optional[torch.Tensor] = None,
        acq_config: Optional[Dict[str, Any]] = None,
        ramp_cost_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_wrapper = model_wrapper
        # Use wrapper as model so acq calls wrapper.posterior()
        self.botorch_model = model_wrapper

        self.acq_type = acq_type
        self.X_current = X_pending if X_pending is not None else X_last
        self.bounds = bounds
        self.acq_config = acq_config or {}
        self.ramp_cost_config = ramp_cost_config or {}

        self.is_mc = acq_type.startswith("q")
        self.base_acq = get_base_acq(
            self.botorch_model, acq_type, X_pending=X_pending, acq_config=acq_config
        )

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        base_val = self.base_acq(X)

        X_s = X.squeeze(1)  # (batch, d)

        pol_pen = get_sigmoid_polarity_penalty(
            X_s,
            self.X_current,
            self.bounds,
            penalty_weight=float(self.ramp_cost_config.get("polarity_penalty", 0.0) or 0.0),
        )

        fav_score = 0.0
        if bool(self.ramp_cost_config.get("use_ramping_favor", False)):
            fav_score = get_knob_excursion_favor(
                X_s,
                self.X_current,
                L_favor=self.ramp_cost_config.get("L_favor", 0.5),
                C_favor=float(self.ramp_cost_config.get("C_favor", 0.0) or 0.0),
            )

        pend_pen = 0.0
        if (not self.is_mc) and bool(self.ramp_cost_config.get("penalize_pending", True)):
            pend_pen = get_pending_point_penalty(
                X_s,
                self.X_current.reshape(1, -1),
                L_penal=self.ramp_cost_config.get("L_penal", 0.1),
                C_penal=float(self.ramp_cost_config.get("C_penal", 0.0) or 0.0),
            )

        return base_val + pol_pen + fav_score + pend_pen
