from __future__ import annotations

from typing import Any, Dict, Optional

from botorch.acquisition import (
    UpperConfidenceBound,
    LogExpectedImprovement,
    qUpperConfidenceBound,
    qLogExpectedImprovement,
)


def get_base_acq(
    botorch_model,
    acq_type: str,
    *,
    X_pending=None,
    acq_config: Optional[Dict[str, Any]] = None,
):
    """Factory for standard BoTorch acquisition functions."""
    acq_config = acq_config or {}
    best_f = acq_config.get("best_f", -1.0)
    beta = acq_config.get("beta", 4.0)

    if acq_type == "Mean":
        return UpperConfidenceBound(botorch_model, beta=0.0)
    if acq_type == "UCB":
        return UpperConfidenceBound(botorch_model, beta=float(beta))
    if acq_type == "LogEI":
        return LogExpectedImprovement(botorch_model, best_f=best_f)
    if acq_type == "qUCB":
        return qUpperConfidenceBound(botorch_model, beta=float(beta), X_pending=X_pending)
    if acq_type == "qLogEI":
        return qLogExpectedImprovement(botorch_model, best_f=best_f, X_pending=X_pending)

    raise ValueError(f"Unknown acq_type={acq_type!r}.")
