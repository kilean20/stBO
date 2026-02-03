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



def get_base_acq(
    botorch_model,
    acq_type: str,
    *,
    X_pending=None,
    acq_config: Optional[Dict[str, Any]] = None,
):
    """
    Factory for standard BoTorch acquisition functions.
    Handles case-insensitivity and standardizes inputs (e.g. EI -> LogEI).
    """
    acq_config = acq_config or {}
    
    # --- 1. Normalize & Case Insensitive Mapping ---
    acq_type = acq_type.lower()
    
    if acq_type == "mean":
        canonical_acq = "Mean"
    elif acq_type == "ucb":
        canonical_acq = "UCB"
    elif acq_type in ["ei", "logei"]:
        canonical_acq = "LogEI"
    elif acq_type == "qucb":
        canonical_acq = "qUCB"
    elif acq_type in ["qei", "qlogei"]:
        canonical_acq = "qLogEI"
    else:
        raise ValueError(f"Unknown acq_type={acq_type!r}.")

    # --- 2. Validate X_pending ---
    # If X_pending is present, we enforce batch (q) acquisition functions
    if X_pending is not None:
        if canonical_acq == 'LogEI':
            canonical_acq = 'qLogEI'
        elif canonical_acq == 'UCB':
            canonical_acq = 'qUCB'
            
    # --- 3. Configuration ---
    best_f = acq_config.get("best_f", -1.0)
    beta = acq_config.get("beta", 4.0)

    # --- 4. Instantiation ---
    if canonical_acq == "Mean":
        return UpperConfidenceBound(botorch_model, beta=0.0)
    
    if canonical_acq == "UCB":
        return UpperConfidenceBound(botorch_model, beta=float(beta))
    
    if canonical_acq == "LogEI":
        return LogExpectedImprovement(botorch_model, best_f=best_f)
    
    if canonical_acq == "qUCB":
        return qUpperConfidenceBound(botorch_model, beta=float(beta), X_pending=X_pending)
    
    if canonical_acq == "qLogEI":
        return qLogExpectedImprovement(botorch_model, best_f=best_f, X_pending=X_pending)
        
    # Should be unreachable due to the else block in step 1, 
    # but good for safety.
    raise ValueError(f"Unknown canonical acq_type={canonical_acq!r}.")