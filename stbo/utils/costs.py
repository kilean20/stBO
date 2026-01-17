from __future__ import annotations

import torch
from typing import Union

__all__ = [
    "get_sigmoid_polarity_penalty",
    "get_knob_excursion_favor",
    "get_pending_point_penalty",
]


def _squeeze_q1(X: torch.Tensor) -> torch.Tensor:
    # BoTorch sometimes passes (batch, q, d) with q=1. We want (batch, d).
    if X.ndim == 3 and X.shape[1] == 1:
        return X.squeeze(1)
    return X


def get_sigmoid_polarity_penalty(
    X: torch.Tensor,
    x_curr: torch.Tensor,
    bounds: torch.Tensor,
    penalty_weight: float = 5.0,
) -> torch.Tensor:
    """Penalty for changing sign ('polarity') relative to current point.

    Returns negative values (penalty) in regions that require polarity flips.
    """
    if penalty_weight == 0:
        return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

    X = _squeeze_q1(X)

    # decay length is 1e-4 of the domain size (per-dimension)
    l_decay = 1e-4 * (bounds[1] - bounds[0])

    curr_sign = torch.sign(x_curr)
    # Mask to ignore dimensions where current value is exactly 0
    active_mask = (x_curr != 0).double()

    # Center shifted by 2*l_decay towards current sign to saturate at 0
    center = 2.0 * l_decay * curr_sign

    dist_into_bad = (center - X) * curr_sign

    val = torch.sigmoid(dist_into_bad / l_decay)

    return -penalty_weight * (val * active_mask).amax(dim=-1)


def get_knob_excursion_favor(
    X: torch.Tensor,
    x_curr: torch.Tensor,
    L_favor: Union[float, torch.Tensor] = 0.5,
    C_favor: float = 5.0,
) -> torch.Tensor:
    """Gaussian-like potential favoring X near x_curr.

    Supports anisotropic L_favor (shape: (d,)).
    """
    if C_favor == 0:
        return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

    X = _squeeze_q1(X)

    diff_sq = (X - x_curr).pow(2)
    # Use max deviation logic (L-inf like) for simultaneous ramping favor
    arg = -(diff_sq / (L_favor**2)).max(dim=-1).values

    return C_favor * torch.exp(arg)


def get_pending_point_penalty(
    X: torch.Tensor,
    X_pending: torch.Tensor | None,
    L_penal: Union[float, torch.Tensor] = 0.1,
    C_penal: float = 2.0,
) -> torch.Tensor:
    """Gaussian repulsion from a pending point.

    Supports anisotropic L_penal (shape: (d,)).
    """
    if X_pending is None or C_penal == 0:
        return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

    X = _squeeze_q1(X)

    dist_sq = (X - X_pending).pow(2)
    arg = -(dist_sq / (L_penal**2)).mean(dim=-1)

    return -C_penal * torch.exp(arg)
