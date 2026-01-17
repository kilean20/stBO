from __future__ import annotations

import warnings
import torch


def configure(
    *,
    default_dtype: torch.dtype | None = torch.float64,
    suppress_warnings: bool = True,
) -> None:
    """Configure common numeric defaults.

    This library's original prototype used float64 and muted warnings for stability.
    We don't force global state on import; call this from your application / examples.
    """
    if suppress_warnings:
        warnings.filterwarnings("ignore")

    if default_dtype is not None:
        torch.set_default_dtype(default_dtype)
