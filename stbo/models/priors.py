from __future__ import annotations

import torch


class PriorMean(torch.nn.Module):
    """Base class for prior mean models used as an additive mean function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
