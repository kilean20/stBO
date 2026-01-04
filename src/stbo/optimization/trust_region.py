from __future__ import annotations

import math
from typing import Optional

import torch


class TrustRegionState:
    """TuRBO-1 style trust region state for maximization."""

    def __init__(
        self,
        dim: int,
        bounds: torch.Tensor,
        *,
        batch_size: int = 1,
        success_tolerance: int = 3,
        length_init: float = 0.1,
        length_min: float = 0.005,
        length_max: float = 1.0,
    ):
        self.dim = int(dim)
        self.bounds = bounds
        self.batch_size = int(batch_size)

        self.success_tolerance = int(success_tolerance)
        self.failure_tolerance = int(
            math.ceil(max(4.0 / self.batch_size, float(self.dim) / self.batch_size))
        )

        self.length = float(length_init)
        self.length_min = float(length_min)
        self.length_max = float(length_max)

        self.center: Optional[torch.Tensor] = None
        self.success_counter = 0
        self.failure_counter = 0
        self.best_value = -float("inf")

    def update(self, y_new: float, x_new: torch.Tensor) -> None:
        improved = True if not math.isfinite(self.best_value) else (
            y_new > self.best_value + 1e-3 * abs(self.best_value) + 1e-9
        )

        if improved:
            self.success_counter += 1
            self.failure_counter = 0
            self.best_value = float(y_new)
            self.center = x_new
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter >= self.success_tolerance:
            self.length = min(self.length * 2.0, self.length_max)
            self.success_counter = 0
        elif self.failure_counter >= self.failure_tolerance:
            self.length = max(self.length / 2.0, self.length_min)
            self.failure_counter = 0

    def get_bounds(self) -> torch.Tensor:
        if self.center is None:
            return self.bounds
        span = self.bounds[1] - self.bounds[0]
        L = self.length * span
        lower = torch.max(self.bounds[0], self.center - L / 2)
        upper = torch.min(self.bounds[1], self.center + L / 2)
        return torch.stack([lower, upper])
