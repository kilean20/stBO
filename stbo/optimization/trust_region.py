from __future__ import annotations
import math
import torch
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrustRegionState:
    dim: int
    bounds: torch.Tensor  # Global bounds (2, d)
    # length is the fraction of the domain to search (0.0 to 1.0)
    length: float = 0.8   
    length_min: float = 0.5**7
    length_max: float = 1.0 # <--- CHANGED: Max 100% of domain
    failure_counter: int = 0
    failure_tolerance: int = 16
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False
    center: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Ensure bounds are on the correct device/dtype
        self.bounds = self.bounds.to(dtype=torch.float64)

    def update(self, y_new: float, x_new: torch.Tensor, adjust_trust_region: bool = True):
        """Update the trust region state based on the new observation."""
        if x_new is None:
            return

        # Check for improvement (using a small epsilon relative to best value)
        # Note: We assume Maximization.
        if y_new > self.best_value + 1e-3 * math.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
            self.best_value = y_new
            self.center = x_new.clone()
        else:
            self.success_counter = 0
            self.failure_counter += 1
            # If we haven't established a center yet, use the current point
            if self.center is None:
                self.center = x_new.clone()

        if adjust_trust_region:
            if self.success_counter >= self.success_tolerance:
                self.length = min(2.0 * self.length, self.length_max)
                self.success_counter = 0
            elif self.failure_counter >= self.failure_tolerance:
                self.length /= 2.0
                self.failure_counter = 0

            if self.length < self.length_min:
                self.restart_triggered = True
                # Prevent collapse if restarts are not handled externally
                self.length = self.length_min * 2.0

    def get_bounds(self) -> torch.Tensor:
        """
        Calculate the trust region bounds in the original (unnormalized) space.
        Scaling: bounds_width = length * global_domain_span
        """
        if self.center is None:
            return self.bounds

        # 1. Calculate the span of the global domain
        domain_span = self.bounds[1] - self.bounds[0]

        # 2. Scale the normalized length to the physical domain
        # radius is half the side length
        physical_radius = (self.length * domain_span) / 2.0

        # 3. Compute raw bounds centered at self.center
        lb = self.center - physical_radius
        ub = self.center + physical_radius

        # 4. Intersect with global bounds to ensure validity
        lb = torch.max(lb, self.bounds[0])
        ub = torch.min(ub, self.bounds[1])

        return torch.stack([lb, ub])