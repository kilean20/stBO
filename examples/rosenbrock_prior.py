import torch
from stbo.models import PriorMean


class RosenbrockPrior(PriorMean):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val = -((1.0 - x[..., 0]) ** 2 + 100.0 * (x[..., 1] - x[..., 0] ** 2) ** 2)
        return val.unsqueeze(-1)
