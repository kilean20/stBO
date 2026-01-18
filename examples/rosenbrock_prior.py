import sys
from pathlib import Path
import torch

# Path injection: Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stbo.models import GPPriorMean

class RosenbrockPrior(GPPriorMean):
    """A prior mean model for the Rosenbrock function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Example implementation logic
        return -((1.0 - x[..., 0])**2 + 100.0 * (x[..., 1] - x[..., 0]**2)**2)