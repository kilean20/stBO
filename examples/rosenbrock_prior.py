import sys
from pathlib import Path
import torch

# Path injection: Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# FIX: Import PriorMean (the correct class name) instead of GPPriorMean
from stbo.models import PriorMean

class RosenbrockPrior(PriorMean):
    """A prior mean model for the Rosenbrock function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the negative Rosenbrock value (since we maximize)
        # x is assumed to be (batch_shape) x q x d
        # Output should be (batch_shape) x q x 1
        
        # Rosenbrock function: (1 - x)^2 + 100 * (y - x^2)^2
        val = -((1.0 - x[..., 0])**2 + 100.0 * (x[..., 1] - x[..., 0]**2)**2)
        
        # Ensure correct output shape (add output dimension)
        return val.unsqueeze(-1)