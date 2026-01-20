import sys
from pathlib import Path
import torch

# Path injection: Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# FIX: Import the correct class 'PriorMean' (not GPPriorMean)
from stbo.models import PriorMean

class RosenbrockPrior(PriorMean):
    """A prior mean model for the Rosenbrock function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_shape) x q x d
        # Rosenbrock global max is at (1, 1, ...) with value 0. 
        # The function usually represents cost, so we negate it for maximization.
        
        # Calculation: -( (1 - x1)^2 + 100 * (x2 - x1^2)^2 )
        # Dimensions are collapsed on the last axis (d), resulting in (batch_shape) x q
        val = -((1.0 - x[..., 0])**2 + 100.0 * (x[..., 1] - x[..., 0]**2)**2)
        
        return val