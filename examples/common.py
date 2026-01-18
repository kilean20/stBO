import os
import sys
from pathlib import Path
import torch

# Path injection: Add the project root to sys.path
# This assumes stbo/ is in the same directory as examples/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stbo import configure

# Ensure plots don't require a display in headless environments
os.environ.setdefault("MPLBACKEND", "Agg")

# Recommended defaults
configure(default_dtype=torch.float64, suppress_warnings=True)

ROSENBROCK2D_BOUNDS = torch.tensor([[-2.0, -1.0], [2.0, 3.0]], dtype=torch.float64)