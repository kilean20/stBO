import os
import torch

from stbo import configure

# Ensure plots don't require a display in headless environments
os.environ.setdefault("MPLBACKEND", "Agg")

# Recommended defaults mirroring the prototype
configure(default_dtype=torch.float64, suppress_warnings=True)

ROSENBROCK2D_BOUNDS = torch.tensor([[-2.0, -2.0], [2.0, 3.0]], dtype=torch.float64)
