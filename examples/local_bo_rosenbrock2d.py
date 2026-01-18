from __future__ import annotations
import sys
from pathlib import Path

# Path injection
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

import common
from stbo.optimization import BOController
from stbo.oracles import RosenbrockOracle
import matplotlib.pyplot as plt

def run(smoke: bool = False, show_plots: bool = False) -> float:
    show_plots = False if smoke else show_plots
    print("=== Example 3: Local init -> Local BO -> FineTune ===")
    oracle = RosenbrockOracle(delay_s=0.0 if smoke else 1.0)
    
    bo = BOController(oracle, common.ROSENBROCK2D_BOUNDS)

    bo.initialize(budget=3 if smoke else 5, local_init=True, ramping_rate=0.2, seed=0)
    # ... rest of the logic
    bo.finalize()
    return bo

if __name__ == "__main__":
    run(smoke=False)