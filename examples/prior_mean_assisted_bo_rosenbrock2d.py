from __future__ import annotations
import sys
from pathlib import Path

# Ensure root and examples directory are in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(ROOT / "examples"))

import common 
# Import the class from the local module (now fixed)
from rosenbrock_prior import RosenbrockPrior
from stbo.optimization import BOController
from stbo.oracles import RosenbrockOracle
import matplotlib.pyplot as plt


def run(smoke: bool = False, show_plots: bool = False) -> float:
    show_plots = False if smoke else show_plots
    print("=== Example 2: Prior mean model ===")
    oracle = RosenbrockOracle(delay_s=0.0 if smoke else 1.0)
    
    # Use common.ROSENBROCK2D_BOUNDS and the corrected Prior
    bo = BOController(oracle, common.ROSENBROCK2D_BOUNDS, prior_mean=RosenbrockPrior())

    bo.initialize(budget=3 if smoke else 5, seed=0)
    n = 3 if smoke else 10
    for _ in range(n):
        bo.step(mode="global", acq_type="qUCB", acq_config={"beta": 4}, fresh_train=True, plot_acq=show_plots)
        if show_plots:
            plt.show()

    bo.finalize()

    best = max(h["y"] for h in bo.history)
    print("Best y:", best)

    if show_plots: 
        bo.plot_history()
        plt.show()

    return bo


if __name__ == "__main__":
    run(smoke=False)