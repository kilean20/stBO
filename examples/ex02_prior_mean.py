from __future__ import annotations

from .common import BOUNDS
from .rosenbrock_prior import RosenbrockPrior
from stbo.optimization import BOController
from stbo.oracles import RosenbrockOracle


def run(smoke: bool = False) -> float:
    print("=== Example 2: Prior mean model ===")
    oracle = RosenbrockOracle(delay_s=0.0 if smoke else 0.05)
    bo = BOController(oracle, BOUNDS, prior_mean=RosenbrockPrior())

    bo.initialize(budget=3 if smoke else 5, seed=0)
    n = 3 if smoke else 10
    for _ in range(n):
        bo.step(mode="global", acq_type="qUCB", acq_config={"beta": 4}, fresh_train=True, plot_acq=False)

    bo.finalize()
    best = max(h["y"] for h in bo.history)
    print("Best y:", best)
    return float(best)


if __name__ == "__main__":
    run(smoke=False)
