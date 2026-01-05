from __future__ import annotations

from .common import ROSENBROCK2D_BOUNDS as BOUNDS
from stbo.optimization import BOController
from stbo.oracles import RosenbrockOracle


def run(smoke: bool = False) -> float:
    print("=== Example 3: Local init -> Local BO -> FineTune ===")
    oracle = RosenbrockOracle(delay_s=0.0 if smoke else 0.05)
    bo = BOController(oracle, BOUNDS)

    bo.initialize(budget=3 if smoke else 5, local_init=True, ramping_rate=0.2, seed=0)
    n1 = 2 if smoke else 5
    n2 = 2 if smoke else 5

    for _ in range(n1):
        bo.step(mode="local", acq_type="UCB", plot_acq=False)
    for _ in range(n2):
        bo.step(mode="local", acq_type="qUCB", plot_acq=False)

    bo.step(mode="finetune", acq_type="qUCB", plot_acq=False)
    bo.finalize()

    best = max(h["y"] for h in bo.history)
    print("Best y:", best)
    return float(best)


if __name__ == "__main__":
    run(smoke=False)
