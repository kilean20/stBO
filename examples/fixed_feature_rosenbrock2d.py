from __future__ import annotations

from .common import ROSENBROCK2D_BOUNDS as BOUNDS
from stbo.optimization import BOController
from stbo.oracles import RosenbrockOracle


def run(smoke: bool = False) -> float:
    print("=== Example 4: Fixed features (x1 fixed at 1.0) ===")
    oracle = RosenbrockOracle(delay_s=0.0 if smoke else 0.05)
    bo = BOController(oracle, BOUNDS)

    bo.initialize(budget=3 if smoke else 5, local_init=True, fixed_features={1: 1.0}, seed=0)

    n = 3 if smoke else 8
    for _ in range(n):
        bo.step(
            mode="local",
            acq_type="qUCB",
            fixed_features={1: 1.0},
            plot_acq=False,
        )

    bo.finalize()
    best = max(h["y"] for h in bo.history)
    print("Best y:", best)

    xs = [h.get("x_set", h["x"])[1] for h in bo.history if "x" in h]
    print("Observed x1 range:", min(xs), max(xs))
    return float(best)


if __name__ == "__main__":
    run(smoke=False)
