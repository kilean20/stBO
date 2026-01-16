from __future__ import annotations

from .common import ROSENBROCK2D_BOUNDS as BOUNDS
from stbo.optimization import BOController
from stbo.oracles import RosenbrockOracle
import matplotlib.pyplot as plt


def run(smoke: bool = False, show_plots: bool = False) -> float:
    show_plots = False if smoke else show_plots
    print("=== Example 4: Resilient feature: x1 is fixed for every n iteration")
    oracle = RosenbrockOracle(delay_s=0.0 if smoke else 1.0)
    bo = BOController(oracle, BOUNDS, adjust_trust_region=False)

    bo.initialize(budget=3 if smoke else 5, local_init=False, fixed_features={1: 0.0}, seed=0)
    bo.initialize(budget=3 if smoke else 5, local_init=False, fixed_features={1: 0.5}, seed=0)
    bo.initialize(budget=3 if smoke else 5, local_init=False, fixed_features={1: 1.0}, seed=0)

    n = 2 if smoke else 5
    for i in range(n):
        bo.step(
            mode="local",
            acq_type="qUCB",
            plot_acq=show_plots,
            )
        fixed_val = bo.X_candidate[1]
        for _ in range(2):
            bo.step(
                mode="local",
                acq_type="qUCB",
                fixed_features={1: fixed_val},
                plot_acq=show_plots,
            )
            if show_plots:
                plt.show()

    bo.finalize()
    best = max(h["y"] for h in bo.history)
    print("Best y:", best)

    xs = [h.get("x_set", h["x"])[1] for h in bo.history if "x" in h]
    print("Observed x1 range:", min(xs), max(xs))

    if show_plots: 
        bo.plot_history()
        plt.show()

    return bo


if __name__ == "__main__":
    run(smoke=False)
