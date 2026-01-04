from __future__ import annotations

import time
import numpy as np


class RosenbrockOracle:
    """A simple oracle with optional readback noise and delay."""

    def __init__(self, *, delay_s: float = 0.05, noise_std: float = 0.001):
        self.x = np.array([0.0, 0.0], dtype=float)
        self.delay_s = float(delay_s)
        self.noise_std = float(noise_std)

    def __call__(self, x=None):
        if x is not None:
            self.x = np.asarray(x, dtype=float)
        if self.delay_s > 0:
            time.sleep(self.delay_s)

        val = -((1.0 - self.x[0]) ** 2 + 100.0 * (self.x[1] - self.x[0] ** 2) ** 2)
        return {
            "x": self.x,
            "x_rd": self.x + np.random.normal(0, self.noise_std, size=2),
            "y": float(val),
        }
