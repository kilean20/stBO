from __future__ import annotations

import math
import numpy as np
import torch
from typing import Dict, Optional

from scipy.stats import qmc


def proximal_ordered_init_sampler(
    n_samples: int,
    bounds: torch.Tensor,
    x0: torch.Tensor,
    ramping_rate: torch.Tensor | float | None = None,
    polarity_change_time: float = 15.0,
    fixed_features: Optional[Dict[int, float]] = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Sobol-initialize, optionally fix dimensions, then greedily order by motion cost."""
    d = bounds.shape[1]

    # 1) Sobol samples in [0,1]^d
    sampler = qmc.Sobol(d=d, seed=seed)
    m = int(np.ceil(np.log2(n_samples)))
    raw = sampler.random(2**m)

    samples = torch.tensor(raw, dtype=torch.float64)
    samples = bounds[0] + samples * (bounds[1] - bounds[0])

    # 2) Fixed features
    if fixed_features:
        for idx, val in fixed_features.items():
            samples[:, idx] = float(val)

    samples = samples[:n_samples]

    if ramping_rate is None:
        ramping_rate = 0.1 * (bounds[1] - bounds[0])

    # 3) Greedy ordering
    ordered = []
    current_x = x0.clone()
    pool = [s for s in samples]

    while pool:
        costs = []
        for cand in pool:
            ramp_time = torch.max(torch.abs(cand - current_x) / ramping_rate).item()
            flip = torch.any((torch.sign(cand) * torch.sign(current_x) < 0))
            pol_time = polarity_change_time if flip else 0.0
            costs.append(max(ramp_time, pol_time))

        best_idx = int(np.argmin(costs))
        best = pool.pop(best_idx)
        ordered.append(best)
        current_x = best

    return torch.stack(ordered)
