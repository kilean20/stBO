from __future__ import annotations

import math
import numpy as np
import torch
from typing import Dict, Optional, Union, List

from scipy.stats import qmc


def proximal_ordered_init_sampler(
    n_samples: int,
    bounds: torch.Tensor,
    x0: torch.Tensor,
    ramping_rate: torch.Tensor | float | List[float] | None = None,
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
    # Ensure samples are on the same device as bounds
    if bounds.device.type != samples.device.type:
        samples = samples.to(bounds.device)

    samples = bounds[0] + samples * (bounds[1] - bounds[0])

    # 2) Fixed features
    if fixed_features:
        for idx, val in fixed_features.items():
            samples[:, idx] = float(val)

    samples = samples[:n_samples]

    # Handle ramping_rate type conversion
    if ramping_rate is None:
        ramping_rate = 0.1 * (bounds[1] - bounds[0])
    elif isinstance(ramping_rate, (list, tuple)):
        # Convert list to tensor for broadcasting
        ramping_rate = torch.tensor(ramping_rate, dtype=samples.dtype, device=samples.device)
    elif isinstance(ramping_rate, torch.Tensor):
        if ramping_rate.device != samples.device:
            ramping_rate = ramping_rate.to(samples.device)
    
    # 3) Greedy ordering
    ordered = []
    current_x = x0.clone()
    pool = [s for s in samples]

    while pool:
        costs = []
        for cand in pool:
            # ramping_rate is now a Tensor or float, so division is safe
            ramp_time = torch.max(torch.abs(cand - current_x) / ramping_rate).item()
            flip = torch.any((torch.sign(cand) * torch.sign(current_x) < 0))
            pol_time = polarity_change_time if flip else 0.0
            costs.append(max(ramp_time, pol_time))

        best_idx = int(np.argmin(costs))
        best = pool.pop(best_idx)
        ordered.append(best)
        current_x = best

    return torch.stack(ordered)