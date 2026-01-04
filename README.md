# stBO (singleTaskBO)

`stBO` is a small, opinionated wrapper around **BoTorch** to run **single-task Bayesian Optimization**
with optional *physics-aware* ramping/polarity/pending-point cost terms and a simple TuRBO-1-style trust region.

## Install (editable)
```bash
pip install -e .[dev]
```

## Quick start
```python
import torch
from stbo.optimization import BOController
from stbo.oracles import RosenbrockOracle

bounds = torch.tensor([[-2.0, -2.0], [2.0, 3.0]], dtype=torch.float64)

bo = BOController(RosenbrockOracle(delay_s=0.0), bounds)
bo.initialize(budget=5)
for _ in range(10):
    bo.step(mode="global", acq_type="qUCB", plot_acq=False)
bo.finalize()

best = max([h["y"] for h in bo.history])
print("Best y:", best)
```

## Examples
See `examples/` for four runnable scripts mirroring the original demos.

- `examples/01_global_then_qucb.py`
- `examples/02_prior_mean.py`
- `examples/03_local_then_finetune.py`
- `examples/04_fixed_features.py`

## Development (tests + lint)
```bash
pytest
ruff check .
```

## Notes
- The import package is `stbo` (lowercase) which is standard in Python.
  The distribution / project name is `stBO` as requested.
