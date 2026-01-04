import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure repo root (which contains ./examples) is importable in CI
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_examples_smoke_run():
    from examples import example1, example2, example3, example4

    assert example1(smoke=True) is not None
    assert example2(smoke=True) is not None
    assert example3(smoke=True) is not None
    assert example4(smoke=True) is not None
