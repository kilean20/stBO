import os
import sys
from pathlib import Path

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_examples_smoke_run():
    import examples

    names = getattr(examples, "__all__", None) or []
    assert names, "examples.__all__ is empty - no runnable examples discovered."

    for name in names:
        fn = getattr(examples, name)
        assert callable(fn), f"examples.{name} is not callable"
        out = fn(smoke=True)
        assert out is not None, f"examples.{name}(smoke=True) returned None"