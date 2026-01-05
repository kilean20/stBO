import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure repo root (which contains ./examples) is importable in CI
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Also ensure src-layout package import works without installation (helpful locally).
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_examples_smoke_run():
    import examples

    # Prefer the explicit export list from examples/__init__.py
    names = getattr(examples, "__all__", None) or []
    assert names, "examples.__all__ is empty - no runnable examples were discovered."

    for name in names:
        fn = getattr(examples, name)
        assert callable(fn), f"examples.{name} is not callable"
        out = fn(smoke=True)
        assert out is not None, f"examples.{name}(smoke=True) returned None"
