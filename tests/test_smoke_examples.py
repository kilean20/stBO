import sys
from pathlib import Path

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_examples_smoke_run():
    import examples

    # Discover and run each example's run() function
    names = getattr(examples, "__all__", None) or []
    assert names, "No runnable examples were discovered."

    for name in names:
        fn = getattr(examples, name)
        assert callable(fn), f"examples.{name} is not callable"
        # The run() function now has the path injection internally
        out = fn(smoke=True)
        assert out is not None