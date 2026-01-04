import os
os.environ.setdefault("MPLBACKEND", "Agg")


def test_examples_smoke_run():
    from examples import example1, example2, example3, example4

    assert example1(smoke=True) is not None
    assert example2(smoke=True) is not None
    assert example3(smoke=True) is not None
    assert example4(smoke=True) is not None
