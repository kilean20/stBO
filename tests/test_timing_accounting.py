from contextlib import ExitStack
from datetime import datetime, timedelta
from unittest.mock import patch

import torch

from stbo.optimization.controller import BOController


class _DoneFuture:
    def result(self):
        return {"x": [0.0], "x_rd": [0.0], "y": 0.0}


class _TrustRegion:
    def update(self, *_args, **_kwargs):
        return None


def _minimal_controller():
    bo = object.__new__(BOController)
    bo.current_future = _DoneFuture()
    bo.t_submit = 100.0
    bo.timing = {"train": [], "search": [], "oracle": [], "total": []}
    bo.train_x = [torch.tensor([0.0], dtype=torch.float64)]
    bo.train_y = [torch.tensor([0.0], dtype=torch.float64)]
    bo.bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    bo.X_candidate = torch.tensor([0.25], dtype=torch.float64)
    bo.X_last = torch.tensor([0.0], dtype=torch.float64)
    bo.gp = object()
    bo.last_acq_object = None
    bo._register_data = lambda _res: None
    bo._update_model = lambda fresh_train=True: None
    bo._submit_job = lambda _x: setattr(bo, "t_submit", 110.0)
    return bo


def test_step_total_is_elapsed_wall_time_for_sync_and_async():
    for asynchro in (False, True):
        bo = _minimal_controller()

        with ExitStack() as stack:
            stack.enter_context(
                patch("stbo.optimization.controller.get_base_acq", return_value=object())
            )
            stack.enter_context(
                patch(
                    "stbo.optimization.controller.RampingCostAwareAcquisition",
                    return_value=object(),
                )
            )
            stack.enter_context(
                patch(
                    "stbo.optimization.controller.optimize_acqf",
                    return_value=(torch.tensor([[0.5]], dtype=torch.float64), None),
                )
            )
            stack.enter_context(
                patch("stbo.optimization.controller.time.time", side_effect=[105.0, 107.0])
            )
            BOController.step(bo, asynchro=asynchro, ramp_cost_config={})

        assert bo.timing["search"] == [2.0]
        assert bo.timing["total"] == [10.0]


def test_total_wall_time_helper_records_elapsed_time():
    bo = object.__new__(BOController)
    bo.timing = {"total": []}

    with patch("stbo.optimization.controller.time.time", return_value=12.5):
        bo._append_total_wall_time(10.0)

    assert bo.timing["total"] == [2.5]


def test_oracle_detail_timing_is_collected_and_aligned():
    bo = object.__new__(BOController)
    bo.timing = {"train": [], "search": [], "oracle": [], "total": []}
    bo._oracle_detail_timing_keys = set()
    bo.t_submit = 1.0
    bo.use_readback = True
    bo.train_x = []
    bo.train_y = []
    bo.history = []
    bo.tr_state = _TrustRegion()
    bo.adjust_trust_region = True

    t0 = datetime(2026, 1, 1, 0, 0, 0)
    BOController._register_data(
        bo,
        {
            "x": [0.0],
            "x_rd": [0.0],
            "y": 1.0,
            "t_start": t0,
            "t_end": t0 + timedelta(seconds=4.0),
            "timing": {"oracle_ensure_set": 1.2, "oracle_read": 2.3},
        },
    )
    BOController._register_data(
        bo,
        {
            "x": [0.0],
            "x_rd": [0.0],
            "y": 1.0,
            "t_start": t0,
            "t_end": t0 + timedelta(seconds=3.0),
            "timing": {"oracle_read": 2.1, "oracle_df_manipulators": 0.05},
        },
    )

    assert bo.timing["oracle"] == [4.0, 3.0]
    assert bo.timing["oracle_ensure_set"] == [1.2, None]
    assert bo.timing["oracle_read"] == [2.3, 2.1]
    assert bo.timing["oracle_df_manipulators"] == [None, 0.05]
