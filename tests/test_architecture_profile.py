"""Timing projection must preserve the registered data/batch/epoch/run budgets."""

import math

import pytest

from roof_pipeline.profile_architectures import project_runtime, timing_summary


def test_projection_counts_partial_batches_and_every_seed():
    result = project_runtime(
        {"median": 0.1, "p90": 0.2},
        {"median": 0.05, "p90": 0.1},
        {"batch_size": 4, "epochs": 40},
        {"train": 4736, "val": 1036},
        [42],
    )
    assert result["train_batches_per_epoch"] == 1184
    assert result["validation_batches_per_epoch"] == 259
    assert result["epoch_seconds"]["lower"] == pytest.approx((118.4 + 12.95) * 1.15)
    assert result["epoch_seconds"]["upper"] == pytest.approx((236.8 + 25.9) * 1.75)
    assert result["run_seconds"]["lower"] == pytest.approx(
        result["epoch_seconds"]["lower"] * 40 + 300
    )
    multiple = project_runtime(
        {"median": 0.1, "p90": 0.2},
        {"median": 0.05, "p90": 0.1},
        {"batch_size": 4, "epochs": 40},
        {"train": 4737, "val": 1037},
        [42, 1337, 2026],
    )
    assert multiple["train_batches_per_epoch"] == 1185
    assert multiple["validation_batches_per_epoch"] == 260
    assert (
        multiple["all_seeds_seconds"]["upper"] == 3 * multiple["run_seconds"]["upper"]
    )


@pytest.mark.parametrize("values", [[], [0], [-1], [math.nan], [math.inf]])
def test_bad_timing_is_rejected(values):
    with pytest.raises(ValueError, match="finite and positive"):
        timing_summary(values)


def test_p90_and_median_do_not_hide_slow_observations():
    summary = timing_summary([1.0] * 8 + [2.0, 5.0])
    assert summary == {"median": 1.0, "p90": 2.0, "min": 1.0, "max": 5.0}
