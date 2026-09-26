"""Check the study figures' observation grain without external data."""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from benchmarks.plot_guide3d import acquisition_values, macro_summary  # noqa: E402


def _row(acquisition, value, *, budget=0.01, method="darkness"):
    return {
        "acquisition_id": acquisition,
        "method": method,
        "budget_fraction": budget,
        "metrics": {"centerline_coverage": value},
        "selection": {"actual_fraction": budget},
    }


def test_unequal_acquisition_sizes_receive_equal_weight():
    # Two zero-coverage views versus one fully covered view: averaging the
    # three frames would yield 1/3, whereas the two acquisitions average 1/2.
    rows = [_row("a", 0), _row("a", 0), _row("b", 1)]
    # Other operating points/methods must not enter the primary comparison.
    rows.extend([_row("a", 1, budget=0.02), _row("a", 1, method="local_contrast")])
    assert acquisition_values(rows, "darkness", "centerline_coverage") == {"a": 0, "b": 1}
    assert macro_summary(rows, "darkness", "centerline_coverage") == {
        "mean": 0.5,
        "minimum": 0,
        "maximum": 1,
        "valid_acquisitions": 2,
        "total_acquisitions": 2,
    }


@pytest.mark.parametrize("missing", [None, np.nan, np.inf])
def test_partial_acquisition_does_not_silently_drop_missing_views(missing):
    rows = [_row("a", missing), _row("a", 0), _row("b", 1)]
    values = acquisition_values(rows, "darkness", "centerline_coverage")
    assert np.isnan(values["a"])
    assert values["b"] == 1
    summary = macro_summary(rows, "darkness", "centerline_coverage")
    assert summary["mean"] == 1
    assert summary["valid_acquisitions"] == 1
    assert summary["total_acquisitions"] == 2


def test_all_missing_acquisitions_remain_undefined():
    rows = [_row("a", None), _row("b", None)]
    summary = macro_summary(rows, "darkness", "centerline_coverage")
    assert all(np.isnan(summary[key]) for key in ("mean", "minimum", "maximum"))
    assert summary["valid_acquisitions"] == 0
    assert summary["total_acquisitions"] == 2
