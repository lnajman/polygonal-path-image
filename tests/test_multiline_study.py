"""Calibration, leakage and comparator checks for the declared experiment."""

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from benchmarks.run_multiline import (  # noqa: E402
    METHODS,
    calibrate,
    calibrate_threshold,
    protocol,
    score_methods,
)


@pytest.mark.parametrize("rate", [0, 0.001, 0.005, 0.01, 0.2, 0.999])
def test_calibration_never_exceeds_background_budget(rate):
    values = np.array([0.0] * 800 + [1.0] * 150 + [2.0] * 40 + [3.0] * 10)
    result = calibrate_threshold(values, rate)
    assert result["selected_background_pixels"] <= int(np.floor(rate * len(values)))
    assert result["achieved_development_rate"] == np.mean(values > result["threshold"])
    assert result["threshold"] >= 0


def test_calibration_preserves_ties_conservatively():
    result = calibrate_threshold(np.array([0, 1, 1, 1, 2]), 0.4)
    assert result["threshold"] == 1
    assert result["selected_background_pixels"] == 1
    assert result["achieved_development_rate"] == 0.2


def test_calibration_exact_order_statistic_without_ties():
    result = calibrate_threshold(np.arange(10), 0.2)
    assert result["threshold"] == 7
    assert result["selected_background_pixels"] == 2


@pytest.mark.parametrize("values", [[], [[1, 2]], [np.nan], [np.inf]])
def test_calibration_rejects_invalid_population(values):
    with pytest.raises(ValueError):
        calibrate_threshold(values, 0.1)


@pytest.mark.parametrize("rate", [-0.01, 1, 2, np.nan, np.inf, True])
def test_calibration_rejects_invalid_rate(rate):
    with pytest.raises(ValueError):
        calibrate_threshold([1, 2, 3], rate)


def test_calibration_rejects_evaluation_before_opening_files(tmp_path):
    with pytest.raises(ValueError, match="development"):
        calibrate([{"split": "evaluation"}], tmp_path)


def test_fixed_protocol_has_no_parameter_selection():
    spec = protocol()
    assert spec["methods"] == list(METHODS)
    assert spec["threshold_selection"]["primary_rate"] == 0.005
    assert spec["frangi"]["gamma"] == 0.02
    assert spec["frangi"]["black_ridges"] is True
    assert spec["ppi"]["segment_length"] == 3
    assert len(spec["manifest"]) == 100


def test_blank_direct_comparators_return_zero_without_normalizing_noise():
    scores, timings = score_methods(np.full((40, 40), 200, dtype=np.uint8))
    assert set(scores) == set(METHODS)
    for method in ("darkness", "local_contrast", "frangi"):
        assert not scores[method].any()
    for method in METHODS:
        assert scores[method].shape == (40, 40)
        assert np.isfinite(scores[method]).all()
        assert (scores[method] >= 0).all()
        assert timings[method]["pipeline_seconds"] >= 0


def test_frangi_detects_declared_dark_ridge_polarity():
    image = np.full((64, 64), 200, dtype=np.uint8)
    image[31:34, 10:54] = 120
    scores, _ = score_methods(image)
    assert scores["frangi"][32, 32] > 0.1
    assert scores["frangi"][32, 32] > scores["frangi"][10, 32]
    assert scores["darkness"][32, 32] == 80
    assert scores["local_contrast"].dtype == np.uint8
