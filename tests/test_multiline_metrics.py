"""Analytic union, gap and complete-ground-truth checks for multi-line scenes."""

import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from benchmarks.multiline_metrics import prepare_multiline, select_at_budget  # noqa: E402
from benchmarks.synthetic import evaluate_prediction  # noqa: E402


def test_crossing_lines_each_receive_coverage_but_crossing_pixel_counts_once():
    roi = np.ones((7, 7), dtype=bool)
    lines = [np.array([[3, 1], [3, 5]]), np.array([[1, 3], [5, 3]])]
    prediction = np.zeros_like(roi)
    prediction[3, 1:6] = True
    prediction[1:6, 3] = True
    result = prepare_multiline(lines, [0.1, 0.1], roi, tolerance=0.5).evaluate(prediction)
    assert result["line_count"] == result["evaluated_line_count"] == 2
    assert result["selected_pixels"] == result["support_pixels"] == 9
    assert result["macro_coverage"] == result["worst_line_coverage"] == 1
    assert result["union_centerline_precision"] == result["selected_support_precision"] == 1
    assert result["outside_support_fraction"] == 0
    assert result["prediction_to_reference_mean_distance"] == 0
    assert result["prediction_to_reference_p95_distance"] == 0
    assert [line["longest_uncovered_arclength"] for line in result["per_line"]] == [0, 0]
    assert [line["reference_to_prediction_p95_distance"] for line in result["per_line"]] == [
        0.5,
        0.5,
    ]


def test_disconnected_lines_never_gain_an_artificial_connecting_segment():
    roi = np.ones((9, 9), dtype=bool)
    lines = [np.array([[2, 0], [2, 2]]), np.array([[6, 6], [6, 8]])]
    prediction = np.zeros_like(roi)
    prediction[4, 4] = True  # Would lie on a connector between concatenated lines.
    result = prepare_multiline(lines, [0.1, 0.1], roi, tolerance=0.5).evaluate(prediction)
    assert result["macro_coverage"] == result["worst_line_coverage"] == 0
    assert result["union_centerline_precision"] == result["selected_support_precision"] == 0
    assert result["selected_outside_support_pixels"] == 1
    assert result["prediction_to_reference_mean_distance"] == pytest.approx(np.sqrt(8))
    assert [line["longest_uncovered_arclength"] for line in result["per_line"]] == [2, 2]


def test_wide_line_support_uses_width_without_relaxing_centerline_tolerance():
    roi = np.ones((11, 7), dtype=bool)
    line = np.array([[5, 0], [5, 6]])
    prediction = np.zeros_like(roi)
    prediction[8, 2] = True  # Width-aware support boundary; beyond tolerance1.
    prediction[10, 2] = True  # Outside declared3sigma support.
    result = prepare_multiline([line], [1], roi, tolerance=1).evaluate(prediction)
    assert result["support_pixels"] == 7 * 7
    assert result["outside_support_pixels"] == 4 * 7
    assert result["selected_support_pixels"] == result["selected_outside_support_pixels"] == 1
    assert result["selected_support_precision"] == 0.5
    assert result["union_centerline_precision"] == 0
    assert result["outside_support_fraction"] == pytest.approx(1 / 28)
    assert result["prediction_to_reference_mean_distance"] == 4
    assert result["prediction_to_reference_p95_distance"] == 5


def test_support_uses_each_lines_own_sigma_before_union():
    roi = np.ones((15, 11), dtype=bool)
    lines = [np.array([[3, 0], [3, 10]]), np.array([[11, 0], [11, 10]])]
    prediction = np.zeros_like(roi)
    prediction[5, 5] = True  # Two pixels from narrow line: outside3*.25.
    prediction[13, 5] = True  # Two pixels from wide line: inside3*1.
    result = prepare_multiline(lines, [0.25, 1], roi, tolerance=0.5).evaluate(prediction)
    assert result["selected_support_precision"] == 0.5
    assert result["selected_outside_support_pixels"] == 1


def test_longest_uncovered_run_uses_arclength_quadrature_not_vertex_count():
    roi = np.ones((5, 9), dtype=bool)
    line = np.array([[2, 0], [2, 0.01], [2, 0.02], [2, 8]])
    prediction = np.zeros_like(roi)
    prediction[2, [0, 1, 5, 6, 7, 8]] = True
    result = prepare_multiline([line], [0.5], roi, tolerance=0.5).evaluate(prediction)
    per_line = result["per_line"][0]
    # Uncovered samples1.75..4.25 span eleven quarter-pixel weights.
    assert per_line["longest_uncovered_arclength"] == 2.75
    assert per_line["coverage"] == pytest.approx((8 - 2.75) / 8)
    assert per_line["centerline_length"] == per_line["evaluated_reference_length"] == 8


def test_roi_holes_break_uncovered_runs_and_do_not_connect_excluded_samples():
    roi = np.ones((3, 9), dtype=bool)
    roi[:, 4] = False
    line = np.array([[1, 0], [1, 8]])
    prediction = np.zeros_like(roi)
    result = prepare_multiline([line], [0.25], roi).evaluate(prediction)
    per_line = result["per_line"][0]
    assert per_line["coverage"] == 0
    assert per_line["evaluated_reference_length"] == 7
    # Left run0..3.25 weighs3.375; right run4.5..8 weighs3.625.
    assert per_line["longest_uncovered_arclength"] == 3.625
    assert per_line["reference_length_fraction"] == 7 / 8


def test_macro_and_worst_coverage_weight_lines_equally_instead_of_length():
    roi = np.ones((8, 13), dtype=bool)
    lines = [np.array([[1, 0], [1, 12]]), np.array([[6, 0], [6, 2]])]
    prediction = np.zeros_like(roi)
    prediction[1] = True
    result = prepare_multiline(lines, [0.1, 0.1], roi, tolerance=0.5).evaluate(prediction)
    assert [line["coverage"] for line in result["per_line"]] == [1, 0]
    assert result["macro_coverage"] == 0.5
    assert result["worst_line_coverage"] == 0


@pytest.mark.parametrize("selected", [False, True])
def test_blank_controls_have_null_reference_metrics_and_roi_background_denominator(selected):
    roi = np.ones((4, 5), dtype=bool)
    prediction = np.zeros_like(roi)
    prediction[0, 0] = selected
    result = prepare_multiline([], [], roi).evaluate(prediction)
    assert result["per_line"] == []
    assert not result["has_target"]
    assert result["macro_coverage"] is None
    assert result["worst_line_coverage"] is None
    assert result["union_centerline_precision"] is None
    assert result["selected_support_precision"] == (0 if selected else None)
    assert result["prediction_to_reference_mean_distance"] is None
    assert result["prediction_to_reference_p95_distance"] is None
    assert result["outside_support_fraction"] == int(selected) / 20
    json.dumps(result, allow_nan=False)


def test_empty_predictions_have_zero_coverage_full_gap_and_null_distances():
    roi = np.ones((5, 7), dtype=bool)
    line = np.array([[2, 1], [2, 5]])
    result = prepare_multiline([line], [0.5], roi).evaluate(np.zeros_like(roi))
    assert result["macro_coverage"] == result["worst_line_coverage"] == 0
    assert result["per_line"][0]["longest_uncovered_arclength"] == 4
    assert result["per_line"][0]["reference_to_prediction_mean_distance"] is None
    assert result["per_line"][0]["reference_to_prediction_p95_distance"] is None
    assert result["selected_support_precision"] is None
    assert result["outside_support_fraction"] == 0


def test_empty_roi_and_outside_references_do_not_create_nonfinite_json():
    roi = np.zeros((5, 7), dtype=bool)
    lines = [np.array([[2, 1], [2, 5]]), np.empty((0, 2))]
    result = prepare_multiline(lines, [0.5, 0.5], roi).evaluate(np.ones_like(roi))
    assert result["selected_fraction"] is None
    assert result["outside_support_fraction"] is None
    assert result["macro_coverage"] is None
    assert all(line["coverage"] is None for line in result["per_line"])
    assert all(line["longest_uncovered_arclength"] is None for line in result["per_line"])
    json.dumps(result, allow_nan=False)


def test_zero_length_repeated_reference_has_unit_coverage_weight_but_zero_gap_length():
    roi = np.ones((5, 5), dtype=bool)
    line = np.array([[2, 2], [2, 2], [2, 2]])
    geometry = prepare_multiline([line], [0.5], roi, tolerance=0)
    missing = geometry.evaluate(np.zeros_like(roi))["per_line"][0]
    assert missing["coverage"] == missing["longest_uncovered_arclength"] == 0
    assert missing["evaluated_reference_length"] == 0
    assert missing["reference_length_fraction"] is None
    prediction = np.zeros_like(roi)
    prediction[2, 2] = True
    recovered = geometry.evaluate(prediction)["per_line"][0]
    assert recovered["coverage"] == 1
    assert recovered["longest_uncovered_arclength"] == 0


@pytest.mark.parametrize("seed", range(4))
def test_one_line_matches_existing_exact_evaluator(seed):
    rng = np.random.default_rng(seed)
    roi = rng.random((13, 17)) > 0.15
    line = rng.uniform([-2, -2], [15, 19], size=(5, 2))
    prediction = rng.random(roi.shape) > 0.9
    result = prepare_multiline([line], [0.5], roi).evaluate(prediction)
    original = evaluate_prediction(prediction, line, roi)
    per_line = result["per_line"][0]
    mapping = {
        "macro_coverage": "centerline_coverage",
        "worst_line_coverage": "centerline_coverage",
        "union_centerline_precision": "selected_pixel_precision",
        "prediction_to_reference_mean_distance": "prediction_to_centerline_mean_distance",
        "prediction_to_reference_p95_distance": "prediction_to_centerline_p95_distance",
    }
    for key, expected_key in mapping.items():
        assert result[key] == pytest.approx(original[expected_key], abs=1e-14)
    assert per_line["reference_to_prediction_mean_distance"] == pytest.approx(
        original["centerline_to_prediction_mean_distance"], abs=1e-14
    )
    assert per_line["reference_to_prediction_p95_distance"] == pytest.approx(
        original["centerline_to_prediction_p95_distance"], abs=1e-14
    )
    json.dumps(result, allow_nan=False)


def test_cache_owns_geometry_and_selection_retains_whole_ties():
    roi = np.ones((5, 5), dtype=bool)
    line = np.array([[2, 0], [2, 4]], dtype=float)
    sigmas = np.array([0.5])
    geometry = prepare_multiline([line], sigmas, roi)
    prediction, selection = select_at_budget(np.ones_like(roi, dtype=float), roi, fraction=0.01)
    assert selection["selected_pixels"] == 25
    assert selection["budget_exceeded_by_ties"] == 24
    original = geometry.evaluate(prediction)
    roi[:] = False
    line[:] = 100
    sigmas[:] = 10
    assert geometry.evaluate(prediction) == original
    for array in (geometry.roi, geometry.pixels, geometry.union_distances, geometry.support):
        assert not array.flags.writeable


def test_public_calibration_masks_are_full_shape_disjoint_and_cover_only_roi():
    roi = np.ones((7, 9), dtype=bool)
    roi[:, 0] = False
    line = np.array([[3, 0], [3, 8]])
    geometry = prepare_multiline([line], [0.5], roi)
    support, outside = geometry.support_mask, geometry.outside_support_mask
    assert support.shape == outside.shape == geometry.union_distance_map.shape == roi.shape
    assert not (support & outside).any()
    np.testing.assert_array_equal(support | outside, roi)
    assert not support[:, 0].any() and not outside[:, 0].any()
    assert np.isinf(geometry.union_distance_map[:, 0]).all()
    assert geometry.union_distance_map[3, 4] == 0
    assert geometry.union_distance_map[0, 4] == 3
    for array in (support, outside, geometry.union_distance_map):
        assert not array.flags.writeable
    blank = prepare_multiline([], [], roi)
    assert not blank.support_mask.any()
    np.testing.assert_array_equal(blank.outside_support_mask, roi)


@pytest.mark.parametrize(
    "arguments",
    [
        {"sigmas": []},
        {"sigmas": [0]},
        {"sigmas": [-1]},
        {"sigmas": [np.nan]},
        {"sigmas": [np.inf]},
        {"sigmas": 1},
        {"centerlines": [np.array([[1, np.nan]])]},
        {"centerlines": [np.array([1, 2])]},
        {"roi": np.ones((5, 5))},
        {"roi": np.ones(5, dtype=bool)},
        {"tolerance": -1},
        {"tolerance": np.inf},
        {"sample_step": 0},
        {"sample_step": 0.5},
        {"support_sigma": 0},
        {"support_sigma": np.nan},
    ],
)
def test_invalid_geometry_inputs_are_rejected(arguments):
    kwargs = {
        "centerlines": [np.array([[2, 0], [2, 4]])],
        "sigmas": [0.5],
        "roi": np.ones((5, 5), dtype=bool),
    }
    kwargs.update(arguments)
    with pytest.raises(ValueError):
        prepare_multiline(**kwargs)


@pytest.mark.parametrize(
    "prediction", [np.ones((5, 5)), np.ones((4, 5), dtype=bool), np.ones(5, dtype=bool)]
)
def test_invalid_predictions_are_rejected(prediction):
    geometry = prepare_multiline([], [], np.ones((5, 5), dtype=bool))
    with pytest.raises(ValueError):
        geometry.evaluate(prediction)
