"""Differential and analytic tests of the scalable acquired-image evaluator."""

import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from benchmarks.guide3d_metrics import (  # noqa: E402
    evaluate_prediction,
    prepare_reference,
    select_at_budget,
)
from benchmarks.synthetic import (  # noqa: E402
    evaluate_prediction as reference_evaluate,
)
from benchmarks.synthetic import (  # noqa: E402
    point_to_polyline_distance,
)
from benchmarks.synthetic import (
    select_at_budget as reference_select,
)


def assert_equivalent(actual, expected):
    assert actual.keys() == expected.keys() | {
        "evaluated_reference_length",
        "reference_length_fraction",
    }
    for key, value in expected.items():
        if isinstance(value, float):
            assert actual[key] == pytest.approx(value, rel=1e-14, abs=1e-14), key
        else:
            assert actual[key] == value, key
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("tolerance", [0.0, 0.5, 2.0])
def test_kdtree_matches_pairwise_search_for_random_curves_and_irregular_rois(seed, tolerance):
    rng = np.random.default_rng(seed)
    roi = rng.random((17, 23)) > 0.2
    prediction = rng.random(roi.shape) > 0.87
    centerline = rng.uniform([-3, -5], [20, 28], size=(8, 2))
    centerline[3] = centerline[2]  # A repeated vertex must not alter weighting.
    actual = evaluate_prediction(prediction, centerline, roi, tolerance=tolerance)
    expected = reference_evaluate(prediction, centerline, roi, tolerance=tolerance)
    assert_equivalent(actual, expected)


@pytest.mark.parametrize(
    "centerline",
    [
        np.empty((0, 2)),
        np.array([[2.0, 2.0]]),
        np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]),
        np.array([[-20.0, -20.0], [-10.0, -10.0]]),
        np.array([[-2.0, 2.0], [7.0, 2.0]]),
    ],
)
@pytest.mark.parametrize("prediction_present", [False, True])
@pytest.mark.parametrize("roi_present", [False, True])
def test_empty_null_degenerate_and_clipped_reference_conventions(
    centerline, prediction_present, roi_present
):
    roi = np.full((5, 5), roi_present, dtype=bool)
    prediction = np.full_like(roi, prediction_present)
    actual = evaluate_prediction(prediction, centerline, roi)
    expected = reference_evaluate(prediction, centerline, roi)
    assert_equivalent(actual, expected)


def test_exact_tolerance_boundary_and_nonuniform_vertices_use_arclength_weights():
    roi = np.ones((3, 5), dtype=bool)
    prediction = np.zeros_like(roi)
    prediction[1, 0] = True
    centerline = np.array([[1, 0], [1, 0.01], [1, 0.02], [1, 4]])
    actual = evaluate_prediction(prediction, centerline, roi, tolerance=1)
    assert actual["centerline_coverage"] == pytest.approx(1.125 / 4)
    assert actual["centerline_to_prediction_mean_distance"] == 2
    assert actual["centerline_to_prediction_p95_distance"] == 3.75
    assert actual["prediction_to_centerline_p95_distance"] == 0
    assert_equivalent(actual, reference_evaluate(prediction, centerline, roi, tolerance=1))


def test_half_pixel_roi_membership_and_holes_match_original_weights():
    roi = np.ones((7, 9), dtype=bool)
    roi[:, :2] = False
    roi[3, 4] = False
    centerline = np.array([[2.5, -1], [2.5, 10]])
    prediction = np.zeros_like(roi)
    prediction[3, 2] = True
    prediction[2, 8] = True
    geometry = prepare_reference(centerline, roi)
    # Row 2.5 rounds upward to row3, whose column4 has a hole. Clipping is
    # sample membership, not subtraction of exact segment/ROI intersection.
    assert not np.any((geometry.samples[:, 1] >= 3.5) & (geometry.samples[:, 1] < 4.5))
    for tolerance in [0, 0.5, 1, 2, 4]:
        assert_equivalent(
            geometry.evaluate(prediction, tolerance=tolerance),
            reference_evaluate(prediction, centerline, roi, tolerance=tolerance),
        )


def test_evaluated_reference_length_reports_clipped_sample_weight():
    roi = np.zeros((3, 5), dtype=bool)
    roi[1, 1:4] = True
    reference = np.array([[1, 0], [1, 4]])
    result = evaluate_prediction(roi, reference, roi)
    # Nearest-pixel ROI membership retains .5 <= column < 3.5: twelve
    # interior samples with weight .25, not the full four-pixel polyline.
    assert result["centerline_length"] == 4
    assert result["evaluated_reference_length"] == 3
    assert result["reference_length_fraction"] == 0.75
    full = evaluate_prediction(roi, reference, np.ones_like(roi))
    assert full["evaluated_reference_length"] == 4
    assert full["reference_length_fraction"] == 1
    empty = evaluate_prediction(roi, reference, np.zeros_like(roi))
    assert empty["evaluated_reference_length"] == 0
    assert empty["reference_length_fraction"] == 0


@pytest.mark.parametrize("reference", [[], [[1, 2]], [[1, 2], [1, 2]]])
def test_empty_and_point_references_have_zero_length_and_null_length_fraction(reference):
    roi = np.ones((3, 5), dtype=bool)
    result = evaluate_prediction(roi, np.array(reference).reshape(-1, 2), roi)
    assert result["evaluated_reference_length"] == 0
    assert result["reference_length_fraction"] is None
    if reference:
        assert result["centerline_coverage"] == 1


def test_cached_reference_map_and_reused_geometry_preserve_exact_measurements():
    roi = np.ones((11, 13), dtype=bool)
    roi[[0, -1]] = False
    centerline = np.array([[1.25, 2], [8.5, 5.75], [8.5, 5.75], [2, 11]])
    pixels = np.indices(roi.shape).reshape(2, -1).T
    distance_map = point_to_polyline_distance(pixels, centerline).reshape(roi.shape)
    geometry = prepare_reference(centerline, roi, distance_map=distance_map)
    for fraction in [0, 0.005, 0.02, 0.5, 1]:
        scores = np.maximum(10 - distance_map, 0)
        prediction, selection = select_at_budget(scores, roi, fraction=fraction)
        expected_prediction, expected_selection = reference_select(scores, roi, fraction=fraction)
        assert selection == expected_selection
        np.testing.assert_array_equal(prediction, expected_prediction)
        expected = reference_evaluate(prediction, centerline, roi, distance_map=distance_map)
        assert_equivalent(geometry.evaluate(prediction), expected)
        assert_equivalent(
            evaluate_prediction(prediction, centerline, roi, distance_map=distance_map), expected
        )


def test_cache_is_not_corrupted_by_mutating_source_arrays():
    roi = np.ones((4, 4), dtype=bool)
    centerline = np.array([[1, 0], [1, 3]], dtype=float)
    distances = np.abs(np.arange(4)[:, None] - 1) * np.ones((4, 4))
    geometry = prepare_reference(centerline, roi, distance_map=distances)
    prediction = np.eye(4, dtype=bool)
    original = geometry.evaluate(prediction)
    roi[:] = False
    centerline[:] = 100
    distances[:] = 100
    assert geometry.evaluate(prediction) == original
    for array in [
        geometry.roi,
        geometry.pixels,
        geometry.distances,
        geometry.samples,
        geometry.weights,
    ]:
        assert not array.flags.writeable


@pytest.mark.parametrize("sample_step", [0.1, 0.2, 0.25])
def test_sample_spacing_retains_p95_and_endpoint_weights(sample_step):
    roi = np.ones((9, 11), dtype=bool)
    prediction = np.zeros_like(roi)
    prediction[[1, 7, 8], [1, 8, 8]] = True
    centerline = np.array([[1.5, 0], [1.5, 0.01], [6.3, 6], [6.3, 6], [8, 9.7]])
    assert_equivalent(
        evaluate_prediction(prediction, centerline, roi, sample_step=sample_step),
        reference_evaluate(prediction, centerline, roi, sample_step=sample_step),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prediction": np.ones((4, 4))},
        {"prediction": np.ones(4, dtype=bool)},
        {"roi": np.ones((4, 4))},
        {"roi": np.ones((3, 4), dtype=bool)},
        {"tolerance": -1},
        {"tolerance": np.nan},
        {"sample_step": 0},
        {"sample_step": 0.5},
        {"sample_step": np.inf},
        {"centerline": [[1, np.nan]]},
        {"centerline": [1, 2]},
        {"distance_map": np.ones((3, 4))},
        {"distance_map": np.full((4, 4), np.nan)},
        {"distance_map": np.full((4, 4), -1.0)},
    ],
)
def test_invalid_inputs_rejected_consistently(kwargs):
    arguments = {
        "prediction": np.ones((4, 4), dtype=bool),
        "centerline": np.array([[1, 1], [1, 3]]),
        "roi": np.ones((4, 4), dtype=bool),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        reference_evaluate(**arguments)
    with pytest.raises(ValueError):
        evaluate_prediction(**arguments)
