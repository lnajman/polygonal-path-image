"""Analytic/toy checks for the benchmark, independent of PPI's implementation."""

import numpy as np
import pytest

from benchmarks.synthetic import (
    evaluate_prediction,
    generate_case,
    point_to_polyline_distance,
    select_at_budget,
)


def test_exact_segment_distances_include_projection_and_endpoint_clamping():
    reference = np.array([[0, 0], [0, 4]], dtype=float)
    points = np.array([[3, 2], [0, 7], [-3, -4], [0, 1.37]])
    expected = [3, 3, 5, 0]
    np.testing.assert_allclose(
        point_to_polyline_distance(points, reference, batch_size=1), expected
    )
    np.testing.assert_allclose(
        point_to_polyline_distance(points, reference, batch_size=256), expected
    )


def test_exact_diagonal_and_multisegment_distances():
    reference = np.array([[0, 0], [4, 4], [4, 8]])
    points = np.array([[2, 0], [5, 6], [4, 10]])
    np.testing.assert_allclose(point_to_polyline_distance(points, reference), [np.sqrt(2), 1, 2])


def test_degenerate_and_empty_reference_distances():
    points = np.array([[3, 4], [0, 0]])
    for reference in (np.array([[0, 0]]), np.array([[0, 0], [0, 0]])):
        np.testing.assert_array_equal(point_to_polyline_distance(points, reference), [5, 0])
    assert np.isinf(point_to_polyline_distance(points, np.empty((0, 2)))).all()
    assert point_to_polyline_distance(np.empty((0, 2)), points).shape == (0,)


def test_budget_includes_entire_tie_block_and_is_transpose_invariant():
    scores = np.array([[10, 5, 0], [5, 5, 1]], dtype=float)
    roi = np.ones(scores.shape, dtype=bool)
    selected, info = select_at_budget(scores, roi, fraction=0.25)
    np.testing.assert_array_equal(selected, [[True, True, False], [True, True, False]])
    assert info["budget_pixels"] == 2
    assert info["selected_pixels"] == 4
    assert info["boundary_tie_pixels"] == 3
    assert info["budget_exceeded_by_ties"] == 2
    assert info["actual_fraction"] == pytest.approx(4 / 6)
    transposed, _ = select_at_budget(scores.T, roi.T, fraction=0.25)
    np.testing.assert_array_equal(selected.T, transposed)


def test_budget_only_includes_positive_finite_roi_scores():
    scores = np.array([[np.inf, np.nan, -2, 0, 7, 9]], dtype=float)
    roi = np.array([[True, True, True, True, True, False]])
    selected, info = select_at_budget(scores, roi, fraction=1)
    np.testing.assert_array_equal(selected, [[False, False, False, False, True, False]])
    assert info["positive_pixels"] == 1
    assert info["threshold"] == 7
    assert info["actual_fraction"] == 0.2


def test_empty_budget_and_empty_roi_have_explicit_results():
    scores = np.ones((2, 3))
    for roi, fraction in [(np.ones_like(scores, dtype=bool), 0), (scores == 0, 0.02)]:
        selected, info = select_at_budget(scores, roi, fraction=fraction)
        assert not selected.any()
        assert info["threshold"] is None
        assert info["actual_fraction"] == (0 if roi.any() else None)


def test_known_precision_false_positive_denominator_and_directional_distances():
    reference = np.array([[2, 0], [2, 4]])
    roi = np.ones((5, 5), dtype=bool)
    prediction = np.zeros_like(roi)
    prediction[2, :] = True
    prediction[0, 0] = True
    result = evaluate_prediction(prediction, reference, roi, tolerance=0.5)
    assert result["selected_pixels"] == 6
    assert result["selected_pixel_precision"] == pytest.approx(5 / 6)
    assert result["off_tube_pixels"] == 20
    assert result["false_positive_pixels"] == 1
    assert result["off_tube_false_positive_rate"] == pytest.approx(1 / 20)
    assert result["centerline_coverage"] == 1
    assert result["prediction_to_centerline_mean_distance"] == pytest.approx(2 / 6)
    # Uniform .25 arclength samples, trapezoidal endpoint weights: distances
    # to integer-column predictions repeat 0, .25, .5, .25, giving mean .25.
    assert result["centerline_to_prediction_mean_distance"] == pytest.approx(0.25)
    assert result["centerline_to_prediction_p95_distance"] == 0.5
    assert result["prediction_to_centerline_p95_distance"] == 2


def test_coverage_and_miss_distance_use_arclength_not_vertex_counts():
    roi = np.ones((3, 5), dtype=bool)
    prediction = np.zeros_like(roi)
    prediction[1, 0] = True
    reference = np.array([[1, 0], [1, 0.01], [1, 0.02], [1, 4]])
    result = evaluate_prediction(prediction, reference, roi, tolerance=1)
    # 17 samples over length4; qualifying samples0,.25,.5,.75,1 have total
    # trapezoidal weight1.125, including the half-weight startpoint.
    assert result["centerline_coverage"] == pytest.approx(1.125 / 4)
    assert result["centerline_to_prediction_mean_distance"] == pytest.approx(2)
    assert result["prediction_to_centerline_mean_distance"] == 0


def test_no_predictions_has_zero_coverage_but_null_precision_and_distances():
    roi = np.ones((5, 5), dtype=bool)
    result = evaluate_prediction(np.zeros_like(roi), np.array([[2, 0], [2, 4]]), roi)
    assert result["centerline_coverage"] == 0
    assert result["selected_pixel_precision"] is None
    assert result["prediction_to_centerline_mean_distance"] is None
    assert result["centerline_to_prediction_mean_distance"] is None
    # Tolerance2 covers the complete ROI, so there is no off-tube denominator.
    assert result["off_tube_false_positive_rate"] is None


def test_no_reference_has_null_reference_metrics_and_all_pixels_are_off_tube():
    roi = np.ones((2, 2), dtype=bool)
    prediction = np.array([[True, False], [False, False]])
    result = evaluate_prediction(prediction, np.empty((0, 2)), roi)
    assert not result["has_target"]
    assert result["centerline_coverage"] is None
    assert result["selected_pixel_precision"] is None
    assert result["centerline_to_prediction_mean_distance"] is None
    assert result["prediction_to_centerline_mean_distance"] is None
    assert result["off_tube_false_positive_rate"] == 0.25


def test_predictions_and_reference_samples_outside_roi_are_excluded():
    roi = np.zeros((5, 5), dtype=bool)
    roi[2, 2] = True
    prediction = np.ones_like(roi)
    result = evaluate_prediction(prediction, np.array([[2, 2]]), roi)
    assert result["selected_pixels"] == 1
    assert result["centerline_samples"] == 1
    assert result["centerline_coverage"] == 1
    assert result["centerline_to_prediction_mean_distance"] == 0
    empty = evaluate_prediction(prediction, np.array([[2, 2]]), np.zeros_like(roi))
    assert empty["selected_fraction"] is None
    assert empty["centerline_coverage"] is None
    assert empty["off_tube_false_positive_rate"] is None


@pytest.mark.parametrize("angle", [0, 22.5, 45, 90, 135])
@pytest.mark.parametrize("bend", [0, 45, 90])
def test_generator_preserves_geometry_and_safe_roi(angle, bend):
    case = generate_case(angle_deg=angle, bend_deg=bend, noise_std=0)
    assert case.image.shape == (96, 96)
    assert case.image.dtype == np.uint8
    assert case.roi.sum() == 64 * 64
    lengths = np.linalg.norm(np.diff(case.centerline, axis=0), axis=1)
    np.testing.assert_allclose(lengths, lengths[0], rtol=1e-11)
    assert lengths.max() <= 0.25 + 1e-12
    assert lengths.sum() == pytest.approx(48, abs=0.001)
    np.testing.assert_allclose(
        (case.centerline.min(axis=0) + case.centerline.max(axis=0)) / 2, [47.5, 47.5]
    )
    assert case.centerline.min() >= 18
    assert case.centerline.max() <= 77
    tangent = case.centerline[97] - case.centerline[95]
    np.testing.assert_allclose(
        tangent / np.linalg.norm(tangent),
        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))],
        atol=1e-12,
    )
    if bend:
        # Circle circumradius from three vertices, independently of generator.
        a, b, c = case.centerline[[0, 96, -1]]
        side1, side2, side3 = np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)
        delta1, delta2 = b - a, c - a
        twice_area = abs(delta1[0] * delta2[1] - delta1[1] * delta2[0])
        radius = side1 * side2 * side3 / (2 * twice_area)
        assert radius == pytest.approx(48 / np.deg2rad(bend), rel=1e-12)


def test_generator_is_seeded_reproducible_and_has_transpose_symmetric_clean_geometry():
    first = generate_case(seed=2012)
    second = generate_case(seed=2012)
    np.testing.assert_array_equal(first.image, second.image)
    assert first.parameters == second.parameters
    assert not np.array_equal(first.image, generate_case(seed=2013).image)
    horizontal = generate_case(angle_deg=0, noise_std=0)
    vertical = generate_case(angle_deg=90, noise_std=0)
    np.testing.assert_array_equal(horizontal.image.T, vertical.image)


def test_generator_zero_contrast_zero_noise_is_constant():
    case = generate_case(contrast=0, noise_std=0)
    np.testing.assert_array_equal(case.image, np.full((96, 96), 200, dtype=np.uint8))


def test_no_target_control_has_empty_reference_and_reuses_noise_field():
    control = generate_case(target_present=False, seed=5, size=128, roi_margin=32)
    zero_contrast = generate_case(contrast=0, seed=5, size=128, roi_margin=32)
    np.testing.assert_array_equal(control.image, zero_contrast.image)
    assert control.centerline.shape == (0, 2)
    assert np.isinf(control.distance_map).all()
    assert control.parameters["target_present"] is False


def test_cached_distances_produce_identical_metrics():
    case = generate_case(noise_std=0)
    selected, _ = select_at_budget(255 - case.image, case.roi)
    uncached = evaluate_prediction(selected, case.centerline, case.roi)
    cached = evaluate_prediction(
        selected, case.centerline, case.roi, distance_map=case.distance_map
    )
    assert uncached == cached


@pytest.mark.parametrize(
    "kwargs", [{"sigma": 0}, {"noise_std": -1}, {"contrast": 201}, {"sample_step": 1}, {"size": 12}]
)
def test_invalid_generator_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_case(**kwargs)


@pytest.mark.parametrize("fraction", [-1, 2, float("nan")])
def test_invalid_selection_budget_is_rejected(fraction):
    with pytest.raises(ValueError):
        select_at_budget(np.ones((2, 2)), np.ones((2, 2), dtype=bool), fraction=fraction)
