"""Analytic blank-path checks; nonzero blank votes are expected behavior."""

import numpy as np
import pytest

import polygonal_path_image as ppi
from benchmarks.diagnose_blank_paths import (
    centered_roi,
    compare_arrays,
    expected_uniform_feasibility,
    make_potential,
    uniform_cost_summary,
    uniform_symmetry_summary,
)


@pytest.mark.parametrize(
    "shape,length,segments", [((9, 9), 1, 3), ((11, 11), 2, 3), ((5, 7), 2, 4)]
)
@pytest.mark.parametrize("intensity", [0, 100, 200, 255])
def test_uniform_paths_match_closed_form_cost_and_feasibility(shape, length, segments, intensity):
    image = np.full(shape, intensity, dtype=np.uint8)
    costs, paths = ppi.compute_ppi(image, length, segments)
    # A cardinal cone must travel L*K rows or columns in one direction.
    expected = np.zeros(shape, dtype=bool)
    for row in range(shape[0]):
        for column in range(shape[1]):
            expected[row, column] = any(
                distance >= length * segments
                for distance in (
                    row,
                    shape[0] - 1 - row,
                    column,
                    shape[1] - 1 - column,
                )
            )
    np.testing.assert_array_equal(expected_uniform_feasibility(shape, length, segments), expected)
    np.testing.assert_array_equal(np.isfinite(costs), expected)
    np.testing.assert_array_equal(
        costs[expected], np.full(expected.sum(), length * segments * intensity)
    )
    assert (paths[~expected] == -1).all()
    summary = uniform_cost_summary(costs, intensity, length, segments)
    assert summary["feasibility_matches"]
    assert summary["all_finite_costs_match"]
    assert summary["maximum_absolute_cost_error"] == (0 if expected.any() else None)


def test_uniform_intensity_changes_costs_but_not_selected_paths_or_votes():
    reference = {}
    for intensity in (0, 100, 200, 255):
        costs, paths = ppi.compute_ppi(make_potential(9, intensity), 1, 3)
        for threshold in (None, 0.5, 0.75):
            filtered = (
                costs if threshold is None else ppi.filter_tortuosity(costs, paths, threshold)
            )
            votes, _ = ppi.voting(filtered, paths)
            if intensity == 0:
                reference[threshold] = paths.copy(), np.isfinite(filtered), votes.copy()
            else:
                old_paths, old_retained, old_votes = reference[threshold]
                np.testing.assert_array_equal(paths, old_paths)
                np.testing.assert_array_equal(np.isfinite(filtered), old_retained)
                np.testing.assert_array_equal(votes, old_votes)
            # Three unit raster steps plus the source contribute four votes.
            assert votes.sum() == 4 * np.isfinite(filtered).sum()
            assert votes.sum() > 0


def test_integer_perturbations_survive_uint8_and_are_deterministic():
    first = make_potential(9, 200, seed=3001)
    repeat = make_potential(9, 200, seed=3001)
    second = make_potential(9, 200, seed=3002)
    np.testing.assert_array_equal(first, repeat)
    assert first.dtype == np.uint8
    assert set(np.unique(first)) == {199, 200, 201}
    assert not np.array_equal(first, second)
    first_costs, first_paths = ppi.compute_ppi(first, 1, 3)
    repeated_costs, repeated_paths = ppi.compute_ppi(repeat, 1, 3)
    np.testing.assert_array_equal(first_costs, repeated_costs)
    np.testing.assert_array_equal(first_paths, repeated_paths)
    for intensity in (0, 255):
        with pytest.raises(ValueError):
            make_potential(9, intensity, seed=3001)


def test_center_roi_stays_aligned_and_symmetry_metrics_do_not_hide_asymmetry():
    roi = centered_roi(9, 5)
    expected = np.zeros((9, 9), dtype=bool)
    expected[2:7, 2:7] = True
    np.testing.assert_array_equal(roi, expected)
    values = np.zeros((9, 9), dtype=np.int64)
    values[3, 4] = 5
    symmetry = uniform_symmetry_summary(values, roi)
    assert symmetry["flip_columns"]["whole_canvas"]["identical"]
    for transform in ("flip_rows", "rotate90"):
        result = symmetry[transform]["center_roi"]
        assert not result["identical"]
        assert result["changed_pixels"] == 2
        assert result["changed_fraction"] == 2 / 25
        assert result["mean_absolute_difference"] == 10 / 25
        assert result["maximum_absolute_difference"] == 5
    with pytest.raises(ValueError):
        compare_arrays(np.zeros((2, 2)), np.zeros((3, 3)))


@pytest.mark.parametrize("size,width", [(8, 5), (5, 9), (9, 0), (True, 1), (9, 3.5)])
def test_center_roi_rejects_ambiguous_alignment_or_invalid_sizes(size, width):
    with pytest.raises(ValueError):
        centered_roi(size, width)
