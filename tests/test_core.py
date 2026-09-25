"""Correctness, compatibility, and public input-contract checks for PPI."""

import numpy as np
import pytest

from polygonal_path_image import bresenham_line, compute_ppi

from .reference import assert_valid_paths, exhaustive_ppi, reference_line


@pytest.mark.parametrize(
    "shape,length,segments,seed",
    [
        ((3, 4), 1, 1, 11),
        ((4, 5), 1, 2, 12),
        ((5, 5), 1, 3, 13),
        ((5, 6), 2, 1, 14),
        ((5, 6), 2, 2, 15),
        ((7, 7), 2, 3, 16),
        ((7, 8), 3, 2, 17),
    ],
)
def test_matches_exhaustive_enumeration(shape, length, segments, seed):
    image = np.random.default_rng(seed).integers(0, 32, shape, dtype=np.uint8)
    costs, coordinates = compute_ppi(image, segment_length=length, nb_segments=segments)
    expected_costs, expected_coordinates = exhaustive_ppi(image, length, segments)
    np.testing.assert_array_equal(costs, expected_costs)
    np.testing.assert_array_equal(coordinates, expected_coordinates)
    assert_valid_paths(image, costs, coordinates, length, segments)


def test_original_overwritten_path_regression():
    # FinalVersion's in-place coordinate reuse produced paths whose actual
    # pixel costs disagreed with the correct cost image on this seed.
    image = np.random.default_rng(0).integers(1, 20, (8, 8), dtype=np.uint8)
    costs, coordinates = compute_ppi(image, segment_length=1, nb_segments=3)
    expected_costs, expected_coordinates = exhaustive_ppi(image, 1, 3)
    np.testing.assert_array_equal(costs, expected_costs)
    np.testing.assert_array_equal(coordinates, expected_coordinates)
    assert_valid_paths(image, costs, coordinates, 1, 3)


@pytest.mark.parametrize("direction", [(-1, 0), (1, 0), (0, 1), (0, -1)])
def test_each_of_the_four_orientations_can_win(direction):
    image = np.full((9, 9), 200, dtype=np.uint8)
    start = (4, 4)
    # Two length-2 cardinal steps form the unique zero-cost path from start.
    for distance in range(1, 5):
        image[start[0] + direction[0] * distance, start[1] + direction[1] * distance] = 0
    costs, coordinates = compute_ppi(image, segment_length=2, nb_segments=2)
    expected = [(start[0] + d * direction[0], start[1] + d * direction[1]) for d in (2, 4)]
    assert costs[start] == 0
    np.testing.assert_array_equal(coordinates[start], expected)
    assert_valid_paths(image, costs, coordinates, 2, 2)


def test_equal_cost_ties_preserve_original_endpoint_and_cone_order():
    image = np.zeros((7, 7), dtype=np.uint8)
    first = compute_ppi(image, segment_length=1, nb_segments=3)
    second = compute_ppi(image, segment_length=1, nb_segments=3)
    expected = exhaustive_ppi(image, 1, 3)
    for actual, repeated, reference in zip(first, second, expected):
        np.testing.assert_array_equal(actual, repeated)
        np.testing.assert_array_equal(actual, reference)
    # H wins across cones; its final endpoint is northwest.
    np.testing.assert_array_equal(first[1][3, 3], [[2, 2], [1, 1], [0, 0]])
    # At the top border H is infeasible; B wins, taking its last (SE) endpoint.
    np.testing.assert_array_equal(first[1][0, 0], [[1, 1], [2, 2], [3, 3]])


@pytest.mark.parametrize("shape,length,segments", [((1, 1), 1, 1), ((3, 3), 1, 3), ((4, 5), 5, 1)])
def test_unreachable_paths_have_infinite_cost_and_only_minus_one_coordinates(
    shape, length, segments
):
    image = np.zeros(shape, dtype=np.uint8)
    costs, coordinates = compute_ppi(image, segment_length=length, nb_segments=segments)
    assert np.isposinf(costs).all()
    assert np.all(coordinates == -1)
    assert_valid_paths(image, costs, coordinates, length, segments)


@pytest.mark.parametrize("shape", [(1, 5), (5, 1)])
def test_singleton_axis_still_allows_paths_along_other_axis(shape):
    image = np.arange(5, dtype=np.uint8).reshape(shape)
    result = compute_ppi(image, segment_length=2, nb_segments=2)
    expected = exhaustive_ppi(image, 2, 2)
    for actual, reference in zip(result, expected):
        np.testing.assert_array_equal(actual, reference)
    assert np.isfinite(result[0]).any()
    assert np.isinf(result[0]).any()
    assert_valid_paths(image, *result, 2, 2)


def test_cost_excludes_start_includes_endpoint_and_counts_joint_once():
    image = np.array([[99, 2, 3, 4, 5]], dtype=np.uint8)
    costs, coordinates = compute_ppi(image, segment_length=2, nb_segments=2)
    assert costs[0, 0] == 2 + 3 + 4 + 5
    np.testing.assert_array_equal(coordinates[0, 0], [[0, 2], [0, 4]])


def test_uint8_summation_does_not_wrap():
    image = np.full((1, 8), 255, dtype=np.uint8)
    costs, coordinates = compute_ppi(image, segment_length=3, nb_segments=2)
    assert costs[0, 0] == 6 * 255
    assert_valid_paths(image, costs, coordinates, 3, 2)


@pytest.mark.parametrize("view_kind", ["transpose", "stride", "reverse"])
def test_noncontiguous_input_and_source_remain_unchanged(view_kind):
    original = np.arange(80, dtype=np.uint8).reshape(8, 10)
    before = original.copy()
    views = {"transpose": original.T, "stride": original[::2, ::2], "reverse": original[::-1, ::-1]}
    image = views[view_kind]
    assert not image.flags.c_contiguous
    result = compute_ppi(image, segment_length=1, nb_segments=2)
    contiguous_result = compute_ppi(image.copy(), segment_length=1, nb_segments=2)
    for actual, expected in zip(result, contiguous_result):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(original, before)
    assert_valid_paths(image, *result, 1, 2)


def test_readonly_input_is_accepted_and_not_modified():
    image = np.arange(30, dtype=np.uint8).reshape(5, 6)
    before = image.copy()
    image.flags.writeable = False
    result = compute_ppi(image, segment_length=1, nb_segments=2)
    np.testing.assert_array_equal(image, before)
    assert_valid_paths(image, *result, 1, 2)


def test_numpy_integer_parameters_are_accepted():
    image = np.ones((4, 5), dtype=np.uint8)
    result = compute_ppi(image, segment_length=np.int64(1), nb_segments=np.int32(2))
    assert_valid_paths(image, *result, 1, 2)


def test_documented_default_parameters():
    image = np.zeros((1, 31), dtype=np.uint8)
    default = compute_ppi(image)
    explicit = compute_ppi(image, segment_length=3, nb_segments=10)
    for actual, expected in zip(default, explicit):
        np.testing.assert_array_equal(actual, expected)
    assert_valid_paths(image, *default, 3, 10)


def test_enormous_segment_length_is_unreachable_without_allocating_endpoints():
    costs, paths = compute_ppi(np.zeros((2, 2), dtype=np.uint8), 10**50, 1)
    assert np.isposinf(costs).all()
    assert np.all(paths == -1)


def test_unrepresentable_path_array_is_rejected_before_allocation():
    with pytest.raises(ValueError, match="too large"):
        compute_ppi(np.zeros((2, 2), dtype=np.uint8), 1, 10**50)


@pytest.mark.parametrize(
    "image",
    [
        None,
        [[1, 2], [3, 4]],
        np.array(2, dtype=np.uint8),
        np.zeros(3, dtype=np.uint8),
        np.zeros((2, 2, 1), dtype=np.uint8),
        np.zeros((0, 2), dtype=np.uint8),
        np.zeros((2, 0), dtype=np.uint8),
    ],
)
def test_invalid_image_structure_is_rejected(image):
    with pytest.raises((TypeError, ValueError)):
        compute_ppi(image, segment_length=1, nb_segments=1)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int8, np.int64, np.uint16, np.bool_, object]
)
def test_non_uint8_images_are_rejected(dtype):
    with pytest.raises((TypeError, ValueError)):
        compute_ppi(np.ones((3, 3), dtype=dtype), segment_length=1, nb_segments=1)


@pytest.mark.parametrize("name", ["segment_length", "nb_segments"])
@pytest.mark.parametrize("value", [0, -1, 1.0, 1.5, True, False, np.bool_(True), "2", None])
def test_parameters_must_be_positive_nonboolean_integers(name, value):
    kwargs = {"segment_length": 1, "nb_segments": 1, name: value}
    with pytest.raises((TypeError, ValueError)):
        compute_ppi(np.zeros((3, 3), dtype=np.uint8), **kwargs)


@pytest.mark.parametrize(
    "start,end,expected",
    [
        ((0, 0), (0, 0), [(0, 0)]),
        ((-2, -3), (-2, 1), [(-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1)]),
        ((2, -1), (-1, -1), [(2, -1), (1, -1), (0, -1), (-1, -1)]),
        ((-2, 2), (1, -1), [(-2, 2), (-1, 1), (0, 0), (1, -1)]),
        ((0, 0), (2, 1), [(0, 0), (1, 1), (2, 1)]),
        ((2, 1), (0, 0), [(2, 1), (1, 0), (0, 0)]),
        ((-3, -2), (-1, -1), [(-3, -2), (-2, -1), (-1, -1)]),
    ],
)
def test_bresenham_exact_signed_coordinates_and_directional_midpoint_ties(start, end, expected):
    assert [tuple(p) for p in bresenham_line(*start, *end)] == expected


@pytest.mark.parametrize("dr,dc", [(r, c) for r in range(-4, 5) for c in range(-4, 5)])
def test_bresenham_all_octants_match_independent_integer_sampling(dr, dc):
    start = (-2, 1)
    end = (start[0] + dr, start[1] + dc)
    line = [tuple(p) for p in bresenham_line(*start, *end)]
    assert line == reference_line(start, end)
    assert line[0] == start
    assert line[-1] == end
    assert len(line) == max(abs(dr), abs(dc)) + 1
    assert all(max(abs(b[0] - a[0]), abs(b[1] - a[1])) == 1 for a, b in zip(line, line[1:]))
