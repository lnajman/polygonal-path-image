"""Small exact postprocessing fixtures, independent of the path optimizer."""

import numpy as np
import pytest

from polygonal_path_image import (
    filter_tortuosity,
    orientation,
    prune_paths,
    voting,
)


def fixture_paths(shape, segments, entries):
    costs = np.full(shape, np.inf)
    paths = np.full((*shape, segments, 2), -1, dtype=np.int64)
    for source, cost, endpoints in entries:
        costs[source] = cost
        paths[source] = endpoints
    return costs, paths


def test_voting_counts_joint_once_and_preserves_arrays():
    costs, paths = fixture_paths((3, 3), 2, [((0, 0), 2, [(0, 2), (2, 2)])])
    before_costs, before_paths = costs.copy(), paths.copy()
    votes, inverted = voting(costs, paths)
    expected = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=np.int64)
    np.testing.assert_array_equal(votes, expected)
    np.testing.assert_array_equal(inverted, 1 - expected)
    assert votes.dtype == inverted.dtype == np.int64
    np.testing.assert_array_equal(costs, before_costs)
    np.testing.assert_array_equal(paths, before_paths)


def test_voting_border_recomputes_inverse_from_cleared_votes():
    costs, paths = fixture_paths((3, 3), 1, [((0, 0), 2, [(0, 2)])])
    votes, inverted = voting(costs, paths, border=1)
    assert not votes.any()
    assert not inverted.any()
    assert not voting(costs, paths, border=5)[0].any()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint64])
def test_voting_accepts_unsigned_integer_border(dtype):
    costs, paths = fixture_paths((3, 3), 1, [((2, 0), 2, [(2, 2)])])
    votes, inverted = voting(costs, paths, border=dtype(1))
    assert not votes.any()
    assert not inverted.any()


def test_tortuosity_includes_first_turn_and_has_no_side_effects():
    costs, paths = fixture_paths(
        (5, 5),
        2,
        [((0, 0), 3, [(0, 2), (0, 4)]), ((1, 0), 5, [(1, 2), (3, 2)])],
    )
    before_costs, before_paths = costs.copy(), paths.copy()
    filtered = filter_tortuosity(costs, paths)
    assert filtered[0, 0] == 3
    assert np.isinf(filtered[1, 0])
    np.testing.assert_array_equal(costs, before_costs)
    np.testing.assert_array_equal(paths, before_paths)
    assert not np.shares_memory(filtered, costs)


def test_tortuosity_single_segment_and_degenerate_segment():
    costs, paths = fixture_paths(
        (3, 3),
        1,
        [((0, 0), 3, [(0, 2)]), ((1, 0), 5, [(1, 0)])],
    )
    filtered = filter_tortuosity(costs, paths, threshold=1)
    assert filtered[0, 0] == 3
    assert np.isinf(filtered[1, 0])


def test_tortuosity_uses_product_of_all_turn_cosines():
    costs, paths = fixture_paths((4, 4), 3, [((0, 0), 3, [(0, 1), (1, 2), (2, 2)])])
    # cos(pi/4)**2 is 0.5, not either individual turn cosine (~0.707).
    assert np.isinf(filter_tortuosity(costs, paths, threshold=0.6)[0, 0])
    assert filter_tortuosity(costs, paths, threshold=0.4)[0, 0] == 3
    assert filter_tortuosity(costs, paths, threshold=0.5)[0, 0] == 3


def test_tortuosity_straight_diagonal_retains_threshold_equality():
    costs, paths = fixture_paths((4, 4), 3, [((0, 0), 3, [(1, 1), (2, 2), (3, 3)])])
    assert filter_tortuosity(costs, paths, threshold=1)[0, 0] == 3


def test_orientation_horizontal_vertical_and_opposite_directions():
    costs, paths = fixture_paths(
        (5, 5),
        1,
        [((0, 0), 1, [(0, 4)]), ((0, 4), 2, [(0, 0)]), ((1, 2), 3, [(4, 2)])],
    )
    angles = orientation(costs, paths)
    np.testing.assert_allclose(angles[0], 0)
    np.testing.assert_allclose(angles[1:, 2], np.pi / 2)
    assert np.isnan(angles[4, 4])


def test_orientation_diagonal_sign_and_squared_length_weights():
    costs, paths = fixture_paths(
        (5, 5),
        1,
        [((0, 0), 1, [(4, 4)]), ((0, 2), 2, [(4, 2)])],
    )
    angles = orientation(costs, paths)
    assert angles[1, 1] == pytest.approx(np.pi / 4)
    # Diagonal length^2=32, vertical length^2=16 at their crossing.
    assert angles[2, 2] == pytest.approx(np.arctan2(32, -16) / 2)
    negative_costs, negative_paths = fixture_paths((3, 3), 1, [((0, 2), 1, [(2, 0)])])
    assert orientation(negative_costs, negative_paths)[1, 1] == pytest.approx(3 * np.pi / 4)


def test_orientation_cancelling_or_zero_length_evidence_is_nan():
    costs, paths = fixture_paths(
        (3, 3),
        1,
        [((1, 0), 1, [(1, 2)]), ((0, 1), 2, [(2, 1)]), ((0, 0), 3, [(0, 0)])],
    )
    angles = orientation(costs, paths)
    assert np.isnan(angles[1, 1])
    assert np.isnan(angles[0, 0])


def test_pruning_duplicates_nonoverlap_and_cost_order():
    costs, paths = fixture_paths(
        (6, 6),
        2,
        [
            ((0, 0), 4, [(0, 1), (0, 2)]),
            ((0, 3), 2, [(0, 1), (0, 2)]),
            ((5, 0), 8, [(5, 1), (5, 2)]),
        ],
    )
    before_costs, before_paths = costs.copy(), paths.copy()
    pruned = prune_paths(costs, paths, fraction=1, distance=0.5)
    assert np.isinf(pruned[0, 0])
    assert pruned[0, 3] == 2
    assert pruned[5, 0] == 8
    assert np.isinf(pruned[1, 1])
    np.testing.assert_array_equal(costs, before_costs)
    np.testing.assert_array_equal(paths, before_paths)
    assert not np.shares_memory(pruned, costs)


def test_pruning_uses_fraction_strict_distance_and_excludes_source():
    costs, paths = fixture_paths(
        (6, 6),
        2,
        [((0, 0), 1, [(2, 0), (4, 0)]), ((0, 1), 2, [(2, 1), (4, 5)])],
    )
    # One out of two candidate endpoints is close; nearby sources do not count.
    assert np.isinf(prune_paths(costs, paths, fraction=0.5, distance=1.01)[0, 1])
    assert prune_paths(costs, paths, fraction=0.75, distance=1.01)[0, 1] == 2
    assert prune_paths(costs, paths, fraction=0.5, distance=1)[0, 1] == 2
    np.testing.assert_array_equal(prune_paths(costs, paths, distance=0), costs)


def test_pruning_ties_are_broken_by_row_major_source_order():
    costs, paths = fixture_paths(
        (3, 3),
        1,
        [((0, 0), 2, [(1, 1)]), ((0, 2), 2, [(1, 1)])],
    )
    pruned = prune_paths(costs, paths, distance=0.5)
    assert pruned[0, 0] == 2
    assert np.isinf(pruned[0, 2])


@pytest.mark.parametrize("function", [voting, filter_tortuosity, orientation, prune_paths])
def test_all_unreachable_paths(function):
    costs, paths = fixture_paths((2, 2), 1, [])
    result = function(costs, paths)
    if function is voting:
        assert not result[0].any() and not result[1].any()
    elif function is orientation:
        assert np.isnan(result).all()
    else:
        assert np.isposinf(result).all()


@pytest.mark.parametrize("function", [voting, filter_tortuosity, orientation, prune_paths])
@pytest.mark.parametrize("endpoint", [(-1, 0), (2, 0), (0, 2)])
def test_reject_out_of_bounds_active_coordinates(function, endpoint):
    costs, paths = fixture_paths((2, 2), 1, [((0, 0), 1, [endpoint])])
    with pytest.raises(ValueError, match="inside the image"):
        function(costs, paths)


@pytest.mark.parametrize("function", [voting, filter_tortuosity, orientation, prune_paths])
@pytest.mark.parametrize("invalid_cost", [np.nan, -np.inf])
def test_reject_invalid_costs(function, invalid_cost):
    costs, paths = fixture_paths((2, 2), 1, [])
    costs[0, 0] = invalid_cost
    with pytest.raises(ValueError, match="finite or positive infinity"):
        function(costs, paths)


@pytest.mark.parametrize("function", [voting, filter_tortuosity, orientation, prune_paths])
def test_reject_mismatched_or_noninteger_paths(function):
    costs, paths = fixture_paths((2, 2), 1, [])
    with pytest.raises(ValueError, match="shape"):
        function(costs, paths[:, :1])
    with pytest.raises(TypeError, match="integer"):
        function(costs, paths.astype(float))
    with pytest.raises(ValueError, match="shape"):
        function(costs, paths[:, :, :0])


@pytest.mark.parametrize(
    "function,kwargs,error",
    [
        (voting, {"border": -1}, ValueError),
        (voting, {"border": 1.5}, TypeError),
        (filter_tortuosity, {"threshold": 1.1}, ValueError),
        (filter_tortuosity, {"threshold": np.nan}, ValueError),
        (prune_paths, {"fraction": 0}, ValueError),
        (prune_paths, {"fraction": 1.1}, ValueError),
        (prune_paths, {"distance": -1}, ValueError),
        (prune_paths, {"distance": np.inf}, ValueError),
    ],
)
def test_reject_invalid_parameters(function, kwargs, error):
    costs, paths = fixture_paths((2, 2), 1, [])
    with pytest.raises(error):
        function(costs, paths, **kwargs)
