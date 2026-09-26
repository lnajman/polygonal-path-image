"""Independent finite-candidate and preservation checks for a research ablation."""

import numpy as np
import pytest

from benchmarks.straight_fallback import minimum_straight_paths, straight_fallback
from polygonal_path_image import compute_ppi, filter_tortuosity, voting
from tests.reference import reference_line


@pytest.mark.parametrize("length,count", [(1, 2), (2, 2), (3, 1)])
def test_straight_minimum_matches_independent_candidate_enumeration(length, count):
    image = np.random.default_rng(101).integers(1, 100, (8, 9), dtype=np.uint8)
    costs, paths = minimum_straight_paths(image, length, count)
    directions = [(dy, dx) for dy in range(-length, length + 1)
                  for dx in range(-length, length + 1) if max(abs(dy), abs(dx)) == length]
    for row, column in np.ndindex(image.shape):
        candidates = []
        for dy, dx in directions:
            end = row + count * dy, column + count * dx
            if not (0 <= end[0] < image.shape[0] and 0 <= end[1] < image.shape[1]):
                continue
            pixels = []
            for step in range(count):
                pixels.extend(reference_line((row + step * dy, column + step * dx),
                                             (row + (step + 1) * dy,
                                              column + (step + 1) * dx))[1:])
            candidates.append(sum(int(image[p]) for p in pixels))
        assert costs[row, column] == (min(candidates) if candidates else np.inf)
        if candidates:
            vertices = np.vstack(((row, column), paths[row, column]))
            assert np.all(np.diff(vertices, axis=0) == np.diff(vertices, axis=0)[0])
            actual = sum(int(image[p]) for a, b in zip(vertices[:-1], vertices[1:], strict=True)
                         for p in reference_line(tuple(a), tuple(b))[1:])
            assert actual == costs[row, column]


def test_fallback_preserves_all_accepted_paths_and_never_fabricates_cheaper_cost():
    image = np.random.default_rng(91).integers(0, 255, (20, 21), dtype=np.uint8)
    original_costs, original_paths = compute_ppi(image, 2, 4)
    kept = filter_tortuosity(original_costs, original_paths, .75)
    scores, costs, paths, changed = straight_fallback(image, segment_length=2, nb_segments=4)
    accepted = np.isfinite(kept)
    assert changed.any()
    np.testing.assert_array_equal(costs[accepted], original_costs[accepted])
    np.testing.assert_array_equal(paths[accepted], original_paths[accepted])
    assert not (changed & accepted).any()
    assert np.all(costs[changed] >= original_costs[changed])
    assert np.isfinite(filter_tortuosity(costs, paths, .75)[changed]).all()
    np.testing.assert_array_equal(scores, voting(costs, paths)[0])


def test_impossible_origins_stay_impossible():
    image = np.zeros((3, 3), dtype=np.uint8)
    scores, costs, paths, changed = straight_fallback(image, segment_length=3, nb_segments=10)
    assert np.isinf(costs).all()
    assert (paths == -1).all()
    assert not scores.any()
    assert not changed.any()


def test_uniform_feasible_path_cost_is_constant_and_input_unchanged():
    image = np.full((15, 17), 200, dtype=np.uint8)
    before = image.copy()
    costs, _ = minimum_straight_paths(image, 2, 4)
    assert np.all(costs[np.isfinite(costs)] == 1600)
    np.testing.assert_array_equal(image, before)
