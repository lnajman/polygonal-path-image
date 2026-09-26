"""Exact toy checks of path-stage tracing, independent of study conclusions."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from benchmarks.diagnose_weak_paths import (  # noqa: E402
    distribution,
    fixed_potentials,
    rasterize_paths,
    source_partitions,
    tortuosity_scores,
)
from polygonal_path_image import filter_tortuosity, voting  # noqa: E402


def toy_paths():
    costs = np.full((5, 6), np.inf)
    paths = np.full((5, 6, 2, 2), -1, dtype=np.int64)
    # Shared joint at(1,3), then a later revisit to the source counts again.
    costs[1, 1] = 5
    paths[1, 1] = [[1, 3], [1, 1]]
    costs[3, 0] = 6
    paths[3, 0] = [[3, 2], [3, 4]]
    return costs, paths


def test_raster_votes_keep_joints_once_and_later_revisits_twice():
    costs, paths = toy_paths()
    raster = rasterize_paths(costs, paths)
    official, _ = voting(costs, paths)
    np.testing.assert_array_equal(raster.votes(), official)
    assert official[1, 1] == official[1, 2] == 2
    assert official[1, 3] == 1
    np.testing.assert_array_equal(np.diff(raster.offsets), [5, 5])
    np.testing.assert_array_equal(raster.votes([True, False])[1], [0, 2, 2, 1, 0, 0])


def test_raster_reconstructs_cost_without_counting_source_at_start():
    costs, paths = toy_paths()
    raster = rasterize_paths(costs, paths)
    image = np.arange(30).reshape(5, 6)
    # Source revisit is paid; only initial source is omitted.
    np.testing.assert_array_equal(
        raster.reconstruct_costs(image), [8 + 9 + 8 + 7, 19 + 20 + 21 + 22]
    )


def test_source_partitions_are_disjoint_complete_and_reproduce_selected_votes():
    costs, paths = toy_paths()
    raster = rasterize_paths(costs, paths)
    weak = np.zeros(costs.shape, dtype=bool)
    weak[1, 1:4] = True
    strong = np.zeros_like(weak)
    strong[1, 3] = strong[3, 4] = True
    groups = source_partitions(raster, weak, strong)
    np.testing.assert_array_equal(sum(group.astype(int) for group in groups.values()), [1, 1])
    assert groups["weak_origin__visits_strong"].tolist() == [True, False]
    assert groups["other_origin__visits_strong"].tolist() == [False, True]
    np.testing.assert_array_equal(
        sum(raster.votes(group) for group in groups.values()), raster.votes()
    )
    np.testing.assert_array_equal(raster.visits(weak), [5, 0])


def test_tortuosity_scores_match_package_for_straight_reversal_and_corner_paths():
    costs, paths = toy_paths()
    scores = tortuosity_scores(costs, paths)
    np.testing.assert_allclose(scores, [-1, 1])
    for threshold in (0.5, 0.75):
        keep = np.isfinite(filter_tortuosity(costs, paths, threshold))[np.isfinite(costs)]
        np.testing.assert_array_equal(keep, scores >= threshold)
    costs[4, 0] = 1
    paths[4, 0] = [[4, 2], [2, 4]]
    assert tortuosity_scores(costs, paths)[-1] == pytest.approx(1 / np.sqrt(2))


def test_degenerate_segment_is_marked_undefined_and_rejected_by_package():
    costs, paths = toy_paths()
    paths[1, 1] = [[1, 1], [1, 3]]
    assert np.isnan(tortuosity_scores(costs, paths)[0])
    assert np.isinf(filter_tortuosity(costs, paths, 0.5)[1, 1])


def test_impossible_paths_produce_zero_votes_and_empty_groups():
    costs = np.full((3, 3), np.inf)
    paths = np.full((3, 3, 2, 2), -1, dtype=np.int64)
    raster = rasterize_paths(costs, paths)
    assert not raster.votes().any()
    assert raster.visits(np.ones((3, 3), dtype=bool)).size == 0
    assert raster.reconstruct_costs(np.ones((3, 3))).size == 0


def test_fixed_local_potential_does_not_normalize_constant_images():
    image = np.full((32, 32), 200, dtype=np.uint8)
    values = fixed_potentials(image)
    np.testing.assert_array_equal(values["raw"], image)
    assert (values["local"] == 255).all()


def test_distribution_has_json_safe_empty_values_and_known_quantiles():
    assert distribution([])["mean"] is None
    result = distribution([0, 1, 2, 3, 4])
    assert result["median"] == 2
    assert result["q25"] == 1
    assert result["zero_fraction"] == 0.2
    with pytest.raises(ValueError):
        distribution([np.nan])
