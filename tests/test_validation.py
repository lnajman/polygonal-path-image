"""Checks that the independent research audit detects invalid reported results."""

import importlib.util
from pathlib import Path

import numpy as np

from polygonal_path_image import compute_ppi, voting

_spec = importlib.util.spec_from_file_location(
    "validate_research", Path(__file__).parents[1] / "scripts" / "validate_research.py"
)
validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validation)


def test_audit_matches_valid_random_paths_and_votes():
    image = np.random.default_rng(2012).integers(0, 256, (11, 13), dtype=np.uint8)
    costs, paths = compute_ppi(image, 2, 3)
    audit, independent_votes, mismatch = validation.audit_paths(image, costs, paths, 2)
    assert audit["finite_paths"] == np.isfinite(costs).sum()
    assert audit["invalid_endpoint_paths"] == 0
    assert audit["invalid_geometry_paths"] == 0
    assert audit["inconsistent_path_costs"] == 0
    assert not mismatch.any()
    np.testing.assert_array_equal(independent_votes, voting(costs, paths)[0])


def test_audit_detects_cost_and_endpoint_corruption():
    image = np.arange(121, dtype=np.uint8).reshape(11, 11)
    costs, paths = compute_ppi(image, 2, 3)
    costs[0, 0] += 1
    paths[0, 1, 0] = (-1, -1)
    audit, _, mismatch = validation.audit_paths(image, costs, paths, 2)
    assert audit["invalid_endpoint_paths"] == 1
    assert audit["invalid_geometry_paths"] == 1
    assert audit["inconsistent_path_costs"] == 2
    assert mismatch[0, 0] and mismatch[0, 1]


def test_audit_detects_invalid_segment_geometry_with_correct_cost():
    image = np.ones((5, 7), dtype=np.uint8)
    costs = np.full(image.shape, np.inf)
    paths = np.full((*image.shape, 2, 2), -1, dtype=np.int64)
    costs[0, 0] = 5
    paths[0, 0] = [(0, 2), (0, 5)]
    audit, votes, mismatch = validation.audit_paths(image, costs, paths, 2)
    assert audit["invalid_geometry_paths"] == 1
    assert audit["inconsistent_path_costs"] == 0
    assert not mismatch.any()
    assert votes.sum() == 6


def test_audit_accepts_empty_finite_set():
    image = np.ones((2, 2), dtype=np.uint8)
    costs, paths = compute_ppi(image, 3, 10)
    audit, votes, mismatch = validation.audit_paths(image, costs, paths, 3)
    assert audit["finite_paths"] == 0
    assert audit["inconsistent_path_costs"] == 0
    assert not votes.any()
    assert not mismatch.any()
