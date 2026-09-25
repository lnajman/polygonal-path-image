"""Numerical postprocessing for polygonal paths, without plotting or I/O.

Path coordinates are ``(row, column)`` endpoint pairs. The first segment starts
at the pixel indexing the path array. Infinite costs mark inactive paths; their
coordinates are ignored. All functions leave their arguments unchanged.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import bresenham_line

__all__ = ["voting", "filter_tortuosity", "orientation", "prune_paths"]


def _validate_paths(costs, paths):
    costs = np.asarray(costs)
    paths = np.asarray(paths)
    if costs.ndim != 2 or 0 in costs.shape:
        raise ValueError("costs must be a nonempty two-dimensional array")
    if costs.dtype.kind not in "iuf":
        raise TypeError("costs must contain real numbers")
    if np.any(np.isnan(costs)) or np.any(np.isneginf(costs)):
        raise ValueError("costs must be finite or positive infinity")
    if (
        paths.ndim != 4
        or paths.shape[:2] != costs.shape
        or paths.shape[2] == 0
        or paths.shape[3] != 2
    ):
        raise ValueError("paths must have shape (height, width, segments, 2)")
    if paths.dtype.kind not in "iu":
        raise TypeError("paths must contain integer pixel coordinates")
    active = np.isfinite(costs)
    endpoints = paths[active]
    if (
        np.any(endpoints < 0)
        or np.any(endpoints[..., 0] >= costs.shape[0])
        or np.any(endpoints[..., 1] >= costs.shape[1])
    ):
        raise ValueError("every finite-cost path endpoint must be inside the image")
    return costs, paths, active


def _real_parameter(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _segments(row, column, endpoints):
    previous = (int(row), int(column))
    for index, endpoint in enumerate(endpoints):
        current = (int(endpoint[0]), int(endpoint[1]))
        pixels = bresenham_line(*previous, *current)
        # The preceding segment already included this joint.
        if index:
            pixels = pixels[1:]
        yield previous, current, pixels
        previous = current


def voting(
    costs: ArrayLike, paths: ArrayLike, *, border: int = 0
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Count rasterized path visits and return ``(votes, max(votes)-votes)``.

    Each finite-cost path contributes one vote per rasterized pixel occurrence.
    A joint shared by consecutive segments is counted once; a later revisit to
    a pixel counts again. ``border`` clears that many rows/columns at every edge
    before computing the inverse. Both returned arrays have dtype ``int64``.
    Unlike the legacy implementation, the default does not clear the border,
    and the inversion maximum is computed after any requested border clearing.
    """
    costs, paths, active = _validate_paths(costs, paths)
    if isinstance(border, (bool, np.bool_)) or not isinstance(border, Integral):
        raise TypeError("border must be an integer")
    border = int(border)
    if border < 0:
        raise ValueError("border must be nonnegative")
    votes = np.zeros(costs.shape, dtype=np.int64)
    for row, column in np.argwhere(active):
        for _, _, pixels in _segments(row, column, paths[row, column]):
            if pixels:
                rows, columns = np.asarray(pixels, dtype=np.intp).T
                np.add.at(votes, (rows, columns), 1)
    if border:
        votes[:border, :] = 0
        votes[-border:, :] = 0
        votes[:, :border] = 0
        votes[:, -border:] = 0
    return votes, votes.max() - votes


def filter_tortuosity(
    costs: ArrayLike, paths: ArrayLike, threshold: float = 0.75
) -> NDArray[np.float64]:
    """Set costs to infinity when the product of turn cosines is too small.

    Turns include the first segment from the source pixel to the first stored
    endpoint. A straight path has score 1. The signed cosines are multiplied
    explicitly, so an even number of reversals can also have positive score;
    this measure is intended for the monotone paths returned by ``compute_ppi``.
    A path with any zero-length segment is rejected. The threshold must be in
    [-1, 1], and equality is retained within floating-point round-off. Return a
    new ``float64`` cost image. The legacy implementation omitted the first turn
    and did not handle zero-length segments.
    """
    costs, paths, active = _validate_paths(costs, paths)
    threshold = _real_parameter(threshold, "threshold")
    if not -1 <= threshold <= 1:
        raise ValueError("threshold must be between -1 and 1")
    result = np.array(costs, dtype=np.float64, copy=True)
    for row, column in np.argwhere(active):
        vertices = np.vstack(((row, column), paths[row, column])).astype(np.float64)
        vectors = np.diff(vertices, axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            result[row, column] = np.inf
            continue
        units = vectors / norms[:, None]
        cosines = np.einsum("ij,ij->i", units[:-1], units[1:])
        score = np.prod(np.clip(cosines, -1.0, 1.0))
        tolerance = 8 * np.finfo(float).eps * max(len(cosines), 1)
        if score < threshold - tolerance:
            result[row, column] = np.inf
    return result


def orientation(costs: ArrayLike, paths: ArrayLike) -> NDArray[np.float64]:
    """Return the axial mean segment orientation at every covered pixel.

    Angles are radians in [0, pi), measured from increasing column toward
    increasing row: horizontal is 0 and vertical is pi/2. Each segment supplies
    its squared Euclidean length as weight to a double-angle circular mean.
    Shared joints belong to the preceding segment. Zero-length segments supply
    no evidence. Uncovered pixels and pixels with cancelling orientation
    evidence contain NaN. Return a new ``float64`` image.
    This replaces the legacy plotting routine's sparse, local estimates with a
    dense orientation field from all active segments.
    """
    costs, paths, active = _validate_paths(costs, paths)
    cosine = np.zeros(costs.shape, dtype=np.float64)
    sine = np.zeros(costs.shape, dtype=np.float64)
    weight = np.zeros(costs.shape, dtype=np.float64)
    for row, column in np.argwhere(active):
        for start, end, pixels in _segments(row, column, paths[row, column]):
            delta_row = float(end[0] - start[0])
            delta_column = float(end[1] - start[1])
            squared_length = delta_row**2 + delta_column**2
            if not pixels or squared_length == 0:
                continue
            # length^2*cos(2*theta) and length^2*sin(2*theta), avoiding
            # trigonometric round-off for exactly axial/diagonal segments.
            rows, columns = np.asarray(pixels, dtype=np.intp).T
            np.add.at(cosine, (rows, columns), delta_column**2 - delta_row**2)
            np.add.at(sine, (rows, columns), 2 * delta_row * delta_column)
            np.add.at(weight, (rows, columns), squared_length)
    magnitude = np.hypot(cosine, sine)
    valid = (weight > 0) & (magnitude > 16 * np.finfo(float).eps * weight)
    result = np.full(costs.shape, np.nan, dtype=np.float64)
    result[valid] = np.mod(0.5 * np.arctan2(sine[valid], cosine[valid]), np.pi)
    return result


def prune_paths(
    costs: ArrayLike,
    paths: ArrayLike,
    *,
    fraction: float = 0.4,
    distance: float = 10.0,
) -> NDArray[np.float64]:
    """Keep the lowest-cost paths while suppressing overlapping endpoint sets.

    Candidates are processed by ascending cost, breaking ties in row-major
    source order. A candidate is suppressed when at least ``fraction`` of its
    endpoint vertices lie strictly closer than ``distance`` to any endpoint
    vertex of one already retained path. Source pixels are excluded from this
    comparison. Fractions from (0, 1] and nonnegative distances are accepted;
    distance 0 suppresses nothing. The result is a new ``float64`` cost image.

    This deterministic global procedure replaces the legacy neighborhood
    heuristic and does not exactly reproduce its pruning decisions. Its worst
    case is quadratic in the number of retained paths and segment count.
    """
    costs, paths, active = _validate_paths(costs, paths)
    fraction = _real_parameter(fraction, "fraction")
    distance = _real_parameter(distance, "distance")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")
    if distance < 0:
        raise ValueError("distance must be nonnegative")
    result = np.array(costs, dtype=np.float64, copy=True)
    if distance == 0:
        return result
    candidates = np.flatnonzero(active)
    order = np.argsort(costs.ravel()[candidates], kind="stable")
    flat_paths = paths.reshape((-1, paths.shape[2], 2))
    retained = []
    for index in candidates[order]:
        endpoints = flat_paths[index].astype(np.float64)
        lower = endpoints.min(axis=0)
        upper = endpoints.max(axis=0)
        suppress = False
        for other, other_lower, other_upper in retained:
            # Bounding boxes often reject separated paths without building
            # their K-by-K endpoint distance matrix.
            gap = np.maximum(np.maximum(lower - other_upper, other_lower - upper), 0)
            if np.hypot(gap[0], gap[1]) >= distance:
                continue
            delta = endpoints[:, None, :] - other[None, :, :]
            near = np.any(np.hypot(delta[..., 0], delta[..., 1]) < distance, axis=1)
            if np.count_nonzero(near) / len(endpoints) >= fraction:
                suppress = True
                break
        if suppress:
            result.flat[index] = np.inf
        else:
            retained.append((endpoints, lower, upper))
    return result
