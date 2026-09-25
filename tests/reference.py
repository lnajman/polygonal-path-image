"""Small exhaustive oracle: no dynamic programming or production helpers.

This deliberately slow implementation is used only on tiny arrays. Enumerating
every endpoint sequence independently catches errors in both optimal costs and
path reconstruction. Pixel sampling uses an integer closed form rather than
the production Bresenham recurrence.
"""

from itertools import product

import numpy as np


def reference_line(start, end):
    """Rasterize with midpoint ties rounded toward the segment's endpoint."""
    dr, dc = end[0] - start[0], end[1] - start[1]
    length = max(abs(dr), abs(dc))
    if not length:
        return [tuple(start)]
    sr = 1 if dr >= 0 else -1
    sc = 1 if dc >= 0 else -1
    return [
        (
            start[0] + sr * ((2 * i * abs(dr) + length) // (2 * length)),
            start[1] + sc * ((2 * i * abs(dc) + length) // (2 * length)),
        )
        for i in range(length + 1)
    ]


def reference_cones(length):
    """Original FinalVersion endpoint order, in image (row, column) axes."""
    ascending = range(-length, length + 1)
    descending = range(length, -length - 1, -1)
    return (
        tuple((-length, c) for c in descending),  # H (north)
        tuple((length, c) for c in ascending),  # B (south)
        tuple((r, length) for r in descending),  # E (east)
        tuple((r, -length) for r in ascending),  # W (west)
    )


def exhaustive_ppi(image, segment_length, nb_segments):
    """Enumerate every fixed-cone polygonal path from every image pixel."""
    rows, columns = image.shape
    costs = np.full(image.shape, np.inf, dtype=np.float64)
    coordinates = np.full((*image.shape, nb_segments, 2), -1, dtype=np.int64)
    for start in product(range(rows), range(columns)):
        for cone in reference_cones(segment_length):
            cone_cost = np.inf
            cone_path = None
            for steps in product(cone, repeat=nb_segments):
                point = start
                path = []
                path_cost = 0
                for dr, dc in steps:
                    endpoint = (point[0] + dr, point[1] + dc)
                    if not (0 <= endpoint[0] < rows and 0 <= endpoint[1] < columns):
                        break
                    path_cost += sum(int(image[p]) for p in reference_line(point, endpoint)[1:])
                    path.append(endpoint)
                    point = endpoint
                else:
                    # The last endpoint wins at each step of an equal-cost path.
                    if path_cost <= cone_cost:
                        cone_cost, cone_path = path_cost, path
            # The first cone wins an equal-cost comparison across cones.
            if cone_cost < costs[start]:
                costs[start] = cone_cost
                coordinates[start] = cone_path
    return costs, coordinates


def assert_valid_paths(image, costs, coordinates, segment_length, nb_segments):
    """Verify geometry and recompute every reported finite path's pixel cost."""
    assert costs.shape == image.shape
    assert costs.dtype == np.float64
    assert coordinates.shape == (*image.shape, nb_segments, 2)
    assert coordinates.dtype == np.int64
    assert not np.isnan(costs).any()
    cones = tuple(set(cone) for cone in reference_cones(segment_length))
    rows, columns = image.shape
    for start in product(range(rows), range(columns)):
        path = coordinates[start]
        if np.isinf(costs[start]):
            assert costs[start] > 0
            assert np.all(path == -1)
            continue
        points = [start] + [tuple(map(int, point)) for point in path]
        assert all(0 <= r < rows and 0 <= c < columns for r, c in points)
        steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
        assert any(all(step in cone for step in steps) for cone in cones)
        reconstructed_cost = sum(
            int(image[pixel])
            for a, b in zip(points, points[1:])
            for pixel in reference_line(a, b)[1:]
        )
        assert reconstructed_cost == costs[start], (
            start,
            costs[start],
            reconstructed_cost,
            path.tolist(),
        )
