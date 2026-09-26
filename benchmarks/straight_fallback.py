"""Experimental replacement of rejected winners by fixed-direction candidates.

Research tooling only: this is not the globally optimal smooth path, and it
does not change the released PPI API. It preserves all accepted original paths.
"""

import numpy as np

from polygonal_path_image import bresenham_line, compute_ppi, filter_tortuosity, voting
from polygonal_path_image.core import _cone_endpoints, _integer


def minimum_straight_paths(image, segment_length=3, nb_segments=10):
    """Minimize cost over paths repeating one original endpoint displacement.

    Candidate order is H/B/E/W and the existing within-cone endpoint order;
    the first minimum wins ties. All segments share a displacement, so their
    continuous polyline is straight. Rasterization/cost exclusion match PPI.
    Impossible origins retain inf costs and -1 coordinates.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2 or not image.size:
        raise ValueError("image must be a nonempty two-dimensional array")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")
    length = _integer(segment_length, "segment_length", positive=True)
    count = _integer(nb_segments, "nb_segments", positive=True)
    rows, columns = image.shape
    costs = np.full(image.shape, np.inf)
    paths = np.full((*image.shape, count, 2), -1, dtype=np.int64)
    if length * count > max(rows - 1, columns - 1):
        return costs, paths
    for displacement in _cone_endpoints(length).reshape(-1, 2):
        dy, dx = map(int, displacement)
        end_y, end_x = count * dy, count * dx
        y0, y1 = max(0, -end_y), min(rows, rows - end_y)
        x0, x1 = max(0, -end_x), min(columns, columns - end_x)
        if y0 >= y1 or x0 >= x1:
            continue
        candidate = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
        segment = bresenham_line(0, 0, dy, dx)[1:]
        for step in range(count):
            for py, px in segment:
                oy, ox = step * dy + py, step * dx + px
                candidate += image[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
        better = candidate < costs[y0:y1, x0:x1]
        costs[y0:y1, x0:x1][better] = candidate[better]
        yy, xx = np.nonzero(better)
        yy, xx = yy + y0, xx + x0
        for step in range(1, count + 1):
            paths[yy, xx, step - 1, 0] = yy + step * dy
            paths[yy, xx, step - 1, 1] = xx + step * dx
    return costs, paths


def straight_fallback(image, *, segment_length=3, nb_segments=10, threshold=0.75):
    """Keep accepted winners; fill rejected origins with the cheapest straight path.

    Returns votes, replacement costs/paths, and a boolean replacement mask.
    Original impossible paths are left impossible. No intensity/contrast gate
    suppresses a blank image; nonzero votes there are expected path occupancy.
    """
    costs, paths = compute_ppi(image, segment_length, nb_segments)
    filtered = filter_tortuosity(costs, paths, threshold)
    straight_costs, straight_paths = minimum_straight_paths(image, segment_length, nb_segments)
    replace = np.isfinite(costs) & ~np.isfinite(filtered) & np.isfinite(straight_costs)
    result_paths = paths.copy()
    result_paths[replace] = straight_paths[replace]
    filtered[replace] = straight_costs[replace]
    votes, _ = voting(filtered, result_paths)
    return votes, filtered, result_paths, replace
