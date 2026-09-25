"""Minimum-cost polygonal paths through grayscale potential images.

Coordinates are always ``(row, column)``. The intensity of the starting pixel
does not contribute to a path's cost; each segment includes its final pixel.
"""

from __future__ import annotations

import sys
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from ._core import compute_ppi_kernel


def _integer(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean")
    result = int(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def bresenham_line(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int]]:
    """Return raster coordinates between two points, including both endpoints.

    Signed integer coordinates are supported. At half-pixel ties this uses the
    same Bresenham convention as the historical Polygonal Path Image code.
    A zero-length line contains its single endpoint.
    """
    row0 = _integer(row0, "row0")
    col0 = _integer(col0, "col0")
    row1 = _integer(row1, "row1")
    col1 = _integer(col1, "col1")
    x, y = row0, col0
    dx, dy = abs(row1 - row0), abs(col1 - col0)
    sx = 1 if row1 > row0 else -1
    sy = 1 if col1 > col0 else -1
    steep = dy > dx
    if steep:
        x, y = y, x
        dx, dy = dy, dx
        sx, sy = sy, sx
    error = 2 * dy - dx
    coordinates = []
    for _ in range(dx):
        coordinates.append((y, x) if steep else (x, y))
        if error >= 0:
            y += sy
            error -= 2 * dx
        x += sx
        error += 2 * dy
    coordinates.append((row1, col1))
    return coordinates


def _cone_endpoints(length: int) -> NDArray[np.int64]:
    """Return endpoint displacements in legacy H, B, E, W and tie order."""
    length = _integer(length, "length", positive=True)
    # Check byte counts with Python integers before NumPy or C-sized conversion.
    if 4 * (2 * length + 1) * 2 * np.dtype(np.int64).itemsize > sys.maxsize:
        raise ValueError("segment_length is too large to represent cone endpoints")
    offsets = np.arange(-length, length + 1, dtype=np.int64)
    endpoints = np.empty((4, offsets.size, 2), dtype=np.int64)
    endpoints[0, :, 0] = -length
    endpoints[0, :, 1] = -offsets
    endpoints[1, :, 0] = length
    endpoints[1, :, 1] = offsets
    endpoints[2, :, 0] = -offsets
    endpoints[2, :, 1] = length
    endpoints[3, :, 0] = offsets
    endpoints[3, :, 1] = -length
    return endpoints


def compute_ppi(
    image: NDArray[np.uint8], segment_length: int = 3, nb_segments: int = 10
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Find a minimum-cost fixed-length polygonal path from every image pixel.

    Parameters
    ----------
    image
        Nonempty two-dimensional NumPy array with dtype ``uint8``. Lower pixel
        intensities are cheaper. Noncontiguous and read-only arrays are accepted.
    segment_length
        Positive integer displacement along a cone's main axis. Each segment
        uses exactly this many raster steps, including diagonal segments.
    nb_segments
        Positive number of segments per path. All segments stay within a single
        one of four cardinal 90-degree cones.

    Returns
    -------
    costs
        ``float64`` array of shape ``(height, width)``. Costs sum raster pixel
        intensities, excluding the starting pixel of each segment and including
        its endpoint. An impossible path has cost ``np.inf``.
    paths
        ``int64`` array of shape ``(height, width, nb_segments, 2)`` containing
        successive segment endpoints, in ``(row, column)`` order. The starting
        pixel is implicit. All coordinates are ``-1`` for an impossible path.

    Notes
    -----
    Equal-cost candidates within a cone select the last endpoint in the legacy
    endpoint order. Equal-cost cones select the first in H (up), B (down), E
    (right), W (left) order. Immutable dynamic-programming backpointers keep
    returned coordinates consistent with their reported costs.

    This function performs no plotting or file I/O and does not modify ``image``.
    Working memory is O(height * width * (segment_length + nb_segments)).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array with dtype uint8")
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if image.size == 0:
        raise ValueError("image must be nonempty")
    if image.dtype != np.dtype(np.uint8):
        raise TypeError("image must have dtype uint8; convert intensities explicitly")
    segment_length = _integer(segment_length, "segment_length", positive=True)
    nb_segments = _integer(nb_segments, "nb_segments", positive=True)
    rows, cols = image.shape
    if rows * cols * nb_segments * 2 * np.dtype(np.int64).itemsize > sys.maxsize:
        raise ValueError("nb_segments is too large to represent the output paths")

    # A cone advances by exactly length pixels on its main axis at each step.
    # Avoid constructing enormous endpoint tables for trivially impossible paths.
    if segment_length * nb_segments > max(rows - 1, cols - 1):
        return (
            np.full((rows, cols), np.inf, dtype=np.float64),
            np.full((rows, cols, nb_segments, 2), -1, dtype=np.int64),
        )
    endpoints = _cone_endpoints(segment_length)
    return compute_ppi_kernel(np.ascontiguousarray(image), endpoints, nb_segments)
