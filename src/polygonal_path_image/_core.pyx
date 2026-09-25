# cython: language_level=3, boundscheck=True, wraparound=False, initializedcheck=True
"""Compiled dynamic programming for :mod:`polygonal_path_image.core`."""

import numpy as np
cimport numpy as cnp
from libc.math cimport INFINITY

cnp.import_array()


cdef double _segment_cost(
    const cnp.uint8_t[:, ::1] image,
    Py_ssize_t row0,
    Py_ssize_t col0,
    Py_ssize_t row1,
    Py_ssize_t col1,
) except -1:
    """Legacy Bresenham raster sum, excluding the first pixel."""
    cdef Py_ssize_t x = row0
    cdef Py_ssize_t y = col0
    cdef Py_ssize_t dx = abs(row1 - row0)
    cdef Py_ssize_t dy = abs(col1 - col0)
    cdef Py_ssize_t sx = 1 if row1 > row0 else -1
    cdef Py_ssize_t sy = 1 if col1 > col0 else -1
    cdef Py_ssize_t error, index, temporary
    cdef bint steep = dy > dx
    cdef double result = 0
    if steep:
        temporary = x
        x = y
        y = temporary
        temporary = dx
        dx = dy
        dy = temporary
        temporary = sx
        sx = sy
        sy = temporary
    error = 2 * dy - dx
    for index in range(dx):
        if index > 0:
            if steep:
                result += image[y, x]
            else:
                result += image[x, y]
        if error >= 0:
            y += sy
            error -= 2 * dx
        x += sx
        error += 2 * dy
    if dx > 0:
        result += image[row1, col1]
    return result


def compute_ppi_kernel(
    const cnp.uint8_t[:, ::1] image,
    const cnp.int64_t[:, :, ::1] endpoints,
    Py_ssize_t nb_segments,
):
    """Compute validated cardinal-cone paths; use ``core.compute_ppi`` publicly.

    Each backpointer belongs to one immutable DP layer. Reconstruction descends
    these layers, so changing another pixel's path cannot invalidate a result.
    Bounds checks deliberately remain enabled.
    """
    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]
    cdef Py_ssize_t num_endpoints = endpoints.shape[1]
    cdef Py_ssize_t cone, iteration, row, col, endpoint, step
    cdef Py_ssize_t next_row, next_col, path_row, path_col
    cdef Py_ssize_t selected_endpoint
    cdef double candidate, minimum

    if rows == 0 or cols == 0 or nb_segments <= 0:
        raise ValueError("image must be nonempty and nb_segments must be positive")
    if endpoints.shape[0] != 4 or endpoints.shape[2] != 2 or num_endpoints == 0:
        raise ValueError("endpoints must have shape (4, number_of_endpoints, 2)")
    # Keep malformed direct calls from overflowing coordinate additions.
    if np.any(np.asarray(endpoints) >= max(rows, cols)) or np.any(
        np.asarray(endpoints) <= -max(rows, cols)
    ):
        raise ValueError("endpoint displacement exceeds the image dimensions")

    result_costs_array = np.full((rows, cols), np.inf, dtype=np.float64)
    result_paths_array = np.full(
        (rows, cols, nb_segments, 2), -1, dtype=np.int64
    )
    cdef double[:, ::1] result_costs = result_costs_array
    cdef cnp.int64_t[:, :, :, ::1] result_paths = result_paths_array
    cdef double[:, :, ::1] sums = np.empty(
        (rows, cols, num_endpoints), dtype=np.float64
    )
    cdef cnp.int64_t[:, :, ::1] choices = np.empty(
        (nb_segments, rows, cols), dtype=np.int64
    )
    cdef double[:, ::1] previous = np.empty((rows, cols), dtype=np.float64)
    cdef double[:, ::1] current = np.empty((rows, cols), dtype=np.float64)
    cdef double[:, ::1] temporary_costs

    for cone in range(4):
        for row in range(rows):
            for col in range(cols):
                previous[row, col] = 0
                for endpoint in range(num_endpoints):
                    next_row = row + endpoints[cone, endpoint, 0]
                    next_col = col + endpoints[cone, endpoint, 1]
                    if 0 <= next_row < rows and 0 <= next_col < cols:
                        sums[row, col, endpoint] = _segment_cost(
                            image, row, col, next_row, next_col
                        )
                    else:
                        sums[row, col, endpoint] = INFINITY

        for iteration in range(nb_segments):
            for row in range(rows):
                for col in range(cols):
                    minimum = INFINITY
                    selected_endpoint = -1
                    for endpoint in range(num_endpoints):
                        if sums[row, col, endpoint] == INFINITY:
                            continue
                        next_row = row + endpoints[cone, endpoint, 0]
                        next_col = col + endpoints[cone, endpoint, 1]
                        candidate = (
                            sums[row, col, endpoint] + previous[next_row, next_col]
                        )
                        # Last equal-cost endpoint wins within each cone.
                        if candidate < INFINITY and candidate <= minimum:
                            minimum = candidate
                            selected_endpoint = endpoint
                    current[row, col] = minimum
                    choices[iteration, row, col] = selected_endpoint
            temporary_costs = previous
            previous = current
            current = temporary_costs

        for row in range(rows):
            for col in range(cols):
                # First equal-cost cone wins across H, B, E, W.
                if previous[row, col] < result_costs[row, col]:
                    result_costs[row, col] = previous[row, col]
                    path_row, path_col = row, col
                    for step in range(nb_segments):
                        selected_endpoint = choices[
                            nb_segments - step - 1, path_row, path_col
                        ]
                        if selected_endpoint < 0:
                            raise RuntimeError("invalid finite-path backpointer")
                        path_row += endpoints[cone, selected_endpoint, 0]
                        path_col += endpoints[cone, selected_endpoint, 1]
                        result_paths[row, col, step, 0] = path_row
                        result_paths[row, col, step, 1] = path_col

    return result_costs_array, result_paths_array
