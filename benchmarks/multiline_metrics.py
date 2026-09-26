"""Exact union geometry and separate-line coverage for complete synthetic truth.

Pixel distances use the union of finite reference polylines, never artificial
segments connecting different lines. Coverage and gaps are computed separately
for each line, so crossings contribute to each line's recovery. The declared
width-aware support is the union of each line's ``support_sigma * sigma`` tube.
The scene generator independently defines the intensity profile and whether
signal is truncated at this declared support boundary.
SciPy is loaded lazily only for an exact nearest-selected-pixel query.
"""

from dataclasses import dataclass

import numpy as np

from benchmarks.synthetic import (
    _points,
    _sample_polyline,
    _weighted_p95,
    point_to_polyline_distance,
    select_at_budget,
)

__all__ = ["MultilineGeometry", "prepare_multiline", "select_at_budget"]


@dataclass(frozen=True)
class _LineGeometry:
    sigma: float
    samples: np.ndarray
    weights: np.ndarray
    sample_indices: np.ndarray
    length: float


def _longest_uncovered(line, uncovered):
    """Sum quadrature weights in the largest consecutive uncovered sample run."""
    if line.length == 0:
        return 0.0
    positions = np.flatnonzero(uncovered)
    if not len(positions):
        return 0.0
    # Original sample indices preserve breaks introduced by ROI clipping, and
    # covered samples introduce their own breaks. Different lines never share
    # this calculation. No connector bridges either kind of missing interval.
    boundaries = np.flatnonzero(np.diff(line.sample_indices[positions]) != 1) + 1
    starts = np.concatenate(([0], boundaries))
    return float(np.max(np.add.reduceat(line.weights[positions], starts)))


@dataclass(frozen=True)
class MultilineGeometry:
    """Cached union distances, declared support, and ordered per-line references.

    Construct through :func:`prepare_multiline`; evaluate any number of boolean
    prediction masks with ``evaluate``. Line order is preserved. Distances and
    lengths use the input image's pixel units. Arrays owned by the cache are
    read-only, so later modification of caller inputs cannot corrupt geometry.
    """

    roi: np.ndarray
    pixels: np.ndarray
    union_distances: np.ndarray
    support: np.ndarray
    lines: tuple
    tolerance: float
    sample_step: float
    support_sigma: float

    @property
    def support_mask(self):
        """Read-only full-image declared-support mask, false outside the ROI."""
        mask = np.zeros(self.roi.shape, dtype=bool)
        mask[self.roi] = self.support
        mask.flags.writeable = False
        return mask

    @property
    def outside_support_mask(self):
        """Read-only full-image ROI pixels outside every declared support tube."""
        mask = np.zeros(self.roi.shape, dtype=bool)
        mask[self.roi] = ~self.support
        mask.flags.writeable = False
        return mask

    @property
    def union_distance_map(self):
        """Read-only full-image union distances; unevaluated outside-ROI pixels are infinity."""
        distances = np.full(self.roi.shape, np.inf)
        distances[self.roi] = self.union_distances
        distances.flags.writeable = False
        return distances

    def evaluate(self, prediction):
        """Measure line recovery and selected pixels against complete references.

        ``outside_support_fraction`` divides selected outside-support pixels by
        *all outside-support ROI pixels*, not by selected pixels. Selection
        precision uses the union, so crossing pixels count once. Macro coverage
        weights evaluated lines equally; lines with no reference samples inside
        the ROI have null coverage and are excluded from this average.

        ``longest_uncovered_arclength`` is the largest sum of trapezoidal sample
        weights in a consecutive uncovered run, an approximation at the declared
        sample step, not an exact continuous gap length. ROI exclusions break
        runs. Zero-length references have zero gap length but unit coverage
        weight, matching the single-line benchmark convention.

        Blank controls have null geometric reference metrics and all ROI pixels
        outside support. Their support precision is zero when pixels are selected
        and null otherwise. With no predictions, nonempty evaluated lines have
        coverage zero and undefined distance/precision metrics remain null.
        """
        prediction = np.asarray(prediction)
        if prediction.ndim != 2 or prediction.dtype != bool:
            raise ValueError("prediction must be a 2D boolean array")
        if prediction.shape != self.roi.shape:
            raise ValueError("prediction must match the ROI shape")
        selections = prediction[self.roi]
        chosen = self.pixels[selections]
        selected_pixels = len(chosen)
        roi_pixels = len(self.pixels)
        evaluated_line_count = sum(bool(len(line.samples)) for line in self.lines)
        has_target = bool(evaluated_line_count)
        tree = None
        if selected_pixels and has_target:
            from scipy.spatial import cKDTree

            tree = cKDTree(chosen)
        per_line = []
        for index, line in enumerate(self.lines):
            coverage = None
            longest_gap = None
            mean_distance = None
            p95_distance = None
            if len(line.samples):
                coverage = 0.0
                uncovered = np.ones(len(line.samples), dtype=bool)
                if tree is not None:
                    distances = tree.query(line.samples, k=1, eps=0.0, p=2, workers=1)[0]
                    uncovered = distances > self.tolerance
                    coverage = float(np.average(~uncovered, weights=line.weights))
                    mean_distance = float(np.average(distances, weights=line.weights))
                    p95_distance = _weighted_p95(distances, line.weights)
                longest_gap = _longest_uncovered(line, uncovered)
            evaluated_length = float(line.weights.sum()) if line.length > 0 else 0.0
            per_line.append(
                {
                    "line_index": index,
                    "sigma_pixels": line.sigma,
                    "centerline_length": line.length,
                    "evaluated_reference_length": evaluated_length,
                    "reference_length_fraction": (
                        evaluated_length / line.length if line.length > 0 else None
                    ),
                    "reference_samples": len(line.samples),
                    "coverage": coverage,
                    "longest_uncovered_arclength": longest_gap,
                    "reference_to_prediction_mean_distance": mean_distance,
                    "reference_to_prediction_p95_distance": p95_distance,
                }
            )
        coverage_values = [line["coverage"] for line in per_line if line["coverage"] is not None]
        inside_centerline_tube = self.union_distances <= self.tolerance
        selected_centerline_pixels = int((selections & inside_centerline_tube).sum())
        selected_support_pixels = int((selections & self.support).sum())
        selected_outside_support_pixels = int((selections & ~self.support).sum())
        outside_support_pixels = int((~self.support).sum())
        union_precision = None
        support_precision = selected_support_pixels / selected_pixels if selected_pixels else None
        prediction_mean = None
        prediction_p95 = None
        if has_target and selected_pixels:
            union_precision = selected_centerline_pixels / selected_pixels
            prediction_mean = float(np.mean(self.union_distances[selections]))
            prediction_p95 = _weighted_p95(
                self.union_distances[selections], np.ones(selected_pixels)
            )
        return {
            "tolerance_pixels": self.tolerance,
            "reference_sample_step": self.sample_step,
            "support_sigma": self.support_sigma,
            "line_count": len(self.lines),
            "evaluated_line_count": evaluated_line_count,
            "has_target": has_target,
            "roi_pixels": roi_pixels,
            "selected_pixels": selected_pixels,
            "selected_fraction": selected_pixels / roi_pixels if roi_pixels else None,
            "per_line": per_line,
            "macro_coverage": float(np.mean(coverage_values)) if coverage_values else None,
            "worst_line_coverage": float(min(coverage_values)) if coverage_values else None,
            "union_centerline_precision": union_precision,
            "centerline_tube_pixels": int(inside_centerline_tube.sum()),
            "selected_centerline_pixels": selected_centerline_pixels,
            "support_pixels": int(self.support.sum()),
            "outside_support_pixels": outside_support_pixels,
            "selected_support_pixels": selected_support_pixels,
            "selected_outside_support_pixels": selected_outside_support_pixels,
            "selected_support_precision": support_precision,
            "outside_support_fraction": (
                selected_outside_support_pixels / outside_support_pixels
                if outside_support_pixels
                else None
            ),
            "prediction_to_reference_mean_distance": prediction_mean,
            "prediction_to_reference_p95_distance": prediction_p95,
        }


def prepare_multiline(
    centerlines, sigmas, roi, *, tolerance=2.0, sample_step=0.25, support_sigma=3.0
):
    """Prepare independent reference lines and their width-aware union support.

    ``centerlines`` is a sequence of finite ``(N, 2)`` row/column arrays;
    ``sigmas`` supplies one strictly positive Gaussian width in pixels per line.
    An empty sequence creates a blank control. Empty or repeated-vertex lines
    are accepted with the same null/point conventions as the original benchmark.

    Reference samples are uniformly spaced in full-polyline arclength with
    trapezoidal weights, then filtered by nearest-pixel ROI membership. Pixel
    distances remain exact distances to the complete finite input polylines.
    Preparation batches temporary segment-distance arrays and retains only one
    union distance vector instead of every line's dense pixel-distance map.
    """
    roi = np.asarray(roi)
    if roi.ndim != 2 or roi.dtype != bool:
        raise ValueError("roi must be a 2D boolean array")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not np.isfinite(sample_step) or not 0 < sample_step <= 0.25:
        raise ValueError("sample_step must be in (0, .25]")
    if not np.isfinite(support_sigma) or support_sigma <= 0:
        raise ValueError("support_sigma must be finite and positive")
    centerlines = [_points(line, "centerline") for line in centerlines]
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if sigmas.shape != (len(centerlines),) or not np.isfinite(sigmas).all() or (sigmas <= 0).any():
        raise ValueError("sigmas must contain one finite positive width per centerline")
    pixels = np.argwhere(roi)
    union_distances = np.full(len(pixels), np.inf)
    support = np.zeros(len(pixels), dtype=bool)
    lines = []
    for centerline, sigma in zip(centerlines, sigmas, strict=True):
        distances = point_to_polyline_distance(pixels, centerline)
        union_distances = np.minimum(union_distances, distances)
        support |= distances <= float(support_sigma) * sigma
        samples, weights, length = _sample_polyline(centerline, sample_step)
        nearest_pixels = np.floor(samples + 0.5).astype(np.int64)
        in_bounds = (
            (nearest_pixels[:, 0] >= 0)
            & (nearest_pixels[:, 0] < roi.shape[0])
            & (nearest_pixels[:, 1] >= 0)
            & (nearest_pixels[:, 1] < roi.shape[1])
        )
        keep = np.zeros(len(samples), dtype=bool)
        keep[in_bounds] = roi[tuple(nearest_pixels[in_bounds].T)]
        arrays = [samples[keep], weights[keep], np.flatnonzero(keep)]
        for array in arrays:
            array.flags.writeable = False
        lines.append(_LineGeometry(float(sigma), *arrays, length))
    roi = roi.copy()
    for array in (roi, pixels, union_distances, support):
        array.flags.writeable = False
    return MultilineGeometry(
        roi,
        pixels,
        union_distances,
        support,
        tuple(lines),
        float(tolerance),
        float(sample_step),
        float(support_sigma),
    )
