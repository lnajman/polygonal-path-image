"""Scalable geometric measurements for acquired Guide3D images.

Definitions match :mod:`benchmarks.synthetic`: exact pixel-to-finite-polyline
distances, reference samples spaced by arclength, trapezoidal sample weights,
nearest-pixel ROI membership, and empirical weighted P95. An exact SciPy
``cKDTree`` query replaces the quadratic reference-sample-to-prediction search.
SciPy is a study dependency, not a package runtime dependency.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from benchmarks.synthetic import (
    _points,
    _sample_polyline,
    _weighted_p95,
    point_to_polyline_distance,
    select_at_budget,
)

__all__ = [
    "ReferenceGeometry",
    "evaluate_prediction",
    "prepare_reference",
    "select_at_budget",
]


@dataclass(frozen=True)
class ReferenceGeometry:
    """Reference-only work cached once per image and evaluation ROI.

    Construct with :func:`prepare_reference`, then reuse ``evaluate`` for
    different score maps, area budgets, and tolerances. Coordinate distances
    and tolerances use the supplied image's pixel units. This helper performs
    no image resizing or conversion to physical/native-resolution units.

    ``centerline_length`` describes the complete input polyline. Samples and
    their weights are restricted by nearest-pixel ROI membership, matching
    the synthetic benchmark; this is not exact geometric segment clipping.
    """

    roi: np.ndarray
    pixels: np.ndarray
    distances: np.ndarray
    samples: np.ndarray
    weights: np.ndarray
    centerline_length: float
    sample_step: float

    def evaluate(self, prediction, *, tolerance=2.0):
        """Return synthetic evaluator measurements plus retained ROI lengths.

        Only selected pixel centers inside the ROI count. A KD-tree query
        uses Euclidean distance, ``eps=0`` (no approximate search), and one
        worker. Empty prediction/reference cases retain the original null
        conventions. Reference distance metrics use the *whole* input
        polyline, even where that polyline extends outside the ROI.
        """
        prediction = np.asarray(prediction)
        if prediction.ndim != 2 or prediction.dtype != bool:
            raise ValueError("prediction must be a 2D boolean array")
        if prediction.shape != self.roi.shape:
            raise ValueError("roi must be a matching boolean array")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        selections = prediction[self.roi]
        inside_tube = self.distances <= tolerance
        chosen = self.pixels[selections]
        roi_pixels = len(self.pixels)
        selected_pixels = len(chosen)
        true_positive_pixels = int((selections & inside_tube).sum())
        false_positive_pixels = int((selections & ~inside_tube).sum())
        off_tube_pixels = int((~inside_tube).sum())
        has_target = bool(len(self.samples))
        coverage = None
        prediction_to_centerline = None
        centerline_to_prediction = None
        prediction_to_centerline_p95 = None
        centerline_to_prediction_p95 = None
        precision = None
        if has_target:
            coverage = 0.0
            if selected_pixels:
                reference_distances = cKDTree(chosen).query(
                    self.samples, k=1, eps=0.0, p=2, workers=1
                )[0]
                coverage = float(np.average(reference_distances <= tolerance, weights=self.weights))
                centerline_to_prediction = float(
                    np.average(reference_distances, weights=self.weights)
                )
                prediction_to_centerline = float(np.mean(self.distances[selections]))
                prediction_to_centerline_p95 = _weighted_p95(
                    self.distances[selections], np.ones(selected_pixels)
                )
                centerline_to_prediction_p95 = _weighted_p95(reference_distances, self.weights)
                precision = true_positive_pixels / selected_pixels
        # A singleton/zero-length reference has unit metric weight but zero
        # physical arclength. Its retained-length fraction is undefined.
        evaluated_reference_length = (
            float(self.weights.sum()) if self.centerline_length > 0 else 0.0
        )
        return {
            "tolerance_pixels": float(tolerance),
            "reference_sample_step": float(self.sample_step),
            "roi_pixels": roi_pixels,
            "selected_pixels": selected_pixels,
            "selected_fraction": selected_pixels / roi_pixels if roi_pixels else None,
            "centerline_samples": len(self.samples),
            "centerline_length": self.centerline_length,
            "evaluated_reference_length": evaluated_reference_length,
            "reference_length_fraction": (
                evaluated_reference_length / self.centerline_length
                if self.centerline_length > 0
                else None
            ),
            "has_target": has_target,
            "centerline_coverage": coverage,
            "prediction_to_centerline_mean_distance": prediction_to_centerline,
            "centerline_to_prediction_mean_distance": centerline_to_prediction,
            "prediction_to_centerline_p95_distance": prediction_to_centerline_p95,
            "centerline_to_prediction_p95_distance": centerline_to_prediction_p95,
            "selected_pixel_precision": precision,
            "true_positive_pixels": true_positive_pixels,
            "false_positive_pixels": false_positive_pixels,
            "off_tube_pixels": off_tube_pixels,
            "off_tube_false_positive_rate": (
                false_positive_pixels / off_tube_pixels if off_tube_pixels else None
            ),
        }


def prepare_reference(centerline, roi, *, sample_step=0.25, distance_map=None):
    """Cache exact distances and reference samples for one image's ROI.

    ``centerline`` contains finite ``(row, column)`` vertices. Empty references
    and repeated vertices are valid. Optional ``distance_map`` must contain
    pixel-center distances to that same reference at the image's resolution;
    its values are trusted, as in the synthetic evaluator. Infinity is valid
    for an empty reference. Only ROI pixels need exact finite-segment queries.

    The finite-segment routine batches both points and segments, bounding its
    temporary arrays. Samples retain the full-polyline trapezoidal weights
    after nearest-pixel ROI filtering, including for holes or boundary cuts.
    """
    roi = np.asarray(roi)
    if roi.ndim != 2 or roi.dtype != bool:
        raise ValueError("roi must be a 2D boolean array")
    if not np.isfinite(sample_step) or not 0 < sample_step <= 0.25:
        raise ValueError("sample_step must be in (0, .25]")
    centerline = _points(centerline, "centerline")
    pixels = np.argwhere(roi)
    if distance_map is None:
        distances = point_to_polyline_distance(pixels, centerline)
    else:
        distance_map = np.asarray(distance_map, dtype=np.float64)
        if (
            distance_map.shape != roi.shape
            or np.isnan(distance_map).any()
            or (distance_map < 0).any()
        ):
            raise ValueError("distance_map must match the image and contain nonnegative distances")
        distances = distance_map[roi]
    samples, weights, polyline_length = _sample_polyline(centerline, sample_step)
    nearest_pixels = np.floor(samples + 0.5).astype(np.int64)
    in_bounds = (
        (nearest_pixels[:, 0] >= 0)
        & (nearest_pixels[:, 0] < roi.shape[0])
        & (nearest_pixels[:, 1] >= 0)
        & (nearest_pixels[:, 1] < roi.shape[1])
    )
    keep = np.zeros(len(samples), dtype=bool)
    keep[in_bounds] = roi[tuple(nearest_pixels[in_bounds].T)]
    # Own and freeze the arrays so changing caller-owned masks cannot corrupt
    # a reference cache reused across configurations and area budgets.
    arrays = [roi.copy(), pixels, distances, samples[keep], weights[keep]]
    for array in arrays:
        array.flags.writeable = False
    return ReferenceGeometry(*arrays, polyline_length, float(sample_step))


def evaluate_prediction(
    prediction, centerline, roi, *, tolerance=2.0, sample_step=0.25, distance_map=None
):
    """Scalable equivalent of ``synthetic.evaluate_prediction`` plus ROI lengths.

    For repeated evaluations of the same reference/ROI, construct
    :func:`prepare_reference` once and call its ``evaluate`` method instead.
    No artificial cap replaces undefined distances; these remain ``None``.
    Additional ``evaluated_reference_length`` and ``reference_length_fraction``
    report retained trapezoidal arclength weight and its fraction of the full
    reference length. A single-point/zero-length reference has evaluated
    length zero and a null length fraction, while retaining unit metric weight.
    """
    prediction = np.asarray(prediction)
    roi = np.asarray(roi)
    if prediction.ndim != 2 or prediction.dtype != bool:
        raise ValueError("prediction must be a 2D boolean array")
    if roi.shape != prediction.shape or roi.dtype != bool:
        raise ValueError("roi must be a matching boolean array")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    geometry = prepare_reference(
        centerline, roi, sample_step=sample_step, distance_map=distance_map
    )
    return geometry.evaluate(prediction, tolerance=tolerance)
