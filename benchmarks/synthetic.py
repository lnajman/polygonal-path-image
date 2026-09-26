"""NumPy-only synthetic curves and explicitly defined geometric measurements.

Coordinates are ``(row, column)``. Angles increase clockwise from the positive
column direction. These are controlled image-processing measurements, not a
reproduction of the clinical evaluation in the MICCAI paper.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class SyntheticCase:
    """An image, a subpixel reference polyline, its evaluation ROI, and inputs."""

    image: np.ndarray
    centerline: np.ndarray
    roi: np.ndarray
    parameters: dict
    distance_map: np.ndarray


def _points(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return result


def _positive_integer(value, name, *, zero_allowed=False):
    minimum = 0 if zero_allowed else 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def point_to_polyline_distance(points, centerline, *, batch_size=256):
    """Exact Euclidean distances to the finite segments of a reference polyline.

    A single reference vertex represents a point. Empty references produce
    infinity. Both axes are batched, bounding temporary pairwise arrays by
    ``batch_size ** 2`` irrespective of image or curve size. Repeated vertices
    are valid zero-length segments.
    """
    points = _points(points, "points")
    centerline = _points(centerline, "centerline")
    batch_size = _positive_integer(batch_size, "batch_size")
    result = np.full(len(points), np.inf, dtype=np.float64)
    if not len(centerline):
        return result
    starts = centerline[:-1] if len(centerline) > 1 else centerline
    ends = centerline[1:] if len(centerline) > 1 else centerline
    for p0 in range(0, len(points), batch_size):
        block = points[p0 : p0 + batch_size]
        best_squared = np.full(len(block), np.inf)
        for s0 in range(0, len(starts), batch_size):
            start = starts[s0 : s0 + batch_size]
            direction = ends[s0 : s0 + batch_size] - start
            length_squared = np.sum(direction * direction, axis=1)
            relative = block[:, None, :] - start[None, :, :]
            numerator = np.sum(relative * direction[None, :, :], axis=2)
            position = np.divide(
                numerator,
                length_squared[None, :],
                out=np.zeros_like(numerator),
                where=length_squared[None, :] != 0,
            )
            np.clip(position, 0, 1, out=position)
            relative -= position[:, :, None] * direction[None, :, :]
            squared = np.sum(relative * relative, axis=2)
            best_squared = np.minimum(best_squared, np.min(squared, axis=1))
        result[p0 : p0 + len(block)] = np.sqrt(best_squared)
    return result


def _nearest_point_distance(points, reference, *, batch_size=256):
    """Exact nearest-point distances with bounded temporary arrays."""
    result = np.full(len(points), np.inf, dtype=np.float64)
    for p0 in range(0, len(points), batch_size):
        block = points[p0 : p0 + batch_size]
        best_squared = np.full(len(block), np.inf)
        for r0 in range(0, len(reference), batch_size):
            delta = block[:, None, :] - reference[None, r0 : r0 + batch_size, :]
            squared = np.sum(delta * delta, axis=2)
            best_squared = np.minimum(best_squared, np.min(squared, axis=1))
        result[p0 : p0 + len(block)] = np.sqrt(best_squared)
    return result


def _sample_polyline(centerline, sample_step):
    """Sample uniformly in polyline arclength and give trapezoidal weights."""
    if len(centerline) < 2:
        return centerline.copy(), np.ones(len(centerline)), 0.0
    lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total == 0:
        return centerline[:1].copy(), np.ones(1), total
    # Remove repeated vertices so interpolation has strictly increasing knots.
    keep = np.concatenate(([True], lengths > 0))
    intervals = int(np.ceil(total / sample_step))
    locations = np.linspace(0, total, intervals + 1)
    sampled = np.column_stack(
        [np.interp(locations, cumulative[keep], centerline[keep, axis]) for axis in range(2)]
    )
    weights = np.full(intervals + 1, total / intervals)
    weights[[0, -1]] *= 0.5
    return sampled, weights, total


def _weighted_p95(values, weights):
    """Smallest observed value reaching 95% of cumulative observation weight."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, 0.95 * cumulative[-1], side="left")
    return float(values[order[index]])


def generate_case(
    *,
    angle_deg=0.0,
    bend_deg=0.0,
    contrast=60.0,
    noise_std=10.0,
    seed=0,
    size=96,
    arclength=48.0,
    sigma=1.25,
    background=200.0,
    roi_margin=16,
    sample_step=0.25,
    target_present=True,
):
    """Generate one dark straight or circular tube in Gaussian intensity noise.

    ``angle_deg`` is the tangent angle at the middle of the arc. ``bend_deg``
    is its total signed turning angle. Reference vertices lie on the analytic
    arc at equally spaced arclength positions, at most ``sample_step`` apart.
    The image uses exact distance to this densely sampled *polyline*, then
    ``background - contrast * exp(-distance**2 / (2*sigma**2))``. Independent
    Gaussian noise is added before rounding and clipping to uint8. Thus the
    actual intensity range can clip; ``noise_std`` describes preclip noise.

    The rotated curve's bounding box is centered on the image. The reference
    plus a two-pixel evaluation tube must fit completely inside the ROI.
    ``target_present=False`` makes a blank/noise-only control with an empty
    reference; zero contrast alone retains a geometrically defined target.
    """
    size = _positive_integer(size, "size")
    roi_margin = _positive_integer(roi_margin, "roi_margin", zero_allowed=True)
    seed = _positive_integer(seed, "seed", zero_allowed=True)
    values = np.array(
        [angle_deg, bend_deg, contrast, noise_std, arclength, sigma, background, sample_step],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("all curve parameters must be finite")
    angle_deg, bend_deg, contrast, noise_std, arclength, sigma, background, sample_step = values
    if arclength <= 0 or sigma <= 0 or not 0 < sample_step <= 0.25:
        raise ValueError("arclength and sigma must be positive; sample_step must be in (0, .25]")
    if abs(bend_deg) > 180:
        raise ValueError("bend_deg must be between -180 and 180")
    if noise_std < 0 or not 0 <= contrast <= background <= 255:
        raise ValueError("require noise_std >= 0 and 0 <= contrast <= background <= 255")
    if 2 * roi_margin >= size:
        raise ValueError("roi_margin leaves no evaluation pixels")
    if not isinstance(target_present, (bool, np.bool_)):
        raise ValueError("target_present must be boolean")
    positions = np.linspace(-arclength / 2, arclength / 2, int(np.ceil(arclength / sample_step)) + 1)
    curvature = np.deg2rad(bend_deg) / arclength
    if curvature == 0:
        x, y = positions, np.zeros_like(positions)
    else:
        x = np.sin(curvature * positions) / curvature
        # The half-angle form avoids cancellation for nearly straight arcs.
        y = 2 * np.sin(curvature * positions / 2) ** 2 / curvature
    angle = np.deg2rad(angle_deg)
    centerline = np.column_stack(
        (x * np.sin(angle) + y * np.cos(angle), x * np.cos(angle) - y * np.sin(angle))
    )
    centerline -= (centerline.max(axis=0) + centerline.min(axis=0)) / 2
    centerline += (size - 1) / 2
    if centerline.min() - 2 < roi_margin or centerline.max() + 2 > size - 1 - roi_margin:
        raise ValueError("reference curve plus its two-pixel tube must fit inside the ROI")
    if not target_present:
        centerline = np.empty((0, 2), dtype=np.float64)
    pixels = np.indices((size, size)).reshape(2, -1).T
    distance = point_to_polyline_distance(pixels, centerline).reshape(size, size)
    clean = background - contrast * np.exp(-(distance * distance) / (2 * sigma * sigma))
    noise = np.random.default_rng(seed).normal(0, noise_std, clean.shape)
    image = np.rint(np.clip(clean + noise, 0, 255)).astype(np.uint8)
    roi = np.zeros(image.shape, dtype=bool)
    roi[roi_margin : size - roi_margin, roi_margin : size - roi_margin] = True
    parameters = {
        "angle_deg": float(angle_deg),
        "bend_deg": float(bend_deg),
        "contrast": float(contrast),
        "noise_std": float(noise_std),
        "seed": seed,
        "size": size,
        "arclength": float(arclength),
        "sigma": float(sigma),
        "background": float(background),
        "roi_margin": roi_margin,
        "sample_step": float(sample_step),
        "target_present": bool(target_present),
    }
    return SyntheticCase(image, centerline, roi, parameters, distance)


def select_at_budget(scores, roi, *, fraction=0.02):
    """Select high, finite, positive scores using a predeclared ROI area budget.

    The nominal count is ``ceil(fraction * ROI pixel count)``. If the cutoff
    score is tied, the complete tied block is included: there is no arbitrary
    coordinate-based tie break. Fewer positive scores yield fewer selections;
    NaN, infinity, zero and negative scores are never selected. Fraction zero
    selects nothing. The actual fraction can exceed the nominal fraction.
    """
    scores = np.asarray(scores)
    roi = np.asarray(roi)
    if scores.ndim != 2 or roi.shape != scores.shape or roi.dtype != bool:
        raise ValueError("scores must be 2D and roi a matching boolean array")
    if not np.issubdtype(scores.dtype, np.number) or np.iscomplexobj(scores):
        raise ValueError("scores must be real numbers")
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    roi_pixels = int(roi.sum())
    budget = int(np.ceil(float(fraction) * roi_pixels))
    candidates = roi & np.isfinite(scores) & (scores > 0)
    positive_pixels = int(candidates.sum())
    selected = np.zeros(scores.shape, dtype=bool)
    threshold = None
    boundary_tie_pixels = 0
    if budget and positive_pixels:
        count = min(budget, positive_pixels)
        candidate_scores = scores[candidates]
        threshold = float(np.partition(candidate_scores, positive_pixels - count)[-count])
        selected = candidates & (scores >= threshold)
        boundary_tie_pixels = int((candidates & (scores == threshold)).sum())
    selected_pixels = int(selected.sum())
    return selected, {
        "requested_fraction": float(fraction),
        "actual_fraction": selected_pixels / roi_pixels if roi_pixels else None,
        "threshold": threshold,
        "positive_pixels": positive_pixels,
        "budget_pixels": budget,
        "selected_pixels": selected_pixels,
        "roi_pixels": roi_pixels,
        "boundary_tie_pixels": boundary_tie_pixels,
        "budget_exceeded_by_ties": max(0, selected_pixels - budget),
    }


def evaluate_prediction(
    prediction, centerline, roi, *, tolerance=2.0, sample_step=0.25, distance_map=None
):
    """Measure a boolean selection against a continuous reference polyline.

    Only selected pixels inside ``roi`` count. Precision is the fraction of
    those pixel centers within ``tolerance`` of the reference's finite
    segments. The off-tube false-positive rate divides selected off-tube
    pixels by *all* off-tube ROI pixels.

    Coverage and the reference-to-prediction mean use reference samples with
    spacing <= ``sample_step`` in polyline arclength and trapezoidal weights.
    Reference samples belong to the ROI if their nearest pixel belongs to it;
    defaults keep the entire synthetic curve within the ROI. Distances to
    predictions use the centers of the selected pixels, not an inferred curve.
    The prediction-to-reference mean instead uses exact segment distances.
    P95 distances are inverse empirical cumulative-distribution quantiles
    (smallest observed distance reaching 95% of weight), with the same weights
    as the respective means. ``distance_map`` may cache pixel-center distances
    to the same reference, as supplied by ``generate_case``.

    With no predictions, coverage is zero for a nonempty reference, precision
    and both mean distances are null (no artificial distance cap). With no
    reference in the ROI, coverage and all reference-dependent means/precision
    are null. An empty reference makes all ROI pixels off-tube. Zero-sized
    denominators produce null. A single reference point has unit weight.
    """
    prediction = np.asarray(prediction)
    roi = np.asarray(roi)
    if prediction.ndim != 2 or prediction.dtype != bool:
        raise ValueError("prediction must be a 2D boolean array")
    if roi.shape != prediction.shape or roi.dtype != bool:
        raise ValueError("roi must be a matching boolean array")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not np.isfinite(sample_step) or not 0 < sample_step <= 0.25:
        raise ValueError("sample_step must be in (0, .25]")
    centerline = _points(centerline, "centerline")
    pixels = np.argwhere(roi)
    if distance_map is None:
        distances = point_to_polyline_distance(pixels, centerline)
    else:
        distance_map = np.asarray(distance_map, dtype=np.float64)
        if distance_map.shape != roi.shape or np.isnan(distance_map).any() or (distance_map < 0).any():
            raise ValueError("distance_map must match the image and contain nonnegative distances")
        distances = distance_map[roi]
    selections = prediction[roi]
    inside_tube = distances <= tolerance
    chosen = pixels[selections]
    roi_pixels = len(pixels)
    selected_pixels = len(chosen)
    true_positive_pixels = int((selections & inside_tube).sum())
    false_positive_pixels = int((selections & ~inside_tube).sum())
    off_tube_pixels = int((~inside_tube).sum())
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
    samples, weights = samples[keep], weights[keep]
    has_target = bool(len(samples))
    coverage = None
    prediction_to_centerline = None
    centerline_to_prediction = None
    prediction_to_centerline_p95 = None
    centerline_to_prediction_p95 = None
    precision = None
    if has_target:
        coverage = 0.0
        if selected_pixels:
            reference_distances = _nearest_point_distance(samples, chosen)
            coverage = float(np.average(reference_distances <= tolerance, weights=weights))
            centerline_to_prediction = float(np.average(reference_distances, weights=weights))
            prediction_to_centerline = float(np.mean(distances[selections]))
            prediction_to_centerline_p95 = _weighted_p95(
                distances[selections], np.ones(selected_pixels)
            )
            centerline_to_prediction_p95 = _weighted_p95(reference_distances, weights)
            precision = true_positive_pixels / selected_pixels
    return {
        "tolerance_pixels": float(tolerance),
        "reference_sample_step": float(sample_step),
        "roi_pixels": roi_pixels,
        "selected_pixels": selected_pixels,
        "selected_fraction": selected_pixels / roi_pixels if roi_pixels else None,
        "centerline_samples": len(samples),
        "centerline_length": polyline_length,
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
