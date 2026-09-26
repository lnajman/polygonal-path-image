"""Trace weak-line voting loss through fixed PPI stages without changing PPI.

The selected single minimum-cost path per source is rasterized once. Exact
integer contribution maps partition sources by weak-band origin and visits to
the strong tube. Package voting/filtering and all reconstructed path costs are
checked, making this a diagnosis of existing behavior rather than an alternative
implementation. Unfiltered stages and raw tau0.5 have no calibrated operating
point; frozen cutoffs56/58 are displayed only as fixed-score diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

from benchmarks.guide3d_metrics import prepare_reference
from benchmarks.multiline_data import generate_multiline, multiline_specs
from benchmarks.run_synthetic import _provenance
from polygonal_path_image import bresenham_line, compute_ppi, filter_tortuosity, voting
from polygonal_path_image.postprocess import _validate_paths

THRESHOLDS = (0.75, 0.5)
SCORE_CUTOFFS = (56, 58)


def distribution(values):
    """Small finite distribution summary, with explicit nulls for empty groups."""
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
            "max": None,
            "mean": None,
            "zero_fraction": None,
        }
    if not np.isfinite(values).all():
        raise ValueError("distribution values must be finite")
    quantiles = np.quantile(values, (0, 0.25, 0.5, 0.75, 0.95, 1))
    return {
        "count": int(values.size),
        **dict(
            zip(
                ("min", "q25", "median", "q75", "q95", "max"),
                (float(value) for value in quantiles),
                strict=True,
            )
        ),
        "mean": float(np.mean(values)),
        "zero_fraction": float(np.mean(values == 0)),
    }


@dataclass(frozen=True)
class RasterPaths:
    """CSR pixel occurrences for active paths; joints once, later revisits retained."""

    shape: tuple
    origins: np.ndarray
    offsets: np.ndarray
    pixels: np.ndarray
    path_ids: np.ndarray

    def votes(self, keep=None):
        """Sum exact integer occurrences from the selected active-path subset."""
        if keep is None:
            keep = np.ones(len(self.origins), dtype=bool)
        keep = np.asarray(keep)
        if keep.dtype != bool or keep.shape != (len(self.origins),):
            raise ValueError("keep must be one boolean per active path")
        selected = self.pixels[keep[self.path_ids]]
        return np.bincount(selected, minlength=int(np.prod(self.shape))).reshape(self.shape)

    def visits(self, mask):
        """Number of selected-mask pixel occurrences in each path."""
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != self.shape:
            raise ValueError("mask must be a matching boolean image")
        return np.bincount(
            self.path_ids, weights=mask.ravel()[self.pixels], minlength=len(self.origins)
        ).astype(np.int64)

    def reconstruct_costs(self, image):
        """Raster sum excluding each source and counting consecutive joints once."""
        image = np.asarray(image)
        if image.shape != self.shape:
            raise ValueError("image must match path shape")
        totals = np.bincount(
            self.path_ids, weights=image.ravel()[self.pixels], minlength=len(self.origins)
        )
        return totals - image[tuple(self.origins.T)]


def rasterize_paths(costs, paths):
    """Prepare reusable path occurrences, validating with the package convention."""
    _, paths, active = _validate_paths(costs, paths)
    origins = np.argwhere(active)
    coordinates, offsets = [], [0]
    width = active.shape[1]
    for row, column in origins:
        previous = (int(row), int(column))
        for index, endpoint in enumerate(paths[row, column]):
            current = tuple(map(int, endpoint))
            segment = bresenham_line(*previous, *current)
            if index:
                segment = segment[1:]
            coordinates.extend(r * width + c for r, c in segment)
            previous = current
        offsets.append(len(coordinates))
    offsets = np.asarray(offsets, dtype=np.int64)
    pixels = np.asarray(coordinates, dtype=np.int64)
    path_ids = np.repeat(np.arange(len(origins), dtype=np.int64), np.diff(offsets))
    return RasterPaths(active.shape, origins, offsets, pixels, path_ids)


def tortuosity_scores(costs, paths):
    """Vectorized path scores for explanation; actual package filters are checked."""
    _, paths, active = _validate_paths(costs, paths)
    origins = np.argwhere(active)
    vertices = np.concatenate((origins[:, None, :], paths[active]), axis=1).astype(float)
    vectors = np.diff(vertices, axis=1)
    norms = np.linalg.norm(vectors, axis=2)
    units = np.divide(
        vectors, norms[..., None], out=np.zeros_like(vectors), where=norms[..., None] != 0
    )
    turns = np.sum(units[:, :-1] * units[:, 1:], axis=2)
    scores = np.prod(np.clip(turns, -1, 1), axis=1)
    scores[np.any(norms == 0, axis=1)] = np.nan
    return scores


def fixed_potentials(image):
    """The historical multiline raw/local input construction, without scoring."""
    value = image.astype(np.float64)
    local = np.maximum(ndimage.gaussian_filter(value, sigma=8, mode="reflect") - value, 0)
    local = np.rint(np.clip(2 * local, 0, 255)).astype(np.uint8)
    return {"raw": image, "local": 255 - local}


def source_partitions(raster, weak_band, strong_support):
    """Partition all path origins by weak-band membership and strong-tube visit."""
    weak_origin = weak_band[tuple(raster.origins.T)]
    visits_strong = raster.visits(strong_support) > 0
    return {
        "weak_origin__visits_strong": weak_origin & visits_strong,
        "weak_origin__no_strong_visit": weak_origin & ~visits_strong,
        "other_origin__visits_strong": ~weak_origin & visits_strong,
        "other_origin__no_strong_visit": ~weak_origin & ~visits_strong,
    }


def _stage_summary(votes, reference, weak_band, roi):
    points = np.floor(reference.samples + 0.5).astype(int)
    sampled_votes = votes[tuple(points.T)]
    weights = reference.weights
    zero_length = float(np.sum(weights[sampled_votes == 0]))
    threshold_coverage = {}
    for cutoff in SCORE_CUTOFFS:
        mask = roi & (votes > cutoff)
        measurement = reference.evaluate(mask, tolerance=2)
        threshold_coverage[str(cutoff)] = {
            "weak_coverage": measurement["centerline_coverage"],
            "selected_roi_pixels": int(mask.sum()),
            "selected_weak_band_pixels": int((mask & weak_band).sum()),
        }
    positive = reference.evaluate(roi & (votes > 0), tolerance=2)
    return {
        "vote_sha256": hashlib.sha256(votes.tobytes()).hexdigest(),
        "weak_band_score_distribution": distribution(votes[weak_band]),
        "weak_reference_nearest_pixel_score_distribution": distribution(sampled_votes),
        "zero_vote_reference_arclength": zero_length,
        "zero_vote_reference_fraction": zero_length / float(weights.sum()),
        "positive_vote_weak_coverage": positive["centerline_coverage"],
        "threshold_coverage": threshold_coverage,
    }


def diagnose_case(spec, potential_name):
    """Run and verify one source/path/filter/vote decomposition."""
    case = generate_multiline(spec)
    image = fixed_potentials(case.image)[potential_name]
    costs, paths = compute_ppi(image, 3, 10)
    raster = rasterize_paths(costs, paths)
    reconstructed = raster.reconstruct_costs(image)
    if not np.array_equal(reconstructed, costs[tuple(raster.origins.T)]):
        raise AssertionError("Path raster does not reproduce exact reported costs")
    scores = tortuosity_scores(costs, paths)
    weak_band = case.roi & (case.distance_maps[0] <= 2)
    strong_support = (
        case.support_masks[1] if len(case.support_masks) > 1 else np.zeros_like(case.roi)
    )
    reference = prepare_reference(case.centerlines[0], case.roi, distance_map=case.distance_maps[0])
    partitions = source_partitions(raster, weak_band, strong_support)
    weak_origin = weak_band[tuple(raster.origins.T)]
    weak_visits = raster.visits(weak_band)
    strong_visits = raster.visits(strong_support)
    lengths = np.diff(raster.offsets)
    masks = {"unfiltered": np.ones(len(raster.origins), dtype=bool)}
    filtered_costs = {"unfiltered": costs}
    for threshold in THRESHOLDS:
        filtered = filter_tortuosity(costs, paths, threshold)
        keep = np.isfinite(filtered[tuple(raster.origins.T)])
        numerical_tolerance = 8 * np.finfo(float).eps * max(paths.shape[2] - 1, 1)
        if not np.array_equal(keep, scores >= threshold - numerical_tolerance):
            raise AssertionError("Diagnostic tortuosity disagrees with package filter")
        masks[f"tau_{threshold:g}"] = keep
        filtered_costs[f"tau_{threshold:g}"] = filtered
    stages = {}
    for stage, keep in masks.items():
        votes = raster.votes(keep)
        official, _ = voting(filtered_costs[stage], paths)
        if not np.array_equal(votes, official):
            raise AssertionError("Diagnostic votes disagree with package voting")
        contributions, total = {}, np.zeros_like(votes)
        for name, group in partitions.items():
            selected = keep & group
            contribution = raster.votes(selected)
            total += contribution
            contributions[name] = {
                "path_count": int(selected.sum()),
                "votes_in_weak_band": int(contribution[weak_band].sum()),
                "votes_in_strong_support": int(contribution[strong_support].sum()),
                "pixel_occurrences_all": int(contribution.sum()),
            }
        if not np.array_equal(total, votes):
            raise AssertionError("Source partitions do not sum exactly to all votes")
        stages[stage] = {
            **_stage_summary(votes, reference, weak_band, case.roi),
            "retained_paths": int(keep.sum()),
            "weak_origin_retained": int((weak_origin & keep).sum()),
            "weak_origin_rejected": int((weak_origin & ~keep).sum()),
            "weak_origin_retained_visits_strong": int(
                (weak_origin & keep & (strong_visits > 0)).sum()
            ),
            "weak_origin_retained_stays_in_weak_band": int(
                (weak_origin & keep & (weak_visits == lengths)).sum()
            ),
            "contributions": contributions,
        }
    # Fixed reference-arclength quarters select the same origins within each
    # weak-alone/strong-neighbor comparison, before looking at path outcomes.
    sample_indices = [
        int(round((len(reference.samples) - 1) * fraction)) for fraction in (0.25, 0.5, 0.75)
    ]
    source_examples = []
    origin_lookup = {tuple(origin): index for index, origin in enumerate(raster.origins)}
    for sample_index in sample_indices:
        origin = tuple(np.floor(reference.samples[sample_index] + 0.5).astype(int))
        index = origin_lookup[origin]
        start, stop = raster.offsets[index : index + 2]
        pixels = np.column_stack(np.unravel_index(raster.pixels[start:stop], raster.shape))
        source_examples.append(
            {
                "origin": list(map(int, origin)),
                "endpoints": paths[origin].tolist(),
                "raster_pixels": pixels.tolist(),
                "cost": float(costs[origin]),
                "reconstructed_cost": float(reconstructed[index]),
                "tortuosity": float(scores[index]),
                "weak_band_pixel_occurrences": int(weak_visits[index]),
                "strong_support_pixel_occurrences": int(strong_visits[index]),
                "retained": {stage: bool(keep[index]) for stage, keep in masks.items()},
            }
        )
    return {
        "case_id": spec["case_id"],
        "pair_group": spec["pair_group"],
        "family": spec["family"],
        "gap": spec["gap"],
        "potential": potential_name,
        "image_sha256": hashlib.sha256(case.image.tobytes()).hexdigest(),
        "potential_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "path_sha256": hashlib.sha256(paths.tobytes()).hexdigest(),
        "finite_path_count": len(raster.origins),
        "weak_origin_count": int(weak_origin.sum()),
        "weak_origin_stays_in_weak_band": int((weak_origin & (weak_visits == lengths)).sum()),
        "weak_origin_departs_weak_band": int((weak_origin & (weak_visits < lengths)).sum()),
        "weak_origin_visits_strong_support": int((weak_origin & (strong_visits > 0)).sum()),
        "weak_origin_tortuosity_distribution": distribution(scores[weak_origin]),
        "weak_origin_fraction_path_in_weak_band": distribution(
            weak_visits[weak_origin] / lengths[weak_origin]
        ),
        "all_path_costs_verified": True,
        "all_stage_votes_verified": True,
        "all_source_partitions_verified": True,
        "stages": stages,
        "source_examples": source_examples,
    }


def snapshot():
    result = _provenance()
    root = Path(__file__).resolve().parents[1]
    result["diagnostic_source_sha256"] = {
        name: hashlib.sha256((root / "benchmarks" / name).read_bytes()).hexdigest()
        for name in ("diagnose_weak_paths.py", "multiline_data.py", "guide3d_metrics.py")
    }
    return result


def run(output, *, context_limit=None):
    """Diagnose all nine evaluation contexts, or a bounded prefix for development."""
    specs = [
        s
        for s in multiline_specs()
        if s["split"] == "evaluation" and s["family"] in ("weak_alone", "weak_near_strong")
    ]
    groups = list(dict.fromkeys(s["pair_group"] for s in specs))
    if context_limit is not None:
        if (
            isinstance(context_limit, bool)
            or not isinstance(context_limit, int)
            or context_limit < 1
        ):
            raise ValueError("context_limit must be a positive integer")
        groups = groups[:context_limit]
        specs = [spec for spec in specs if spec["pair_group"] in groups]
    sources = snapshot()
    results = []
    for spec in specs:
        for potential in ("raw", "local"):
            results.append(diagnose_case(spec, potential))
            print(
                f"Diagnosed {len(results)}/{2 * len(specs)}: {spec['case_id']} {potential}",
                flush=True,
            )
    if snapshot() != sources:
        raise RuntimeError("Measured source changed during diagnosis")
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": sources,
        "protocol": {
            "contexts": groups,
            "segment_length": 3,
            "nb_segments": 10,
            "tortuosity_thresholds": list(THRESHOLDS),
            "fixed_score_cutoffs": list(SCORE_CUTOFFS),
            "historical_operating_points": {
                "raw_tau0.75": 56,
                "local_tau0.75": 56,
                "local_tau0.5": 58,
            },
            "zero_vote_arclength": "trapezoidal weak-reference sample weights whose nearest image pixel has zero votes; distinct from tolerance-based positive-vote coverage",
            "origin_partition": "all finite source paths: origin within2pixels of weak reference versus every other origin; then visit versus no visit to strong3sigma support",
            "path_pixel_counting": "include source, shared consecutive joint once, later visits count again",
            "examples": "fixed weak-reference arclength quarters, midpoint and three-quarters",
            "limits": [
                "existing selected path only, not enumeration of alternative paths",
                "unfiltered/raw-relaxed cutoffs are diagnostic, not recalibrated accuracy claims",
                "nine contexts share seeds; no independence claim",
                "flat-background voting is expected because paths remain admissible; this report diagnoses weak-line loss",
            ],
        },
        "results": results,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-limit", type=int)
    args = parser.parse_args()
    run(args.output, context_limit=args.context_limit)


if __name__ == "__main__":
    main()
