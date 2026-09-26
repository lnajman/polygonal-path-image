"""Diagnose deterministic votes on constant potentials without changing PPI.

Every feasible path on a constant image has equal cost. The released algorithm
returns one optimum per source, not every tied optimum. Votes are therefore
expected on blanks; this study describes their dependence on tie selection,
finite boundaries, tortuosity and tiny discrete perturbations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import polygonal_path_image as ppi

SEGMENT_LENGTH = 3
NB_SEGMENTS = 10
ROI_WIDTH = 80
HISTORICAL_THRESHOLDS = {"tau075": 56, "tau050": 58}


def centered_roi(size, width=ROI_WIDTH):
    """Return a centered square ROI; only equal-parity size/width are accepted."""
    for value in (size, width):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("size and width must be positive integers")
    if width <= 0 or size < width or (size - width) % 2:
        raise ValueError("ROI must fit and have the same parity as its canvas")
    margin = (size - width) // 2
    result = np.zeros((size, size), dtype=bool)
    result[margin : margin + width, margin : margin + width] = True
    return result


def make_potential(size, intensity, *, seed=None):
    """Constant uint8 potential, or constant plus seeded integer -1/0/+1 noise.

    This perturbation changes stored uint8 pixels. It changes many cost ties
    without guaranteeing that every optimal path becomes unique.
    """
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    if (
        isinstance(intensity, (bool, np.bool_))
        or not isinstance(intensity, (int, np.integer))
        or not 0 <= intensity <= 255
    ):
        raise ValueError("intensity must be an integer from 0 to 255")
    image = np.full((size, size), intensity, dtype=np.int16)
    if seed is not None:
        if not 1 <= intensity <= 254:
            raise ValueError("Perturbed intensity must leave room for both integer offsets")
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        image += np.random.default_rng(int(seed)).integers(-1, 2, image.shape, dtype=np.int16)
    return image.astype(np.uint8)


def expected_uniform_feasibility(shape, segment_length, nb_segments):
    """A cardinal cone needs L*K available main-axis steps from its source."""
    rows, columns = np.indices(shape)
    available = np.maximum.reduce((rows, shape[0] - 1 - rows, columns, shape[1] - 1 - columns))
    return available >= segment_length * nb_segments


def uniform_cost_summary(costs, intensity, segment_length=SEGMENT_LENGTH, nb_segments=NB_SEGMENTS):
    """Check cost L*K*c on feasible sources, including c=0 and impossible paths."""
    costs = np.asarray(costs)
    feasible = np.isfinite(costs)
    theoretical = expected_uniform_feasibility(costs.shape, segment_length, nb_segments)
    expected = segment_length * nb_segments * intensity
    return {
        "expected_finite_cost": expected,
        "finite_source_count": int(feasible.sum()),
        "expected_finite_source_count": int(theoretical.sum()),
        "feasibility_matches": bool(np.array_equal(feasible, theoretical)),
        "all_finite_costs_match": bool(np.all(costs[feasible] == expected)),
        "maximum_absolute_cost_error": float(np.max(np.abs(costs[feasible] - expected)))
        if feasible.any()
        else None,
    }


def array_sha256(array):
    """Hash C-order values; report shapes/dtypes alongside each use."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _region_values(votes, roi):
    values = votes[roi]
    unique, counts = np.unique(values, return_counts=True)
    return {
        "pixels": int(values.size),
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "sum": int(values.sum()),
        "positive_pixels": int(np.count_nonzero(values)),
        "count_strictly_above_56": int(np.count_nonzero(values > 56)),
        "count_strictly_above_58": int(np.count_nonzero(values > 58)),
        "value_histogram": [
            {"votes": int(value), "pixels": int(count)}
            for value, count in zip(unique, counts, strict=True)
        ],
        "sha256": array_sha256(values),
    }


def compare_arrays(first, second):
    """Scalar comparison of two aligned integer maps or center crops."""
    if first.shape != second.shape:
        raise ValueError("Compared maps must have identical shape")
    difference = first.astype(np.float64) - second.astype(np.float64)
    return {
        "identical": bool(np.array_equal(first, second)),
        "changed_pixels": int(np.count_nonzero(difference)),
        "changed_fraction": float(np.mean(difference != 0)),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
    }


def uniform_symmetry_summary(votes, roi):
    """Compare output-map symmetries for a transformation-invariant input.

    A flip or quarter turn leaves a uniform input exactly unchanged. Comparing
    its single deterministic output with the transformed output is therefore
    sufficient for this uniform-input equivariance diagnostic. This does not
    claim general equivariance for nonconstant inputs.
    """
    return {
        name: {
            "whole_canvas": compare_arrays(votes, transformed),
            "center_roi": compare_arrays(votes[roi], transformed[roi]),
        }
        for name, transformed in (
            ("flip_rows", votes[::-1]),
            ("flip_columns", votes[:, ::-1]),
            ("rotate90", np.rot90(votes)),
        )
    }


def _first_step_histogram(paths, active):
    source = np.argwhere(active)
    differences = paths[active][:, 0] - source
    values, counts = np.unique(differences, axis=0, return_counts=True)
    return [
        {"delta_row": int(value[0]), "delta_column": int(value[1]), "sources": int(count)}
        for value, count in zip(values, counts, strict=True)
    ]


def _provenance():
    root = Path(__file__).resolve().parents[1]
    package = Path(ppi.__file__).parent
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    sources = {
        f"src/polygonal_path_image/{name}": hashlib.sha256(
            (package / name).read_bytes()
        ).hexdigest()
        for name in ("__init__.py", "core.py", "_core.pyx", "postprocess.py")
    }
    sources["benchmarks/diagnose_blank_paths.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    kernel = Path(sys.modules["polygonal_path_image._core"].__file__)
    return {
        "git_head_at_execution": revision,
        "source_sha256": sources,
        "compiled_kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "note": "Source hashes identify executed files, including files not yet committed at the recorded Git revision.",
    }


def run_diagnostics(*, cache_dir=None):
    """Run eight core cases and twenty filtering/voting configurations."""
    started = time.perf_counter()
    provenance = _provenance()
    cases = [(128, intensity, None, (None, 0.5, 0.75)) for intensity in (0, 100, 200, 255)]
    cases.extend((size, 200, None, (0.5, 0.75)) for size in (192, 256))
    cases.extend((128, 200, seed, (0.5, 0.75)) for seed in (3001, 3002))
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "study_id": "blank-path-ties-v1",
        "segment_length": SEGMENT_LENGTH,
        "nb_segments": NB_SEGMENTS,
        "uniform_128_intensities": [0, 100, 200, 255],
        "canvas_sizes": [128, 192, 256],
        "center_roi_width": ROI_WIDTH,
        "uniform_128_filters": [None, 0.5, 0.75],
        "larger_canvas_filters": [0.5, 0.75],
        "perturbation_filters": [0.5, 0.75],
        "perturbation": {
            "background": 200,
            "integer_offsets": [-1, 0, 1],
            "seeds": [3001, 3002],
            "distribution": "independent equiprobable integer draws using NumPy default_rng",
            "does_not_guarantee_unique_optima": True,
        },
        "historical_cutoffs": HISTORICAL_THRESHOLDS,
        "historical_cutoff_scope": "Fixed cutoffs from the preceding all-lines study, not calibrated confidence or per-image error guarantees.",
        "voting": "border=0; sources and segment endpoints count, shared joints counted once",
        "uniform_cost_formula": "Every feasible path costs segment_length * nb_segments * intensity = 30*c.",
        "vote_conservation": "Every retained path contributes L*K+1 = 31 raster-pixel occurrences; sum over the complete canvas.",
        "tie_policy_from_unchanged_source": "Last equal-cost endpoint wins within each cone; first equal-cost cone wins across H,B,E,W.",
        "scope": "The API selects one optimum per source; it does not enumerate all tied paths.",
        "boundary_comparison": "Aligned centered 80x80 ROIs; margins are 24,56,88 pixels on 128,192,256 canvases.",
        "interpretation": "Blank votes are expected. Their existence is not a correctness bug; spatial pattern and calibrated-score interpretation are separate questions.",
    }
    records, maps, path_arrays, retained_arrays = [], {}, {}, {}
    for size, intensity, seed, thresholds in cases:
        case_id = (
            f"uniform_n{size}_c{intensity}"
            if seed is None
            else f"perturbed_n{size}_c{intensity}_seed{seed}"
        )
        image = make_potential(size, intensity, seed=seed)
        roi = centered_roi(size)
        costs, paths = ppi.compute_ppi(image, SEGMENT_LENGTH, NB_SEGMENTS)
        finite = np.isfinite(costs)
        record = {
            "case_id": case_id,
            "size": size,
            "intensity": intensity,
            "seed": seed,
            "image_sha256": array_sha256(image),
            "paths_sha256": array_sha256(paths),
            "finite_source_count": int(finite.sum()),
            "finite_cost_minimum": float(costs[finite].min()),
            "finite_cost_maximum": float(costs[finite].max()),
            "distinct_finite_costs": int(np.unique(costs[finite]).size),
            "uniform_cost_check": uniform_cost_summary(costs, intensity) if seed is None else None,
            "first_step_histogram": _first_step_histogram(paths, finite),
            "center_source_first_step_histogram": _first_step_histogram(paths, finite & roi),
            "vote_maps": [],
        }
        arrays = {"image": image, "roi": roi, "costs": costs, "paths": paths}
        path_arrays[case_id] = paths
        for threshold in thresholds:
            key = (
                "unfiltered" if threshold is None else ("tau050" if threshold == 0.5 else "tau075")
            )
            filtered = (
                costs if threshold is None else ppi.filter_tortuosity(costs, paths, threshold)
            )
            retained = np.isfinite(filtered)
            votes, _ = ppi.voting(filtered, paths, border=0)
            expected_votes = int(retained.sum()) * (SEGMENT_LENGTH * NB_SEGMENTS + 1)
            if int(votes.sum()) != expected_votes:
                raise AssertionError("Whole-canvas vote conservation failed")
            map_id = f"{case_id}_{key}"
            maps[map_id] = votes
            retained_arrays[map_id] = retained
            arrays[f"votes_{key}"] = votes
            arrays[f"retained_{key}"] = retained
            record["vote_maps"].append(
                {
                    "map_id": map_id,
                    "filter": key,
                    "tortuosity_threshold": threshold,
                    "retained_sources": int(retained.sum()),
                    "retained_source_fraction": float(retained.mean()),
                    "retained_sources_in_roi": int((retained & roi).sum()),
                    "retained_mask_sha256": array_sha256(retained),
                    "whole_canvas": _region_values(votes, np.ones(votes.shape, dtype=bool)),
                    "center_roi": _region_values(votes, roi),
                    "expected_vote_sum": expected_votes,
                    "vote_conservation_holds": True,
                    "uniform_input_symmetry": uniform_symmetry_summary(votes, roi)
                    if seed is None
                    else None,
                }
            )
        if cache_dir is not None:
            np.savez_compressed(cache_dir / f"{case_id}.npz", **arrays)
        records.append(record)
        print(f"Completed {case_id}", flush=True)
    invariant = []
    for intensity in (0, 100, 200, 255):
        case_id = f"uniform_n128_c{intensity}"
        for key in ("unfiltered", "tau050", "tau075"):
            other, baseline = f"{case_id}_{key}", f"uniform_n128_c200_{key}"
            invariant.append(
                {
                    "intensity": intensity,
                    "filter": key,
                    "paths_identical_to_intensity_200": bool(
                        np.array_equal(path_arrays[case_id], path_arrays["uniform_n128_c200"])
                    ),
                    "retained_masks_identical": bool(
                        np.array_equal(retained_arrays[other], retained_arrays[baseline])
                    ),
                    "votes_identical": bool(np.array_equal(maps[other], maps[baseline])),
                }
            )
    boundaries = []
    for key in ("tau050", "tau075"):
        for first_size, second_size in ((128, 192), (128, 256), (192, 256)):
            first = maps[f"uniform_n{first_size}_c200_{key}"][centered_roi(first_size)]
            second = maps[f"uniform_n{second_size}_c200_{key}"][centered_roi(second_size)]
            boundaries.append(
                {
                    "filter": key,
                    "first_size": first_size,
                    "second_size": second_size,
                    **compare_arrays(first, second),
                }
            )
    perturbations = []
    baseline_paths = path_arrays["uniform_n128_c200"]
    roi = centered_roi(128)
    for seed in (3001, 3002):
        case_id = f"perturbed_n128_c200_seed{seed}"
        changed_paths = np.any(path_arrays[case_id] != baseline_paths, axis=(2, 3))
        for key in ("tau050", "tau075"):
            votes = maps[f"{case_id}_{key}"]
            baseline = maps[f"uniform_n128_c200_{key}"]
            perturbations.append(
                {
                    "seed": seed,
                    "filter": key,
                    "source_paths_changed_fraction": float(changed_paths.mean()),
                    "center_source_paths_changed_fraction": float(changed_paths[roi].mean()),
                    "whole_canvas_votes": compare_arrays(votes, baseline),
                    "center_roi_votes": compare_arrays(votes[roi], baseline[roi]),
                }
            )
    final_provenance = _provenance()
    for key in ("source_sha256", "compiled_kernel_sha256"):
        if final_provenance[key] != provenance[key]:
            raise RuntimeError("Executed source or compiled kernel changed during diagnostics")
    return {
        "schema_version": 1,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "provenance": provenance,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "package": importlib.metadata.version("polygonal-path-image"),
            "platform": platform.platform(),
        },
        "core_case_count": len(records),
        "vote_map_count": sum(len(r["vote_maps"]) for r in records),
        "cases": records,
        "constant_intensity_comparisons": invariant,
        "center_roi_canvas_comparisons": boundaries,
        "integer_perturbation_comparisons": perturbations,
        "duration_seconds": time.perf_counter() - started,
        "timing_note": "Observed elapsed time, not a controlled performance benchmark.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/benchmarks/diagnostics/blank-paths.json")
    )
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    report = run_diagnostics(cache_dir=args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {report['vote_map_count']} vote-map summaries to {args.output}")


if __name__ == "__main__":
    main()
