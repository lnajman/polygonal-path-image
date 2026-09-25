#!/usr/bin/env python3
"""Reproducible numerical validation; this does not establish clinical efficacy.

Run from an installed development environment::

    python scripts/validate_research.py --output work/validation.json
    python scripts/validate_research.py --image example.pgm --output work/report.json

``--image`` may be repeated. A deterministic synthetic image is always included.
Image files require the ``examples`` extra (Pillow); .npy inputs need only NumPy.
To compare a separately compiled historical FinalVersion module, supply both
``--legacy-python`` and ``--legacy-module-dir``. The legacy interpreter must have
NumPy, Pillow and matplotlib and be able to import PPIpython from that directory.
No input files are modified. Arrays, worker logs, and environment records are
saved beside the JSON report in a directory named ``<report-stem>-data``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_image() -> np.ndarray:
    """A fixed noisy dark curve, generated without optional dependencies."""
    rng = np.random.default_rng(2012)
    rows, columns = np.indices((64, 80))
    centre = 32 + 10 * np.sin(columns / 15)
    image = 210 - 170 * np.exp(-(((rows - centre) / 1.8) ** 2))
    return np.rint(np.clip(image + rng.normal(0, 5, image.shape), 0, 255)).astype(np.uint8)


def _load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
    else:
        from PIL import Image

        with Image.open(path) as opened:
            image = np.asarray(opened)
    if image.ndim != 2 or image.dtype != np.uint8 or not image.size:
        raise ValueError(f"{path}: input must already be a nonempty uint8 grayscale image")
    return np.ascontiguousarray(image)


def audit_paths(image, costs, paths, segment_length):
    """Independently check geometry, raster costs, and votes of every finite path.

    Rasterization uses integer midpoint rounding, not the production Bresenham
    recurrence. Grouping equal-length segments and bounded vector batches keeps
    the audit usable for full images, including corrupted historical paths.
    """
    active = np.isfinite(costs)
    start = np.argwhere(active)
    vertices = paths[active].astype(np.int64, copy=False)
    valid = np.all((vertices >= 0) & (vertices < np.array(image.shape)), axis=(1, 2))
    geometry_valid = valid.copy()
    starts_valid = start[valid]
    endpoints = vertices[valid]
    total = np.zeros(len(starts_valid), dtype=np.int64)
    previous = starts_valid
    common_cone = np.ones((len(previous), 4), dtype=bool)
    votes = np.bincount(previous[:, 0] * image.shape[1] + previous[:, 1], minlength=image.size)
    flat_image = image.ravel()
    for segment in range(paths.shape[2]):
        endpoint = endpoints[:, segment]
        delta = endpoint - previous
        common_cone &= np.column_stack(
            (
                (delta[:, 0] == -segment_length) & (abs(delta[:, 1]) <= segment_length),
                (delta[:, 0] == segment_length) & (abs(delta[:, 1]) <= segment_length),
                (delta[:, 1] == segment_length) & (abs(delta[:, 0]) <= segment_length),
                (delta[:, 1] == -segment_length) & (abs(delta[:, 0]) <= segment_length),
            )
        )
        lengths = np.max(np.abs(delta), axis=1)
        for length in np.unique(lengths):
            if not length:
                continue
            indices = np.flatnonzero(lengths == length)
            batch_size = max(1, 262144 // int(length))
            steps = np.arange(1, length + 1)[None, :, None]
            for first in range(0, len(indices), batch_size):
                selection = indices[first : first + batch_size]
                changes = delta[selection, None, :]
                pixels = previous[selection, None, :] + np.sign(changes) * (
                    (2 * steps * np.abs(changes) + length) // (2 * length)
                )
                flat = pixels[:, :, 0] * image.shape[1] + pixels[:, :, 1]
                total[selection] += flat_image[flat].sum(axis=1, dtype=np.int64)
                votes += np.bincount(flat.ravel(), minlength=image.size)
        previous = endpoint
    geometry_valid[valid] = np.any(common_cone, axis=1)
    reconstructed = np.full(costs.shape, np.nan)
    reconstructed[tuple(starts_valid.T)] = total
    mismatch = active & (reconstructed != costs)
    examples = []
    for row, column in np.argwhere(mismatch)[:5]:
        value = reconstructed[row, column]
        examples.append(
            {
                "source": [int(row), int(column)],
                "reported_cost": float(costs[row, column]),
                "recomputed_cost": float(value) if np.isfinite(value) else None,
                "endpoints": paths[row, column].tolist(),
            }
        )
    differences = np.abs(
        reconstructed[active & np.isfinite(reconstructed)]
        - costs[active & np.isfinite(reconstructed)]
    )
    return (
        {
            "finite_paths": int(active.sum()),
            "invalid_endpoint_paths": int((~valid).sum()),
            "invalid_geometry_paths": int((~geometry_valid).sum()),
            "inconsistent_path_costs": int(mismatch.sum()),
            "max_absolute_path_cost_error": float(differences.max(initial=0)),
            "first_inconsistencies": examples,
        },
        votes.reshape(image.shape),
        mismatch,
    )


def _peak_memory():
    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        return {"peak_rss_bytes": None, "peak_rss_note": "resource module unavailable"}
    # Darwin reports bytes; Linux reports KiB. Do not guess other OS conventions.
    multiplier = {"Darwin": 1, "Linux": 1024}.get(platform.system())
    return {
        "peak_rss_bytes": int(raw * multiplier) if multiplier else None,
        "ru_maxrss_raw": raw,
        "ru_maxrss_unit": {"Darwin": "bytes", "Linux": "KiB"}.get(platform.system(), "unknown"),
        "peak_rss_scope": "worker lifetime including imports, core, postprocessing and audit",
    }


def _worker(args):
    image = np.load(args.worker_input, allow_pickle=False)
    if args.worker == "legacy":
        sys.path.insert(0, str(Path(args.legacy_module_dir).resolve()))
        import PPIpython as implementation
    else:
        import polygonal_path_image as implementation

    kernel = sys.modules.get("polygonal_path_image._core", implementation)
    environment = {
        "python": sys.version,
        "executable": Path(sys.executable).name,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "module_file": Path(implementation.__file__).name,
        "kernel_sha256": digest(Path(kernel.__file__)),
    }
    if args.worker == "modern":
        environment["package_version"] = importlib.metadata.version("polygonal-path-image")
        environment["implementation_source_sha256"] = {
            filename: digest(Path(implementation.__file__).parent / filename)
            for filename in ("__init__.py", "core.py", "_core.pyx", "postprocess.py")
        }
    else:
        source = Path(args.legacy_module_dir) / "PPIpython.pyx"
        if source.exists():
            environment["compatibility_source_sha256"] = digest(source)
        transformations = Path(args.legacy_module_dir) / "transformations.txt"
        if transformations.exists():
            environment["compatibility_transformations"] = transformations.read_text()
    timings = {}
    before = time.perf_counter()
    costs, paths = implementation.compute_ppi(image, args.segment_length, args.nb_segments)
    timings["compute_ppi_seconds"] = time.perf_counter() - before
    before = time.perf_counter()
    audit, independently_rasterized_votes, mismatch = audit_paths(
        image, costs, paths, args.segment_length
    )
    timings["independent_path_audit_seconds"] = time.perf_counter() - before
    arrays = {"costs": costs, "paths": paths, "path_cost_mismatch": mismatch}
    errors = {}
    before = time.perf_counter()
    if args.worker == "modern":
        votes, inverse = implementation.voting(costs, paths)
        arrays.update(votes=votes, inverse=inverse)
        audit["votes_match_independent_rasterization"] = bool(
            np.array_equal(votes, independently_rasterized_votes)
        )
        arrays["filtered_costs"] = implementation.filter_tortuosity(
            costs, paths, args.tortuosity_threshold
        )
    else:
        for name, trial_costs, trial_paths in [("raw", costs, paths)]:
            arrays[f"{name}_votes"], arrays[f"{name}_inverse"] = implementation.voting(
                trial_costs, trial_paths
            )
            try:
                arrays[f"{name}_filtered_costs"] = implementation.voting_without_tortuosity(
                    trial_costs, trial_paths, args.tortuosity_threshold
                )
            except (ValueError, ZeroDivisionError) as error:
                errors[f"{name}_tortuosity"] = str(error)
        # Identical coordinates isolate API/definition differences from the
        # corrected dynamic-programming reconstruction.
        with np.load(args.modern_result, allow_pickle=False) as modern:
            trial_costs, trial_paths = modern["costs"], modern["paths"]
        arrays["corrected_path_votes"], arrays["corrected_path_inverse"] = implementation.voting(
            trial_costs, trial_paths
        )
        arrays["corrected_path_filtered_costs"] = implementation.voting_without_tortuosity(
            trial_costs, trial_paths, args.tortuosity_threshold
        )
    timings["postprocessing_seconds"] = time.perf_counter() - before
    metadata = {"environment": environment, "timings": timings, "path_audit": audit}
    metadata.update(_peak_memory())
    if errors:
        metadata["legacy_postprocessing_errors"] = errors
    np.savez_compressed(args.worker_output, **arrays)
    Path(args.worker_output).with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )


def _run_worker(args, backend, image_path, output_path, modern_result=None):
    interpreter = args.legacy_python if backend == "legacy" else sys.executable
    command = [
        interpreter,
        str(Path(__file__).resolve()),
        "--worker",
        backend,
        "--worker-input",
        str(image_path),
        "--worker-output",
        str(output_path),
        "--segment-length",
        str(args.segment_length),
        "--nb-segments",
        str(args.nb_segments),
        "--tortuosity-threshold",
        str(args.tortuosity_threshold),
    ]
    if backend == "legacy":
        command += [
            "--legacy-module-dir",
            args.legacy_module_dir,
            "--modern-result",
            str(modern_result),
        ]
    print(f"Validating {image_path.stem}: {backend}", flush=True)
    started = time.perf_counter()
    with output_path.with_suffix(".log").open("w") as log:
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=args.timeout,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
    metadata = json.loads(output_path.with_suffix(".json").read_text())
    metadata["timings"]["worker_wall_seconds"] = time.perf_counter() - started
    metadata["artifact"] = str(output_path.name)
    return metadata


def compare_results(modern_path, legacy_path):
    with (
        np.load(modern_path, allow_pickle=False) as modern,
        np.load(legacy_path, allow_pickle=False) as legacy,
    ):
        modern_costs, legacy_costs = modern["costs"], legacy["costs"]
        both_finite = np.isfinite(modern_costs) & np.isfinite(legacy_costs)
        difference = np.abs(modern_costs[both_finite] - legacy_costs[both_finite])
        bordered_votes = modern["votes"].copy()
        bordered_votes[:2] = bordered_votes[-2:] = 0
        bordered_votes[:, :2] = bordered_votes[:, -2:] = 0
        modern_keep = np.isfinite(modern["filtered_costs"])
        legacy_keep = np.isfinite(legacy["corrected_path_filtered_costs"])
        inverse_offset = legacy["corrected_path_inverse"] - (bordered_votes.max() - bordered_votes)
        return {
            "cost_arrays_exactly_equal": bool(np.array_equal(modern_costs, legacy_costs)),
            "finite_masks_equal": bool(
                np.array_equal(np.isfinite(modern_costs), np.isfinite(legacy_costs))
            ),
            "different_cost_pixels": int(np.count_nonzero(modern_costs != legacy_costs)),
            "max_absolute_cost_difference": float(difference.max(initial=0)),
            "different_finite_path_coordinates": int(
                np.count_nonzero(
                    np.any(modern["paths"] != legacy["paths"], axis=(2, 3)) & both_finite
                )
            ),
            "voting_on_identical_corrected_paths": {
                "legacy_votes_equal_modern_with_border_2": bool(
                    np.array_equal(legacy["corrected_path_votes"], bordered_votes)
                ),
                "legacy_inverse_minus_modern_border_2_inverse_unique": np.unique(
                    inverse_offset
                ).tolist(),
                "modern_uncleared_maximum": int(modern["votes"].max()),
                "cleared_maximum": int(bordered_votes.max()),
            },
            "tortuosity_on_identical_corrected_paths": {
                "threshold": "recorded in report parameters",
                "modern_retained": int(modern_keep.sum()),
                "legacy_retained": int(legacy_keep.sum()),
                "legacy_only_retained": int(np.count_nonzero(legacy_keep & ~modern_keep)),
                "modern_only_retained": int(np.count_nonzero(modern_keep & ~legacy_keep)),
                "different_decisions": int(np.count_nonzero(modern_keep != legacy_keep)),
            },
            "raw_legacy_votes_different_from_corrected_votes_border_2": int(
                np.count_nonzero(legacy["raw_votes"] != bordered_votes)
            ),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path, default=Path("validation.json"))
    parser.add_argument("--segment-length", type=int, default=3)
    parser.add_argument("--nb-segments", type=int, default=10)
    parser.add_argument("--tortuosity-threshold", type=float, default=0.75)
    parser.add_argument("--legacy-python")
    parser.add_argument("--legacy-module-dir")
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="maximum seconds per worker, including postprocessing (default 600)",
    )
    parser.add_argument("--worker", choices=("modern", "legacy"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-input", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--modern-result", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(args)
        return
    if bool(args.legacy_python) != bool(args.legacy_module_dir):
        parser.error("--legacy-python and --legacy-module-dir must be supplied together")
    if args.segment_length <= 0 or args.nb_segments <= 0:
        parser.error("--segment-length and --nb-segments must be positive")
    if not -1 <= args.tortuosity_threshold <= 1:
        parser.error("--tortuosity-threshold must be between -1 and 1")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = output.parent / f"{output.stem}-data"
    data.mkdir(exist_ok=True)
    inputs = [("synthetic_curve_seed_2012", synthetic_image(), None)]
    inputs.extend((path.stem, _load_image(path), path.resolve()) for path in args.image)
    report = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(Path(__file__)),
        "parameters": {
            "segment_length": args.segment_length,
            "nb_segments": args.nb_segments,
            "tortuosity_threshold": args.tortuosity_threshold,
        },
        "array_directory": data.name,
        "method": "Full resolution uint8 inputs without rescaling or preprocessing; every finite "
        "path checked with independent integer midpoint rasterization. Single timings "
        "are observations, not a controlled performance benchmark.",
        "limitations": [
            "Agreement with historical costs is a compatibility check, not an independent "
            "proof of optimality; the test suite supplies exhaustive small-image oracles.",
            "The supplied images have no ground-truth labels here. No paper figure, clinical "
            "accuracy, sensitivity, specificity, or segmentation claim is reproduced.",
            "The historical NPY outputs are not used because their producing parameters are unknown.",
            "Orientation and pruning were intentionally redesigned and are not compared numerically "
            "to the historical plotting and local-pruning routines.",
        ],
        "cases": [],
    }
    for index, (name, image, source) in enumerate(inputs):
        stem = f"{index:02d}-{name}"
        input_path = data / f"{stem}-input.npy"
        np.save(input_path, image, allow_pickle=False)
        case = {
            "name": name,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "source_file": source.name if source else None,
            "source_file_sha256": digest(source) if source else None,
            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "input_artifact": input_path.name,
        }
        modern_path = data / f"{stem}-modern.npz"
        case["modern"] = _run_worker(args, "modern", input_path, modern_path)
        audit = case["modern"]["path_audit"]
        case["correctness_checks_passed"] = (
            audit["invalid_endpoint_paths"] == 0
            and audit["invalid_geometry_paths"] == 0
            and audit["inconsistent_path_costs"] == 0
            and audit["votes_match_independent_rasterization"]
        )
        if args.legacy_python:
            legacy_path = data / f"{stem}-legacy.npz"
            case["legacy"] = _run_worker(args, "legacy", input_path, legacy_path, modern_path)
            case["comparison"] = compare_results(modern_path, legacy_path)
            case["correctness_checks_passed"] &= case["comparison"]["cost_arrays_exactly_equal"]
            case["correctness_checks_passed"] &= case["comparison"][
                "voting_on_identical_corrected_paths"
            ]["legacy_votes_equal_modern_with_border_2"]
        report["cases"].append(case)
        report["correctness_checks_passed"] = all(
            item["correctness_checks_passed"] for item in report["cases"]
        )
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {output}")
    if not report["correctness_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
