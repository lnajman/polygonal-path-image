"""Run a declared synthetic stress-test panel against the installed PPI package.

From a checkout: python -m benchmarks.run_synthetic --output benchmark-output/results.json
The full panel needs only NumPy and the installed package. It generates every
input locally; no clinical data, image download, or parameter fitting is used.
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
from benchmarks.synthetic import evaluate_prediction, generate_case, select_at_budget

BUDGETS = (0.005, 0.01, 0.02, 0.05)
SEEDS = (2012, 2013, 2014)
CONFIGURATIONS = {
    "baseline": (3, 10, 0.75),
    "short_segments": (1, 10, 0.75),
    "long_segments": (5, 10, 0.75),
    "short_paths": (3, 5, 0.75),
    "long_paths": (3, 15, 0.75),
    "tau_0": (3, 10, 0.0),
    "tau_050": (3, 10, 0.5),
    "tau_095": (3, 10, 0.95),
    "matched_L2_K15": (2, 15, 0.75),
    "matched_L5_K6": (5, 6, 0.75),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def experiment_panel(suite="full"):
    """Declare all cases before observing any result; keep paired noise seeds."""
    if suite not in ("full", "smoke"):
        raise ValueError("suite must be full or smoke")
    cases = {}

    def add(block, configs=("baseline",), **changes):
        parameters = dict(
            size=128,
            roi_margin=32,
            arclength=48.0,
            sigma=1.25,
            background=200.0,
            sample_step=0.25,
            angle_deg=0.0,
            bend_deg=0.0,
            contrast=60.0,
            noise_std=15.0,
            seed=2012,
            target_present=True,
        )
        parameters.update(changes)
        case_id = digest(json.dumps(parameters, sort_keys=True).encode())[:16]
        case = cases.setdefault(case_id, {"parameters": parameters, "contexts": []})
        case["contexts"].append({"block": block, "config_ids": list(configs)})

    if suite == "smoke":
        add("direction", angle_deg=30.0, bend_deg=45.0, contrast=120.0, noise_std=0.0)
        add("parameters", ("baseline", "short_paths", "tau_0"), angle_deg=30.0, bend_deg=90.0)
        add("controls", target_present=False, contrast=0.0, noise_std=0.0)
        return cases

    # Geometry panel: one noise-free realization, not 3 identical replicates.
    for bend in (0.0, 45.0, 90.0):
        for angle in range(0, 180, 15):
            add("direction", angle_deg=float(angle), bend_deg=bend, contrast=120.0, noise_std=0.0)
    for angle in (0.0, 30.0, 45.0):
        for contrast in (30.0, 60.0, 120.0):
            for noise in (5.0, 15.0, 30.0):
                for seed in SEEDS:
                    add(
                        "robustness",
                        angle_deg=angle,
                        bend_deg=45.0,
                        contrast=contrast,
                        noise_std=noise,
                        seed=seed,
                    )
    for angle in (0.0, 30.0, 45.0):
        for bend in (0.0, 90.0):
            add("parameters", tuple(CONFIGURATIONS), angle_deg=angle, bend_deg=bend)
    add("controls", target_present=False, contrast=0.0, noise_std=0.0)
    for noise in (5.0, 15.0, 30.0):
        for seed in SEEDS:
            add("controls", target_present=False, contrast=0.0, noise_std=noise, seed=seed)
    for angle in (0.0, 30.0, 45.0):
        for bend in (-45.0, -90.0):
            add("reflection", angle_deg=angle, bend_deg=bend)
    return cases


def run_pipeline(
    image, *, segment_length=3, nb_segments=10, tortuosity_threshold=0.75, core_result=None
):
    """Return votes and observed timings; optionally reuse a measured core call.

    Pipeline time covers compute_ppi + filter_tortuosity + voting only. Generation,
    reference geometry, score thresholding, metrics, imports, and plotting are
    excluded. Reusing core_result avoids repeating identical threshold-only runs.
    """
    if core_result is None:
        started = time.perf_counter()
        costs, paths = ppi.compute_ppi(image, segment_length, nb_segments)
        core_seconds = time.perf_counter() - started
    else:
        costs, paths, core_seconds = core_result
    started = time.perf_counter()
    filtered = ppi.filter_tortuosity(costs, paths, tortuosity_threshold)
    filter_seconds = time.perf_counter() - started
    started = time.perf_counter()
    votes, _ = ppi.voting(filtered, paths)
    voting_seconds = time.perf_counter() - started
    timings = {
        "core_seconds": core_seconds,
        "filter_seconds": filter_seconds,
        "voting_seconds": voting_seconds,
        "pipeline_seconds": core_seconds + filter_seconds + voting_seconds,
        "path_array_bytes": int(paths.nbytes),
        "finite_path_fraction": float(np.isfinite(costs).mean()),
        "retained_path_fraction": float(np.isfinite(filtered).mean()),
    }
    return votes, timings, (costs, paths, core_seconds)


def _provenance():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    package = Path(ppi.__file__).parent
    return {
        "git_head_at_execution": revision,
        "benchmark_source_sha256": {
            name: digest((root / "benchmarks" / name).read_bytes())
            for name in ("synthetic.py", "run_synthetic.py")
        },
        "implementation_source_sha256": {
            name: digest((package / name).read_bytes())
            for name in ("__init__.py", "core.py", "_core.pyx", "postprocess.py")
        },
        "compiled_kernel_sha256": digest(
            Path(sys.modules["polygonal_path_image._core"].__file__).read_bytes()
        ),
        "note": "Source hashes identify the executed files, including benchmark files added after the recorded Git commit.",
    }


def run_benchmark(*, suite="full", save_arrays=None, progress=True):
    started = time.perf_counter()
    provenance = _provenance()
    definitions = experiment_panel(suite)
    rows, scenes = [], []
    if save_arrays is not None:
        save_arrays = Path(save_arrays)
        save_arrays.mkdir(parents=True, exist_ok=True)
    for index, (case_id, definition) in enumerate(definitions.items(), 1):
        if progress:
            print(f"Scene {index}/{len(definitions)}: {case_id}", flush=True)
        case = generate_case(**definition["parameters"])
        scenes.append(
            {
                "case_id": case_id,
                "parameters": case.parameters,
                "contexts": definition["contexts"],
                "image_sha256": digest(case.image.tobytes()),
                "centerline_sha256": digest(case.centerline.tobytes()),
            }
        )
        configs = dict.fromkeys(
            name for context in definition["contexts"] for name in context["config_ids"]
        )
        started_baseline = time.perf_counter()
        darkness = np.maximum(case.parameters["background"] - case.image.astype(float), 0.0)
        baseline_seconds = time.perf_counter() - started_baseline
        results = {
            "darkness": (
                darkness,
                {
                    "core_seconds": 0.0,
                    "filter_seconds": 0.0,
                    "voting_seconds": 0.0,
                    "pipeline_seconds": baseline_seconds,
                    "path_array_bytes": 0,
                    "finite_path_fraction": None,
                    "retained_path_fraction": None,
                },
            )
        }
        core_cache = {}
        for name in configs:
            length, segments, threshold = CONFIGURATIONS[name]
            key = (length, segments)
            scores, timing, core = run_pipeline(
                case.image,
                segment_length=length,
                nb_segments=segments,
                tortuosity_threshold=threshold,
                core_result=core_cache.get(key),
            )
            core_cache[key] = core
            results[name] = (scores, timing)
        del core_cache
        if save_arrays is not None:
            np.savez_compressed(
                save_arrays / f"{case_id}.npz",
                image=case.image,
                centerline=case.centerline,
                roi=case.roi,
                **{f"scores_{name}": value[0] for name, value in results.items()},
            )
        measured = {}
        for name, (scores, timing) in results.items():
            measured[name] = []
            for budget in BUDGETS:
                mask, selection = select_at_budget(scores, case.roi, fraction=budget)
                metrics = evaluate_prediction(
                    mask,
                    case.centerline,
                    case.roi,
                    tolerance=2.0,
                    distance_map=case.distance_map,
                )
                if (
                    metrics["true_positive_pixels"] + metrics["false_positive_pixels"]
                    != metrics["selected_pixels"]
                ):
                    raise AssertionError("selection count does not reconcile")
                measured[name].append(
                    {
                        "case_id": case_id,
                        **case.parameters,
                        "config_id": name,
                        "method": "darkness" if name == "darkness" else "ppi",
                        "segment_length": None if name == "darkness" else CONFIGURATIONS[name][0],
                        "nb_segments": None if name == "darkness" else CONFIGURATIONS[name][1],
                        "tortuosity_threshold": None
                        if name == "darkness"
                        else CONFIGURATIONS[name][2],
                        "budget_fraction": budget,
                        **metrics,
                        **timing,
                        **{f"selection_{key}": value for key, value in selection.items()},
                        "score_max_roi": float(scores[case.roi].max()),
                        "score_sha256": digest(scores.tobytes()),
                        "prediction_sha256": digest(mask.tobytes()),
                    }
                )
        for context in definition["contexts"]:
            for name in ["darkness", *context["config_ids"]]:
                rows.extend({"block": context["block"], **row} for row in measured[name])
    scientific = [
        {key: value for key, value in row.items() if not key.endswith("_seconds")} for row in rows
    ]
    if _provenance() != provenance:
        raise RuntimeError("Source files changed while the benchmark was running; rerun it")
    return {
        "schema_version": 1,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "protocol": {
            "canvas_shape": [128, 128],
            "roi_shape": [64, 64],
            "roi_margin": 32,
            "curve_arclength_pixels": 48,
            "tube_sigma_pixels": 1.25,
            "background": 200,
            "tolerance_pixels": 2.0,
            "reference_sample_step_pixels": 0.25,
            "budget_fractions": list(BUDGETS),
            "primary_budget_fraction": 0.02,
            "noise_seeds": list(SEEDS),
            "direction_angles_degrees": list(range(0, 180, 15)),
            "configurations": {
                name: dict(
                    segment_length=value[0], nb_segments=value[1], tortuosity_threshold=value[2]
                )
                for name, value in CONFIGURATIONS.items()
            },
            "threshold_rule": "Positive scores only; ceil(budget*ROI size) rank; include every cutoff tie; report actual selected fraction.",
            "baseline": "max(200 - image.astype(float), 0), evaluated on the identical ROI and nominal area budgets.",
            "coverage": "Arclength-weighted fraction of dense reference samples within 2 pixels of a selected pixel center.",
            "false_positive_rate": "Selected pixel centers farther than 2 pixels from the finite reference polyline / all such ROI pixel centers.",
            "localization": "Report selected-pixel-to-polyline and reference-sample-to-selected-pixel distances separately; null where undefined.",
            "timing": "Single observations of core + filtering + voting. Core timing reused across threshold-only configs on the same image. Excludes generation, metric computation, IO and imports; not a controlled speedup study.",
            "uncertainty": "Ranges across specified scenarios/seeds describe observed spread, not confidence intervals or clinical uncertainty.",
        },
        "limitations": [
            "Synthetic Gaussian tubes and Gaussian noise are a targeted stress test, not a representative clinical population.",
            "Nominal area budgets are ranking probes, not calibrated detection thresholds; ties can enlarge the selected area and noise-only controls can still produce detections.",
            "The fixed canvas and ROI do not eliminate boundary effects from paths originating outside the ROI.",
            "L*K counts main-axis raster steps, not common Euclidean arclength; changing L at fixed K also changes total raster length.",
            "The small negative-bend control is not an exhaustive rotation/reflection invariance test.",
            "Coverage and pixel precision have different populations and are not combined into an F1 score.",
            "Parameter scenarios have one paired seed; the robustness panel has three fixed seeds. No parameters were selected using these outputs.",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "package_version": importlib.metadata.version("polygonal-path-image"),
        },
        "provenance": provenance,
        "duration_seconds": time.perf_counter() - started,
        "scene_count": len(scenes),
        "row_count": len(rows),
        "non_timing_rows_sha256": digest(
            json.dumps(scientific, sort_keys=True, allow_nan=False).encode()
        ),
        "scenes": scenes,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("full", "smoke"), default="full")
    parser.add_argument("--output", type=Path, default=Path("benchmark-output/results.json"))
    parser.add_argument(
        "--save-arrays", type=Path, help="Optional local score/input arrays for audits"
    )
    args = parser.parse_args()
    report = run_benchmark(suite=args.suite, save_arrays=args.save_arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {report['row_count']} rows from {report['scene_count']} scenes to {args.output}")


if __name__ == "__main__":
    main()
