"""Reproducible all-line study with development-frozen score thresholds.

No PPI algorithm or parameter is fitted in this experiment. Thresholds alone
are calibrated using development background pixels, before evaluation images
are processed. Cache files and the protocol are bound to source/version hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.filters import frangi

from benchmarks.multiline_data import generate_multiline, multiline_specs
from benchmarks.multiline_metrics import prepare_multiline
from benchmarks.run_synthetic import _provenance, run_pipeline
from benchmarks.synthetic import select_at_budget

BACKGROUND_RATES = (0.001, 0.005, 0.01, 0.02)
BUDGETS = (0.01, 0.02, 0.05)
PRIMARY_RATE = 0.005
METHODS = (
    "darkness",
    "local_contrast",
    "frangi",
    "ppi_raw_default",
    "ppi_local_default",
    "ppi_local_relaxed",
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def digest_json(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def protocol(suite="full"):
    """Declare the full experiment before computing scores or viewing results."""
    return {
        "study_id": "multiline-v1",
        "suite": suite,
        "manifest": multiline_specs(suite),
        "methods": list(METHODS),
        "darkness": "max(200-image,0); knows the generator's constant background",
        "local_contrast": "rint(clip(2*max(Gaussian(image,sigma=8,mode=reflect)-image,0),0,255)); fixed gain, no per-image normalization",
        "frangi": {
            "implementation": "skimage.filters.frangi",
            "input": "image.astype(float64)/255",
            "sigmas": [0.8, 1.25, 2.0],
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 0.02,
            "black_ridges": True,
            "mode": "reflect",
            "note": "Fixed gamma permits a common score threshold and avoids image-wise maximum normalization; no parameter search. Alpha is unused in 2D.",
        },
        "ppi": {
            "segment_length": 3,
            "nb_segments": 10,
            "default_tortuosity_threshold": 0.75,
            "relaxed_tortuosity_threshold": 0.5,
            "raw_potential": "original uint8 image",
            "local_potential": "255 minus the same quantized local_contrast score",
            "score": "unweighted path votes, border=0, no score normalization",
        },
        "reference": {
            "centerline_tolerance_pixels": 2.0,
            "sample_step_pixels": 0.25,
            "support_radius": "3 times each individual line sigma; union across all lines",
            "junctions": "line coverage counts each incident reference; pixel measures use union",
        },
        "threshold_selection": {
            "development_background_rates": list(BACKGROUND_RATES),
            "primary_rate": PRIMARY_RATE,
            "calibration_population": "pooled ROI pixels outside all declared line supports in every development scene, including controls",
            "rule": "nonnegative order-statistic cutoff; select finite scores strictly greater than cutoff; preserve ties as a block; achieved development rate cannot exceed requested rate",
            "evaluation": "same scalar threshold for every evaluation image; never recalibrate on evaluation",
        },
        "budget_selection": {
            "fractions": list(BUDGETS),
            "rule": "finite positive scores; ceil(fraction*ROI pixels); include whole cutoff tie block",
            "purpose": "secondary ranking diagnostic; not an absence test",
        },
        "aggregation": "equal scene means; per-scene coverage is equal-line mean. Also report worst-line coverage, families, paired weak-line changes and controls separately. Shared noise/orientation contexts are not independent observations.",
        "limits": [
            "dark finite Gaussian tubes with declared scales, not every possible line appearance",
            "synthetic constant background and independent Gaussian noise, not clinical images",
            "same scene families in development and evaluation; distinct orientations and noise seeds, not unseen families",
            "matched requested development background rates do not guarantee matched evaluation rates",
            "continuity is a sampled uncovered-length diagnostic, not recovered graph connectivity",
            "two-pixel coverage tolerance may bridge nearby predictions; no identity assignment at junctions",
            "image-edge influences may reach the ROI because PPI paths and Gaussian support exceed the 24-pixel inset",
            "no algorithm tuning, no exact MICCAI reproduction, no exhaustive comparator tuning",
        ],
    }


def provenance():
    result = _provenance()
    root = Path(__file__).resolve().parents[1]
    result["benchmark_source_sha256"].update(
        {
            name: digest((root / "benchmarks" / name).read_bytes())
            for name in ("multiline_data.py", "multiline_metrics.py", "run_multiline.py")
        }
    )
    result["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("polygonal-path-image", "numpy", "scipy", "scikit-image")
    }
    result["python"] = platform.python_version()
    return result


def calibrate_threshold(background_scores, target_rate):
    """Choose an observed cutoff with at most the requested strict exceedances.

    Scores are not rounded or tie-broken. For n samples, at most floor(rate*n)
    may exceed the cutoff. A nonnegative floor prevents selecting zero scores.
    """
    values = np.asarray(background_scores, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("background_scores must be a nonempty finite vector")
    if isinstance(target_rate, bool) or not np.isfinite(target_rate) or not 0 <= target_rate < 1:
        raise ValueError("target_rate must be in [0,1)")
    allowed = int(np.floor(target_rate * values.size))
    index = values.size - allowed - 1
    cutoff = max(0.0, float(np.partition(values, index)[index]))
    return {
        "threshold": cutoff,
        "target_rate": float(target_rate),
        "background_pixels": int(values.size),
        "selected_background_pixels": int(np.count_nonzero(values > cutoff)),
        "achieved_development_rate": float(np.mean(values > cutoff)),
    }


def score_methods(image):
    """Fixed methods with matching local potential and direct local baseline."""
    value = image.astype(np.float64)
    scores = {}
    timings = {}
    start = time.perf_counter()
    scores["darkness"] = np.maximum(200.0 - value, 0.0)
    timings["darkness"] = {"pipeline_seconds": time.perf_counter() - start}
    start = time.perf_counter()
    local = np.maximum(ndimage.gaussian_filter(value, sigma=8, mode="reflect") - value, 0)
    scores["local_contrast"] = np.rint(np.clip(2 * local, 0, 255)).astype(np.uint8)
    local_seconds = time.perf_counter() - start
    timings["local_contrast"] = {"pipeline_seconds": local_seconds}
    start = time.perf_counter()
    scores["frangi"] = frangi(
        value / 255,
        sigmas=(0.8, 1.25, 2.0),
        alpha=0.5,
        beta=0.5,
        gamma=0.02,
        black_ridges=True,
        mode="reflect",
    )
    timings["frangi"] = {"pipeline_seconds": time.perf_counter() - start}
    raw, timing, _ = run_pipeline(image)
    scores["ppi_raw_default"], timings["ppi_raw_default"] = raw, timing
    potential = 255 - scores["local_contrast"]
    local_default, timing, core = run_pipeline(potential)
    timing = dict(timing, preprocessing_seconds=local_seconds)
    timing["pipeline_seconds"] += local_seconds
    scores["ppi_local_default"], timings["ppi_local_default"] = local_default, timing
    local_relaxed, timing, _ = run_pipeline(
        potential, tortuosity_threshold=0.5, core_result=core
    )
    timing = dict(timing, preprocessing_seconds=local_seconds)
    timing["pipeline_seconds"] += local_seconds
    scores["ppi_local_relaxed"], timings["ppi_local_relaxed"] = local_relaxed, timing
    for name, score in scores.items():
        if score.shape != image.shape or not np.isfinite(score).all() or (score < 0).any():
            raise ValueError(f"Invalid score map for {name}")
    return scores, timings


def _calculate_case(arguments):
    spec, directory, fingerprint = arguments
    directory = Path(directory)
    metadata_path = directory / f"{spec['case_id']}.json"
    arrays_path = directory / f"{spec['case_id']}.npz"
    if metadata_path.exists() and arrays_path.exists():
        saved = json.loads(metadata_path.read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Cached scene belongs to a different protocol/source/runtime")
        if digest(arrays_path.read_bytes()) != saved["arrays_sha256"]:
            raise ValueError("Cached arrays checksum mismatch")
        return saved["scene"]
    case = generate_multiline(spec)
    scores, timings = score_methods(case.image)
    scene = {
        "case_id": spec["case_id"],
        "parameters": case.parameters,
        "line_names": case.line_names,
        "sigmas": [float(v) for v in case.sigmas],
        "image_sha256": digest(case.image.tobytes()),
        "centerline_sha256": [digest(line.tobytes()) for line in case.centerlines],
        "score_sha256": {name: digest(value.tobytes()) for name, value in scores.items()},
        "timings": timings,
    }
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **scores)
    write_json(metadata_path, {
        "fingerprint": fingerprint,
        "arrays_sha256": digest(arrays_path.read_bytes()),
        "scene": scene,
    })
    return scene


def calculate_split(specs, cache, fingerprint, workers):
    tasks = [(spec, str(cache), fingerprint) for spec in specs]
    if workers == 1:
        iterator = map(_calculate_case, tasks)
        scenes = []
        for index, scene in enumerate(iterator, 1):
            print(f"Scored {index}/{len(specs)}: {scene['case_id']}", flush=True)
            scenes.append(scene)
        return scenes
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scenes = []
        for index, scene in enumerate(pool.map(_calculate_case, tasks), 1):
            print(f"Scored {index}/{len(specs)}: {scene['case_id']}", flush=True)
            scenes.append(scene)
        return scenes


def reference_for(case):
    return prepare_multiline(case.centerlines, case.sigmas, case.roi)


def calibrate(specs, cache):
    if any(spec["split"] != "development" for spec in specs):
        raise ValueError("Threshold calibration accepts development scenes only")
    values = {method: [] for method in METHODS}
    for spec in specs:
        case = generate_multiline(spec)
        ref = reference_for(case)
        background = case.roi & ~ref.support_mask
        with np.load(cache / f"{spec['case_id']}.npz", allow_pickle=False) as arrays:
            for method in METHODS:
                values[method].append(arrays[method][background])
    return {
        method: [calibrate_threshold(np.concatenate(parts), rate) for rate in BACKGROUND_RATES]
        for method, parts in values.items()
    }


def measure(specs, cache, thresholds):
    rows = []
    for index, spec in enumerate(specs, 1):
        case = generate_multiline(spec)
        ref = reference_for(case)
        with np.load(cache / f"{spec['case_id']}.npz", allow_pickle=False) as arrays:
            for method in METHODS:
                score = arrays[method]
                selections = []
                for definition in thresholds[method]:
                    mask = case.roi & (score > definition["threshold"])
                    selections.append(("frozen_threshold", definition["target_rate"], mask,
                                       {"score_threshold": definition["threshold"]}))
                for budget in BUDGETS:
                    mask, detail = select_at_budget(score, case.roi, fraction=budget)
                    selections.append(("area_budget", budget, mask, detail))
                for selection, operating_point, mask, detail in selections:
                    metrics = ref.evaluate(mask)
                    for line, name in zip(metrics["per_line"], case.line_names, strict=True):
                        line["line_name"] = name
                    rows.append({
                        "case_id": spec["case_id"],
                        "split": spec["split"],
                        "family": spec["family"],
                        "pair_group": spec["pair_group"],
                        "gap": spec["gap"],
                        "method": method,
                        "selection": selection,
                        "operating_point": operating_point,
                        "selection_detail": detail,
                        "metrics": metrics,
                    })
        print(f"Measured {index}/{len(specs)}: {spec['case_id']}", flush=True)
    return rows


def run(*, suite, work_dir, output, workers=1, phase="all"):
    if isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be positive")
    work_dir = Path(work_dir)
    cache = work_dir / "cache"
    definition = protocol(suite)
    sources = provenance()
    fingerprint = digest_json({"protocol": definition, "provenance": sources})
    frozen_path = work_dir / "protocol.json"
    frozen = {"protocol": definition, "provenance": sources, "fingerprint": fingerprint}
    if frozen_path.exists():
        if json.loads(frozen_path.read_text()) != frozen:
            raise ValueError("Work directory already contains a different frozen protocol")
    else:
        write_json(frozen_path, frozen)
    if phase == "prepare":
        return frozen
    manifest = definition["manifest"]
    development = [spec for spec in manifest if spec["split"] == "development"]
    evaluation = [spec for spec in manifest if spec["split"] == "evaluation"]
    start = time.perf_counter()
    scenes = calculate_split(development, cache, fingerprint, workers)
    thresholds = calibrate(development, cache)
    threshold_path = work_dir / "thresholds.json"
    threshold_record = {"fingerprint": fingerprint, "thresholds": thresholds}
    if threshold_path.exists() and json.loads(threshold_path.read_text()) != threshold_record:
        raise ValueError("Development thresholds differ from the saved calibration")
    write_json(threshold_path, threshold_record)
    rows = measure(development, cache, thresholds)
    if phase == "all":
        # Evaluation image processing starts only after thresholds are on disk.
        scenes += calculate_split(evaluation, cache, fingerprint, workers)
        rows += measure(evaluation, cache, thresholds)
    if provenance() != sources or protocol(suite) != definition:
        raise RuntimeError("Sources or protocol changed during the experiment")
    result = {
        **frozen,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "environment": {"platform": platform.platform(), "workers": workers},
        "wall_seconds_this_invocation": time.perf_counter() - start,
        "timing_note": "Pipeline timings are observations under concurrent workers, not controlled speed comparisons; cached results can be reused.",
        "thresholds": thresholds,
        "scenes": scenes,
        "rows": rows,
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("full", "smoke"), default="full")
    parser.add_argument("--work-dir", type=Path, default=Path("benchmark-output/multiline"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-output/multiline.json"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--phase", choices=("prepare", "development", "all"), default="all")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
