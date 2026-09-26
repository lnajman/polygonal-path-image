"""Frozen, acquisition-separated Guide3D pilot; raw data stay outside the checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

from benchmarks.guide3d_data import (
    ANNOTATION_SHA256,
    ANNOTATION_URL,
    ARCHIVE_SHA256,
    ARCHIVE_URL,
    load_image,
    read_index,
    sha256_file,
)
from benchmarks.guide3d_metrics import prepare_reference, select_at_budget
from benchmarks.run_synthetic import _provenance, run_pipeline

DEV_ACQUISITIONS = ("0-bca-angle-1", "0-bca-straight-1", "1-bca-angle-1")
BUDGETS = (0.0025, 0.005, 0.01, 0.02)
CONFIGURATIONS = {
    "ppi_local_default": (3, 10, 0.75),
    "ppi_local_relaxed": (3, 10, 0.5),
    "ppi_local_short": (3, 5, 0.75),
}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_manifest(records, pairs_per_acquisition=10):
    """Sample positions uniformly in each sorted acquisition, retaining both views."""
    if (
        isinstance(pairs_per_acquisition, bool)
        or not isinstance(pairs_per_acquisition, int)
        or pairs_per_acquisition < 1
    ):
        raise ValueError("pairs_per_acquisition must be a positive integer")
    acquisitions = sorted({record.acquisition_id for record in records})
    if not set(DEV_ACQUISITIONS).issubset(acquisitions):
        raise ValueError("Missing declared development acquisitions")
    manifest = []
    for acquisition in acquisitions:
        group = [record for record in records if record.acquisition_id == acquisition]
        frames = sorted({record.frame_number for record in group})
        for frame in frames:
            if sorted(record.camera for record in group if record.frame_number == frame) != [
                "camera1",
                "camera2",
            ]:
                raise ValueError("Each frame must have exactly two distinct camera views")
        if len(frames) < pairs_per_acquisition:
            raise ValueError("Insufficient paired frames")
        selected = {
            frames[i]
            for i in np.rint(np.linspace(0, len(frames) - 1, pairs_per_acquisition)).astype(int)
        }
        for record in group:
            if record.frame_number in selected:
                manifest.append(
                    {
                        "case_id": record.case_id,
                        "acquisition_id": acquisition,
                        "frame_number": record.frame_number,
                        "camera": record.camera,
                        "image_path": record.image_path,
                        "fluid": record.fluid,
                        "guidewire_type": record.guidewire_type,
                        "split": "development" if acquisition in DEV_ACQUISITIONS else "evaluation",
                    }
                )
    return sorted(manifest, key=lambda row: row["case_id"])


def preprocess(image):
    """2x2 area averaging and image-only FOV/local-contrast construction."""
    if image.shape != (1024, 1024) or image.dtype != np.uint8:
        raise ValueError("Expected a native 1024x1024 uint8 image")
    working = np.rint(image.reshape(512, 2, 512, 2).mean(axis=(1, 3))).astype(np.uint8)
    labels, count = ndimage.label(working > 5)
    if not count:
        raise ValueError("Empty field of view")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    fov = ndimage.binary_fill_holes(labels == sizes.argmax())
    distance = ndimage.distance_transform_edt(np.pad(fov, 1))[1:-1, 1:-1]
    roi = distance > 16
    if not roi.any():
        raise ValueError("Empty evaluation ROI")
    filled = np.where(fov, working, np.median(working[fov])).astype(np.float64)
    local = np.maximum(ndimage.gaussian_filter(filled, sigma=8, mode="reflect") - filled, 0)
    scale = max(float(np.quantile(local[roi], 0.995)), 1.0)
    local_score = np.rint(255 * np.clip(local / scale, 0, 1)).astype(np.uint8)
    local_score[~fov] = 0
    raw_potential = np.where(fov, working, 255).astype(np.uint8)
    return (
        working,
        roi,
        raw_potential,
        local_score,
        {
            "fov_pixels": int(fov.sum()),
            "roi_pixels": int(roi.sum()),
            "local_scale_graylevels": scale,
            "local_saturated_roi_fraction": float(np.mean(local_score[roi] == 255)),
        },
    )


def protocol(manifest):
    return {
        "study_id": "guide3d-pilot-v1",
        "data_scope": "acquired fluoroscopic phantom images",
        "sources": {
            "archive_url": ARCHIVE_URL,
            "archive_sha256": ARCHIVE_SHA256,
            "annotation_url": ANNOTATION_URL,
            "annotation_sha256": ANNOTATION_SHA256,
            "license": "CC-BY-NC-4.0",
        },
        "manifest": manifest,
        "development_acquisitions": list(DEV_ACQUISITIONS),
        "sampling": "10 rounded linspace positions among sorted paired frame IDs per acquisition",
        "native_size": [1024, 1024],
        "working_size": [512, 512],
        "native_pixels_per_working_pixel": 2.0,
        "image_resampling": "2x2 arithmetic mean, numpy.rint to uint8",
        "coordinate_transform": "working row,column = (native row,column - 0.5) / 2",
        "fov": "largest 4-connected component of working image >5; fill holes",
        "roi": "FOV pixels with Euclidean distance >16 working pixels from outside FOV or image",
        "raw_potential": "working intensity; outside FOV replaced by 255",
        "local_contrast": "max(Gaussian(sigma=8,reflect)-image,0); outside FOV filled with FOV median before Gaussian; scale=max(ROI 99.5th percentile,1); rint(255*clip(contrast/scale,0,1)); outside FOV score=0",
        "local_potential": "255 minus the same quantized local contrast used by the direct baseline",
        "configurations": {key: list(value) for key, value in CONFIGURATIONS.items()},
        "raw_ppi_configuration": [3, 10, 0.75],
        "budgets": list(BUDGETS),
        "primary_budget": 0.01,
        "tolerance_working_pixels": 2,
        "reference_sample_step_working_pixels": 0.25,
        "selection": "finite positive scores; ceil(budget*ROI pixels); retain entire cutoff tie block",
        "development_objective": "maximum acquisition-macro centerline coverage at nominal 1%; ties: lower actual area, then configuration insertion order",
        "aggregation": "mean of views within acquisition, then equally weighted acquisitions",
        "timing_scope": "observed pipeline wall times under concurrent workers; not controlled speed measurements",
        "limitations": [
            "phantom only",
            "full tip-and-shaft annotations",
            "no absence controls",
            "one fluid/straight acquisition, held out",
            "no claimed MICCAI reproduction",
        ],
    }


def select_configuration(rows):
    summaries = []
    expected_cases = None
    for order, method in enumerate(CONFIGURATIONS):
        selected = [r for r in rows if r["method"] == method and r["budget_fraction"] == 0.01]
        cases = {(r["acquisition_id"], r["case_id"]) for r in selected}
        if len(cases) != len(selected):
            raise ValueError("Duplicate development case")
        if expected_cases is None:
            expected_cases = cases
        elif cases != expected_cases:
            raise ValueError("Candidate configurations must use identical development cases")
        for row in selected:
            for metric in ("centerline_coverage", "selected_fraction"):
                value = row["metrics"][metric]
                if value is None or not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid development metric")
        groups = sorted({r["acquisition_id"] for r in selected})
        if groups != sorted(DEV_ACQUISITIONS):
            raise ValueError("Incomplete development groups")
        values = []
        for group in groups:
            group_rows = [r for r in selected if r["acquisition_id"] == group]
            values.append(
                (
                    np.mean([r["metrics"]["centerline_coverage"] for r in group_rows]),
                    np.mean([r["metrics"]["selected_fraction"] for r in group_rows]),
                )
            )
        summaries.append(
            {
                "method": method,
                "coverage": float(np.mean([v[0] for v in values])),
                "actual_fraction": float(np.mean([v[1] for v in values])),
                "order": order,
            }
        )
    best = min(summaries, key=lambda row: (-row["coverage"], row["actual_fraction"], row["order"]))
    return {"selected_method": best["method"], "development_scores": summaries}


def run_case(arguments):
    record, data_root, methods, fingerprint, cache_dir = arguments
    cache = Path(cache_dir) / (record.case_id.replace(":", "_") + ".json")
    if cache.exists():
        result = json.loads(cache.read_text())
        if result["fingerprint"] != fingerprint or result["methods"] != methods:
            raise ValueError(f"Stale checkpoint: {cache}")
        return result
    started = time.perf_counter()
    native = load_image(record, data_root)
    working, roi, raw_potential, local_score, preprocessing = preprocess(native)
    centerline = (record.centerline - 0.5) / 2
    reference = prepare_reference(centerline, roi)
    score_maps = {"darkness": 255 - working, "local_contrast": local_score}
    timings = {}
    raw_votes, timings["ppi_raw_default"], _ = run_pipeline(raw_potential)
    score_maps["ppi_raw_default"] = raw_votes
    local_potential = 255 - local_score
    cached_core = None
    for method in methods:
        if method not in CONFIGURATIONS:
            continue
        length, segments, threshold = CONFIGURATIONS[method]
        reuse = cached_core if segments == 10 else None
        votes, timings[method], core = run_pipeline(
            local_potential,
            segment_length=length,
            nb_segments=segments,
            tortuosity_threshold=threshold,
            core_result=reuse,
        )
        if segments == 10:
            cached_core = core
        score_maps[method] = votes
    rows = []
    for method in methods:
        for budget in BUDGETS:
            prediction, selection = select_at_budget(score_maps[method], roi, fraction=budget)
            rows.append(
                {
                    "case_id": record.case_id,
                    "acquisition_id": record.acquisition_id,
                    "camera": record.camera,
                    "frame_number": record.frame_number,
                    "method": method,
                    "budget_fraction": budget,
                    "selection": selection,
                    "metrics": reference.evaluate(prediction),
                }
            )
    result = {
        "fingerprint": fingerprint,
        "case_id": record.case_id,
        "methods": methods,
        "image_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
        "preprocessing": preprocessing,
        "timings": timings,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(cache, result)
    return result


def run_phase(records, manifest, phase, methods, args, fingerprint):
    selected = {row["case_id"] for row in manifest if row["split"] == phase}
    jobs = [
        (record, str(args.data_root), methods, fingerprint, str(args.work_dir / phase))
        for record in records
        if record.case_id in selected
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, job): job[0].case_id for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{phase} {len(results)}/{len(jobs)} {result['case_id']}", flush=True)
    results.sort(key=lambda result: result["case_id"])
    return {
        "rows": [row for result in results for row in result["rows"]],
        "cases": [
            {
                key: value
                for key, value in result.items()
                if key not in ("rows", "fingerprint", "methods")
            }
            for result in results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phase", choices=("prepare", "development", "all"), default="all")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if sha256_file(args.annotations) != ANNOTATION_SHA256:
        raise ValueError("Unexpected annotation hash")
    audit = json.loads(args.audit.read_text())
    if (
        audit.get("archive_actual_sha256") != ARCHIVE_SHA256
        or not audit.get("archive_integrity_verified")
        or audit.get("annotation_actual_sha256") != ANNOTATION_SHA256
        or audit.get("missing_images")
        or audit.get("out_of_bounds_annotations")
    ):
        raise ValueError("Expected a successful audit of the pinned data")
    records = read_index(args.annotations)
    manifest = make_manifest(records)
    definition = protocol(manifest)
    provenance = _provenance()
    provenance["study_source_sha256"] = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ("run_guide3d.py", "guide3d_data.py", "guide3d_metrics.py")
    }
    fingerprint = digest_json(
        {
            "protocol": definition,
            "study_source": provenance["study_source_sha256"],
            "implementation": provenance["implementation_source_sha256"],
            "benchmark_source": provenance["benchmark_source_sha256"],
            "compiled_kernel": provenance["compiled_kernel_sha256"],
        }
    )
    frozen = args.work_dir / "protocol.json"
    if frozen.exists() and json.loads(frozen.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Protocol or executed sources changed: use a fresh work directory")
    if not frozen.exists():
        write_json(
            frozen,
            {
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "fingerprint": fingerprint,
                "protocol": definition,
            },
        )
    if args.phase == "prepare":
        print(f"Frozen {len(manifest)} views: {frozen}")
        return
    base = ["darkness", "local_contrast", "ppi_raw_default"]
    development = run_phase(
        records, manifest, "development", base + list(CONFIGURATIONS), args, fingerprint
    )
    selection = select_configuration(development["rows"])
    write_json(args.work_dir / "selection.json", selection)
    output = {
        "protocol": definition,
        "protocol_freeze": json.loads(frozen.read_text()),
        "audit": audit,
        "selection": selection,
        "development": development,
        "provenance": provenance,
        "environment": {
            "platform": platform.platform(),
            "workers": args.workers,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("polygonal-path-image", "numpy", "scipy", "Pillow")
            },
        },
    }
    if args.phase == "all":
        methods = base + ["ppi_local_default"]
        if selection["selected_method"] not in methods:
            methods.append(selection["selected_method"])
        output["evaluation"] = run_phase(
            records, manifest, "evaluation", methods, args, fingerprint
        )
    final_provenance = _provenance()
    for key in (
        "implementation_source_sha256",
        "benchmark_source_sha256",
        "compiled_kernel_sha256",
    ):
        if final_provenance[key] != provenance[key]:
            raise RuntimeError("Executed implementation changed during this run")
    for name, expected in provenance["study_source_sha256"].items():
        if sha256_file(Path(__file__).with_name(name)) != expected:
            raise RuntimeError("Study source changed during this run")
    output["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, output)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
