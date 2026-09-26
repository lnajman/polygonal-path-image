"""Development-calibrated validation of one frozen straight-fallback candidate.

Preparation and development never generate fresh evaluation images. The explicit
``all`` phase uses the same saved thresholds on the predeclared fresh panel.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path

import numpy as np

from benchmarks.multiline_data import _spec, generate_multiline, multiline_specs
from benchmarks.run_multiline import (
    BACKGROUND_RATES,
    calibrate_threshold,
    digest,
    digest_json,
    reference_for,
    write_json,
)
from benchmarks.run_multiline import (
    METHODS as BASE_METHODS,
)
from benchmarks.run_multiline import (
    protocol as base_protocol,
)
from benchmarks.run_multiline import (
    provenance as base_provenance,
)
from benchmarks.run_multiline import (
    score_methods as base_score_methods,
)
from benchmarks.straight_fallback import straight_fallback
from benchmarks.synthetic import select_at_budget

CANDIDATE = "ppi_raw_straight_fallback"
METHODS = (*BASE_METHODS, CANDIDATE)
BUDGETS = (0.02,)
PRIMARY_RATE = 0.005


def fresh_manifest(plan):
    """Build identifiers and parameters only; do not render evaluation images."""
    angles = plan["evaluation_angles_degrees"]
    seeds = plan["evaluation_seeds"]
    if angles != [19, 53, 107] or seeds != [4101, 4102]:
        raise ValueError("This frozen experiment requires angles 19/53/107 and seeds 4101/4102")
    if plan["line_scenes"] != 48:
        raise ValueError("The predeclared panel must contain 48 line scenes")
    if plan["noise_rule"] != "5 if angle_index + seed_index is even, otherwise 12":
        raise ValueError("Unexpected fresh-panel noise rule")
    templates = [
        spec
        for spec in multiline_specs()
        if spec["split"] == "development"
        and spec["seed"] == 1001
        and spec["family"] not in ("blank", "noise")
    ]
    names = [
        spec["family"] + (f"_gap{int(spec['gap'])}" if spec["gap"] is not None else "")
        for spec in templates
    ]
    if names != plan["families"]:
        raise ValueError("The predeclared families differ from the unchanged geometry")
    specs = [
        _spec(
            "evaluation",
            template["family"],
            angle,
            seed,
            5 if (i + j) % 2 == 0 else 12,
            gap=None if template["gap"] is None else int(template["gap"]),
        )
        for i, angle in enumerate(angles)
        for j, seed in enumerate(seeds)
        for template in templates
    ]
    specs.append(_spec("evaluation", "blank", None, seeds[0], 0))
    specs.extend(
        _spec("evaluation", "noise", None, seed, noise) for seed in seeds for noise in (5, 12)
    )
    return specs


def protocol(plan_path):
    """Bind the previously written plan to an explicit, reproducible candidate."""
    plan_bytes = Path(plan_path).read_bytes()
    plan = json.loads(plan_bytes)
    definition = copy.deepcopy(base_protocol())
    development = [s for s in definition["manifest"] if s["split"] == "development"]
    definition.update(
        {
            "study_id": "straight-fallback-v1",
            "suite": "fresh-validation",
            "predeclared_plan": plan,
            "predeclared_plan_sha256": digest(plan_bytes),
            "manifest": development + fresh_manifest(plan),
            "methods": list(METHODS),
            "experimental_candidate": {
                "method": CANDIDATE,
                "input": "original uint8 image, same as ppi_raw_default",
                "segment_length": 3,
                "nb_segments": 10,
                "tortuosity_threshold": 0.75,
                "replacement": "preserve every accepted original winner; rejected finite origins use the cheapest feasible fixed-displacement straight candidate",
                "directions": "repeat each existing cone endpoint displacement ten times; 24 distinct directions at segment_length=3",
                "ties": "first candidate in H/B/E/W and legacy endpoint enumeration wins equal costs",
                "no_candidate": "leave the rejected origin inactive; original impossible origins stay inactive",
                "votes": "unweighted visits, border=0; no blank suppression or intensity gate",
                "scope": "research-only path-selection fallback, not a globally optimal tortuosity-constrained path or exact paper reproduction",
            },
        }
    )
    definition["budget_selection"]["fractions"] = list(BUDGETS)
    definition["limits"] += [
        "Candidate informed by the previous panel and original development; only the new evaluation seed/orientation combinations are fresh",
        "Fresh evaluation has six shared orientation/seed contexts and only two underlying Gaussian draws, not 48 independent replicates",
        "The deterministic blank is intentionally shared across development and evaluation; it is a functional control, not an independent replicate",
        "Fallback favours straight structures; curved/branch recovery and actual background selection remain necessary guardrails",
        "No candidate or parameter retuning is permitted after fresh evaluation",
    ]
    return definition


def provenance():
    record = base_provenance()
    directory = Path(__file__).resolve().parent
    record["benchmark_source_sha256"].update(
        {
            name: digest((directory / name).read_bytes())
            for name in ("straight_fallback.py", "run_fallback_validation.py")
        }
    )
    return record


def score_methods(image):
    scores, timings = base_score_methods(image)
    start = time.perf_counter()
    votes, costs, _, replacements = straight_fallback(
        image, segment_length=3, nb_segments=10, threshold=0.75
    )
    scores[CANDIDATE] = votes
    timings[CANDIDATE] = {"pipeline_seconds": time.perf_counter() - start}
    if votes.shape != image.shape or not np.isfinite(votes).all() or (votes < 0).any():
        raise ValueError("Invalid candidate score map")
    diagnostics = {
        "replaced_origins": int(replacements.sum()),
        "active_origins_after_fallback": int(np.isfinite(costs).sum()),
        "total_origins": int(image.size),
    }
    return scores, timings, diagnostics


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
    scores, timings, diagnostics = score_methods(case.image)
    if set(scores) != set(METHODS):
        raise ValueError("Score maps must contain every declared method exactly once")
    scene = {
        "case_id": spec["case_id"],
        "parameters": case.parameters,
        "line_names": case.line_names,
        "sigmas": [float(v) for v in case.sigmas],
        "image_sha256": digest(case.image.tobytes()),
        "centerline_sha256": [digest(line.tobytes()) for line in case.centerlines],
        "score_sha256": {name: digest(value.tobytes()) for name, value in scores.items()},
        "timings": timings,
        "experimental_diagnostics": diagnostics,
    }
    directory.mkdir(parents=True, exist_ok=True)
    temporary = arrays_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **scores)
    temporary.replace(arrays_path)
    write_json(
        metadata_path,
        {
            "fingerprint": fingerprint,
            "arrays_sha256": digest(arrays_path.read_bytes()),
            "scene": scene,
        },
    )
    return scene


def calculate_split(specs, cache, fingerprint, workers):
    tasks = [(spec, str(cache), fingerprint) for spec in specs]
    if workers == 1:
        iterator = map(_calculate_case, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(_calculate_case, tasks)
    try:
        scenes = []
        for index, scene in enumerate(iterator, 1):
            print(f"Scored {index}/{len(specs)}: {scene['case_id']}", flush=True)
            scenes.append(scene)
        return scenes
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def calibrate(specs, cache):
    if not specs or any(spec["split"] != "development" for spec in specs):
        raise ValueError("Threshold calibration accepts nonempty development scenes only")
    populations = {method: [] for method in METHODS}
    for spec in specs:
        case = generate_multiline(spec)
        ref = reference_for(case)
        background = case.roi & ~ref.support_mask
        with np.load(cache / f"{spec['case_id']}.npz", allow_pickle=False) as arrays:
            for method in METHODS:
                populations[method].append(arrays[method][background])
    return {
        method: [calibrate_threshold(np.concatenate(parts), rate) for rate in BACKGROUND_RATES]
        for method, parts in populations.items()
    }


def measure(specs, cache, thresholds):
    rows = []
    for index, spec in enumerate(specs, 1):
        case = generate_multiline(spec)
        ref = reference_for(case)
        with np.load(cache / f"{spec['case_id']}.npz", allow_pickle=False) as arrays:
            for method in METHODS:
                score = arrays[method]
                selections = [
                    (
                        "frozen_threshold",
                        definition["target_rate"],
                        case.roi & (score > definition["threshold"]),
                        {"score_threshold": definition["threshold"]},
                    )
                    for definition in thresholds[method]
                ]
                for budget in BUDGETS:
                    mask, detail = select_at_budget(score, case.roi, fraction=budget)
                    selections.append(("area_budget", budget, mask, detail))
                for selection, point, mask, detail in selections:
                    metrics = ref.evaluate(mask)
                    for line, name in zip(metrics["per_line"], case.line_names, strict=True):
                        line["line_name"] = name
                    rows.append(
                        {
                            "case_id": spec["case_id"],
                            "split": spec["split"],
                            "family": spec["family"],
                            "pair_group": spec["pair_group"],
                            "gap": spec["gap"],
                            "method": method,
                            "selection": selection,
                            "operating_point": point,
                            "selection_detail": detail,
                            "metrics": metrics,
                        }
                    )
        print(f"Measured {index}/{len(specs)}: {spec['case_id']}", flush=True)
    return rows


def run(*, plan, work_dir, output, workers=1, phase="prepare"):
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if phase not in ("prepare", "development", "all"):
        raise ValueError("phase must be prepare, development or all")
    work_dir = Path(work_dir)
    cache = work_dir / "cache"
    definition = protocol(plan)
    sources = provenance()
    fingerprint = digest_json({"protocol": definition, "provenance": sources})
    frozen = {"protocol": definition, "provenance": sources, "fingerprint": fingerprint}
    frozen_path = work_dir / "protocol.json"
    if frozen_path.exists():
        if json.loads(frozen_path.read_text()) != frozen:
            raise ValueError("Work directory already contains a different frozen protocol")
    else:
        write_json(frozen_path, frozen)
    if phase == "prepare":
        return frozen
    start = time.perf_counter()
    development = [s for s in definition["manifest"] if s["split"] == "development"]
    evaluation = [s for s in definition["manifest"] if s["split"] == "evaluation"]
    scenes = calculate_split(development, cache, fingerprint, workers)
    thresholds = calibrate(development, cache)
    threshold_path = work_dir / "thresholds.json"
    threshold_record = {"fingerprint": fingerprint, "thresholds": thresholds}
    if threshold_path.exists() and json.loads(threshold_path.read_text()) != threshold_record:
        raise ValueError("Development thresholds differ from the saved calibration")
    write_json(threshold_path, threshold_record)
    rows = measure(development, cache, thresholds)
    if phase == "all":
        # The persisted calibration is the only threshold source for fresh images.
        saved = json.loads(threshold_path.read_text())
        if saved != threshold_record:
            raise ValueError("Saved calibration changed before evaluation")
        scenes += calculate_split(evaluation, cache, fingerprint, workers)
        rows += measure(evaluation, cache, saved["thresholds"])
    if provenance() != sources or protocol(plan) != definition:
        raise RuntimeError("Sources or protocol changed during the experiment")
    result = {
        **frozen,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "environment": {"platform": platform.platform(), "workers": int(workers)},
        "wall_seconds_this_invocation": time.perf_counter() - start,
        "timing_note": "Concurrent, cacheable observations; not controlled speed comparisons.",
        "thresholds": thresholds,
        "scenes": scenes,
        "rows": rows,
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--phase", choices=("prepare", "development", "all"), default="prepare")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
