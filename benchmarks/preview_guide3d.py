"""Render two fixed development examples locally; never save data imagery in Git.

Uses the frozen study's preprocessing, native-to-working coordinate transform,
PPI configurations, ROI and geometric metrics. This qualitative check is not a
representative sample of the held-out acquisitions and does not tune settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from benchmarks.guide3d_data import (  # noqa: E402
    ANNOTATION_SHA256,
    ANNOTATION_URL,
    ARCHIVE_URL,
    load_image,
    read_index,
    sha256_file,
)
from benchmarks.guide3d_metrics import prepare_reference, select_at_budget  # noqa: E402
from benchmarks.run_guide3d import CONFIGURATIONS, preprocess  # noqa: E402
from benchmarks.run_synthetic import run_pipeline  # noqa: E402
from benchmarks.synthetic import evaluate_prediction as original_evaluate_prediction  # noqa: E402

ACQUISITION = "0-bca-angle-1"
BUDGET = 0.01
REFERENCE_COLOR = "#ed3737"
SELECTION_COLOR = "#00d5ee"
ROI_COLOR = "#ffcb45"


def _percentage(value):
    return "undefined" if value is None else f"{100 * value:.2f}%"


def verify_reference_metrics(prediction, centerline, roi, metrics):
    """Check this full-size local baseline against the original exact evaluator."""
    original = original_evaluate_prediction(prediction, centerline, roi, tolerance=2)
    largest_difference = 0.0
    for key, expected in original.items():
        actual = metrics[key]
        if isinstance(expected, float):
            difference = abs(actual - expected)
            largest_difference = max(largest_difference, difference)
            if difference > 1e-12:
                raise AssertionError(f"Reference evaluator mismatch: {key}")
        elif actual != expected:
            raise AssertionError(f"Reference evaluator mismatch: {key}")
    return {
        "status": "passed",
        "method": "local_contrast",
        "reference_evaluator": "benchmarks.synthetic.evaluate_prediction",
        "checked_metric_count": len(original),
        "absolute_tolerance": 1e-12,
        "maximum_absolute_difference": largest_difference,
    }


def render_preview(record, data_root, output_dir):
    """Save one four-panel figure and return its source/measurement metadata."""
    native = load_image(record, data_root)
    working, roi, raw_potential, local_score, preprocessing = preprocess(native)
    centerline = (record.centerline - 0.5) / 2
    reference = prepare_reference(centerline, roi)
    raw_votes, _, _ = run_pipeline(raw_potential)
    length, segments, threshold = CONFIGURATIONS["ppi_local_default"]
    local_votes, _, _ = run_pipeline(
        255 - local_score,
        segment_length=length,
        nb_segments=segments,
        tortuosity_threshold=threshold,
    )
    methods = [
        ("local_contrast", "Direct local contrast", local_score),
        ("ppi_raw_default", "PPI on raw intensity", raw_votes),
        ("ppi_local_default", "PPI on local contrast", local_votes),
    ]
    measurements = []
    reference_check = None
    figure, axes = plt.subplots(1, 4, figsize=(20, 6.35), layout="constrained")
    figure.set_constrained_layout_pads(w_pad=0.06, h_pad=0.12, wspace=0.025)
    figure.suptitle(
        f"Guide3D development example: {record.acquisition_id} / {record.camera} / "
        f"frame {record.frame_number}\n"
        "Fixed first-frame check; not a representative held-out example",
        fontsize=15,
        fontweight="bold",
    )
    for ax in axes:
        ax.imshow(working, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.contour(roi, levels=[0.5], colors=[ROI_COLOR], linewidths=0.8)
        ax.set_xlim(-0.5, working.shape[1] - 0.5)
        ax.set_ylim(working.shape[0] - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title("Acquired image + manual polyline\n512 x 512 working pixels", fontsize=12)
    for ax, (method, label, scores) in zip(axes[1:], methods, strict=True):
        prediction, selection = select_at_budget(scores, roi, fraction=BUDGET)
        metrics = reference.evaluate(prediction, tolerance=2)
        if method == "local_contrast":
            reference_check = verify_reference_metrics(prediction, centerline, roi, metrics)
        overlay = np.zeros((*working.shape, 4), dtype=float)
        overlay[prediction] = [0, 213 / 255, 238 / 255, 0.85]
        ax.imshow(overlay, interpolation="nearest")
        ax.set_title(
            f"{label}\n"
            f"Actual area {_percentage(selection['actual_fraction'])}; "
            f"coverage {_percentage(metrics['centerline_coverage'])}",
            fontsize=11,
        )
        measurements.append({"method": method, "selection": selection, "metrics": metrics})
    for ax in axes:
        ax.plot(centerline[:, 1], centerline[:, 0], color=REFERENCE_COLOR, linewidth=1.05)
    figure.legend(
        handles=[
            Line2D([0], [0], color=REFERENCE_COLOR, linewidth=2, label="Manual reference polyline"),
            Patch(facecolor=SELECTION_COLOR, label="Selected pixels (nominal 1% ROI area)"),
            Line2D([0], [0], color=ROI_COLOR, linewidth=2, label="Evaluation ROI boundary"),
        ],
        loc="outside lower center",
        bbox_to_anchor=(0.5, -0.014),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    figure.text(
        0.5,
        0.058,
        "Coverage: reference arclength within 2 working pixels (4 native pixels) of selections. "
        "Whole cutoff ties retained.\n"
        "Guide3D: Tudor Jianu et al., ACCV 2024 | CC BY-NC 4.0 | "
        "Modified: 2x2 averaging, overlays and derived measurements.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    # Reserve an explicit footer band independently of Matplotlib's outside
    # legend layout, keeping attribution visible in the saved artifact.
    figure.get_layout_engine().set(rect=(0, 0.14, 1, 0.86))
    path = Path(output_dir) / f"development_{record.acquisition_id}_{record.camera}.png"
    figure.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "case_id": record.case_id,
        "image_path": record.image_path,
        "native_image_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
        "figure": path.name,
        "preprocessing": preprocessing,
        "methods": measurements,
        "reference_evaluator_check": reference_check,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    repository = Path(__file__).resolve().parents[1]
    if output_dir.is_relative_to(repository):
        parser.error(
            "Write Guide3D image previews outside the repository; their license is separate"
        )
    if sha256_file(args.annotations) != ANNOTATION_SHA256:
        parser.error("Expected the pinned Guide3D manual annotations")
    records = [r for r in read_index(args.annotations) if r.acquisition_id == ACQUISITION]
    if not records:
        parser.error("Missing fixed development acquisition")
    first_frame = min(record.frame_number for record in records)
    examples = sorted(
        [record for record in records if record.frame_number == first_frame],
        key=lambda record: record.camera,
    )
    if [record.camera for record in examples] != ["camera1", "camera2"]:
        parser.error("Expected exactly two paired development views")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for record in examples:
        result = render_preview(record, args.data_root, output_dir)
        results.append(result)
        print(f"Rendered {record.case_id}: {output_dir / result['figure']}", flush=True)
    metadata = {
        "purpose": "Fixed development-only qualitative examples, not representative held-out claims",
        "source_archive": ARCHIVE_URL,
        "source_annotations": ANNOTATION_URL,
        "annotation_sha256": ANNOTATION_SHA256,
        "license": "CC-BY-NC-4.0",
        "nominal_budget": BUDGET,
        "tolerance_working_pixels": 2,
        "native_pixels_per_working_pixel": 2,
        "results": results,
    }
    (output_dir / "preview-metrics.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Guide3D development examples\n\n"
        "These two figures show the first annotated frame of development acquisition "
        "`0-bca-angle-1`, one view from each camera. They are fixed qualitative checks, "
        "not representative examples selected from the held-out evaluation.\n\n"
        + "\n".join(f"- [{result['case_id']}]({result['figure']})" for result in results)
        + "\n\nEach figure compares direct local contrast, PPI on raw intensity, and "
        "PPI on local contrast. Both PPI methods use segment length 3, 10 segments, "
        "and tortuosity threshold 0.75. The nominal selection budget is 1% of the "
        "evaluation ROI. Whole score ties are retained, so actual areas can exceed "
        "1%; each panel reports its actual area and reference coverage. Coverage "
        "uses a tolerance of 2 working pixels (4 native pixels). The colored ROI "
        "boundary describes evaluation support, not an absence of path boundary effects.\n\n"
        "Measurements are saved in [preview-metrics.json](preview-metrics.json). "
        "The preprocessing and geometric measurements are exactly those used by "
        "the frozen Guide3D pilot. No settings were tuned using these previews.\n\n"
        "For each camera, all shared metric fields for the direct-local-contrast "
        "selection were independently checked against the original exact evaluator "
        "at the full 512 x 512 working resolution (absolute tolerance 1e-12). "
        "The adjacent JSON records this check.\n\n"
        "## Source and license\n\n"
        "Source: Tudor Jianu et al., *Guide3D: A Bi-planar X-ray Dataset for 3D Shape "
        "Reconstruction*, ACCV 2024. "
        "[Project](https://airvlab.github.io/guide3d/), "
        "[paper](https://arxiv.org/abs/2410.22224), "
        "[dataset](https://huggingface.co/datasets/airvlab/guide3d).\n\n"
        "The source images and these derived image previews are covered by "
        "[Creative Commons Attribution-NonCommercial 4.0 International]"
        "(https://creativecommons.org/licenses/by-nc/4.0/). "
        "This is separate from the software repository's BSD license. "
        "No endorsement by the dataset creators is implied.\n\n"
        "Modifications: native 1024 x 1024 grayscale images were reduced by 2x2 "
        "area averaging to 512 x 512; manual annotation coordinates were transformed "
        "to working pixel centers; ROI contours, prediction selections, explanatory "
        "labels, and derived measurements were added. "
        "Pinned source URLs and source hashes are recorded in the adjacent JSON.\n\n"
        "These image files are stored outside the public software checkout. "
        "Recreate them with `python -m benchmarks.preview_guide3d --annotations "
        "PATH/TO/raw.json --data-root PATH/TO/guide3d --output-dir "
        "PATH/OUTSIDE/REPOSITORY`. Install the repository's `study` and `benchmark` "
        "extras to supply SciPy, Pillow, and Matplotlib.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
