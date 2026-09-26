"""Figures and transparent summaries of the complete-reference line study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "darkness": "Darkness",
    "local_contrast": "Local contrast",
    "frangi": "Frangi",
    "ppi_raw_default": "PPI raw, τ=0.75",
    "ppi_local_default": "PPI local, τ=0.75",
    "ppi_local_relaxed": "PPI local, τ=0.50",
}
COLORS = dict(zip(LABELS, ("#757575", "#009E73", "#0072B2", "#E69F00", "#CC79A7", "#D55E00")))
FAMILY_LABELS = {
    "crossing": "Crossing",
    "branch": "Branch",
    "parallel": "Parallel",
    "mixed_width": "Mixed widths",
    "curved_pair": "Curved pair",
    "weak_alone": "Weak alone",
    "weak_near_strong": "Weak + strong",
}


def select_rows(result, *, method=None, selection="frozen_threshold", point=0.005,
                split="evaluation", target=True):
    return [
        row for row in result["rows"]
        if row["split"] == split and row["selection"] == selection
        and row["operating_point"] == point
        and (method is None or row["method"] == method)
        and (target is None or row["metrics"]["has_target"] == target)
    ]


def mean_metric(rows, key):
    values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
    return float(np.mean(values)) if values else None


def _mean_or_none(values):
    return float(np.mean(values)) if values else None


def primary_summary(result):
    summary = {}
    for method in LABELS:
        rows = select_rows(result, method=method)
        summary[method] = {
            "line_scenes": len(rows),
            "scenes_with_selections": sum(bool(r["metrics"]["selected_pixels"]) for r in rows),
            "metric_scene_counts": {},
        }
        for key in ("macro_coverage", "worst_line_coverage", "outside_support_fraction",
                    "selected_support_precision", "union_centerline_precision",
                    "prediction_to_reference_mean_distance", "selected_fraction"):
            summary[method][key] = mean_metric(rows, key)
            summary[method]["metric_scene_counts"][key] = sum(
                r["metrics"][key] is not None for r in rows
            )
        summary[method]["mean_longest_gap"] = _mean_or_none([
            np.mean([line["longest_uncovered_arclength"] for line in row["metrics"]["per_line"]])
            for row in rows
        ])
        summary[method]["mean_reference_to_prediction_distance"] = _mean_or_none([
            np.mean([line["reference_to_prediction_mean_distance"]
                     for line in row["metrics"]["per_line"]
                     if line["reference_to_prediction_mean_distance"] is not None])
            for row in rows if row["metrics"]["selected_pixels"]
        ])
    return summary


def weak_pairs(result, *, method, selection="frozen_threshold", point=0.005):
    rows = select_rows(result, method=method, selection=selection, point=point)
    alone = {r["pair_group"]: r for r in rows if r["family"] == "weak_alone"}
    records = []
    for row in rows:
        if row["family"] != "weak_near_strong":
            continue
        baseline = alone[row["pair_group"]]
        weak = next(line for line in row["metrics"]["per_line"] if line["line_name"] == "weak")
        single = next(line for line in baseline["metrics"]["per_line"]
                      if line["line_name"] == "weak")
        records.append({
            "pair_group": row["pair_group"], "gap": row["gap"],
            "alone_coverage": single["coverage"], "paired_coverage": weak["coverage"],
            "change": weak["coverage"] - single["coverage"],
            "gap_change": weak["longest_uncovered_arclength"] - single["longest_uncovered_arclength"],
        })
    return records


def _axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


def plot_tradeoff(result):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    rates = result["protocol"]["threshold_selection"]["development_background_rates"]
    for method, label in LABELS.items():
        rows = [select_rows(result, method=method, point=rate) for rate in rates]
        background = [100 * mean_metric(group, "outside_support_fraction") for group in rows]
        for ax, key in zip(axes, ("macro_coverage", "worst_line_coverage"), strict=True):
            coverage = [100 * mean_metric(group, key) for group in rows]
            ax.plot(background, coverage, "o-", color=COLORS[method], label=label, ms=4)
            primary = rates.index(0.005)
            ax.scatter(background[primary], coverage[primary], s=85,
                       facecolors="none", edgecolors=COLORS[method], zorder=5)
    for ax, title in zip(axes, ("Mean of all lines", "Worst line in each scene"), strict=True):
        ax.set(title=title, xlabel="Evaluation pixels selected outside all supports (%)",
               ylabel="Centerline coverage (%)", ylim=(0, 102))
        _axes_style(ax)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("72 evaluation line scenes · frozen development thresholds\n"
                 "Open rings: primary 0.5% development background setting", fontsize=12)
    return fig


def plot_families(result):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    y = np.arange(len(FAMILY_LABELS))
    for index, (method, label) in enumerate(LABELS.items()):
        rows = select_rows(result, method=method)
        offset = (index - 2.5) * 0.10
        groups = [[r for r in rows if r["family"] == family] for family in FAMILY_LABELS]
        coverage = [100 * mean_metric(group, "macro_coverage") for group in groups]
        gaps = [np.mean([np.mean([line["longest_uncovered_arclength"]
                                 for line in row["metrics"]["per_line"]]) for row in group])
                for group in groups]
        axes[0].scatter(coverage, y + offset, color=COLORS[method], label=label, s=25)
        axes[1].scatter(gaps, y + offset, color=COLORS[method], label=label, s=25)
    for ax in axes:
        ax.set_yticks(y, FAMILY_LABELS.values())
        ax.invert_yaxis()
        _axes_style(ax)
    axes[0].set(xlabel="Mean per-line coverage (%)", xlim=(0, 102))
    axes[1].set(xlabel="Mean longest uncovered run (pixels)", xlim=(0, None))
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3)
    fig.suptitle("Every line is a target · primary frozen threshold\n"
                 "9 scenes per family; weak + strong has 18 (two separations)", fontsize=12)
    return fig


def plot_weak_neighbours(result):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), layout="constrained")
    for ax, selection, point, title in (
        (axes[0], "frozen_threshold", 0.005, "Frozen score threshold (primary)"),
        (axes[1], "area_budget", 0.02, "Fixed 2% area budget (ranking diagnostic)"),
    ):
        for index, (method, label) in enumerate(LABELS.items()):
            pairs = weak_pairs(result, method=method, selection=selection, point=point)
            for gap, offset, marker in ((6, -0.16, "o"), (10, 0.16, "s")):
                changes = [100 * p["change"] for p in pairs if p["gap"] == gap]
                x = index + offset
                ax.plot([x, x], [min(changes), max(changes)], color=COLORS[method], lw=1)
                ax.scatter(x, np.mean(changes), color=COLORS[method], marker=marker, s=42,
                           label=f"{gap}-pixel separation" if index == 0 else None)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_xticks(range(len(LABELS)), LABELS.values(), rotation=35, ha="right")
        ax.set(title=title, ylabel="Weak-line coverage change (percentage points)", ylim=(-105, 25))
        _axes_style(ax)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Adding a strong neighbour to the same weak line and noise\n"
                 "Points: mean of 9 matched pairs; bars: observed range, not confidence intervals",
                 fontsize=12)
    return fig


def plot_controls(result):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    specifications = {spec["case_id"]: spec for spec in result["protocol"]["manifest"]}
    for index, (method, label) in enumerate(LABELS.items()):
        rows = select_rows(result, method=method, target=False)
        for ax, family in zip(axes, ("blank", "noise"), strict=True):
            group = [r for r in rows if r["family"] == family]
            for row in group:
                noise = specifications[row["case_id"]]["noise_std"]
                x = index + ({0: 0, 5: -0.12, 12: 0.12}[noise])
                ax.scatter(x, 100 * row["metrics"]["selected_fraction"], color=COLORS[method],
                           marker="o" if noise < 12 else "s", alpha=0.85, s=32)
            ax.set_xticks(range(len(LABELS)), LABELS.values(), rotation=35, ha="right")
            ax.set_ylabel("ROI selected (%)")
            _axes_style(ax)
    for ax in axes:
        heights = [float(point[1]) for collection in ax.collections
                   for point in collection.get_offsets()]
        ax.set_ylim(0, max(0.05, max(heights, default=0) * 1.12))
    axes[0].set_title("One uniform blank")
    axes[1].set_title("Six noise-only images\nCircles: σ=5; squares: σ=12")
    fig.suptitle("Absence controls · same primary frozen thresholds as line scenes", fontsize=12)
    return fig


def plot_examples(result, cache_dir):
    """Predeclared first evaluation context; no cherry-picking by measured outcome."""
    from benchmarks.multiline_data import generate_multiline

    manifest = result["protocol"]["manifest"]
    families = ("crossing", "mixed_width", "weak_near_strong")
    chosen = [next(spec for spec in manifest if spec["split"] == "evaluation"
                   and spec["angle_deg"] == 0 and spec["seed"] == 2001
                   and spec["family"] == family
                   and (family != "weak_near_strong" or spec["gap"] == 6))
              for family in families]
    methods = ("local_contrast", "frangi", "ppi_raw_default", "ppi_local_relaxed")
    fig, axes = plt.subplots(3, 5, figsize=(13, 8), layout="constrained")
    for row, spec in enumerate(chosen):
        case = generate_multiline(spec)
        for ax in axes[row]:
            ax.imshow(case.image, cmap="gray", vmin=100, vmax=220)
            ax.set_xlim(22.5, 104.5)
            ax.set_ylim(104.5, 22.5)
            ax.set_xticks([])
            ax.set_yticks([])
        for line in case.centerlines:
            axes[row, 0].plot(line[:, 1], line[:, 0], color="#E64B35", lw=0.8)
        axes[row, 0].set_ylabel(FAMILY_LABELS[spec["family"]], fontsize=10)
        with np.load(Path(cache_dir) / f"{spec['case_id']}.npz", allow_pickle=False) as arrays:
            for column, method in enumerate(methods, 1):
                threshold = next(t["threshold"] for t in result["thresholds"][method]
                                 if t["target_rate"] == 0.005)
                prediction = case.roi & (arrays[method] > threshold)
                overlay = np.zeros((*prediction.shape, 4))
                overlay[prediction] = (0, 0.85, 1, 0.85)
                axes[row, column].imshow(overlay)
        if row == 0:
            axes[row, 0].set_title("All reference lines", fontsize=10)
            for column, method in enumerate(methods, 1):
                axes[row, column].set_title(LABELS[method], fontsize=10)
    fig.suptitle("Fixed examples: first evaluation orientation/seed, noise σ=5\n"
                 "Red: complete centerlines · Cyan: selected pixels at the primary frozen threshold",
                 fontsize=12)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("docs/benchmarks/multiline/results.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("docs/benchmarks/multiline/assets"))
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    builders = {"tradeoff": plot_tradeoff, "families": plot_families,
                "weak_neighbours": plot_weak_neighbours, "controls": plot_controls}
    for name, builder in builders.items():
        figure = builder(result)
        figure.savefig(args.output_dir / f"{name}.png", dpi=160)
        plt.close(figure)
    if args.cache_dir:
        figure = plot_examples(result, args.cache_dir)
        figure.savefig(args.output_dir / "examples.png", dpi=160)
        plt.close(figure)
    print(json.dumps(primary_summary(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
