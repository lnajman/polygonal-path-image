"""Plot the committed Guide3D pilot's scalar results without downloading images.

Run from a checkout with NumPy and Matplotlib installed::

    python -m benchmarks.plot_guide3d --input docs/benchmarks/guide3d/results.json

Each acquisition receives equal weight. Within an acquisition, each selected
2D view receives equal weight. A displayed range is the observed range of
acquisition means, never a confidence interval or a pooled-frame percentile.
Only the guidewire is annotated. Selections away from it are unclassified for
general line enhancement, not established false detections of other lines.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = {
    "darkness": ("Raw darkness", "#656565", "o", ":"),
    "local_contrast": ("Local contrast", "#b07a12", "s", ":"),
    "ppi_raw_default": ("PPI · raw default", "#32669b", "^", "--"),
    "ppi_local_default": ("PPI · local default", "#965695", "D", "-"),
    "ppi_local_relaxed": ("PPI · local relaxed", "#d05d29", "v", "-"),
    "ppi_local_short": ("PPI · local short", "#627539", "P", "-"),
}
PRIMARY_BUDGET = 0.01
NATIVE_SCALE = 2.0


def set_style():
    """Use one readable static style in the notebook and exported figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.facecolor": "white",
        }
    )


def load_results(path):
    """Read and check the row grain and the development/evaluation separation."""
    report = json.loads(Path(path).read_text())
    protocol = report["protocol"]
    if not np.isclose(protocol["primary_budget"], PRIMARY_BUDGET):
        raise ValueError("The figure labels require the declared 1% primary area budget")
    if not np.isclose(protocol["native_pixels_per_working_pixel"], NATIVE_SCALE):
        raise ValueError("The localization labels require two native pixels per working pixel")
    if not np.isclose(protocol["tolerance_working_pixels"], 2):
        raise ValueError("The coverage labels require a two-working-pixel tolerance")
    manifest = {entry["case_id"]: entry for entry in protocol["manifest"]}
    if len(manifest) != len(protocol["manifest"]):
        raise ValueError("duplicate cases in the frozen manifest")
    split_acquisitions = {}
    for split in ("development", "evaluation"):
        rows = report[split]["rows"]
        if not rows:
            raise ValueError(f"{split} has no measurements")
        grain = [(row["case_id"], row["method"], row["budget_fraction"]) for row in rows]
        if len(grain) != len(set(grain)):
            raise ValueError(f"duplicate case/method/budget rows in {split}")
        expected_cases = {case for case, entry in manifest.items() if entry["split"] == split}
        if {row["case_id"] for row in rows} != expected_cases:
            raise ValueError(f"measurement cases disagree with the frozen {split} manifest")
        if {row["budget_fraction"] for row in rows} != set(protocol["budgets"]):
            raise ValueError(f"area budgets disagree with the frozen {split} protocol")
        for row in rows:
            if any(
                row[key] != manifest[row["case_id"]][key]
                for key in ("acquisition_id", "camera", "frame_number")
            ):
                raise ValueError("measurement grouping disagrees with the frozen manifest")
        split_acquisitions[split] = {row["acquisition_id"] for row in rows}
        if set(row["method"] for row in rows) - METHODS.keys():
            raise ValueError("unknown method label; update the figure's method definitions")
        # A method cannot silently lose difficult views or area budgets.
        populations = defaultdict(set)
        for row in rows:
            populations[row["method"]].add((row["case_id"], row["budget_fraction"]))
        if len({frozenset(population) for population in populations.values()}) != 1:
            raise ValueError(f"methods have unequal case/budget populations in {split}")
    if split_acquisitions["development"] & split_acquisitions["evaluation"]:
        raise ValueError("an acquisition occurs in both development and evaluation")
    return report


def methods_in(rows):
    """Use stable method order and labels across all figures."""
    present = {row["method"] for row in rows}
    return [method for method in METHODS if method in present]


def method_label(method):
    return METHODS[method][0]


def _value(row, metric):
    if metric == "actual_fraction":
        return row["selection"]["actual_fraction"]
    return row["metrics"][metric]


def acquisition_values(rows, method, metric, budget_fraction=PRIMARY_BUDGET):
    """Return acquisition means, retaining null groups as NaN.

    Null frame measurements do not become zeros. A partially observed
    acquisition is also NaN, avoiding a silent change in its frame population.
    Callers can inspect ``valid_acquisitions`` in :func:`macro_summary`.
    """
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] == method and np.isclose(row["budget_fraction"], budget_fraction):
            grouped[row["acquisition_id"]].append(_value(row, metric))
    return {
        name: (
            float(np.mean(values))
            if values and all(value is not None and np.isfinite(value) for value in values)
            else np.nan
        )
        for name, values in sorted(grouped.items())
    }


def macro_summary(rows, method, metric, budget_fraction=PRIMARY_BUDGET):
    """Mean and observed range of acquisition means, with availability counts."""
    groups = acquisition_values(rows, method, metric, budget_fraction)
    values = np.asarray(list(groups.values()), dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(finite)) if len(finite) else np.nan,
        "minimum": float(np.min(finite)) if len(finite) else np.nan,
        "maximum": float(np.max(finite)) if len(finite) else np.nan,
        "valid_acquisitions": len(finite),
        "total_acquisitions": len(values),
    }


def _context(rows):
    acquisitions = len({row["acquisition_id"] for row in rows})
    views = len({row["case_id"] for row in rows})
    return f"{acquisitions} held-out acquisitions · {views} 2D views · equal acquisition weights"


def _decorate(fig, title, rows, *, note):
    fig.suptitle(title, x=0.02, ha="left", fontsize=16, fontweight="bold", y=0.985)
    fig.text(0.02, 0.925, _context(rows), ha="left", color="#555555", fontsize=10)
    fig.text(0.02, 0.025, note, ha="left", color="#555555", fontsize=9)


def plot_budget_curves(report):
    """Compare guidewire coverage and off-guidewire selection against area."""
    rows = report["evaluation"]["rows"]
    methods = methods_in(rows)
    budgets = sorted({row["budget_fraction"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3))
    for method in methods:
        label, color, marker, linestyle = METHODS[method]
        x = [
            100 * macro_summary(rows, method, "actual_fraction", budget)["mean"]
            for budget in budgets
        ]
        for ax, metric in zip(
            axes, ("centerline_coverage", "off_tube_false_positive_rate"), strict=True
        ):
            y = [100 * macro_summary(rows, method, metric, budget)["mean"] for budget in budgets]
            ax.plot(
                x, y, label=label, color=color, marker=marker, linestyle=linestyle, linewidth=1.8
            )
            primary = next(
                i for i, budget in enumerate(budgets) if np.isclose(budget, PRIMARY_BUDGET)
            )
            ax.plot(
                x[primary],
                y[primary],
                marker=marker,
                markersize=10,
                markerfacecolor="none",
                markeredgecolor=color,
            )
    axes[0].set_ylabel("Annotated guidewire covered (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_ylabel("Off-guidewire ROI pixels selected (%)")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("Actual selected ROI area (%)")
        ax.set_xlim(left=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    _decorate(
        fig,
        "Guide3D: guidewire coverage and off-guidewire selection",
        rows,
        note="Points: nominal area budgets 0.25%, 0.5%, 1%, 2%; open rings: 1%. Complete cutoff ties are included.\nCoverage uses a 4-native-pixel tolerance. Lines connect measured operating points; they are not fitted curves.\nOnly the guidewire is annotated; selections elsewhere are unclassified for general line enhancement.",
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.74, wspace=0.30)
    return fig


def plot_acquisitions(report, budget_fraction=PRIMARY_BUDGET):
    """Expose acquisition variation instead of treating frames as independent."""
    rows = report["evaluation"]["rows"]
    methods = methods_in(rows)
    acquisitions = sorted({row["acquisition_id"] for row in rows})
    values = np.array(
        [
            [
                100 * acquisition_values(rows, method, "centerline_coverage", budget_fraction)[acq]
                for method in methods
            ]
            for acq in acquisitions
        ]
    )
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    palette = plt.colormaps["cividis"].copy()
    palette.set_bad("#eeeeee")
    graphic = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=100, cmap=palette, aspect="auto")
    ax.set_yticks(np.arange(len(acquisitions)), acquisitions)
    ax.set_xticks(
        np.arange(len(methods)), [method_label(method).replace(" · ", "\n") for method in methods]
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(acquisitions)):
        for j in range(len(methods)):
            value = values[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}" if np.isfinite(value) else "—",
                ha="center",
                va="center",
                color="white" if value < 45 else "#171717",
                fontsize=11,
            )
    colorbar = fig.colorbar(graphic, ax=ax, fraction=0.035, pad=0.035)
    colorbar.set_label("Annotated guidewire covered (%)")
    _decorate(
        fig,
        f"Guide3D: guidewire coverage by acquisition at {100 * budget_fraction:g}% nominal area",
        rows,
        note="Each cell averages 20 selected 2D views (10 paired frames). Tolerance: 4 native pixels.\nAcquisition IDs retain the source's fluid/wire labels. The fluid/straight condition has one held-out acquisition.",
    )
    fig.subplots_adjust(left=0.23, right=0.91, bottom=0.13, top=0.79)
    return fig


def _summary_plot(report, specifications, title, note, *, budget_fraction=PRIMARY_BUDGET):
    rows = report["evaluation"]["rows"]
    methods = methods_in(rows)
    fig, axes = plt.subplots(1, len(specifications), figsize=(12, 5.8))
    positions = np.arange(len(methods))
    for ax, (metric, xlabel, factor, upper) in zip(axes, specifications, strict=True):
        for position, method in zip(positions, methods, strict=True):
            stats = macro_summary(rows, method, metric, budget_fraction)
            mean, low, high = factor * np.array([stats["mean"], stats["minimum"], stats["maximum"]])
            _, color, marker, _ = METHODS[method]
            if np.isfinite(mean):
                ax.plot([low, high], [position, position], color=color, linewidth=1.3)
                ax.plot(mean, position, marker=marker, color=color, markersize=8)
            else:
                ax.text(
                    0.01, position, "unavailable", transform=ax.get_yaxis_transform(), va="center"
                )
        ax.set_yticks(positions)
        ax.set_yticklabels([method_label(method) for method in methods])
        ax.invert_yaxis()
        ax.set_ylim(len(methods) - 0.5, -0.5)
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0, right=upper)
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    _decorate(fig, title, rows, note=note)
    fig.subplots_adjust(left=0.21, right=0.975, bottom=0.22, top=0.83, wspace=0.25)
    return fig


def plot_precision(report, budget_fraction=PRIMARY_BUDGET):
    """Show guidewire association; other lines have no labels in this study.

    The function name and stored metric keys are retained for compatibility.
    They do not make the measurements precision or false-positive rates for
    general line enhancement.
    """
    return _summary_plot(
        report,
        [
            ("selected_pixel_precision", "Selected pixels near guidewire (%)", 100, 100),
            ("off_tube_false_positive_rate", "Off-guidewire ROI pixels selected (%)", 100, None),
        ],
        f"Guide3D: selection relative to the guidewire at {100 * budget_fraction:g}% nominal area",
        "Points: mean of acquisition means. Lines: observed minimum–maximum acquisition means, not confidence intervals.\nLeft denominator: all selected pixels. Right: ROI pixels outside the 4-native-pixel guidewire tube.\nSelections outside this annotation are unclassified; these are not all-line precision or false-positive rates.",
        budget_fraction=budget_fraction,
    )


def plot_localization(report, budget_fraction=PRIMARY_BUDGET):
    """Show distances to the guidewire, not errors against all visible lines."""
    return _summary_plot(
        report,
        [
            (
                "centerline_to_prediction_mean_distance",
                "Guidewire → selected pixels (native px)",
                NATIVE_SCALE,
                None,
            ),
            (
                "prediction_to_centerline_mean_distance",
                "Selected pixels → guidewire (native px)",
                NATIVE_SCALE,
                None,
            ),
        ],
        f"Guide3D: mean distances to the guidewire at {100 * budget_fraction:g}% nominal area",
        "Points: mean of acquisition means. Lines: observed minimum–maximum acquisition means, not confidence intervals.\nNative 1024×1024 pixel units (2 × working-grid distances); no physical calibration. Means are not a pooled P95.\nOnly the guidewire is annotated; distance away from it does not establish a false response to a line.",
        budget_fraction=budget_fraction,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/benchmarks/guide3d/results.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/guide3d/assets"))
    args = parser.parse_args(argv)
    report = load_results(args.input)
    set_style()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, function in (
        ("budget_curves", plot_budget_curves),
        ("acquisitions", plot_acquisitions),
        ("selection_quality", plot_precision),
        ("localization", plot_localization),
    ):
        figure = function(report)
        destination = args.output / f"{name}.png"
        figure.savefig(destination)
        plt.close(figure)
        print(destination)


if __name__ == "__main__":
    main()
