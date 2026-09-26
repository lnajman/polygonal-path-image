"""Inspectable figures for source-path, filtering and blank-image diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = (("weak_alone", None), ("weak_near_strong", 6), ("weak_near_strong", 10))
SCENARIO_LABELS = ("Weak alone", "Strong, gap 6", "Strong, gap 10")
STAGES = {"unfiltered": ("Before filtering", "#666666"),
          "tau_0.75": ("Tortuosity ≥ 0.75", "#0072B2"),
          "tau_0.5": ("Tortuosity ≥ 0.50", "#D55E00")}


def _group(weak, potential, family, gap):
    return [r for r in weak["results"] if r["potential"] == potential
            and r["family"] == family and r["gap"] == gap]


def plot_weak_stages(weak):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey="row", layout="constrained")
    for column, potential in enumerate(("raw", "local")):
        groups = [_group(weak, potential, family, gap) for family, gap in SCENARIOS]
        for stage, (label, color) in STAGES.items():
            values = [np.mean([r["stages"][stage]["weak_band_score_distribution"]["mean"]
                               for r in group]) for group in groups]
            axes[0, column].plot(range(3), values, "o-", label=label, color=color)
        for key, label, color in (
            ("positive", "Any positive vote within 2 pixels", "#009E73"),
            ("threshold", "Votes > 56 within 2 pixels", "#0072B2"),
        ):
            values = [100 * np.mean([
                r["stages"]["tau_0.75"]["positive_vote_weak_coverage"] if key == "positive"
                else r["stages"]["tau_0.75"]["threshold_coverage"]["56"]["weak_coverage"]
                for r in group]) for group in groups]
            axes[1, column].plot(range(3), values, "o-", label=label, color=color)
        axes[0, column].set(title=f"{potential.capitalize()} potential", yscale="log",
                            ylabel="Mean votes per weak-band pixel (log scale)")
        axes[1, column].set(ylabel="Weak-line coverage (%)", ylim=(-3, 103))
    for ax in axes.flat:
        ax.set_xticks(range(3), SCENARIO_LABELS)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.17))
    count = len(weak["protocol"]["contexts"])
    fig.suptitle("Weak-line evidence changes before and after filtering\n"
                 f"Equal means over {count} existing contexts; weak band ≤ 2 pixels; lower panels τ = 0.75",
                 fontsize=11)
    return fig


def plot_source_paths(weak):
    from benchmarks.multiline_data import generate_multiline, multiline_specs

    specs = {s["case_id"]: s for s in multiline_specs()}
    first_group = weak["protocol"]["contexts"][0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5), layout="constrained")
    for ax, (family, gap), title in zip(axes, SCENARIOS, SCENARIO_LABELS, strict=True):
        record = next(r for r in weak["results"] if r["potential"] == "raw"
                      and r["pair_group"] == first_group and r["family"] == family
                      and r["gap"] == gap)
        case = generate_multiline(specs[record["case_id"]])
        source = record["source_examples"][1]
        origin = np.asarray(source["origin"])
        vertices = np.vstack((origin, source["endpoints"]))
        ax.imshow(case.image, cmap="gray", vmin=100, vmax=220)
        ax.plot(vertices[:, 1], vertices[:, 0], "o-", color="#00BFC4", lw=2, ms=3)
        ax.scatter(origin[1], origin[0], marker="*", s=130, color="#E69F00", zorder=5)
        ax.set(xlim=(24, 104), ylim=(104, 24), xticks=[], yticks=[],
               title=f"{title}\nCost {source['cost']:.0f}; τ = {source['tortuosity']:.3f}")
        status = "kept" if source["retained"]["tau_0.75"] else "rejected"
        ax.set_xlabel(f"{source['weak_band_pixel_occurrences']}/31 pixels near weak line; {status}")
    fig.suptitle("Same fixed midpoint source, raw input · first diagnostic context\n"
                 "Cyan: selected minimum-cost path before filtering; star: source", fontsize=12)
    return fig


def plot_blank_summary(blank):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    cases = [c for c in blank["cases"] if c["seed"] is None and c["intensity"] == 200]
    for case in cases:
        mapping = next(m for m in case["vote_maps"] if m["tortuosity_threshold"] == .75)
        histogram = mapping["center_roi"]["value_histogram"]
        axes[0].plot([p["votes"] for p in histogram], [p["pixels"] for p in histogram],
                     "o-", label=f"{case['size']} × {case['size']} canvas", ms=4)
    axes[0].axvline(56, color="#666666", ls="--", lw=1, label="Historical cutoff 56")
    axes[0].set(xlabel="Votes at an ROI pixel", ylabel="ROI pixel count", yscale="log")
    axes[0].legend(fontsize=8)
    variants = [next(c for c in blank["cases"] if c["case_id"] == name) for name in
                ("uniform_n128_c200", "perturbed_n128_c200_seed3001", "perturbed_n128_c200_seed3002")]
    for offset, threshold, color in ((-.16, .75, "#0072B2"), (.16, .5, "#D55E00")):
        fractions = [100 * next(m for m in c["vote_maps"]
                                if m["tortuosity_threshold"] == threshold)["retained_source_fraction"]
                     for c in variants]
        axes[1].bar(np.arange(3) + offset, fractions, width=.3, label=f"τ ≥ {threshold}", color=color)
    axes[1].set(xticks=range(3), xticklabels=("Uniform", "±1, seed 3001", "±1, seed 3002"),
               ylabel="Retained source paths (%)", ylim=(0, 105))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Blank votes are expected; their concentration depends on boundaries and ties\n"
                 "Left: fixed centered 80 × 80 ROI, τ ≥ 0.75; right: actual integer perturbations",
                 fontsize=11)
    return fig


def plot_blank_maps(blank, cache_dir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for ax, size in zip(axes, (128, 192, 256), strict=True):
        filename = Path(cache_dir) / f"uniform_n{size}_c200.npz"
        with np.load(filename, allow_pickle=False) as saved:
            votes = saved["votes_tau075"]
        margin = (size - 80) // 2
        roi = votes[margin:margin + 80, margin:margin + 80]
        shown = ax.imshow(roi, cmap="viridis", vmin=0, vmax=61)
        ax.set(xticks=[], yticks=[], title=f"{size} × {size} canvas\nROI range {roi.min()}–{roi.max()}")
    fig.colorbar(shown, ax=axes, label="Votes (common scale)", shrink=.75)
    fig.suptitle("Identical constant intensity 200 · same centered 80 × 80 ROI · τ ≥ 0.75",
                 fontsize=12)
    return fig


def plot_fallback(result):
    from benchmarks.plot_multiline import COLORS, LABELS, mean_metric, select_rows, weak_pairs

    labels = {**LABELS, "ppi_raw_straight_fallback": "Raw PPI + straight fallback"}
    colors = {**COLORS, "ppi_raw_straight_fallback": "#6F4C9B"}
    rates = (.001, .005, .01, .02)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    for method, label in labels.items():
        groups = [select_rows(result, method=method, point=rate) for rate in rates]
        x = [100 * mean_metric(group, "outside_support_fraction") for group in groups]
        y = [100 * mean_metric(group, "macro_coverage") for group in groups]
        axes[0].plot(x, y, "o-", color=colors[method], label=label, ms=4)
        axes[0].scatter(x[1], y[1], s=90, facecolors="none", edgecolors=colors[method])
        pairs = weak_pairs(result, method=method)
        alone = {p["pair_group"]: p["alone_coverage"] for p in pairs}
        values = [100 * np.mean(list(alone.values()))]
        values.extend(100 * np.mean([p["paired_coverage"] for p in pairs if p["gap"] == gap])
                      for gap in (6, 10))
        axes[1].plot(range(3), values, "o-", color=colors[method], label=label, ms=4)
    axes[0].set(xlabel="Evaluation selection outside all supports (%)",
                ylabel="Mean per-line coverage (%)", ylim=(-2, 102),
                title="Four frozen development thresholds")
    axes[1].set(xticks=range(3), xticklabels=SCENARIO_LABELS,
                ylabel="Weak-line coverage (%)", ylim=(-2, 102),
                title="Paired weak-line recovery at primary thresholds")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.17), ncol=2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Fresh panel: 48 line scenes, six paired contexts · one unchanged experimental fallback\n"
                 "Open rings: 0.5% development background target; achieved evaluation rates differ",
                 fontsize=11)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("docs/benchmarks/diagnostics"))
    parser.add_argument("--blank-cache", type=Path)
    args = parser.parse_args()
    weak = json.loads((args.directory / "weak-paths.json").read_text())
    blank = json.loads((args.directory / "blank-paths.json").read_text())
    assets = args.directory / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name, figure in (("weak-stages", plot_weak_stages(weak)),
                         ("source-paths", plot_source_paths(weak)),
                         ("blank-summary", plot_blank_summary(blank))):
        figure.savefig(assets / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
    if args.blank_cache:
        figure = plot_blank_maps(blank, args.blank_cache)
        figure.savefig(assets / "blank-maps.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
    validation = args.directory / "fallback-validation.json"
    if validation.exists():
        figure = plot_fallback(json.loads(validation.read_text()))
        figure.savefig(assets / "fallback-validation.png", dpi=160, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
