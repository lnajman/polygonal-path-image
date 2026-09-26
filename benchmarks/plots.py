"""Portable Matplotlib figures for the synthetic benchmark and its tutorial.

These helpers consume the benchmark's saved rows; they do not rerun PPI or
silently drop missing results. Callers own saving and closing the figures.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter

COLORS = {"ppi": "#12628A", "darkness": "#C66A20", "truth": "#E5B64C"}
METHOD_LABELS = {"ppi": "PPI votes", "darkness": "Raw darkness"}
METHOD_STYLES = {"ppi": ("o", "-"), "darkness": ("s", "--")}


def set_style() -> None:
    """Apply a restrained, legible style shared by notebook and static figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#26333A",
            "axes.labelcolor": "#26333A",
            "axes.edgecolor": "#7D888E",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#DDE2E4",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "savefig.facecolor": "white",
            "savefig.dpi": 160,
        }
    )


def _rows(results, block, budget_fraction):
    return [
        row
        for row in results["rows"]
        if row["block"] == block
        and np.isclose(row["budget_fraction"], budget_fraction, rtol=0, atol=1e-12)
    ]


def _value(row, metric):
    value = row[metric]
    if value is None or not np.isfinite(value):
        raise ValueError(f"Missing/nonfinite {metric} in case {row['case_id']}")
    return float(value)


def _rate_axis(ax, upper=1.03):
    ax.set_ylim(0, upper)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def plot_case(case, votes, ppi_mask, darkness_mask):
    """Show the generated image, ground truth, score, and both selections."""
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 4), layout="constrained")
    axes[0].imshow(case.image, cmap="gray", vmin=0, vmax=255)
    axes[0].plot(
        case.centerline[:, 1],
        case.centerline[:, 0],
        color=COLORS["truth"],
        lw=1,
        label="Known centerline",
    )
    axes[0].contour(case.roi, levels=[0.5], colors=["white"], linewidths=1, linestyles="dashed")
    axes[0].set_title("Input and evaluation region")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    artist = axes[1].imshow(votes, cmap="cividis", interpolation="nearest")
    axes[1].contour(case.roi, levels=[0.5], colors=["white"], linewidths=1, linestyles="dashed")
    axes[1].set_title("Votes after tortuosity filter")
    fig.colorbar(artist, ax=axes[1], shrink=0.7, label="Path visits")
    for ax, mask, method in zip(axes[2:], [ppi_mask, darkness_mask], ["ppi", "darkness"]):
        ax.imshow(case.image, cmap="gray", vmin=0, vmax=255, alpha=0.35)
        ax.plot(case.centerline[:, 1], case.centerline[:, 0], color=COLORS["truth"], lw=2)
        selected = np.argwhere(mask)
        ax.scatter(
            selected[:, 1], selected[:, 0], color=COLORS[method], s=4, marker="s", linewidths=0
        )
        fraction = mask.sum() / case.roi.sum()
        ax.set_title(f"{METHOD_LABELS[method]} selection\n{fraction:.2%} of evaluation region")
        ax.contour(case.roi, levels=[0.5], colors=["#73818A"], linewidths=1, linestyles="dashed")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle("One synthetic curve: compare scores at the same requested pixel budget")
    return fig


def _metrics():
    return [
        ("centerline_coverage", "Centerline coverage (↑ better)"),
        ("off_tube_false_positive_rate", "Off-tube false-positive rate (↓ better)"),
    ]


def _metric_upper(rows, metric):
    if metric == "centerline_coverage":
        return 1.03
    return max(0.005, max(_value(row, metric) for row in rows) * 1.1)


def plot_direction(results, *, budget_fraction=0.02):
    """Separate coverage and background errors with one column per bend."""
    rows = _rows(results, "direction", budget_fraction)
    bends = sorted({row["bend_deg"] for row in rows})
    if not bends:
        raise ValueError("No direction rows for the requested budget")
    fig, axes = plt.subplots(
        2,
        len(bends),
        figsize=(12, 6.5),
        squeeze=False,
        layout="constrained",
        sharey="row",
        sharex=True,
    )
    for metric_index, (metric, label) in enumerate(_metrics()):
        for ax, bend in zip(axes[metric_index], bends):
            for method in METHOD_LABELS:
                selected = sorted(
                    (row for row in rows if row["bend_deg"] == bend and row["method"] == method),
                    key=lambda row: row["angle_deg"],
                )
                marker, linestyle = METHOD_STYLES[method]
                ax.plot(
                    [row["angle_deg"] for row in selected],
                    [_value(row, metric) for row in selected],
                    color=COLORS[method],
                    marker=marker,
                    linestyle=linestyle,
                    markersize=4,
                    label=METHOD_LABELS[method],
                )
            if metric_index == 0:
                ax.set_title(f"Total bend: {bend:g}°")
            else:
                ax.set_xlabel("Midpoint tangent angle (degrees)")
            ax.set_xticks([0, 45, 90, 135, 165])
            _rate_axis(ax, _metric_upper(rows, metric))
        axes[metric_index, 0].set_ylabel(label)
    axes[0, -1].legend(loc="lower left")
    fig.suptitle(f"Direction sensitivity · clean curves · requested budget {budget_fraction:.1%}")
    return fig


def plot_robustness(results, *, budget_fraction=0.02):
    """Display means and observed ranges across the angle/seed scenarios."""
    rows = _rows(results, "robustness", budget_fraction)
    contrasts = sorted({row["contrast"] for row in rows})
    if not contrasts:
        raise ValueError("No robustness rows for the requested budget")
    fig, axes = plt.subplots(
        2,
        len(contrasts),
        figsize=(12, 7),
        squeeze=False,
        layout="constrained",
        sharey="row",
        sharex=True,
    )
    for metric_index, (metric, label) in enumerate(_metrics()):
        for ax, contrast in zip(axes[metric_index], contrasts):
            for method in METHOD_LABELS:
                grouped = defaultdict(list)
                for row in rows:
                    if row["contrast"] == contrast and row["method"] == method:
                        grouped[row["noise_std"]].append(_value(row, metric))
                noises = sorted(grouped)
                means = [np.mean(grouped[noise]) for noise in noises]
                low = [np.min(grouped[noise]) for noise in noises]
                high = [np.max(grouped[noise]) for noise in noises]
                marker, linestyle = METHOD_STYLES[method]
                ax.fill_between(noises, low, high, color=COLORS[method], alpha=0.11)
                ax.plot(
                    noises,
                    means,
                    color=COLORS[method],
                    marker=marker,
                    linestyle=linestyle,
                    label=METHOD_LABELS[method],
                )
            if metric_index == 0:
                ax.set_title(f"Contrast: {contrast:g} gray levels")
            else:
                ax.set_xlabel("Noise standard deviation (gray levels)")
            ax.set_xticks(noises)
            _rate_axis(ax, _metric_upper(rows, metric))
        axes[metric_index, 0].set_ylabel(label)
    axes[0, -1].legend(loc="lower left")
    fig.suptitle(
        f"Noise and contrast · requested budget {budget_fraction:.1%}\n"
        "Lines: scenario means; bands: observed minimum–maximum, not confidence intervals"
    )
    return fig


def plot_parameters(results, *, budget_fraction=0.02):
    """Retain each scene in parameter matrices, without combining unlike rates."""
    rows = [row for row in _rows(results, "parameters", budget_fraction) if row["method"] == "ppi"]
    configs = list(dict.fromkeys(row["config_id"] for row in rows))
    scenes = sorted({(row["angle_deg"], row["bend_deg"]) for row in rows})
    if not configs or not scenes:
        raise ValueError("No parameter rows for the requested budget")
    labels = []
    for config in configs:
        row = next(row for row in rows if row["config_id"] == config)
        threshold = row["tortuosity_threshold"]
        tau = "off" if threshold is None else f"{threshold:g}"
        labels.append(f"L={row['segment_length']}\nK={row['nb_segments']}\nτ={tau}")
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), layout="constrained")
    for ax, (metric, metric_label) in zip(axes, _metrics()):
        grid = np.full((len(scenes), len(configs)), np.nan)
        for row in rows:
            key = (row["angle_deg"], row["bend_deg"])
            position = (scenes.index(key), configs.index(row["config_id"]))
            if np.isfinite(grid[position]):
                raise ValueError(f"Duplicate parameter cell for {key}, {row['config_id']}")
            grid[position] = _value(row, metric)
        if not np.all(np.isfinite(grid)):
            raise ValueError("Parameter matrix is incomplete")
        upper = 1 if metric == "centerline_coverage" else max(grid.max(), 0.005)
        artist = ax.imshow(grid, cmap="cividis", norm=Normalize(0, upper), aspect="auto")
        ax.set_xticks(np.arange(len(configs)), labels)
        ax.set_yticks(np.arange(len(scenes)), [f"{angle:g}° / {bend:g}°" for angle, bend in scenes])
        ax.set_ylabel("Midpoint angle / total bend")
        ax.set_title(metric_label)
        for (i, j), value in np.ndenumerate(grid):
            text = f"{value:.0%}" if metric == "centerline_coverage" else f"{value:.1%}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value / upper < 0.5 else "#152A35",
                fontsize=9,
            )
        colorbar = fig.colorbar(artist, ax=ax, shrink=0.85)
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[-1].set_xlabel("Segment length L; number of segments K; tortuosity threshold τ")
    fig.suptitle(
        f"Parameter sensitivity · contrast 60, noise σ=15 · requested budget {budget_fraction:.1%}"
    )
    return fig


def plot_runtime(results, *, budget_fraction=0.02):
    """Show mean measured stages for each configuration, with each case once."""
    rows = [row for row in _rows(results, "parameters", budget_fraction) if row["method"] == "ppi"]
    configs = list(dict.fromkeys(row["config_id"] for row in rows))
    if not configs:
        raise ValueError("No parameter timing rows for the requested budget")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    left = np.zeros(len(configs))
    stages = [
        ("core_seconds", "Path computation", "#12628A"),
        ("filter_seconds", "Tortuosity filter", "#B9CDD8"),
        ("voting_seconds", "Voting", "#C66A20"),
    ]
    for key, label, color in stages:
        values = [
            np.mean([_value(row, key) for row in rows if row["config_id"] == config])
            for config in configs
        ]
        ax.barh(configs, values, left=left, color=color, label=label, height=0.7)
        left += values
    ax.invert_yaxis()
    ax.set_xlabel("Mean elapsed seconds per image (one observation per scene/configuration)")
    ax.set_ylabel("PPI configuration")
    ax.set_xlim(left=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.suptitle("Where execution time goes · six parameter scenes per configuration")
    return fig


def plot_budget_curves(results, *, angle_deg=30, bend_deg=45, contrast=60, noise_std=15, seed=2012):
    """Show the four budgets for one declared scene against actual selected area."""
    scene = dict(
        angle_deg=angle_deg, bend_deg=bend_deg, contrast=contrast, noise_std=noise_std, seed=seed
    )
    rows = [
        row
        for row in results["rows"]
        if row["block"] == "robustness" and all(row[key] == value for key, value in scene.items())
    ]
    if not rows:
        raise ValueError("No robustness rows match the requested budget-curve scene")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), layout="constrained", sharex=True)
    for ax, (metric, label) in zip(axes, _metrics()):
        for method in METHOD_LABELS:
            selected = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: row["budget_fraction"],
            )
            if not selected or len({row["budget_fraction"] for row in selected}) != len(selected):
                raise ValueError(f"Missing or duplicate budget rows for {method}")
            marker, linestyle = METHOD_STYLES[method]
            ax.plot(
                [_value(row, "selected_fraction") for row in selected],
                [_value(row, metric) for row in selected],
                color=COLORS[method],
                marker=marker,
                linestyle=linestyle,
                label=METHOD_LABELS[method],
            )
            if metric == "centerline_coverage":
                for row in selected:
                    ax.annotate(
                        f"{row['budget_fraction']:.1%}",
                        (_value(row, "selected_fraction"), _value(row, metric)),
                        xytext=(0, 8 if method == "ppi" else -15),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color=COLORS[method],
                    )
        ax.set_xlabel("Actual selected ROI area")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel(label)
        _rate_axis(ax, 1.1 if metric == "centerline_coverage" else _metric_upper(rows, metric))
        ax.set_xlim(left=0)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axes[-1].legend(loc="upper left")
    fig.suptitle(
        f"Area-budget tradeoff · one scene: angle {angle_deg:g}°, bend {bend_deg:g}°, "
        f"contrast {contrast:g}, noise σ={noise_std:g}\n"
        f"Seed {seed}; labels are requested budgets, coordinates use actual selected area"
    )
    return fig
