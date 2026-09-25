#!/usr/bin/env python3
"""Plot a successful numerical comparison from ``validate_research.py``.

Requires the optional ``examples`` dependencies and the local array directory
written by the validation command, including historical comparison results::

    python scripts/plot_validation.py work/report.json --output work/comparison.png

Neither the report nor its input arrays are modified. A report containing only
the synthetic case is supported when it includes a historical comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from polygonal_path_image import voting


def load_results(report_path):
    """Check report status and artifacts before making equality claims in a plot."""
    report = json.loads(report_path.read_text())
    if not report.get("cases") or report.get("correctness_checks_passed") is not True:
        raise ValueError("the report must contain successful validation cases")
    folder = report_path.parent / report["array_directory"]
    cases = []
    for case in report["cases"]:
        if (
            case.get("correctness_checks_passed") is not True
            or "legacy" not in case
            or case.get("comparison", {}).get("cost_arrays_exactly_equal") is not True
        ):
            raise ValueError(f"{case['name']}: a successful historical cost comparison is required")
        image = np.load(folder / case["input_artifact"], allow_pickle=False)
        if hashlib.sha256(image.tobytes()).hexdigest() != case["pixel_sha256"]:
            raise ValueError(f"{case['name']}: input pixels do not match the report hash")
        with (
            np.load(folder / case["modern"]["artifact"], allow_pickle=False) as modern,
            np.load(folder / case["legacy"]["artifact"], allow_pickle=False) as legacy,
        ):
            cost = modern["costs"]
            errors = legacy["path_cost_mismatch"]
            if not np.array_equal(cost, legacy["costs"]):
                raise ValueError(f"{case['name']}: artifact costs disagree with the report")
            if (
                cost.shape != image.shape
                or errors.shape != image.shape
                or modern["path_cost_mismatch"].any()
                or np.count_nonzero(errors)
                != case["legacy"]["path_audit"]["inconsistent_path_costs"]
            ):
                raise ValueError(f"{case['name']}: artifact path checks disagree with the report")
            votes = voting(modern["filtered_costs"], modern["paths"])[0]
        cases.append((case, image, cost, votes, errors))
    return report, cases


def plot_results(report, cases, output):
    # Keep matplotlib optional and let --help work without it installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(
        len(cases), 4, figsize=(17, 3 * len(cases) + 1.2), layout="constrained", squeeze=False
    )
    parameters = report["parameters"]
    segment_length = parameters["segment_length"]
    nb_segments = parameters["nb_segments"]
    samples = segment_length * nb_segments
    largest_vote = max(1, max(row[3].max() for row in cases))
    labels = [
        "Input intensity (uint8)",
        f"Sum of {samples} sampled pixel intensities",
        "Number of path visits",
        "Recomputed path cost differs from reported cost",
    ]
    for row, (case, image, cost, votes, errors) in enumerate(cases):
        arrays = [image, np.ma.masked_invalid(cost), votes, errors]
        colormaps = ["gray", "gray", "magma", ListedColormap(["#f3f3f3", "#ad531d"])]
        maxima = [255, 255 * samples, largest_vote, 1]
        count = case["legacy"]["path_audit"]["inconsistent_path_costs"]
        finite = case["legacy"]["path_audit"]["finite_paths"]
        error_label = f"{count:,} / {finite:,} finite paths"
        if finite:
            error_label += f" ({count / finite:.1%})"
        titles = [
            case["name"] + f"\n{image.shape[0]} × {image.shape[1]} pixels",
            "PPI cost: modern = historical\nexactly equal at every source pixel",
            f"Modern votes after tortuosity ≥ {parameters['tortuosity_threshold']}\n"
            "linear vote count; no border clearing",
            "Historical inconsistent path costs\n" + error_label,
        ]
        for column, (array, cmap, maximum, title) in enumerate(
            zip(arrays, colormaps, maxima, titles)
        ):
            ax = axes[row, column]
            plotted = ax.imshow(array, cmap=cmap, vmin=0, vmax=maximum, interpolation="nearest")
            ax.set_title(title, fontsize=10, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == len(cases) - 1:
                bar = figure.colorbar(
                    plotted,
                    ax=axes[:, column],
                    orientation="horizontal",
                    fraction=0.04,
                    pad=0.035,
                    shrink=0.85,
                )
                bar.ax.tick_params(labelsize=9)
                bar.set_label(labels[column], fontsize=9)
                if column == 3:
                    bar.set_ticks([0, 1], labels=["No", "Yes"])
    figure.suptitle("Polygonal Path Image · numerical validation", fontsize=17)
    figure.supxlabel(
        f"Segment length {segment_length}, {nb_segments} segments. "
        "Shared scales within each column; infinite costs are blank. "
        "Modern paths: zero cost inconsistencies in every case.\n"
        "Full-resolution inputs; no preprocessing. These checks establish numerical compatibility, "
        "not reproduction of clinical results from MICCAI 2012.",
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, facecolor="white")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report, cases = load_results(args.report)
        plot_results(report, cases, args.output)
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.error(str(error))
    print(args.output)


if __name__ == "__main__":
    main()
