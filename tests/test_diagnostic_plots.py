"""Check figure measurements and cohort selection rather than aesthetic details."""

from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from benchmarks import multiline_data  # noqa: E402
from benchmarks.plot_path_diagnostics import (  # noqa: E402
    SCENARIOS,
    STAGES,
    plot_blank_maps,
    plot_blank_summary,
    plot_fallback,
    plot_source_paths,
    plot_weak_stages,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def weak_fixture():
    specs = [
        spec
        for spec in multiline_data.multiline_specs()
        if spec["split"] == "evaluation" and spec["family"] in ("weak_alone", "weak_near_strong")
    ]
    contexts = list(dict.fromkeys(spec["pair_group"] for spec in specs))
    rows = []
    for spec in specs:
        context_index = contexts.index(spec["pair_group"])
        scenario_index = SCENARIOS.index((spec["family"], spec["gap"]))
        for potential_index, potential in enumerate(("raw", "local")):
            stages = {}
            for stage_index, stage in enumerate(STAGES):
                stages[stage] = {
                    "weak_band_score_distribution": {
                        "mean": 100 * (potential_index + 1)
                        + 10 * stage_index
                        + 3 * scenario_index
                        + context_index
                    },
                    "positive_vote_weak_coverage": 0.6
                    + 0.01 * context_index
                    + 0.02 * scenario_index,
                    "threshold_coverage": {
                        "56": {"weak_coverage": 0.1 + 0.01 * context_index + 0.03 * scenario_index}
                    },
                }
            example = {
                "origin": [59, 64],
                "endpoints": [[59, 64 - 3 * step] for step in range(1, 11)],
                "cost": 1000 + 100 * context_index + scenario_index,
                "tortuosity": 0.65,
                "retained": {"tau_0.75": False},
                "weak_band_pixel_occurrences": 31,
            }
            rows.append(
                {
                    "case_id": spec["case_id"],
                    "family": spec["family"],
                    "gap": spec["gap"],
                    "pair_group": spec["pair_group"],
                    "potential": potential,
                    "stages": stages,
                    "source_examples": [example, example, example],
                }
            )
    # A different storage order must not turn the first illustrative context
    # into the entire aggregate or change which fixed example is illustrated.
    return {"protocol": {"contexts": contexts}, "results": rows[::-1]}


def test_weak_stage_figures_use_all_nine_contexts_and_preserve_stage_units():
    data = weak_fixture()
    figure = plot_weak_stages(data)
    assert len(figure.axes) == 4
    assert "9 existing contexts" in figure._suptitle.get_text()
    for column, potential in enumerate(("raw", "local")):
        top = figure.axes[column]
        assert top.get_yscale() == "log"
        for stage_index, stage in enumerate(STAGES):
            expected = []
            for family, gap in SCENARIOS:
                values = [
                    row["stages"][stage]["weak_band_score_distribution"]["mean"]
                    for row in data["results"]
                    if row["potential"] == potential
                    and row["family"] == family
                    and row["gap"] == gap
                ]
                assert len(values) == 9
                expected.append(sum(values) / 9)
            np.testing.assert_allclose(top.lines[stage_index].get_ydata(), expected)
        bottom = figure.axes[2 + column]
        np.testing.assert_allclose(bottom.lines[0].get_ydata(), [64, 66, 68])
        np.testing.assert_allclose(bottom.lines[1].get_ydata(), [14, 17, 20])
        assert bottom.get_ylim() == (-3, 103)


def test_path_illustration_uses_first_declared_context_and_fixed_midpoint_only(monkeypatch):
    data = weak_fixture()
    generated = []

    def fake_generate(spec):
        generated.append(spec)
        return SimpleNamespace(image=np.full((128, 128), 200, dtype=np.uint8))

    monkeypatch.setattr(multiline_data, "generate_multiline", fake_generate)
    figure = plot_source_paths(data)
    assert len(generated) == 3
    assert {spec["pair_group"] for spec in generated} == {data["protocol"]["contexts"][0]}
    assert "first diagnostic context" in figure._suptitle.get_text()
    for index, ax in enumerate(figure.axes):
        assert f"Cost {1000 + index}" in ax.get_title()
        np.testing.assert_array_equal(
            ax.lines[0].get_xdata(), [64 - 3 * step for step in range(11)]
        )
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), np.full(11, 59))
        assert "31/31 pixels" in ax.get_xlabel()
        assert "rejected" in ax.get_xlabel()


def test_blank_summary_shows_pixel_histograms_and_source_retention_as_different_units():
    cases = []
    for size in (128, 192, 256):
        cases.append(
            {
                "case_id": f"uniform_n{size}_c200",
                "size": size,
                "intensity": 200,
                "seed": None,
                "vote_maps": [
                    {
                        "tortuosity_threshold": threshold,
                        "center_roi": {
                            "value_histogram": [
                                {"votes": 2, "pixels": 6000},
                                {"votes": 60, "pixels": 400},
                            ]
                        },
                        "retained_source_fraction": 0.7 if threshold == 0.75 else 0.9,
                    }
                    for threshold in (0.75, 0.5)
                ],
            }
        )
    for seed, fractions in ((3001, (0.1, 0.3)), (3002, (0.2, 0.4))):
        cases.append(
            {
                "case_id": f"perturbed_n128_c200_seed{seed}",
                "size": 128,
                "intensity": 200,
                "seed": seed,
                "vote_maps": [
                    {"tortuosity_threshold": threshold, "retained_source_fraction": fraction}
                    for threshold, fraction in zip((0.75, 0.5), fractions, strict=True)
                ],
            }
        )
    figure = plot_blank_summary({"cases": cases})
    histogram, retention = figure.axes
    for line in histogram.lines[:3]:
        np.testing.assert_array_equal(line.get_xdata(), [2, 60])
        np.testing.assert_array_equal(line.get_ydata(), [6000, 400])
    assert histogram.get_ylabel() == "ROI pixel count"
    assert retention.get_ylabel() == "Retained source paths (%)"
    np.testing.assert_allclose(
        [patch.get_height() for patch in retention.patches], [70, 10, 20, 90, 30, 40]
    )


def test_blank_maps_crop_same_center_roi_and_share_color_scale(tmp_path):
    for size, value in ((128, 4), (192, 7), (256, 10)):
        pixels = np.zeros((size, size), dtype=np.int64)
        margin = (size - 80) // 2
        pixels[margin : margin + 80, margin : margin + 80] = value
        np.savez(tmp_path / f"uniform_n{size}_c200.npz", votes_tau075=pixels)
    figure = plot_blank_maps({}, tmp_path)
    for ax, value in zip(figure.axes[:3], (4, 7, 10), strict=True):
        displayed = ax.images[0]
        assert displayed.get_array().shape == (80, 80)
        assert np.all(displayed.get_array() == value)
        assert displayed.get_clim() == (0, 61)


def test_fallback_plot_uses_achieved_error_rates_and_matched_weak_examples():
    methods = [
        "darkness",
        "local_contrast",
        "frangi",
        "ppi_raw_default",
        "ppi_local_default",
        "ppi_local_relaxed",
        "ppi_raw_straight_fallback",
    ]
    rows = []
    for method_index, method in enumerate(methods):
        for rate_index, rate in enumerate((0.001, 0.005, 0.01, 0.02)):
            for context in range(6):
                for scenario, (family, gap) in enumerate(SCENARIOS):
                    weak_coverage = (
                        (0.8, 0.6, 0.2)[scenario] + 0.01 * method_index + 0.005 * context
                    )
                    rows.append(
                        {
                            "split": "evaluation",
                            "selection": "frozen_threshold",
                            "operating_point": rate,
                            "method": method,
                            "family": family,
                            "gap": gap,
                            "pair_group": f"context{context}",
                            "metrics": {
                                "has_target": True,
                                "macro_coverage": 0.2
                                + 0.05 * rate_index
                                + 0.01 * method_index
                                + 0.01 * context
                                + 0.001 * scenario,
                                "outside_support_fraction": 0.002
                                + 0.001 * rate_index
                                + 0.0001 * method_index
                                + 0.0001 * context,
                                "per_line": [
                                    {
                                        "line_name": "weak",
                                        "coverage": weak_coverage,
                                        "longest_uncovered_arclength": 10 * (1 - weak_coverage),
                                    }
                                ],
                            },
                        }
                    )
    # Distinct development values must never influence the fresh result curves.
    rows.extend(
        {
            **row,
            "split": "development",
            "metrics": {
                **row["metrics"],
                "macro_coverage": 0,
                "outside_support_fraction": 1,
            },
        }
        for row in list(rows)
    )
    figure = plot_fallback({"rows": rows})
    tradeoff, paired = figure.axes
    assert len(tradeoff.lines) == len(paired.lines) == 7
    candidate = tradeoff.lines[-1]
    np.testing.assert_allclose(candidate.get_xdata(), [0.285, 0.385, 0.485, 0.585])
    np.testing.assert_allclose(candidate.get_ydata(), [28.6, 33.6, 38.6, 43.6])
    # The emphasized point is requested development0.5%, plotted at the
    # achieved evaluation rate0.385%, not falsely placed at x=0.5%.
    np.testing.assert_allclose(tradeoff.collections[-1].get_offsets(), [[0.385, 33.6]])
    np.testing.assert_allclose(paired.lines[-1].get_ydata(), [87.25, 67.25, 27.25])
    assert "achieved evaluation rates differ" in figure._suptitle.get_text()
