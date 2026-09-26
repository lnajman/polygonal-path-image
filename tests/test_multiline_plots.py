"""Independent observation-grain, null and plotting-limit regressions."""

import copy
import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.plot_multiline import (  # noqa: E402
    LABELS,
    mean_metric,
    plot_controls,
    primary_summary,
    select_rows,
    weak_pairs,
)


def _row(
    case_id,
    *,
    method="darkness",
    split="evaluation",
    selection="frozen_threshold",
    point=0.005,
    target=True,
    family="parallel",
    pair_group=None,
    gap=None,
    metrics=None,
):
    return {
        "case_id": case_id,
        "method": method,
        "split": split,
        "selection": selection,
        "operating_point": point,
        "family": family,
        "pair_group": pair_group,
        "gap": gap,
        "metrics": {"has_target": target, **(metrics or {})},
    }


def _metrics(
    coverages,
    longest_gaps,
    *,
    reference_distances,
    prediction_distance,
    precision,
    selected_pixels=10,
):
    return {
        "macro_coverage": float(np.mean(coverages)),
        "worst_line_coverage": min(coverages),
        "outside_support_fraction": 0.02 if selected_pixels else 0,
        "selected_support_precision": precision,
        "union_centerline_precision": precision,
        "prediction_to_reference_mean_distance": prediction_distance,
        "selected_fraction": selected_pixels / 100,
        "selected_pixels": selected_pixels,
        "per_line": [
            {
                "coverage": c,
                "longest_uncovered_arclength": g,
                "reference_to_prediction_mean_distance": d,
            }
            for c, g, d in zip(coverages, longest_gaps, reference_distances, strict=True)
        ],
    }


def test_select_rows_keeps_split_selection_operating_point_and_control_grains_separate():
    wanted = _row("a")
    control = _row("control", target=False)
    distractors = [
        _row("a", method="frangi"),
        _row("a", split="development"),
        _row("a", selection="area_budget"),
        _row("a", point=0.01),
    ]
    result = {"rows": [wanted, control, *distractors]}
    assert select_rows(result, method="darkness") == [wanted]
    assert select_rows(result, method="darkness", target=False) == [control]
    assert select_rows(result, method="darkness", target=None) == [wanted, control]


def test_mean_metric_excludes_undefined_values_but_preserves_real_zero():
    rows = [{"metrics": {"value": value}} for value in (None, 0.0, 1.0)]
    assert mean_metric(rows, "value") == 0.5
    assert mean_metric(rows[:1], "value") is None
    assert mean_metric([], "value") is None


def test_primary_summary_weights_scenes_equally_after_line_averaging():
    # Scene A has three lines, B has one. The scene mean is .5 coverage,
    # whereas pooling all four lines would produce .25 coverage.
    a = _metrics(
        [0, 0, 0], [6, 12, 18], reference_distances=[1, 2, 3], prediction_distance=4, precision=0.25
    )
    b = _metrics([1], [0], reference_distances=[10], prediction_distance=8, precision=0.75)
    rows = [
        _row(case, method=method, metrics=copy.deepcopy(metrics))
        for method in LABELS
        for case, metrics in (("a", a), ("b", b))
    ]
    summary = primary_summary({"rows": rows})
    for values in summary.values():
        assert values["line_scenes"] == 2
        assert values["macro_coverage"] == 0.5
        assert values["worst_line_coverage"] == 0.5
        assert values["mean_longest_gap"] == 6
        assert values["mean_reference_to_prediction_distance"] == 6
        assert values["prediction_to_reference_mean_distance"] == 6
        assert values["selected_support_precision"] == 0.5


def test_zero_selection_scenes_stay_in_coverage_and_gaps_but_not_undefined_distances():
    found = _metrics([1], [0], reference_distances=[2], prediction_distance=3, precision=0.8)
    empty = _metrics(
        [0],
        [56],
        reference_distances=[None],
        prediction_distance=None,
        precision=None,
        selected_pixels=0,
    )
    rows = [
        _row(case, method=method, metrics=copy.deepcopy(metrics))
        for method in LABELS
        for case, metrics in (("found", found), ("empty", empty))
    ]
    summary = primary_summary({"rows": rows})
    for values in summary.values():
        assert values["line_scenes"] == 2
        assert values["macro_coverage"] == 0.5
        assert values["mean_longest_gap"] == 28
        assert values["selected_support_precision"] == 0.8
        assert values["mean_reference_to_prediction_distance"] == 2
        assert values["prediction_to_reference_mean_distance"] == 3
        assert values["scenes_with_selections"] == 1
        assert values["metric_scene_counts"]["macro_coverage"] == 2
        assert values["metric_scene_counts"]["selected_support_precision"] == 1
        assert values["metric_scene_counts"]["prediction_to_reference_mean_distance"] == 1


def test_all_undefined_localization_is_json_null_not_nan():
    empty = _metrics(
        [0],
        [56],
        reference_distances=[None],
        prediction_distance=None,
        precision=None,
        selected_pixels=0,
    )
    summary = primary_summary(
        {"rows": [_row("empty", method=method, metrics=copy.deepcopy(empty)) for method in LABELS]}
    )
    for values in summary.values():
        assert values["macro_coverage"] == 0
        assert values["mean_longest_gap"] == 56
        assert values["selected_support_precision"] is None
        assert values["mean_reference_to_prediction_distance"] is None
        assert values["prediction_to_reference_mean_distance"] is None
        assert values["scenes_with_selections"] == 0
        assert values["metric_scene_counts"]["macro_coverage"] == 1
        assert values["metric_scene_counts"]["selected_support_precision"] == 0
    json.dumps(summary, allow_nan=False)


def test_weak_comparisons_match_pair_id_and_named_line_not_order_or_line_index():
    def line(name, coverage, gap):
        return {"line_name": name, "coverage": coverage, "longest_uncovered_arclength": gap}

    rows = [
        _row(
            "b_added",
            family="weak_near_strong",
            pair_group="b",
            gap=6,
            metrics={"per_line": [line("strong", 1, 0), line("weak", 0.5, 2)]},
        ),
        _row(
            "a_alone",
            family="weak_alone",
            pair_group="a",
            metrics={"per_line": [line("weak", 0.9, 1)]},
        ),
        _row(
            "a_added10",
            family="weak_near_strong",
            pair_group="a",
            gap=10,
            metrics={"per_line": [line("weak", 0.8, 2), line("strong", 1, 0)]},
        ),
        _row(
            "b_alone",
            family="weak_alone",
            pair_group="b",
            metrics={"per_line": [line("weak", 0.2, 8)]},
        ),
        _row(
            "a_added6",
            family="weak_near_strong",
            pair_group="a",
            gap=6,
            metrics={"per_line": [line("strong", 1, 0), line("weak", 0.3, 3)]},
        ),
    ]
    pairs = {
        (pair["pair_group"], pair["gap"]): pair
        for pair in weak_pairs({"rows": rows}, method="darkness")
    }
    assert pairs["a", 6]["change"] == pytest.approx(-0.6)
    assert pairs["a", 10]["change"] == pytest.approx(-0.1)
    assert pairs["b", 6]["change"] == pytest.approx(0.3)
    assert pairs["a", 6]["gap_change"] == 2
    assert pairs["b", 6]["gap_change"] == -6
    assert len(pairs) == 3


def test_control_axes_include_later_methods_and_all_noise_levels():
    # The first method has only zeros. Limits must not freeze before later
    # methods add positive responses; this hid blank-PPI selections originally.
    specs = [{"case_id": "blank", "noise_std": 0}]
    specs.extend(
        {"case_id": f"noise{seed}_{noise}", "noise_std": noise}
        for seed in range(3)
        for noise in (5, 12)
    )
    rows = []
    for index, method in enumerate(LABELS):
        for spec in specs:
            value = 0 if index == 0 else (index + spec["noise_std"] / 10) / 100
            rows.append(
                _row(
                    spec["case_id"],
                    method=method,
                    target=False,
                    family="blank" if spec["noise_std"] == 0 else "noise",
                    metrics={"selected_fraction": value},
                )
            )
    figure = plot_controls({"rows": rows, "protocol": {"manifest": specs}})
    try:
        assert len(figure.axes) == 2
        assert len(figure.axes[0].collections) == 6
        assert len(figure.axes[1].collections) == 36
        for axis in figure.axes:
            low, high = axis.get_ylim()
            for collection in axis.collections:
                y = np.asarray(collection.get_offsets())[:, 1]
                assert (y >= low).all()
                assert (y <= high).all()
    finally:
        plt.close(figure)
