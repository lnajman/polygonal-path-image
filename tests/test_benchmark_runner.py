"""Protocol checks and a tiny real execution of the synthetic benchmark runner."""

import json
from collections import Counter, defaultdict

import pytest

from benchmarks import run_synthetic as runner


def _context_cases(panel, block):
    return [
        definition
        for definition in panel.values()
        if any(context["block"] == block for context in definition["contexts"])
    ]


def test_declared_panel_has_expected_coverage_and_unique_measurement_grain():
    panel = runner.experiment_panel()
    expected_scenes = {
        "direction": 36,
        "robustness": 81,
        "parameters": 6,
        "controls": 10,
        "reflection": 6,
    }
    actual = Counter(
        context["block"] for definition in panel.values() for context in definition["contexts"]
    )
    assert actual == expected_scenes
    assert len(panel) == 139
    grains = []
    ppi_evaluations = 0
    for case_id, definition in panel.items():
        configurations = {
            name for context in definition["contexts"] for name in context["config_ids"]
        }
        ppi_evaluations += len(configurations)
        for context in definition["contexts"]:
            assert len(context["config_ids"]) == len(set(context["config_ids"]))
            for name in ["darkness", *context["config_ids"]]:
                for budget in runner.BUDGETS:
                    grains.append((context["block"], case_id, name, budget))
    assert ppi_evaluations == 193
    assert len(grains) == len(set(grains)) == 1328
    assert runner.BUDGETS == (0.005, 0.01, 0.02, 0.05)


def test_direction_panel_is_noise_free_with_complete_declared_angles_and_bends():
    cases = _context_cases(runner.experiment_panel(), "direction")
    observed = {(case["parameters"]["angle_deg"], case["parameters"]["bend_deg"]) for case in cases}
    assert observed == {(angle, bend) for angle in range(0, 180, 15) for bend in (0, 45, 90)}
    for case in cases:
        assert case["parameters"]["noise_std"] == 0
        assert case["parameters"]["contrast"] == 120
        assert case["contexts"][0]["config_ids"] == ["baseline"]


def test_robustness_seeds_are_paired_across_all_declared_conditions():
    paired = defaultdict(set)
    for case in _context_cases(runner.experiment_panel(), "robustness"):
        p = case["parameters"]
        assert p["bend_deg"] == 45
        paired[p["angle_deg"], p["contrast"], p["noise_std"]].add(p["seed"])
    assert set(paired) == {
        (angle, contrast, noise)
        for angle in (0, 30, 45)
        for contrast in (30, 60, 120)
        for noise in (5, 15, 30)
    }
    assert all(seeds == {2012, 2013, 2014} for seeds in paired.values())


def test_parameter_panel_compares_fixed_inputs_and_preserves_matched_raster_lengths():
    panel = runner.experiment_panel()
    cases = _context_cases(panel, "parameters")
    assert {
        (case["parameters"]["angle_deg"], case["parameters"]["bend_deg"]) for case in cases
    } == {(angle, bend) for angle in (0, 30, 45) for bend in (0, 90)}
    for case in cases:
        p = case["parameters"]
        assert (p["contrast"], p["noise_std"], p["seed"]) == (60, 15, 2012)
        assert set(case["contexts"][0]["config_ids"]) == set(runner.CONFIGURATIONS)
    assert len(set(runner.CONFIGURATIONS.values())) == 10
    for name in ("baseline", "matched_L2_K15", "matched_L5_K6"):
        length, segments, threshold = runner.CONFIGURATIONS[name]
        assert length * segments == 30
        assert threshold == 0.75
    assert {
        runner.CONFIGURATIONS[name][0] for name in ("short_segments", "baseline", "long_segments")
    } == {1, 3, 5}
    assert {
        runner.CONFIGURATIONS[name][1] for name in ("short_paths", "baseline", "long_paths")
    } == {5, 10, 15}
    assert {
        runner.CONFIGURATIONS[name][2] for name in ("tau_0", "tau_050", "baseline", "tau_095")
    } == {0, 0.5, 0.75, 0.95}
    for case in panel.values():
        p = case["parameters"]
        assert (p["size"], p["roi_margin"], p["arclength"]) == (128, 32, 48)
        # At least one signed main-axis displacement fits from every source.
        assert (
            max(length * segments for length, segments, _ in runner.CONFIGURATIONS.values())
            <= (p["size"] - 1) // 2
        )


def test_controls_have_no_target_and_negative_bends_have_paired_positive_cases():
    panel = runner.experiment_panel()
    controls = _context_cases(panel, "controls")
    assert {(case["parameters"]["noise_std"], case["parameters"]["seed"]) for case in controls} == {
        (0, 2012),
        *((noise, seed) for noise in (5, 15, 30) for seed in (2012, 2013, 2014)),
    }
    assert all(not case["parameters"]["target_present"] for case in controls)
    parameters = [case["parameters"] for case in panel.values()]
    for case in _context_cases(panel, "reflection"):
        counterpart = dict(case["parameters"])
        assert counterpart["bend_deg"] in (-45, -90)
        counterpart["bend_deg"] *= -1
        assert counterpart in parameters


def test_tiny_run_reuses_core_and_produces_reproducible_finite_json(monkeypatch, tmp_path):
    def tiny_panel(suite):
        parameters = dict(
            size=16,
            roi_margin=2,
            arclength=4,
            noise_std=0,
            seed=2012,
        )
        return {
            "target": {
                "parameters": parameters,
                "contexts": [{"block": "tiny", "config_ids": ["baseline", "tau_0", "short_paths"]}],
            },
            "control": {
                "parameters": {**parameters, "target_present": False},
                "contexts": [{"block": "tiny", "config_ids": ["baseline"]}],
            },
        }

    monkeypatch.setattr(runner, "experiment_panel", tiny_panel)
    monkeypatch.setattr(
        runner,
        "CONFIGURATIONS",
        {"baseline": (1, 2, 0.75), "tau_0": (1, 2, 0), "short_paths": (1, 1, 0.75)},
    )
    monkeypatch.setattr(runner, "_provenance", lambda: {"test": "tiny real kernel"})
    actual_compute = runner.ppi.compute_ppi
    calls = []

    def counted_compute(image, length, segments):
        calls.append((length, segments))
        return actual_compute(image, length, segments)

    monkeypatch.setattr(runner.ppi, "compute_ppi", counted_compute)
    first = runner.run_benchmark(suite="smoke", save_arrays=tmp_path, progress=False)
    assert Counter(calls) == {(1, 2): 2, (1, 1): 1}
    assert first["scene_count"] == 2
    assert first["row_count"] == 24
    assert set(path.name for path in tmp_path.glob("*.npz")) == {"target.npz", "control.npz"}
    rows = first["rows"]
    for row in rows:
        assert row["selected_pixels"] == row["true_positive_pixels"] + row["false_positive_pixels"]
        assert row["selected_pixels"] == row["selection_selected_pixels"]
        assert row["selected_fraction"] == row["selection_actual_fraction"]
        assert row["pipeline_seconds"] >= 0
        if row["method"] == "ppi":
            assert row["pipeline_seconds"] == pytest.approx(
                row["core_seconds"] + row["filter_seconds"] + row["voting_seconds"]
            )
    control_darkness = [
        row for row in rows if row["case_id"] == "control" and row["method"] == "darkness"
    ]
    assert len(control_darkness) == 4
    assert all(row["selected_pixels"] == 0 for row in control_darkness)
    assert all(row["centerline_coverage"] is None for row in control_darkness)
    assert all(row["selected_pixel_precision"] is None for row in control_darkness)
    assert all(row["off_tube_false_positive_rate"] == 0 for row in control_darkness)
    json.dumps(first, allow_nan=False)
    second = runner.run_benchmark(suite="smoke", progress=False)
    assert first["non_timing_rows_sha256"] == second["non_timing_rows_sha256"]


def test_unknown_suite_is_rejected():
    with pytest.raises(ValueError, match="suite"):
        runner.experiment_panel("unknown")
