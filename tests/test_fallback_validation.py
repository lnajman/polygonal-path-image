"""Freeze/calibration boundaries without processing the fresh evaluation panel."""

import json
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from benchmarks import run_fallback_validation as runner  # noqa: E402
from benchmarks.multiline_data import multiline_specs  # noqa: E402


@pytest.fixture
def plan(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "evaluation_angles_degrees": [19, 53, 107],
                "evaluation_seeds": [4101, 4102],
                "noise_rule": "5 if angle_index + seed_index is even, otherwise 12",
                "line_scenes": 48,
                "families": [
                    "crossing",
                    "branch",
                    "parallel",
                    "mixed_width",
                    "curved_pair",
                    "weak_alone",
                    "weak_near_strong_gap6",
                    "weak_near_strong_gap10",
                ],
            }
        )
    )
    return path


def test_manifest_preserves_development_and_predeclares_fresh_contexts(plan, monkeypatch):
    def forbid_render(_):
        pytest.fail("Defining the panel must not render any evaluation image")

    monkeypatch.setattr(runner, "generate_multiline", forbid_render)
    definition = runner.protocol(plan)
    manifest = definition["manifest"]
    dev = [s for s in manifest if s["split"] == "development"]
    fresh = [s for s in manifest if s["split"] == "evaluation"]
    assert dev == [s for s in multiline_specs() if s["split"] == "development"]
    assert len(dev) == 21 and len(fresh) == 53
    assert len({s["case_id"] for s in manifest}) == 74
    targets = [s for s in fresh if s["family"] not in ("blank", "noise")]
    assert len(targets) == 48
    assert len({(s["angle_deg"], s["seed"]) for s in targets}) == 6
    assert {s["seed"] for s in targets} == {4101, 4102}
    assert Counter(s["family"] for s in fresh)["noise"] == 4
    assert Counter(s["family"] for s in fresh)["blank"] == 1
    for spec in targets:
        i = [19, 53, 107].index(spec["angle_deg"])
        j = [4101, 4102].index(spec["seed"])
        assert spec["noise_std"] == (5 if (i + j) % 2 == 0 else 12)
    groups = Counter(s["pair_group"] for s in targets if s["pair_group"] is not None)
    assert len(groups) == 6 and set(groups.values()) == {3}
    assert definition["predeclared_plan_sha256"] == runner.digest(plan.read_bytes())


@pytest.mark.parametrize(
    "field,value",
    [
        ("evaluation_angles_degrees", [0, 37, 79]),
        ("evaluation_seeds", [2001, 2002]),
        ("noise_rule", "noise chosen from results"),
        ("line_scenes", 72),
        ("families", ["weak_alone"]),
    ],
)
def test_rejects_drift_from_predeclared_panel(plan, field, value):
    content = json.loads(plan.read_text())
    content[field] = value
    plan.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        runner.protocol(plan)


def test_calibration_refuses_evaluation_before_loading_images(tmp_path, monkeypatch):
    def forbid_render(_):
        pytest.fail("Calibration attempted to render evaluation data")

    monkeypatch.setattr(runner, "generate_multiline", forbid_render)
    with pytest.raises(ValueError, match="development"):
        runner.calibrate([{"split": "evaluation"}], tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        runner.calibrate([], tmp_path)


def test_candidate_addition_preserves_baselines_and_fixed_parameters(monkeypatch):
    image = np.full((3, 4), 100, dtype=np.uint8)
    baselines = {method: np.full(image.shape, i) for i, method in enumerate(runner.BASE_METHODS)}
    monkeypatch.setattr(runner, "base_score_methods", lambda _: (baselines.copy(), {}))

    def candidate(value, **parameters):
        assert value is image
        assert parameters == {"segment_length": 3, "nb_segments": 10, "threshold": 0.75}
        mask = np.zeros(image.shape, dtype=bool)
        mask[0, 0] = True
        return np.ones(image.shape, dtype=np.int64), np.zeros(image.shape), None, mask

    monkeypatch.setattr(runner, "straight_fallback", candidate)
    scores, timings, diagnostics = runner.score_methods(image)
    for method in runner.BASE_METHODS:
        assert scores[method] is baselines[method]
    assert set(scores) == set(runner.METHODS)
    assert diagnostics == {
        "replaced_origins": 1,
        "active_origins_after_fallback": 12,
        "total_origins": 12,
    }
    assert timings[runner.CANDIDATE]["pipeline_seconds"] >= 0


def test_cache_binds_fingerprint_and_array_contents(tmp_path, monkeypatch):
    # Only an original development blank is generated, never a fresh image.
    spec = next(
        s for s in multiline_specs() if s["split"] == "development" and s["family"] == "blank"
    )
    calls = []

    def score(image):
        calls.append(True)
        return {name: np.zeros(image.shape) for name in runner.METHODS}, {}, {}

    monkeypatch.setattr(runner, "score_methods", score)
    first = runner._calculate_case((spec, tmp_path, "first"))
    assert runner._calculate_case((spec, tmp_path, "first")) == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="different protocol"):
        runner._calculate_case((spec, tmp_path, "second"))
    path = tmp_path / f"{spec['case_id']}.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        runner._calculate_case((spec, tmp_path, "first"))


@pytest.fixture
def mocked_run(tmp_path, monkeypatch):
    """Use metadata stand-ins to test scheduling without scoring actual images."""
    work = tmp_path / "run"
    manifest = [
        {"case_id": "dev", "split": "development"},
        {"case_id": "fresh", "split": "evaluation"},
    ]
    monkeypatch.setattr(runner, "protocol", lambda _: {"manifest": manifest})
    monkeypatch.setattr(runner, "provenance", lambda: {"source": "frozen"})
    events = []
    thresholds = {method: [{"threshold": 10.0, "target_rate": 0.005}] for method in runner.METHODS}

    def calculate(specs, cache, fingerprint, workers):
        split = specs[0]["split"]
        if split == "evaluation":
            saved = json.loads((work / "thresholds.json").read_text())
            assert saved["thresholds"] == thresholds
            assert saved["fingerprint"] == fingerprint
        events.append(("score", split))
        return [{"case_id": s["case_id"]} for s in specs]

    def calibrate(specs, cache):
        assert all(s["split"] == "development" for s in specs)
        events.append(("calibrate", "development"))
        return thresholds

    def measure(specs, cache, definitions):
        assert definitions == thresholds
        events.append(("measure", specs[0]["split"]))
        return [{"split": s["split"]} for s in specs]

    monkeypatch.setattr(runner, "calculate_split", calculate)
    monkeypatch.setattr(runner, "calibrate", calibrate)
    monkeypatch.setattr(runner, "measure", measure)
    return (
        {"plan": tmp_path / "unused.json", "work_dir": work, "output": tmp_path / "result.json"},
        events,
        thresholds,
    )


def test_prepare_default_does_not_score_anything(mocked_run):
    arguments, events, _ = mocked_run
    runner.run(**arguments)
    assert not events
    assert (arguments["work_dir"] / "protocol.json").is_file()
    assert not arguments["output"].exists()


def test_development_phase_never_scores_fresh_panel(mocked_run):
    arguments, events, _ = mocked_run
    result = runner.run(**arguments, phase="development")
    assert events == [
        ("score", "development"),
        ("calibrate", "development"),
        ("measure", "development"),
    ]
    assert result["scenes"] == [{"case_id": "dev"}]
    assert {row["split"] for row in result["rows"]} == {"development"}


def test_all_phase_persists_thresholds_before_fresh_scoring(mocked_run):
    arguments, events, _ = mocked_run
    runner.run(**arguments, phase="all")
    assert events == [
        ("score", "development"),
        ("calibrate", "development"),
        ("measure", "development"),
        ("score", "evaluation"),
        ("measure", "evaluation"),
    ]


def test_changed_calibration_rejected_before_fresh_scoring(mocked_run):
    arguments, events, thresholds = mocked_run
    runner.run(**arguments, phase="development")
    events.clear()
    thresholds[runner.CANDIDATE][0]["threshold"] = 11.0
    with pytest.raises(ValueError, match="saved calibration"):
        runner.run(**arguments, phase="all")
    assert not any(split == "evaluation" for _, split in events)


def test_source_change_rejects_cache_namespace(mocked_run, monkeypatch):
    arguments, events, _ = mocked_run
    runner.run(**arguments, phase="prepare")
    monkeypatch.setattr(runner, "provenance", lambda: {"source": "changed"})
    with pytest.raises(ValueError, match="frozen protocol"):
        runner.run(**arguments, phase="all")
    assert not events


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_workers_rejected_before_preparing(mocked_run, workers):
    arguments, events, _ = mocked_run
    with pytest.raises(ValueError, match="positive integer"):
        runner.run(**arguments, workers=workers)
    assert not events
