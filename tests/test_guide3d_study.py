"""Small, data-free checks of the acquisition-separated Guide3D study protocol."""

from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("PIL")

from benchmarks import run_guide3d as runner  # noqa: E402
from benchmarks.guide3d_data import Guide3DRecord  # noqa: E402


def records_fixture():
    records = []
    for acquisition in (*runner.DEV_ACQUISITIONS, "0-bca-straight-9", "1-bca-straight-1"):
        for frame in (3, 8, 15, 21, 44):
            for camera in ("camera1", "camera2"):
                records.append(
                    Guide3DRecord(
                        acquisition,
                        frame,
                        camera,
                        f"{acquisition}-{camera}/{frame:03d}.png",
                        int(acquisition[0]),
                        "straight" if "straight" in acquisition else "angle",
                        np.array([[100.5, 200.5], [200.5, 400.5]]),
                    )
                )
    return records


def selection_fixture():
    rows = []
    for method in runner.CONFIGURATIONS:
        for acquisition in runner.DEV_ACQUISITIONS:
            for frame in (3, 8):
                for camera in ("camera1", "camera2"):
                    rows.append(
                        {
                            "case_id": f"{acquisition}:{frame:04d}:{camera}",
                            "acquisition_id": acquisition,
                            "camera": camera,
                            "frame_number": frame,
                            "method": method,
                            "budget_fraction": 0.01,
                            "metrics": {"centerline_coverage": 0.5, "selected_fraction": 0.012},
                        }
                    )
    return rows


def test_uniform_frame_manifest_keeps_views_paired_and_acquisitions_separate():
    records = records_fixture()
    manifest = runner.make_manifest(records, pairs_per_acquisition=4)
    assert len(manifest) == 5 * 4 * 2
    assert len({row["case_id"] for row in manifest}) == len(manifest)
    groups, pairs, acquisition_splits = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in manifest:
        groups[row["acquisition_id"]].add(row["frame_number"])
        pairs[row["acquisition_id"], row["frame_number"]].add(row["camera"])
        acquisition_splits[row["acquisition_id"]].add(row["split"])
    assert all(frames == {3, 8, 21, 44} for frames in groups.values())
    assert all(cameras == {"camera1", "camera2"} for cameras in pairs.values())
    assert all(len(splits) == 1 for splits in acquisition_splits.values())
    assert {
        group for group, splits in acquisition_splits.items() if splits == {"development"}
    } == set(runner.DEV_ACQUISITIONS)
    assert runner.make_manifest(records[::-1], pairs_per_acquisition=4) == manifest


def test_manifest_rejects_missing_development_acquisitions_and_insufficient_pairs():
    records = records_fixture()
    incomplete = [r for r in records if r.acquisition_id != runner.DEV_ACQUISITIONS[0]]
    with pytest.raises(ValueError):
        runner.make_manifest(incomplete, pairs_per_acquisition=4)
    with pytest.raises(ValueError):
        runner.make_manifest(records, pairs_per_acquisition=6)


@pytest.mark.parametrize("pairs", [0, -1, 1.5, True])
def test_manifest_requires_positive_integer_pair_count(pairs):
    with pytest.raises(ValueError):
        runner.make_manifest(records_fixture(), pairs_per_acquisition=pairs)


@pytest.mark.parametrize("malformation", ["missing_camera", "duplicate_camera", "unknown_camera"])
def test_manifest_rejects_broken_pair_cardinality(malformation):
    records = records_fixture()
    if malformation == "missing_camera":
        records.pop(0)
    elif malformation == "duplicate_camera":
        records.append(records[0])
    else:
        records[0] = replace(records[0], camera="camera3")
    with pytest.raises(ValueError):
        runner.make_manifest(records, pairs_per_acquisition=4)


def test_preprocess_uses_two_by_two_area_averaging_and_round_to_even():
    native = np.full((1024, 1024), 100, dtype=np.uint8)
    native[100:102, 100:102] = [[0, 1], [2, 3]]  # mean1.5 rounds to2.
    native[102:104, 100:102] = [[1, 2], [3, 4]]  # mean2.5 also rounds to2.
    working, roi, raw, local, info = runner.preprocess(native)
    assert working.shape == (512, 512)
    assert working.dtype == raw.dtype == local.dtype == np.uint8
    assert roi.dtype == bool
    assert working[50, 50] == working[51, 50] == 2
    assert info["fov_pixels"] == 512 * 512  # Internal dark holes are filled.
    assert info["roi_pixels"] == 480 * 480
    assert not roi[:16].any() and not roi[-16:].any()
    assert not roi[:, :16].any() and not roi[:, -16:].any()
    np.testing.assert_array_equal(raw, working)
    assert local[50, 50] > 0


def test_preprocess_fov_discards_isolated_island_but_fills_dark_interior_holes():
    working = np.zeros((512, 512), dtype=np.uint8)
    working[80:430, 100:420] = 80
    working[160:180, 200:220] = 0
    working[10:15, 10:15] = 150
    native = np.repeat(np.repeat(working, 2, axis=0), 2, axis=1)
    actual, roi, raw, local, info = runner.preprocess(native)
    np.testing.assert_array_equal(actual, working)
    assert info["fov_pixels"] == 350 * 320
    assert info["roi_pixels"] == (350 - 32) * (320 - 32)
    assert roi[170, 210]
    assert not roi[12, 12]
    assert raw[12, 12] == raw[0, 0] == 255
    assert raw[170, 210] == 0
    assert local[12, 12] == local[0, 0] == 0
    assert local[170, 210] > 0
    assert not roi[80:96].any()


def test_uniform_image_has_no_local_contrast_and_no_nonfinite_preprocessing_values():
    working, roi, raw, local, info = runner.preprocess(np.full((1024, 1024), 80, dtype=np.uint8))
    assert not local.any()
    assert info["local_scale_graylevels"] == 1
    assert info["local_saturated_roi_fraction"] == 0
    np.testing.assert_array_equal(raw, working)
    assert roi.sum() == 480 * 480


@pytest.mark.parametrize(
    "native",
    [
        np.zeros((512, 512), dtype=np.uint8),
        np.zeros((1024, 1024), dtype=np.float64),
        np.zeros((1024, 1024), dtype=np.uint8),
    ],
)
def test_preprocess_rejects_wrong_resolution_dtype_or_empty_field(native):
    with pytest.raises(ValueError):
        runner.preprocess(native)


def test_configuration_selection_uses_coverage_then_area_then_declared_order():
    methods = list(runner.CONFIGURATIONS)
    rows = selection_fixture()
    assert runner.select_configuration(rows)["selected_method"] == methods[0]
    for row in rows:
        if row["method"] == methods[1]:
            row["metrics"]["selected_fraction"] = 0.011
    assert runner.select_configuration(rows)["selected_method"] == methods[1]
    for row in rows:
        if row["method"] == methods[2]:
            row["metrics"]["centerline_coverage"] = 0.51
            row["metrics"]["selected_fraction"] = 0.1
    assert runner.select_configuration(rows)["selected_method"] == methods[2]


def test_configuration_selection_uses_acquisition_macro_average_and_primary_budget_only():
    rows = selection_fixture()
    methods = list(runner.CONFIGURATIONS)
    for row in rows:
        if row["method"] == methods[1]:
            row["metrics"]["centerline_coverage"] = (
                1.0 if row["acquisition_id"] == runner.DEV_ACQUISITIONS[0] else 0.3
            )
    # Duplicate frame observations equally for all candidate methods in just
    # one acquisition, with new case IDs: these must not change group weight.
    extra = []
    for row in rows:
        if row["acquisition_id"] == runner.DEV_ACQUISITIONS[1]:
            extra.append(
                {
                    **row,
                    "case_id": row["case_id"] + "-extra",
                    "frame_number": row["frame_number"] + 100,
                }
            )
    rows.extend(extra)
    ignored = [{**row, "budget_fraction": 0.02} for row in selection_fixture()]
    result = runner.select_configuration(rows + ignored)
    assert result["selected_method"] == methods[1]
    summary = next(r for r in result["development_scores"] if r["method"] == methods[1])
    assert summary["coverage"] == pytest.approx((1 + 0.3 + 0.3) / 3)


@pytest.mark.parametrize(
    "malformation",
    ["missing_acquisition", "held_out_acquisition", "missing_case", "duplicate_case"],
)
def test_configuration_selection_rejects_incomplete_or_contaminated_candidate_rows(malformation):
    rows = selection_fixture()
    if malformation == "missing_acquisition":
        rows = [r for r in rows if r["acquisition_id"] != runner.DEV_ACQUISITIONS[0]]
    elif malformation == "held_out_acquisition":
        rows.append({**rows[0], "acquisition_id": "1-bca-straight-1", "case_id": "held-out"})
    elif malformation == "missing_case":
        rows.pop(0)
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError):
        runner.select_configuration(rows)


def test_run_case_scales_pixel_centers_and_reuses_matching_core_without_changing_budgets(
    monkeypatch, tmp_path
):
    record = records_fixture()[0]
    native = np.full((1024, 1024), 100, dtype=np.uint8)
    working = np.arange(64, dtype=np.uint8).reshape(8, 8) + 50
    roi = np.ones((8, 8), dtype=bool)
    local = np.arange(64, dtype=np.uint8).reshape(8, 8)
    raw = working.copy()
    monkeypatch.setattr(runner, "load_image", lambda *_: native)
    monkeypatch.setattr(runner, "preprocess", lambda _: (working, roi, raw, local, {}))
    monkeypatch.setattr(runner, "BUDGETS", (0.01, 0.02))
    captured_references, pipeline_calls = [], []
    original_prepare = runner.prepare_reference

    def capture_reference(centerline, mask):
        captured_references.append(centerline)
        # Translate this fixture's reference back inside the tiny test ROI;
        # the assertion below checks the unmodified actual scaling passed in.
        return original_prepare(centerline - centerline[0] + 2, mask)

    def fake_pipeline(image, **kwargs):
        pipeline_calls.append((image.copy(), kwargs))
        return np.ones(image.shape, dtype=np.int64), {"pipeline_seconds": 0.0}, ("cached",)

    monkeypatch.setattr(runner, "prepare_reference", capture_reference)
    monkeypatch.setattr(runner, "run_pipeline", fake_pipeline)
    methods = ["darkness", "local_contrast", "ppi_raw_default", *runner.CONFIGURATIONS]
    arguments = (record, str(tmp_path), methods, "frozen-fingerprint", str(tmp_path / "cache"))
    result = runner.run_case(arguments)
    np.testing.assert_array_equal(captured_references[0], [[50, 100], [100, 200]])
    assert len(result["rows"]) == len(methods) * 2
    assert Counter(row["method"] for row in result["rows"]) == dict.fromkeys(methods, 2)
    assert len(pipeline_calls) == 4
    np.testing.assert_array_equal(pipeline_calls[0][0], raw)
    for image, _ in pipeline_calls[1:]:
        np.testing.assert_array_equal(image, 255 - local)
    assert pipeline_calls[1][1]["core_result"] is None
    assert pipeline_calls[2][1]["core_result"] == ("cached",)
    assert pipeline_calls[3][1]["core_result"] is None
    for row in result["rows"]:
        if row["method"].startswith("ppi_"):
            assert row["metrics"]["selected_pixels"] == 64  # Whole cutoff tie retained.
            assert row["selection"]["budget_exceeded_by_ties"] > 0
    assert runner.run_case(arguments) == result
    assert len(pipeline_calls) == 4  # Valid checkpoint bypasses recomputation.
    with pytest.raises(ValueError, match="Stale checkpoint"):
        runner.run_case((*arguments[:3], "changed-fingerprint", arguments[4]))
    with pytest.raises(ValueError, match="Stale checkpoint"):
        runner.run_case((record, str(tmp_path), methods[:-1], arguments[3], arguments[4]))
