"""Independent checks for the complete-label, paired-noise synthetic generator."""

from collections import Counter, defaultdict

import numpy as np
import pytest

from benchmarks.multiline_data import generate_multiline, multiline_specs
from benchmarks.synthetic import point_to_polyline_distance


def _spec(family, *, angle=0, gap=None, seed=2001):
    return next(
        spec
        for spec in multiline_specs()
        if spec["family"] == family
        and spec["angle_deg"] == angle
        and spec["gap"] == gap
        and spec["seed"] == seed
    )


def test_frozen_manifest_count_grain_and_disjoint_partitions():
    specs = multiline_specs()
    assert len(specs) == 100
    assert len({spec["case_id"] for spec in specs}) == 100
    assert specs == multiline_specs()
    assert Counter(spec["split"] for spec in specs) == {"development": 21, "evaluation": 79}
    assert Counter(spec["family"] for spec in specs) == {
        "crossing": 11,
        "branch": 11,
        "parallel": 11,
        "mixed_width": 11,
        "curved_pair": 11,
        "weak_alone": 11,
        "weak_near_strong": 22,
        "blank": 2,
        "noise": 10,
    }
    development = [spec for spec in specs if spec["split"] == "development"]
    evaluation = [spec for spec in specs if spec["split"] == "evaluation"]
    assert {spec["seed"] for spec in development} == {1001, 1002}
    assert {spec["seed"] for spec in evaluation} == {2001, 2002, 2003}
    assert {spec["angle_deg"] for spec in development if spec["angle_deg"] is not None} == {13}
    assert {spec["angle_deg"] for spec in evaluation if spec["angle_deg"] is not None} == {
        0,
        37,
        79,
    }
    # Controls have no geometry; the two noiseless blanks are intentionally identical.
    for angle in (0, 37, 79):
        assert {spec["noise_std"] for spec in evaluation if spec["angle_deg"] == angle} == {5, 12}
    for seed in (2001, 2002, 2003):
        assert {
            spec["noise_std"]
            for spec in evaluation
            if spec["seed"] == seed and spec["angle_deg"] is not None
        } == {5, 12}
    pairs = defaultdict(list)
    for spec in specs:
        if spec["pair_group"] is not None:
            pairs[spec["pair_group"]].append(spec)
    assert len(pairs) == 11
    for group in pairs.values():
        assert Counter(spec["family"] for spec in group) == {"weak_alone": 1, "weak_near_strong": 2}
        assert {spec["gap"] for spec in group} == {None, 6, 10}
        assert (
            len(
                {
                    (spec["split"], spec["angle_deg"], spec["seed"], spec["noise_std"])
                    for spec in group
                }
            )
            == 1
        )


def test_smoke_contains_valid_calibration_and_evaluation_pairs_and_controls():
    specs = multiline_specs("smoke")
    assert len(specs) == 10
    assert Counter(spec["split"] for spec in specs) == {"development": 5, "evaluation": 5}
    for split in ("development", "evaluation"):
        assert Counter(spec["family"] for spec in specs if spec["split"] == split) == {
            "weak_alone": 1,
            "weak_near_strong": 2,
            "blank": 1,
            "noise": 1,
        }
    assert all(spec in multiline_specs() for spec in specs)
    with pytest.raises(ValueError):
        multiline_specs("large")


def test_determinism_and_pixel_generation_preserve_exact_component_arrays():
    spec = _spec("mixed_width", angle=37)
    first, second = generate_multiline(spec), generate_multiline(spec)
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.noise, second.noise)
    np.testing.assert_array_equal(first.clean_image, second.clean_image)
    np.testing.assert_array_equal(
        first.image, np.rint(np.clip(first.clean_image + first.noise, 0, 255)).astype(np.uint8)
    )
    assert first.image.dtype == np.uint8
    assert first.image.shape == (128, 128)
    assert first.roi.dtype == bool
    assert first.roi.sum() == 80 * 80
    assert first.parameters["sigmas"] == [0.8, 1.25, 2.0]
    assert first.parameters["contrasts"] == [50, 50, 50]
    assert first.parameters["gaussian_cutoff_sigmas"] == 3
    assert not first.image.flags.writeable
    assert not first.noise.flags.writeable
    assert not first.centerlines[0].flags.writeable
    # Generation must not mutate the caller's serializable manifest entry.
    assert "line_names" not in spec


@pytest.mark.parametrize("angle,seed", [(13, 1001), (0, 2001), (37, 2002), (79, 2003)])
def test_weak_pairing_preserves_geometry_noise_and_pixels_away_from_added_line(angle, seed):
    alone = generate_multiline(_spec("weak_alone", angle=angle, seed=seed))
    for gap in (6, 10):
        together = generate_multiline(_spec("weak_near_strong", angle=angle, gap=gap, seed=seed))
        assert alone.line_names == ["weak"]
        assert together.line_names == ["weak", "strong"]
        assert alone.parameters["pair_group"] == together.parameters["pair_group"]
        np.testing.assert_array_equal(alone.centerlines[0], together.centerlines[0])
        np.testing.assert_array_equal(alone.distance_maps[0], together.distance_maps[0])
        np.testing.assert_array_equal(alone.support_masks[0], together.support_masks[0])
        np.testing.assert_array_equal(alone.noise, together.noise)
        assert (together.clean_image <= alone.clean_image).all()
        unchanged = ~together.support_masks[1]
        np.testing.assert_array_equal(alone.clean_image[unchanged], together.clean_image[unchanged])
        np.testing.assert_array_equal(alone.image[unchanged], together.image[unchanged])
        # Parallel line translation has the requested centerline separation.
        assert np.linalg.norm(
            together.centerlines[1][0] - together.centerlines[0][0]
        ) == pytest.approx(gap)


@pytest.mark.parametrize("angle", [0, 13, 37, 79])
def test_every_geometry_has_complete_finite_support_inside_fixed_roi(angle):
    # One noise instance per distinct geometry; noise cannot alter support bounds.
    seed = 1001 if angle == 13 else 2001
    specs = [
        spec for spec in multiline_specs() if spec["angle_deg"] == angle and spec["seed"] == seed
    ]
    assert len(specs) == 8
    for spec in specs:
        case = generate_multiline(spec)
        assert (
            len(case.centerlines)
            == len(case.sigmas)
            == len(case.line_names)
            == len(case.support_masks)
        )
        union = np.logical_or.reduce(case.support_masks)
        assert not union[~case.roi].any()
        np.testing.assert_array_equal(case.clean_image[~union], np.full((~union).sum(), 200.0))
        assert (case.clean_image[union] < 200).all()
        for line, sigma, support, distance in zip(
            case.centerlines, case.sigmas, case.support_masks, case.distance_maps, strict=True
        ):
            assert np.isfinite(line).all() and line.shape[1] == 2
            assert (line.min(axis=0) - 3 * sigma >= 24).all()
            assert (line.max(axis=0) + 3 * sigma <= 103).all()
            np.testing.assert_array_equal(support, distance <= 3 * sigma)


def test_crossing_uses_maximum_darkness_without_additive_intersection():
    case = generate_multiline(_spec("crossing"))
    profiles = [
        np.where(support, contrast * np.exp(-(distance**2) / (2 * sigma**2)), 0)
        for support, distance, sigma, contrast in zip(
            case.support_masks,
            case.distance_maps,
            case.sigmas,
            case.parameters["contrasts"],
            strict=True,
        )
    ]
    np.testing.assert_array_equal(case.clean_image, 200 - np.maximum.reduce(profiles))
    assert case.clean_image.min() >= 150  # Additive crossing would approach 100.
    assert np.any(case.support_masks[0] & case.support_masks[1])


def test_disconnected_lines_do_not_create_implicit_connectors():
    case = generate_multiline(_spec("mixed_width"))
    # Midpoint of a spurious connector from first line's end to second line's start.
    midpoint = (case.centerlines[0][-1] + case.centerlines[1][0]) / 2
    distances = [
        point_to_polyline_distance(midpoint[None, :], line)[0] for line in case.centerlines
    ]
    np.testing.assert_allclose(distances, [5, 5, 15])
    assert all(distance > 3 * sigma for distance, sigma in zip(distances, case.sigmas, strict=True))
    assert case.clean_image[58, 63] == 200
    # The deliberately wrong concatenation would draw a line through that gap.
    joined = np.concatenate(case.centerlines)
    assert point_to_polyline_distance(midpoint[None, :], joined)[0] == pytest.approx(0)


def test_branch_has_three_separate_arms_with_one_shared_junction():
    case = generate_multiline(_spec("branch"))
    assert len(case.centerlines) == 3
    np.testing.assert_array_equal(case.centerlines[0][-1], case.centerlines[1][0])
    np.testing.assert_array_equal(case.centerlines[0][-1], case.centerlines[2][0])
    np.testing.assert_allclose(case.parameters["line_lengths"], [28, 28, 28])


def test_controls_have_empty_references_and_no_clean_signal():
    for spec in multiline_specs():
        if spec["family"] not in ("blank", "noise"):
            continue
        case = generate_multiline(spec)
        assert case.centerlines == case.sigmas == case.line_names == case.support_masks == []
        assert case.parameters["line_count"] == 0
        assert (case.clean_image == 200).all()
        if spec["family"] == "blank":
            assert (case.image == 200).all()
            assert (case.noise == 0).all()
        else:
            assert np.unique(case.image).size > 1


@pytest.mark.parametrize(
    "changes",
    [
        {"noise_std": -1},
        {"noise_std": float("nan")},
        {"seed": -1},
        {"seed": True},
        {"family": "unknown"},
        {"angle_deg": float("inf")},
        {"gap": 3},
        {"size": 64},
        {"roi_margin": 0},
        {"gaussian_cutoff_sigmas": 4},
        {"split": "test"},
    ],
)
def test_invalid_or_changed_frozen_inputs_raise(changes):
    spec = _spec("parallel").copy()
    spec.update(changes)
    with pytest.raises(ValueError):
        generate_multiline(spec)
