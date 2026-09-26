"""Complete, finite-support synthetic annotations for an all-lines study.

Every intended dark structure has its own independent finite polyline. Tube
profiles are truncated at three sigma: the noiseless image is exactly background
outside their union. Overlaps use maximum darkness, not additive contrast.
Coordinates are (row, column); orientation increases clockwise from +column.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from benchmarks.synthetic import point_to_polyline_distance

SIZE = 128
ROI_MARGIN = 24
BACKGROUND = 200.0
SAMPLE_STEP = 0.25
GAUSSIAN_CUTOFF_SIGMAS = 3.0
FAMILIES = (
    "crossing",
    "branch",
    "parallel",
    "mixed_width",
    "curved_pair",
    "weak_alone",
    "weak_near_strong",
)


@dataclass(frozen=True)
class MultilineCase:
    """One image with every intended line and its operational tube support.

    ``clean_image`` and ``noise`` retain the unrounded float64 components of
    ``image = rint(clip(clean_image + noise, 0, 255))``. Pair comparisons use
    exactly the same noise array, rather than merely the same noise level.
    ``distance_maps`` contain distances to each finite polyline independently;
    concatenating disconnected lines would create spurious connecting segments.
    """

    image: np.ndarray
    centerlines: list[np.ndarray]
    sigmas: list[float]
    roi: np.ndarray
    line_names: list[str]
    parameters: dict
    clean_image: np.ndarray
    noise: np.ndarray
    distance_maps: list[np.ndarray]
    support_masks: list[np.ndarray]


def _spec(split, family, angle, seed, noise_std, *, gap=None):
    control = family in ("blank", "noise")
    context = f"{split}_a{angle:03d}_s{seed}_n{noise_std:02d}" if not control else split
    variant = family + (f"_gap{gap:02d}" if gap is not None else "")
    case_id = (
        f"{context}_{variant}" if not control else f"{split}_{family}_s{seed}_n{noise_std:02d}"
    )
    return {
        "case_id": case_id,
        "split": split,
        "family": family,
        "angle_deg": None if control else float(angle),
        "seed": seed,
        "noise_std": float(noise_std),
        "gap": None if gap is None else float(gap),
        "pair_group": context if family in ("weak_alone", "weak_near_strong") else None,
        "size": SIZE,
        "roi_margin": ROI_MARGIN,
        "background": BACKGROUND,
        "sample_step": SAMPLE_STEP,
        "gaussian_cutoff_sigmas": GAUSSIAN_CUTOFF_SIGMAS,
    }


def multiline_specs(suite="full"):
    """Return the predeclared 100-scene panel or a fixed ten-scene smoke subset.

    Full panel: 21 development scenes (16 line scenes + 5 controls), and 79
    evaluation scenes (72 line scenes + 7 controls). The 72 evaluation line
    scenes span nine orientation/noise contexts, each containing eight variants;
    they are not 72 independent noise realizations. A weak-only scene is reused
    for its two near-strong comparisons.

    Development uses angle 13 and seeds 1001/1002 at noise 5/12 respectively.
    Evaluation crosses angles 0/37/79 with seeds 2001/2002/2003, assigning noise
    5 or 12 by orientation-index + seed-index parity. Both noise levels occur
    for every evaluation orientation and seed, but this is not a full factorial.
    Controls include one deterministic blank per split and each split's seeds
    at both noise levels. The identical blanks are deliberately not independent
    replicates, nor members of the development/evaluation geometry comparison.
    """
    if suite not in ("full", "smoke"):
        raise ValueError("suite must be 'full' or 'smoke'")
    contexts = [("development", 13, 1001, 5), ("development", 13, 1002, 12)]
    contexts.extend(
        ("evaluation", angle, seed, 5 if (i + j) % 2 == 0 else 12)
        for i, angle in enumerate((0, 37, 79))
        for j, seed in enumerate((2001, 2002, 2003))
    )
    variants = [(family, None) for family in FAMILIES if family != "weak_near_strong"]
    variants.extend([("weak_near_strong", 6), ("weak_near_strong", 10)])
    specs = [
        _spec(split, family, angle, seed, noise, gap=gap)
        for split, angle, seed, noise in contexts
        for family, gap in variants
    ]
    for split, seeds in (("development", (1001, 1002)), ("evaluation", (2001, 2002, 2003))):
        specs.append(_spec(split, "blank", None, seeds[0], 0))
        specs.extend(
            _spec(split, "noise", None, seed, noise) for seed in seeds for noise in (5, 12)
        )
    if suite == "smoke":
        # A paired weak-line trio and both control types in each split.
        contexts = {"development_a013_s1001_n05", "evaluation_a000_s2001_n05"}
        controls = {
            "development_blank_s1001_n00",
            "development_noise_s1001_n05",
            "evaluation_blank_s2001_n00",
            "evaluation_noise_s2001_n05",
        }
        return [
            spec for spec in specs if spec["pair_group"] in contexts or spec["case_id"] in controls
        ]
    return specs


def _straight(y=0.0, half_length=28.0):
    """A local (x, y) segment; two vertices suffice for exact rendering."""
    return np.array([[-half_length, y], [half_length, y]], dtype=np.float64)


def _arc(y_offset, sign, sample_step):
    arclength, bend = 54.0, np.deg2rad(70.0)
    position = np.linspace(-arclength / 2, arclength / 2, int(np.ceil(arclength / sample_step)) + 1)
    curvature = bend / arclength
    x = np.sin(curvature * position) / curvature
    y = y_offset + sign * 2 * np.sin(curvature * position / 2) ** 2 / curvature
    return np.column_stack((x, y))


def _local_geometry(family, gap, sample_step):
    """Construct independently annotated polylines in a fixed shared frame."""
    if family == "crossing":
        angle = np.deg2rad(65.0)
        endpoint = 28 * np.array([np.cos(angle), np.sin(angle)])
        return (
            [_straight(), np.stack((-endpoint, endpoint))],
            [1.25, 1.25],
            [50, 50],
            ["crossing_a", "crossing_b"],
        )
    if family == "branch":
        angle = np.deg2rad(50.0)
        upper = 28 * np.array([np.cos(angle), -np.sin(angle)])
        lower = 28 * np.array([np.cos(angle), np.sin(angle)])
        lines = [np.array([[-28, 0], [0, 0]]), np.stack(([0, 0], upper)), np.stack(([0, 0], lower))]
        return lines, [1.25] * 3, [50] * 3, ["stem", "branch_upper", "branch_lower"]
    if family == "parallel":
        return (
            [_straight(y) for y in (-8, 0, 8)],
            [1.25] * 3,
            [50] * 3,
            ["upper", "middle", "lower"],
        )
    if family == "mixed_width":
        return (
            [_straight(y, 27) for y in (-10, 0, 10)],
            [0.8, 1.25, 2.0],
            [50] * 3,
            ["thin", "medium", "wide"],
        )
    if family == "curved_pair":
        return (
            [_arc(-10, 1, sample_step), _arc(10, -1, sample_step)],
            [1.25, 1.25],
            [50, 50],
            ["curve_upper", "curve_lower"],
        )
    if family == "weak_alone":
        return [_straight(-5)], [1.25], [20], ["weak"]
    if family == "weak_near_strong":
        return [_straight(-5), _straight(-5 + gap)], [1.25, 1.25], [20, 80], ["weak", "strong"]
    if family in ("blank", "noise"):
        return [], [], [], []
    raise ValueError(f"Unknown multiline family: {family!r}")


def generate_multiline(spec):
    """Render a deterministic manifest scene without label-dependent image crops.

    A tube's signal and declared support both end at three sigma. The sharp
    cutoff discards residual darkness of approximately 1.1% of line contrast,
    a deliberate modelling limitation. All supports fit within the fixed ROI.
    At crossings or overlaps, the darker tube wins via a pointwise maximum;
    annotations retain both contributing line identities.
    """
    if not isinstance(spec, dict):
        raise ValueError("spec must be a dictionary")
    parameters = dict(spec)
    required = ("case_id", "split", "family", "angle_deg", "seed", "noise_std", "gap", "pair_group")
    if any(key not in spec for key in required):
        raise ValueError("spec is missing manifest fields")
    if spec["split"] not in ("development", "evaluation"):
        raise ValueError("Unknown split")
    for key, expected in (
        ("size", SIZE),
        ("roi_margin", ROI_MARGIN),
        ("background", BACKGROUND),
        ("sample_step", SAMPLE_STEP),
        ("gaussian_cutoff_sigmas", GAUSSIAN_CUTOFF_SIGMAS),
    ):
        if parameters.get(key, expected) != expected:
            raise ValueError(f"The frozen geometry requires {key}={expected}")
        parameters[key] = expected
    seed = spec["seed"]
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    noise_std = float(spec["noise_std"])
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    family = spec["family"]
    control = family in ("blank", "noise")
    angle_deg = 0.0 if control else float(spec["angle_deg"])
    if not np.isfinite(angle_deg):
        raise ValueError("angle_deg must be finite")
    gap = spec["gap"]
    if family == "weak_near_strong":
        if gap not in (6, 10):
            raise ValueError("weak_near_strong gap must be 6 or 10 pixels")
    elif gap is not None:
        raise ValueError("Only weak_near_strong scenes have a gap parameter")
    if family == "blank" and noise_std != 0:
        raise ValueError("Blank controls must have zero noise")
    if family == "noise" and noise_std <= 0:
        raise ValueError("Noise controls must have positive noise")
    lines, sigmas, contrasts, names = _local_geometry(family, gap, SAMPLE_STEP)
    angle = np.deg2rad(angle_deg)
    centerlines = []
    for line, sigma in zip(lines, sigmas, strict=True):
        line = np.asarray(line, dtype=np.float64)
        x, y = line.T
        centerline = np.column_stack(
            (x * np.sin(angle) + y * np.cos(angle), x * np.cos(angle) - y * np.sin(angle))
        )
        centerline += (SIZE - 1) / 2
        radius = GAUSSIAN_CUTOFF_SIGMAS * sigma
        if (centerline.min(axis=0) - radius < ROI_MARGIN).any() or (
            centerline.max(axis=0) + radius > SIZE - 1 - ROI_MARGIN
        ).any():
            raise ValueError("Every reference and its full tube support must fit inside the ROI")
        centerlines.append(centerline)
    roi = np.zeros((SIZE, SIZE), dtype=bool)
    roi[ROI_MARGIN : SIZE - ROI_MARGIN, ROI_MARGIN : SIZE - ROI_MARGIN] = True
    pixels = np.indices((SIZE, SIZE)).reshape(2, -1).T
    darkness = np.zeros((SIZE, SIZE), dtype=np.float64)
    distance_maps, supports = [], []
    for centerline, sigma, contrast in zip(centerlines, sigmas, contrasts, strict=True):
        distance = point_to_polyline_distance(pixels, centerline).reshape(SIZE, SIZE)
        support = distance <= GAUSSIAN_CUTOFF_SIGMAS * sigma
        profile = np.where(support, contrast * np.exp(-(distance**2) / (2 * sigma**2)), 0.0)
        np.maximum(darkness, profile, out=darkness)
        distance_maps.append(distance)
        supports.append(support)
    clean = BACKGROUND - darkness
    noise = np.random.default_rng(int(seed)).normal(0, noise_std, clean.shape)
    image = np.rint(np.clip(clean + noise, 0, 255)).astype(np.uint8)
    parameters.update(
        {
            "sigmas": [float(sigma) for sigma in sigmas],
            "contrasts": [float(contrast) for contrast in contrasts],
            "line_names": names.copy(),
            "line_count": len(centerlines),
            "line_lengths": [
                float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()) for line in centerlines
            ],
            "signal_composition": "pointwise maximum of truncated Gaussian tube darkness",
            "geometry_frame": "fixed image center; no scene-dependent recentering",
        }
    )
    for array in [image, roi, clean, noise, *centerlines, *distance_maps, *supports]:
        array.setflags(write=False)
    return MultilineCase(
        image,
        centerlines,
        list(sigmas),
        roi,
        names,
        parameters,
        clean,
        noise,
        distance_maps,
        supports,
    )
