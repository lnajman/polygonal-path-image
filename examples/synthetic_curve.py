"""Generate or load a grayscale image and save a reproducible PPI demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from polygonal_path_image import compute_ppi, filter_tortuosity, voting


def synthetic_image() -> np.ndarray:
    """Return a seeded 96-by-96 image with a dark, gently curved structure."""
    rng = np.random.default_rng(42)
    rows, cols = np.indices((96, 96))
    center = 48 + 16 * np.sin((cols - 12) / 27)
    signal = 190 - 165 * np.exp(-0.5 * ((rows - center) / 1.4) ** 2)
    return np.clip(signal + rng.normal(0, 12, signal.shape), 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("example-output"))
    parser.add_argument("--input", type=Path, help="Optional image, converted to grayscale uint8")
    parser.add_argument("--segment-length", type=int, default=3)
    parser.add_argument("--segments", type=int, default=10)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if args.input is None:
        image = synthetic_image()
    else:
        with Image.open(args.input) as source:
            image = np.asarray(source.convert("L"))
    costs, paths = compute_ppi(image, args.segment_length, args.segments)
    accepted = filter_tortuosity(costs, paths, threshold=0.75)
    votes, _ = voting(accepted, paths)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, array in [("input", image), ("costs", costs), ("paths", paths), ("votes", votes)]:
        np.save(args.output / f"{name}.npy", array, allow_pickle=False)
    Image.fromarray(image).save(args.output / "input.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    for ax, data, title, cmap in zip(
        axes,
        [image, np.ma.masked_invalid(costs), votes],
        ["Input potential", "Minimum path cost", "Path votes (straightness ≥ 0.75)"],
        ["gray", "viridis", "magma"],
    ):
        artist = ax.imshow(data, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
        fig.colorbar(artist, ax=ax, shrink=0.72)
    fig.savefig(args.output / "comparison.png", dpi=160)
    plt.close(fig)
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
