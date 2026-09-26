"""Export the report figures without rerunning path computations.

Run from a repository checkout with ``python -m benchmarks.plot_synthetic``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from . import plots  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/benchmarks/synthetic-v0.1.0.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmark-output/figures"))
    args = parser.parse_args()
    results = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    plots.set_style()
    for name in ("direction", "robustness", "parameters", "runtime", "budget_curves"):
        figure = getattr(plots, f"plot_{name}")(results)
        destination = args.output / f"{name}.png"
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
