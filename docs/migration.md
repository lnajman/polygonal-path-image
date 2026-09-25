# Migrating the historical PPI code

The supported import is now `polygonal_path_image`. This is a new versioned API;
the old `PPIpython` import and plotting demonstration functions are not installed.

| Historical usage | Modern usage |
| --- | --- |
| `PPIpython.compute_ppi(image, length, count)` | `compute_ppi(image, length, count)` |
| `nb_segment=...` keyword | `nb_segments=...` |
| `voting(costs, paths)` with automatic border removal | `voting(costs, paths, border=2)` |
| `voting_without_tortuosity(costs, paths, tortuosityMin)` | `filter_tortuosity(costs, paths, threshold=...)` |
| `orientation(image, costs, paths)` draws arrows | `orientation(costs, paths)` returns an angle image |
| `pruning(...)` mutates costs and returns random colors | `prune_paths(...)` returns a filtered cost image |
| `example()` reads, plots, profiles, and writes fixed names | `examples/synthetic_curve.py` with explicit output directory |

## Correctness changes

- **Reconstruction fixed:** the historical recurrence copied costs per iteration
  but read and overwrote coordinates in place. A deterministic 8-by-8 example
  returned 7 coordinate paths inconsistent with their reported costs. Immutable
  backpointers now reconstruct the same optimal cost using consistent endpoints.
- **Impossible paths:** all endpoint coordinates are now `-1`, rather than a mix
  of zeros, stale coordinates, or partial sentinels.
- **Precise costs:** accumulation uses float64 rather than the legacy float32
  temporaries, avoiding premature precision loss on longer paths.
- **Signed raster coordinates:** Bresenham no longer reflects negative coordinates
  through `abs()`. Paths inside images retain the historical raster convention.
- **Tortuosity:** the first segment from the source pixel now participates in
  the score. Degenerate segments are rejected without division by zero.
- **Voting:** no border is removed by default. `border=2` requests the historical
  border width. Inversion uses the maximum after clearing the border.
- **Orientation:** computes a full numeric axial field using double-angle
  averaging instead of a sampled plotting routine with quadrant ambiguities.
- **Pruning:** uses global, deterministic endpoint comparisons, without the old
  local-window heuristic, input mutation, or random drawing.

The cone endpoint order and within-cone/across-cone tie rules of FinalVersion are
preserved. Postprocessing corrections deliberately change some historical outputs;
old saved voting images are not treated as unquestioned correctness fixtures.

## Build changes

The package uses Python 3, Cython 3, NumPy 1.26/2-compatible types, and modern
isolated builds. Historical `.so`, `.o`, generated C, profiling files, backups,
PDFs, and ZIPs are not installed. The old code is preserved only for provenance
under `research/legacy/` and is excluded from package distributions.
