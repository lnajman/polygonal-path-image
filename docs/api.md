# API and algorithm conventions

All coordinates use `(row, column)`. Import public functions from
`polygonal_path_image`; `_core` is a private implementation detail.

## compute_ppi(image, segment_length=3, nb_segments=10)

Accepts a nonempty 2D uint8 NumPy array, including noncontiguous and read-only
arrays. Length parameters must be positive integers, not booleans. Returns
`(costs, paths)` with shapes `(H, W)` and `(H, W, nb_segments, 2)`.

Each path stays inside a single cone. For length L, endpoints in tie-breaking
order are:

| Cone | Relative endpoints |
| --- | --- |
| Up (H) | `(-L, c)` for `c = L, ..., -L` |
| Down (B) | `(L, c)` for `c = -L, ..., L` |
| Right (E) | `(r, L)` for `r = L, ..., -L` |
| Left (W) | `(r, -L)` for `r = -L, ..., L` |

The last equal-cost endpoint wins within a cone; the first equal-cost cone wins
in H, B, E, W order. Segment cost is the sum along the Bresenham rasterization,
excluding its starting pixel and including its ending pixel. This is a raster
length convention, not Euclidean length normalization.

Dynamic programming keeps immutable per-iteration backpointers. Reconstruction
follows those layers rather than reading paths overwritten by neighboring pixels.
The legacy cost definition and tie rules are preserved. Impossible paths have
infinite cost and all endpoints equal to `(-1, -1)`.

The source intensity is omitted from the whole path's cost. Add
`image.astype(float)` to the returned cost image if all-pixel cost rankings are
needed across sources, for example before pruning. See the
[paper comparison and validation](validation.md).

## bresenham_line(row0, col0, row1, col1)

Returns a list of integer pixel coordinates, including both endpoints. Signed
coordinates are supported; this helper does not clip against an image. Half-pixel
ties use the historical convention and need not rasterize identically after
reversing a segment.

## voting(costs, paths, *, border=0)

Returns `(votes, inverted_votes)`, both int64 arrays. Every finite-cost path votes
for each raster pixel it visits. Shared joints count once. An optional nonnegative
integer `border` zeros that many outer rows and columns. Inversion is computed
after border clearing as `votes.max() - votes`.

## filter_tortuosity(costs, paths, threshold=0.75)

Returns a copy of the cost image with rejected paths marked `inf`. The straightness
score is the product of cosine values between every consecutive pair of segments,
including the first segment from the source pixel. A score below `threshold` is
rejected. The threshold lies in `[-1, 1]`; paths with a zero-length segment are
rejected. A nondegenerate one-segment path has score 1.
Equality is retained with a small roundoff tolerance scaled to the number of
turns; see the scientific comparison for the paper's strict threshold convention.

## orientation(costs, paths)

Returns an axial orientation image in radians modulo pi. Zero is horizontal;
pi/2 is vertical. Evidence comes from segments of finite-cost paths that pass
through each pixel, weighted by squared segment length and combined using
double-angle averaging. Opposite directions describe the same axis. Uncovered
pixels or exactly cancelling evidence have `NaN` orientation. Shared joints
belong to the preceding segment. This function returns data; it does not plot.

## prune_paths(costs, paths, *, fraction=0.4, distance=10.0)

Greedily retains the lowest-cost active path and suppresses similar paths. A path
is suppressed if at least `fraction` of its endpoint vertices have Euclidean
distance strictly less than `distance` from any endpoint of a retained path.
The source pixel is not an endpoint vertex. Ties are resolved by source row-major
order. The fraction is in `(0, 1]`; distance is finite and nonnegative.

Returns a new cost image; suppressed entries are `inf`. This deterministic global
comparison replaces the old demonstration's local search-window heuristic and
random RGB output. It can be expensive for large images.

## Common postprocessing validation

Cost images must be nonempty 2D real arrays containing finite values or positive
infinity. NaN and negative infinity are rejected. Endpoint arrays must be integer
arrays of shape `(H, W, K, 2)` with positive K. Coordinates belonging to
finite-cost paths must lie inside the image. Inactive paths may contain the
`-1` sentinel or their original coordinates after filtering. Inputs are not mutated.
