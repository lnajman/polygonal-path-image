# Changelog

## Unreleased

- Added source-path diagnostics for weak/strong-line pairs and uniform images,
  with exact cost/vote reconciliation, boundary/tie controls and an executed
  notebook. Clarified that nonzero blank votes are expected path occupancy.
- Added a research-only straight-candidate fallback after tortuosity rejection,
  with separate development calibration and a predefined fresh validation panel.
- Added a complete-reference multi-line synthetic study with crossings, branches,
  mixed widths, paired weak/strong lines, development-frozen thresholds, Frangi
  and local-contrast comparisons, continuity metrics and blank/noise controls.
- Corrected Guide3D interpretation: off-guidewire selections are unclassified
  for all-line enhancement, not established false positives. Historical
  measurements and the frozen acquisition protocol remain unchanged.
- Added a Guide3D acquired-image phantom pilot with pinned data auditing,
  acquisition-separated development/evaluation, local-contrast comparisons,
  exact accelerated geometry metrics, scalar results and an executed notebook.
- Added optional study dependencies and fixture/notebook CI without dataset downloads.
- Clarified that the original MICCAI 2012 clinical dataset was private.
- Added a reproducible synthetic benchmark with continuous reference curves,
  explicit selection budgets, noise-only controls, and an image-darkness baseline.
- Added independent geometry/metric tests and a runnable parameter tutorial,
  including saved figures, source hashes, full measurements, and notebook CI.

## 0.1.0

- Initial Python 3 package based on the FinalVersion implementation by
  Paula Agregán Reboredo, Vincent Bismuth, and Laurent Najman.
- Corrected path reconstruction with immutable dynamic-programming backpointers.
- Added validated array-based computation and deterministic postprocessing APIs.
- Added a synthetic example, migration guide, attribution, and citation metadata.
- Added exhaustive small-image correctness tests and installed-wheel CI.
- Added isolated source/wheel builds compatible with NumPy 1.26 and NumPy 2.
- Validated full-resolution synthetic and historical images with independent
  path-cost and geometry audits; published reproducible results and method limits.
- Added 20 tested release wheels for CPython 3.10–3.14 across Linux, Windows,
  and both macOS architectures, plus an explicit PyPI Trusted Publishing workflow.
