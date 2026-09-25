# Changelog

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
