# Authors and provenance

The authors of this package and its original Python/Cython PPI implementation are:

- Paula Agregán Reboredo
- Vincent Bismuth
- Laurent Najman

The original implementation was developed during Paula Agregán Reboredo's
student work. Vincent Bismuth and Laurent Najman were her advisors and also
contributed to the code. The work is documented in her 2013–2014 thesis,
*Implementation and development of the technique polygonal path image applied
to guide-wires segmentation in medical images*.

Laurent Najman maintains this package. The 2026 modernization was prepared with
Codex assistance: Python 3/Cython 3 compatibility, corrected path reconstruction,
validation, postprocessing APIs, packaging, documentation, and automated tests.

The package is based on the supplied `FinalVersion/PPIpython.pyx`. The earlier
`pythonPPI-V0.9` snapshot was used for comparison. Original source snapshots are
kept in `research/legacy/` for provenance and are not imported by the package or
included in its wheel or source distribution.

The polygonal path image method is described by Vincent Bismuth, Régis Vaillant,
Hugues Talbot, and Laurent Najman in
[*Curvilinear Structure Enhancement with the Polygonal Path Image - Application
to Guide-Wire Segmentation in X-Ray Fluoroscopy*](https://hal.science/hal-00741956v1/),
MICCAI 2012, Part II, LNCS 7511, pp. 9–16.
DOI: [10.1007/978-3-642-33418-4_2](https://doi.org/10.1007/978-3-642-33418-4_2).
This package follows the original four-cone implementation; it does not
claim to reproduce every variant in the original research.
