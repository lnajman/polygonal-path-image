# Authors and provenance

Paula Agregán Reboredo wrote the original Python/Cython PPI implementation
and the accompanying C++ Dijkstra experiment for her 2013–2014 thesis,
*Implementation and development of the technique polygonal path image applied
to guide-wires segmentation in medical images*. The thesis names supervisors
Julio Martín Herrero and Laurent Najman.

Laurent Najman maintains this package. The 2026 modernization was prepared with
Codex assistance: Python 3/Cython 3 compatibility, corrected path reconstruction,
validation, postprocessing APIs, packaging, documentation, and automated tests.

The package is based on the supplied `FinalVersion/PPIpython.pyx`. The earlier
`pythonPPI-V0.9` snapshot was used for comparison. Original source snapshots are
kept in `research/legacy/` for provenance and are not imported by the package or
included in its wheel or source distribution.

The polygonal path image method is described by Vincent Bismuth, Régis Vaillant,
Hugues Talbot, and Laurent Najman in *Curvilinear structure enhancement with the
polygonal path image—Application to guide-wire segmentation in X-ray fluoroscopy*,
MICCAI 2012. This package follows Paula's four-cone implementation; it does not
claim to reproduce every variant in the original research.
