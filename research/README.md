# Research provenance

The maintained package derives from the supplied `FinalVersion/PPIpython.pyx`.
The earlier `pythonPPI-V0.9` source is retained for comparison. `legacy/` contains
byte-for-byte copies of the Python/Cython source and build scripts (plus the
earlier README). These files are historical documents, not runnable package code.

`source-manifest.json` records names, sizes, and SHA-256 checksums for every file
in both original folders. Original folders were preserved without modification.
The full research collection remains in those folders rather than in package
distributions or Git history.

The original collection also contains:

- Paula Agregán Reboredo's 88-page thesis `PFC_paris.pdf`,
- a separate Olena/Milena C++ Dijkstra experiment,
- two grayscale PGM images and saved NumPy voting maps,
- generated C, Python 2/Intel macOS binaries and object files,
- profiling output, backups, and duplicate ZIP archives.

The archive in V0.9 duplicates its loose source exactly. The source in
FinalVersion's `reportandcodes.zip` matches its backup `PPIpython.pyx~`; the loose
source has a corrected Pillow import and added demonstration argument defaults.

The thesis describes both approaches and selects the Python/Cython PPI approach
for its results and performance. The modern package follows its four-cone
implementation. It does not bundle the C++ experiment, research PDF, saved
outputs, old binaries, or historical input images. The shipped example generates
a deterministic synthetic image instead.

See [AUTHORS.md](../AUTHORS.md) for attribution and [migration notes](../docs/migration.md)
for intentional behavioral changes.
