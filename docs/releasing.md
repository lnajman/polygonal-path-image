# Building and releasing

The `Wheels` workflow builds an isolated source distribution, then builds every
wheel from that same archive. It runs the package's tests against the installed
wheel outside the source tree. All wheels run the suite with NumPy 2; Python
3.10–3.12 wheels also run with NumPy 1.26.4.

## Supported binary distributions

| Platform | Architecture | Build runner | Python versions |
| --- | --- | --- | --- |
| Linux, glibc 2.28+ | x86-64 | `ubuntu-24.04`, manylinux_2_28 container | CPython 3.10–3.14 |
| Windows | x86-64 | `windows-2022`, MSVC | CPython 3.10–3.14 |
| macOS | Intel x86-64 | `macos-15-intel`, Xcode | CPython 3.10–3.14 |
| macOS | Apple Silicon arm64 | `macos-14`, Xcode | CPython 3.10–3.14 |

The workflow produces 20 wheels and one source distribution. Every architecture
is tested natively. Linux ARM, Windows ARM, musl-based Linux (including Alpine),
PyPy, and free-threaded Python are not part of this binary release matrix. The
minimum macOS version is encoded in each wheel filename; pip selects a compatible
wheel automatically. The NumPy release selected by pip may have its own higher
minimum operating-system requirement.

The runner labels are listed in [GitHub's runner documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job).
Build selection and isolated wheel testing use
[cibuildwheel](https://cibuildwheel.pypa.io/en/stable/options/).

## Validate before tagging

1. Update package version, citation version/date, and changelog; commit the final
   source and documentation to `main`.
2. Run `gh workflow run wheels.yml --ref main` and inspect that specific run.
   The regular `Tests` workflow must also pass for the same commit.
3. Download the combined `release-dist` artifact from the successful Wheels run:

   ```bash
   gh run download RUN_ID --name release-dist --dir release-dist
   python scripts/check_release_artifacts.py release-dist \
       --version 0.1.0 --complete --check-checksums
   ```

Replace `RUN_ID` and `0.1.0` with the selected run and version. The final artifact
check verifies all 20 platform/interpreter combinations, embedded versions,
authors, license files, compiled extension presence, source contents, and SHA-256
checksums. Artifacts expire after 90 days; permanent release assets are attached
in the next step.

## Publish a GitHub release

After validation, create and push an annotated version tag on that same commit:

```bash
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

The tag starts a fresh Wheels run and must match the version in `pyproject.toml`.
Wait for both Wheels and Tests to succeed at the tag's commit. Download the
`release-dist` artifact **from the tag-triggered run**, verify it as above, then
attach all distributions and `SHA256SUMS` to a GitHub release:

```bash
gh release create v0.1.0 --verify-tag --title "v0.1.0" \
    --notes-file release-notes.md release-dist/*
```

Prepare `release-notes.md` from the changelog and validation findings. Include
known scientific limitations; automated correctness checks do not establish
reproduction of the published MICCAI results.

GitHub assets can be used before PyPI publication. Download the wheel matching
the user's Python and operating system, then install it with:

```bash
python -m pip install ./polygonal_path_image-0.1.0-<python>-<abi>-<platform>.whl
```

Replace the placeholder filename with the downloaded wheel's actual filename.
The wheel already contains the extension and does not require a C compiler.
The `.tar.gz` source archive still requires a compiler and isolated build
dependencies when installed.

## Configure PyPI Trusted Publishing once

A PyPI project owner must register a
[pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
for a new project, or
[add a publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
to the existing project:

| PyPI field | Value |
| --- | --- |
| Project name | `polygonal-path-image` |
| Owner | `lnajman` |
| Repository | `polygonal-path-image` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

Create the matching GitHub environment `pypi`. It can restrict deployments to
`main` and require a maintainer's review. No PyPI password or API token belongs
in repository secrets. The short-lived upload credential is issued through
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Publish to PyPI explicitly

After publishing the GitHub release and configuring Trusted Publishing, dispatch
the separate publishing workflow **from `main`**, naming the release tag and its
successful, tag-triggered Wheels run:

```bash
gh workflow run publish.yml --ref main \
    -f tag=v0.1.0 -f wheel_run_id=RUN_ID
```

The workflow verifies that the tag names a stable, published GitHub release and
that the specified run belongs to this repository's `wheels.yml`, was triggered
by a push, succeeded, and built the exact tagged commit. It downloads only that
run's artifacts, checks versions and checksums, and uploads the verified package
files in a separate job with `id-token: write` and the `pypi` environment. The
publishing job does not check out or execute project code. PyPI digital
attestations are enabled.

A branch push, pull request, wheel build, or GitHub release alone does not publish
to PyPI. If PyPI setup has not been completed, GitHub releases remain usable.
Once a version is published to PyPI, its distribution files cannot be replaced;
a correction needs a new version. For a partial network failure, inspect PyPI
before retrying: already uploaded filenames will be rejected rather than skipped.

After the successful publishing run, test installation in a clean environment
without falling back to compilation:

```bash
python -m pip install --only-binary=:all: polygonal-path-image==0.1.0
python -c "import numpy as np; from polygonal_path_image import compute_ppi; print(compute_ppi(np.zeros((8, 8), dtype=np.uint8), 1, 2)[0].shape)"
```
