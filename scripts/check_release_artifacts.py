"""Check release coverage and metadata without importing or installing the package.

Use Python 3.11+; this tool uses only the standard library. Wheels are tested by
cibuildwheel before reaching this check. This script checks their packaging, not
whether their compiled code runs on a different operating system.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import tomllib

PACKAGE = "polygonal_path_image"
PYTHONS = {f"cp3{minor}" for minor in range(10, 15)}
PLATFORMS = {"linux-x86_64", "windows-x86_64", "macos-x86_64", "macos-arm64"}
AUTHORS = ("Paula Agregán Reboredo", "Vincent Bismuth", "Laurent Najman")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_metadata(data: bytes, version: str, filename: str) -> None:
    metadata = BytesParser(policy=default).parsebytes(data)
    name = metadata["Name"] or ""
    require(name.replace("-", "_") == PACKAGE, f"{filename}: wrong package name")
    require(metadata["Version"] == version, f"{filename}: wrong metadata version")
    require(metadata["Requires-Python"] == ">=3.10", f"{filename}: wrong Python requirement")
    require(metadata["License-Expression"] == "BSD-3-Clause", f"{filename}: wrong license")
    author = str(metadata["Author"] or "")
    require(all(name in author for name in AUTHORS), f"{filename}: missing author credit")


def wheel_platform(platform_tags: str) -> str:
    tags = platform_tags.split(".")
    if all(
        re.fullmatch(r"manylinux(?:1|2010|2014|_2_(?:[0-9]|1[0-9]|2[0-8]))_x86_64", tag)
        for tag in tags
    ):
        return "linux-x86_64"
    if tags == ["win_amd64"]:
        return "windows-x86_64"
    for arch in ("x86_64", "arm64"):
        if all(re.fullmatch(rf"macosx_\d+_\d+_{arch}", tag) for tag in tags):
            return f"macos-{arch}"
    raise ValueError(f"Unexpected wheel platform: {platform_tags}")


def check_wheel(path: Path, version: str) -> tuple[str, str]:
    parts = path.stem.split("-")
    require(len(parts) == 5, f"{path.name}: expected five wheel filename components")
    name, wheel_version, python_tag, abi_tag, platform_tags = parts
    require(name == PACKAGE, f"{path.name}: wrong wheel name")
    require(wheel_version == version, f"{path.name}: wrong filename version")
    require(python_tag in PYTHONS, f"{path.name}: unsupported Python tag")
    require(abi_tag == python_tag, f"{path.name}: unexpected ABI tag")
    platform = wheel_platform(platform_tags)
    with zipfile.ZipFile(path) as wheel:
        require(wheel.testzip() is None, f"{path.name}: corrupt ZIP entry")
        members = wheel.namelist()
        info = f"{PACKAGE}-{version}.dist-info"
        check_metadata(wheel.read(f"{info}/METADATA"), version, path.name)
        require(f"{info}/licenses/LICENSE" in members, f"{path.name}: missing license file")
        require(f"{info}/licenses/AUTHORS.md" in members, f"{path.name}: missing credits")
        require(
            any(
                entry.startswith(f"{PACKAGE}/_core.") and entry.endswith((".so", ".pyd"))
                for entry in members
            ),
            f"{path.name}: missing compiled extension",
        )
    return python_tag, platform


def check_sdist(path: Path, version: str) -> None:
    require(path.name == f"{PACKAGE}-{version}.tar.gz", f"{path.name}: wrong source filename")
    root = f"{PACKAGE}-{version}"
    with tarfile.open(path, "r:gz") as archive:
        members = set(archive.getnames())
        for required in (
            "pyproject.toml",
            "setup.py",
            "LICENSE",
            "AUTHORS.md",
            "CITATION.cff",
            "src/polygonal_path_image/_core.pyx",
            "tests/test_core.py",
            "tests/reference.py",
            "scripts/validate_research.py",
        ):
            require(f"{root}/{required}" in members, f"{path.name}: missing {required}")
        package_info = archive.extractfile(f"{root}/PKG-INFO")
        require(package_info is not None, f"{path.name}: missing PKG-INFO")
        check_metadata(package_info.read(), version, path.name)
        project_file = archive.extractfile(f"{root}/pyproject.toml")
        require(project_file is not None, f"{path.name}: missing pyproject.toml")
        project = tomllib.loads(project_file.read().decode("utf-8"))
        require(project["project"]["version"] == version, f"{path.name}: wrong source version")
        require(
            "cibuildwheel" in project.get("tool", {}), f"{path.name}: missing wheel test config"
        )
        require(
            not any(member.startswith(f"{root}/research/") for member in members),
            f"{path.name}: research archive should not be in the distribution",
        )


def check_artifacts(directory: Path, version: str, complete: bool) -> list[Path]:
    require(directory.is_dir(), f"Missing distribution directory: {directory}")
    files = sorted(directory.iterdir())
    packages = [path for path in files if path.suffix == ".whl" or path.name.endswith(".tar.gz")]
    require(bool(packages), "No package distributions found")
    unexpected = [path.name for path in files if path not in packages and path.name != "SHA256SUMS"]
    require(not unexpected, f"Unexpected release files: {unexpected}")
    seen = set()
    sdists = 0
    for path in packages:
        require(path.is_file() and not path.is_symlink(), f"Not a regular file: {path}")
        if path.suffix == ".whl":
            target = check_wheel(path, version)
            require(target not in seen, f"Duplicate platform/Python combination: {target}")
            seen.add(target)
        else:
            check_sdist(path, version)
            sdists += 1
    require(sdists == 1, f"Expected one source distribution, found {sdists}")
    if complete:
        expected = {(python, platform) for python in PYTHONS for platform in PLATFORMS}
        require(seen == expected, f"Incomplete wheel coverage; missing {sorted(expected - seen)}")
    return packages


def checksums(packages: list[Path]) -> str:
    return "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in packages
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--complete", action="store_true", help="Require all 20 supported wheels")
    checksum_mode = parser.add_mutually_exclusive_group()
    checksum_mode.add_argument("--write-checksums", action="store_true")
    checksum_mode.add_argument("--check-checksums", action="store_true")
    args = parser.parse_args()
    try:
        packages = check_artifacts(args.directory, args.version, args.complete)
        manifest = args.directory / "SHA256SUMS"
        expected = checksums(packages)
        if args.check_checksums:
            require(manifest.read_text(encoding="ascii") == expected, "SHA256SUMS does not match")
        if args.write_checksums:
            manifest.write_text(expected, encoding="ascii")
    except (ValueError, OSError, KeyError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Release validation failed: {error}\n")
    print(f"Verified {len(packages)} distributions for version {args.version}")


if __name__ == "__main__":
    main()
