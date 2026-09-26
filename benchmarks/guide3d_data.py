"""Pinned Guide3D acquisition, annotation parsing, and safe image access.

The data remain under CC BY-NC 4.0, independently of this repository's BSD
license. Nothing in this module executes code obtained from the data sources.
Original annotation points are (x, y); public records expose (row, column).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

HF_REVISION = "1234a083c1b85a6f0cb84fb15b6a68c86044d253"
ANNOTATION_REVISION = "e034a71ca5cce0f5c64d67f6fc5302ce98d64481"
ARCHIVE_URL = f"https://huggingface.co/datasets/airvlab/guide3d/resolve/{HF_REVISION}/guide3d.zip"
ANNOTATION_URL = (
    f"https://raw.githubusercontent.com/airvlab/guide3d/{ANNOTATION_REVISION}"
    "/data/annotations/raw.json"
)
ARCHIVE_SIZE = 4_067_349_834
ARCHIVE_SHA256 = "90adb69e69ff8ec128130da33d390cd9b23c51e122d092229c549b1f33d3c236"
ANNOTATION_SHA256 = "5b6fa440c15ca66da276fe3619cd9122e5ab8d2ca662768c737e141991abdfc3"
LICENSE = "CC-BY-NC-4.0"


@dataclass(frozen=True)
class Guide3DRecord:
    """One annotated 2D view; paired views share acquisition and frame IDs."""

    acquisition_id: str
    frame_number: int
    camera: str
    image_path: str
    fluid: int
    guidewire_type: str
    centerline: np.ndarray

    @property
    def case_id(self):
        return f"{self.acquisition_id}:{self.frame_number:04d}:{self.camera}"

    @property
    def pair_id(self):
        return f"{self.acquisition_id}:{self.frame_number:04d}"


def sha256_file(path, *, chunk_size=1024 * 1024):
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Expected a nonempty relative POSIX path")
    result = PurePosixPath(value)
    if result.is_absolute() or ".." in result.parts or ":" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return result


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def read_index(annotation_path):
    """Read manual 2D polylines without modifying or deduplicating their vertices.

    Require paired views and unique acquisition/frame/image identifiers. Repeated
    vertices are legitimate zero-length segments and are retained. This parser
    validates finite coordinates; image-bound checks occur when loading images.
    """
    acquisitions = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    if not isinstance(acquisitions, list) or not acquisitions:
        raise ValueError("Annotations must be a nonempty acquisition list")
    records = []
    task_ids, image_paths = set(), set()
    for acquisition in acquisitions:
        task = acquisition["task"]
        if not isinstance(task, str) or not task or task in task_ids:
            raise ValueError("Acquisition IDs must be unique nonempty strings")
        task_ids.add(task)
        frames = acquisition["frames"]
        if not isinstance(frames, list) or len(frames) != acquisition["frame_count"]:
            raise ValueError(f"Frame count mismatch in {task}")
        frame_ids = set()
        for frame in frames:
            frame_number = _integer(frame["frame_number"], "frame_number")
            if frame_number in frame_ids:
                raise ValueError(f"Duplicate frame number in {task}")
            frame_ids.add(frame_number)
            for camera in ("camera1", "camera2"):
                if camera not in frame:
                    raise ValueError(f"Missing paired {camera} in {task}/{frame_number}")
                annotation = frame[camera]
                image_path = str(_relative_path(annotation["image"]))
                if image_path in image_paths:
                    raise ValueError(f"Duplicate image path: {image_path}")
                image_paths.add(image_path)
                points = np.asarray(annotation["points"], dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
                    raise ValueError(f"Polyline must have shape (N>=2, 2): {image_path}")
                if not np.isfinite(points).all() or (points < 0).any():
                    raise ValueError(f"Invalid polyline coordinates: {image_path}")
                centerline = np.ascontiguousarray(points[:, ::-1])
                centerline.setflags(write=False)
                records.append(
                    Guide3DRecord(
                        task,
                        frame_number,
                        camera,
                        image_path,
                        _integer(acquisition["fluid"], "fluid"),
                        str(acquisition["guidewire_type"]),
                        centerline,
                    )
                )
    return sorted(records, key=lambda record: record.case_id)


def safe_extract(archive_path, destination, *, members=None, max_bytes=32 * 1024**3):
    """Extract regular files, rejecting traversal, symlinks, duplicates and bombs.

    ``members`` optionally limits extraction to exact archive names. Existing
    regular files are replaced; symlink destinations are rejected. ZIP CRCs are
    checked while streaming files, which are renamed only after complete reads.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    requested = None if members is None else set(members)
    with zipfile.ZipFile(archive_path) as archive:
        entries = [
            entry
            for entry in archive.infolist()
            if requested is None or entry.filename in requested
        ]
        names = [entry.filename for entry in entries]
        if requested is not None and requested - set(names):
            raise ValueError("Requested archive members are missing")
        if len(entries) > 100_000 or sum(entry.file_size for entry in entries) > max_bytes:
            raise ValueError("Archive exceeds extraction limits")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member names")
        validated = []
        targets = set()
        for entry in entries:
            relative = _relative_path(entry.filename)
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError(f"Archive member is not a regular file: {entry.filename}")
            target = destination.joinpath(*relative.parts)
            if target in targets:
                raise ValueError("Duplicate normalized archive destinations")
            targets.add(target)
            if target.is_symlink() or not target.resolve().is_relative_to(destination):
                raise ValueError(f"Unsafe extraction destination: {entry.filename}")
            validated.append((entry, target))
        extracted = []
        for entry, target in validated:
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            if temporary.is_symlink():
                raise ValueError("Unsafe temporary extraction destination")
            try:
                with archive.open(entry) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            extracted.append(target)
    return extracted


def download_verified(
    url, destination, expected_sha256, *, expected_size=None, max_bytes=5 * 1024**3
):
    """Resume a streaming HTTPS download, then check size and SHA-256 before use.

    An interrupted transfer keeps its ``.part`` file. A server that ignores a
    Range request safely restarts the transfer. Existing final files must pass
    the same integrity checks. This function never reads credentials.
    """
    if not url.startswith("https://"):
        raise ValueError("Dataset downloads require HTTPS")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise ValueError("Existing download has unexpected size")
        if sha256_file(destination) != expected_sha256:
            raise ValueError("Existing download failed SHA-256 verification")
        return destination
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > max_bytes:
        raise ValueError("Partial download exceeds size limit")
    # A complete .part file may be left after interruption before its rename.
    if expected_size is None or offset != expected_size:
        headers = {"User-Agent": "polygonal-path-image Guide3D benchmark"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status == 206:
                if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected byte range")
                mode = "ab"
            elif response.status == 200:
                offset, mode = 0, "wb"
            else:
                raise ValueError(f"Unexpected download status: {response.status}")
            with partial.open(mode) as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    offset += len(block)
                    if offset > max_bytes or (expected_size is not None and offset > expected_size):
                        raise ValueError("Download exceeds expected size")
                    output.write(block)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise ValueError("Downloaded file has unexpected size")
    if sha256_file(partial) != expected_sha256:
        raise ValueError("Downloaded file failed SHA-256 verification")
    partial.replace(destination)
    return destination


def resolve_image_path(record, data_root):
    """Resolve an annotation's image path beneath an explicit image directory."""
    root = Path(data_root).resolve()
    relative = _relative_path(record.image_path)
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Image path escapes the dataset root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_image(record, data_root):
    """Load original grayscale uint8 pixels; no contrast or coordinate changes."""
    from PIL import Image

    with Image.open(resolve_image_path(record, data_root)) as source:
        image = np.array(source)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected an unmodified grayscale uint8 Guide3D image")
    if (record.centerline >= np.asarray(image.shape)[None, :]).any():
        raise ValueError(f"Annotation lies outside image bounds: {record.case_id}")
    return image


def audit_dataset(annotation_path, data_root, *, hash_images=True):
    """Audit all PNGs and manual polylines, preserving original observations.

    Pixel hashes detect exact duplicate decoded images, not approximate temporal
    similarity. A full decode checks image mode, dimensions and bounds. Hashing
    is optional for quick local checks; published audits should leave it enabled.
    """
    from collections import Counter, defaultdict

    from PIL import Image

    records = read_index(annotation_path)
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    images = sorted(root.rglob("*.png"))
    actual = {str(path.relative_to(root)) for path in images}
    expected = {record.image_path for record in records}
    shape_counts, mode_counts, dtype_counts = Counter(), Counter(), Counter()
    pixel_hashes, file_hashes = defaultdict(list), defaultdict(list)
    shapes = {}
    global_min, global_max = 255, 0
    for path in images:
        relative = str(path.relative_to(root))
        with Image.open(path) as source:
            mode_counts[source.mode] += 1
            image = np.array(source)
        shape_counts[str(tuple(image.shape))] += 1
        dtype_counts[str(image.dtype)] += 1
        shapes[relative] = image.shape
        global_min = min(global_min, int(image.min()))
        global_max = max(global_max, int(image.max()))
        if hash_images:
            file_hashes[sha256_file(path)].append(relative)
            pixel_hashes[hashlib.sha256(image.tobytes()).hexdigest()].append(relative)
    repeated_vertices = []
    invalid_bounds = []
    polylines = defaultdict(list)
    acquisition_counts = Counter(record.acquisition_id for record in records)
    for record in records:
        points = record.centerline
        repeated = int(np.sum(np.all(np.diff(points, axis=0) == 0, axis=1)))
        if repeated:
            repeated_vertices.append({"case_id": record.case_id, "zero_length_segments": repeated})
        shape = shapes.get(record.image_path)
        if shape is not None and (
            len(shape) != 2 or (points >= np.asarray(shape[:2])[None, :]).any()
        ):
            invalid_bounds.append(record.case_id)
        polylines[hashlib.sha256(points.tobytes()).hexdigest()].append(record.case_id)
    duplicate_polylines = [cases for cases in polylines.values() if len(cases) > 1]
    duplicate_images = [paths for paths in pixel_hashes.values() if len(paths) > 1]
    duplicate_files = [paths for paths in file_hashes.values() if len(paths) > 1]
    return {
        "schema_version": 1,
        "dataset": "Guide3D",
        "license": LICENSE,
        "huggingface_revision": HF_REVISION,
        "annotation_revision": ANNOTATION_REVISION,
        "archive_expected_sha256": ARCHIVE_SHA256,
        "annotation_actual_sha256": sha256_file(annotation_path),
        "image_count": len(images),
        "annotated_image_count": len(records),
        "paired_frame_count": len({record.pair_id for record in records}),
        "annotated_acquisition_count": len(acquisition_counts),
        "annotated_images_per_acquisition": dict(sorted(acquisition_counts.items())),
        "missing_images": sorted(expected - actual),
        "unannotated_images": sorted(actual - expected),
        "image_shape_counts": dict(shape_counts),
        "image_mode_counts": dict(mode_counts),
        "image_dtype_counts": dict(dtype_counts),
        "pixel_min": global_min if images else None,
        "pixel_max": global_max if images else None,
        "out_of_bounds_annotations": invalid_bounds,
        "annotations_with_zero_length_segments": repeated_vertices,
        "zero_length_segment_count": sum(
            item["zero_length_segments"] for item in repeated_vertices
        ),
        "duplicate_polyline_groups": duplicate_polylines,
        "duplicate_polyline_excess_count": sum(len(group) - 1 for group in duplicate_polylines),
        "image_hashing_performed": hash_images,
        "duplicate_pixel_image_groups": duplicate_images if hash_images else None,
        "duplicate_file_groups": duplicate_files if hash_images else None,
        "duplicate_pixel_image_excess_count": sum(len(group) - 1 for group in duplicate_images)
        if hash_images
        else None,
        "duplicate_file_excess_count": sum(len(group) - 1 for group in duplicate_files)
        if hash_images
        else None,
        "notes": [
            "Coordinates are exposed as (row, column), converted from upstream (x, y).",
            "Repeated vertices and repeated temporal annotations are retained.",
            "Image hashing detects exact duplicates only; neighboring frames are not independent samples.",
            "Only manually annotated acquisitions are eligible for supervised evaluation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--extract", action="store_true", help="Extract the verified archive")
    parser.add_argument(
        "--audit", action="store_true", help="Audit all extracted images and annotations"
    )
    arguments = parser.parse_args()
    archive = download_verified(
        ARCHIVE_URL,
        arguments.destination / "guide3d.zip",
        ARCHIVE_SHA256,
        expected_size=ARCHIVE_SIZE,
    )
    annotation = download_verified(
        ANNOTATION_URL,
        arguments.destination / "raw.json",
        ANNOTATION_SHA256,
        max_bytes=50 * 1024**2,
    )
    records = read_index(annotation)
    print(
        f"Verified {len(records)} annotated views in {len({r.acquisition_id for r in records})} acquisitions"
    )
    if arguments.extract:
        files = safe_extract(archive, arguments.destination / "extracted")
        print(f"Extracted {len(files)} regular files")
    if arguments.audit:
        report = audit_dataset(annotation, arguments.destination / "extracted" / "guide3d")
        report["archive_actual_sha256"] = ARCHIVE_SHA256
        report["archive_size_bytes"] = archive.stat().st_size
        report["archive_integrity_verified"] = True
        output = arguments.destination / "audit.json"
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote audit to {output}")


if __name__ == "__main__":
    main()
