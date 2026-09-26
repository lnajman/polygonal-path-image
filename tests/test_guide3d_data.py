"""Small fixtures for pinned data parsing, paired identities and safe IO."""

import copy
import errno
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.guide3d_data import (
    download_verified,
    load_image,
    read_index,
    resolve_image_path,
    safe_extract,
)


def annotation():
    return [
        {
            "task": "0-bca-angle-1",
            "fluid": 0,
            "guidewire_type": "angle",
            "video_number": 1,
            "frame_count": 1,
            "frames": [
                {
                    "frame_number": 0,
                    "camera1": {
                        "image": "0-bca-angle-1-2/000.png",
                        "points": [[1, 2], [3, 4], [3, 4]],
                    },
                    "camera2": {"image": "0-bca-angle-1-1/000.png", "points": [[2, 1], [4, 3]]},
                }
            ],
        }
    ]


def write_annotation(tmp_path, data):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_reader_preserves_camera_mapping_pairing_and_repeated_vertices(tmp_path):
    records = read_index(write_annotation(tmp_path, annotation()))
    assert len(records) == 2
    assert records[0].camera == "camera1"
    assert records[0].image_path.endswith("-2/000.png")
    assert records[0].pair_id == records[1].pair_id
    assert records[0].case_id != records[1].case_id
    np.testing.assert_array_equal(records[0].centerline, [[2, 1], [4, 3], [4, 3]])
    assert not records[0].centerline.flags.writeable


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_camera",
        "duplicate_image",
        "duplicate_frame",
        "duplicate_acquisition",
        "frame_count",
        "nan",
        "negative",
        "singleton",
        "traversal",
    ],
)
def test_reader_rejects_ambiguous_or_invalid_annotations(tmp_path, corruption):
    data = annotation()
    frame = data[0]["frames"][0]
    if corruption == "missing_camera":
        del frame["camera2"]
    elif corruption == "duplicate_image":
        frame["camera2"]["image"] = frame["camera1"]["image"]
    elif corruption == "duplicate_frame":
        data[0]["frames"].append(copy.deepcopy(frame))
        data[0]["frame_count"] = 2
    elif corruption == "duplicate_acquisition":
        data.append(copy.deepcopy(data[0]))
    elif corruption == "frame_count":
        data[0]["frame_count"] = 2
    elif corruption == "nan":
        frame["camera1"]["points"][0][0] = float("nan")
    elif corruption == "negative":
        frame["camera1"]["points"][0][0] = -1
    elif corruption == "singleton":
        frame["camera1"]["points"] = [[1, 2]]
    elif corruption == "traversal":
        frame["camera1"]["image"] = "../secret.png"
    with pytest.raises(ValueError):
        read_index(write_annotation(tmp_path, data))


def make_archive(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members:
            archive.writestr(name, contents)
    return path


def test_safe_extraction_streams_selected_files(tmp_path):
    archive = make_archive(
        tmp_path / "data.zip", [("images/a.png", b"abc"), ("images/b.png", b"def")]
    )
    root = tmp_path / "output"
    files = safe_extract(archive, root, members=["images/a.png"])
    assert files == [root / "images/a.png"]
    assert files[0].read_bytes() == b"abc"
    assert not (root / "images/b.png").exists()
    with pytest.raises(ValueError, match="missing"):
        safe_extract(archive, root, members=["absent"])
    with pytest.raises(ValueError, match="limits"):
        safe_extract(archive, root, max_bytes=1)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute", "dir\\escape"])
def test_safe_extraction_rejects_unsafe_paths(tmp_path, name):
    archive = make_archive(tmp_path / "data.zip", [(name, b"a")])
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "output")


def make_directory_symlink(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        if error.errno in (errno.EPERM, errno.EACCES) or getattr(error, "winerror", None) == 1314:
            pytest.skip("Creating filesystem symlinks is not permitted on this platform")
        raise


def test_safe_extraction_rejects_zip_and_destination_symlinks(tmp_path):
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = make_archive(tmp_path / "data.zip", [(info, b"../outside")])
    with pytest.raises(ValueError, match="regular"):
        safe_extract(archive, tmp_path / "output")
    root = tmp_path / "output"
    outside = tmp_path / "outside"
    outside.mkdir()
    make_directory_symlink(root / "images", outside)
    archive = make_archive(tmp_path / "data.zip", [("images/image.png", b"a")])
    with pytest.raises(ValueError, match="destination"):
        safe_extract(archive, root)
    assert not (outside / "image.png").exists()


def test_safe_extraction_rejects_duplicate_member_names(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate"):
        archive = make_archive(tmp_path / "data.zip", [("a", b"one"), ("a", b"two")])
    with pytest.raises(ValueError, match="Duplicate"):
        safe_extract(archive, tmp_path / "output")


def test_image_loader_keeps_pixel_values_and_checks_bounds(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    records = read_index(write_annotation(tmp_path, annotation()))
    image = np.arange(36, dtype=np.uint8).reshape(6, 6)
    path = tmp_path / records[0].image_path
    path.parent.mkdir()
    Image.fromarray(image).save(path)
    np.testing.assert_array_equal(load_image(records[0], tmp_path), image)
    Image.fromarray(image[:3, :3]).save(path)
    with pytest.raises(ValueError, match="bounds"):
        load_image(records[0], tmp_path)
    Image.fromarray(np.repeat(image[:, :, None], 3, axis=2)).save(path)
    with pytest.raises(ValueError, match="grayscale"):
        load_image(records[0], tmp_path)


def test_image_path_rejects_symlink_escape(tmp_path):
    record = read_index(write_annotation(tmp_path, annotation()))[0]
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "000.png").write_bytes(b"not an image")
    make_directory_symlink(data_root / Path(record.image_path).parts[0], outside)
    with pytest.raises(ValueError, match="escapes"):
        resolve_image_path(record, data_root)


class Response(io.BytesIO):
    def __init__(self, data, status=200, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {} if content_range is None else {"Content-Range": content_range}


@pytest.mark.parametrize("resume", [False, True])
def test_download_resumes_or_safely_restarts_when_range_ignored(tmp_path, monkeypatch, resume):
    content = b"abcdef"
    target = tmp_path / "data.bin"
    target.with_name("data.bin.part").write_bytes(b"abc")

    def response(request, timeout):
        assert request.headers["Range"] == "bytes=3-"
        return Response(b"def", 206, "bytes 3-5/6") if resume else Response(content)

    monkeypatch.setattr("urllib.request.urlopen", response)
    download_verified(
        "https://example.invalid/data", target, hashlib.sha256(content).hexdigest(), expected_size=6
    )
    assert target.read_bytes() == content
    assert not target.with_name("data.bin.part").exists()


@pytest.mark.parametrize("failure", ["hash", "size", "range", "limit"])
def test_download_integrity_errors_do_not_publish_final_file(tmp_path, monkeypatch, failure):
    target = tmp_path / "data.bin"
    content = b"abcdef"
    expected_size = 7 if failure == "size" else 6
    expected_hash = "incorrect" if failure == "hash" else hashlib.sha256(content).hexdigest()
    if failure == "range":
        target.with_name("data.bin.part").write_bytes(b"abc")

        def response(request, timeout):
            return Response(b"ef", 206, "bytes 4-5/6")
    else:

        def response(request, timeout):
            return Response(content)

    monkeypatch.setattr("urllib.request.urlopen", response)
    with pytest.raises(ValueError):
        download_verified(
            "https://example.invalid/data",
            target,
            expected_hash,
            expected_size=expected_size,
            max_bytes=2 if failure == "limit" else 10,
        )
    assert not target.exists()


def test_audit_counts_missing_unannotated_and_duplicate_images(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from benchmarks.guide3d_data import audit_dataset

    data = annotation()
    annotation_path = write_annotation(tmp_path, data)
    record = read_index(annotation_path)[0]
    image_root = tmp_path / "images"
    image_path = image_root / record.image_path
    image_path.parent.mkdir(parents=True)
    pixels = np.arange(36, dtype=np.uint8).reshape(6, 6)
    Image.fromarray(pixels).save(image_path)
    Image.fromarray(pixels).save(image_root / "unannotated.png")
    report = audit_dataset(annotation_path, image_root)
    assert report["image_count"] == 2
    assert report["annotated_image_count"] == 2
    assert report["paired_frame_count"] == 1
    assert report["annotated_acquisition_count"] == 1
    assert report["missing_images"] == [data[0]["frames"][0]["camera2"]["image"]]
    assert report["unannotated_images"] == ["unannotated.png"]
    assert report["duplicate_pixel_image_excess_count"] == 1
    assert report["zero_length_segment_count"] == 1
    assert report["out_of_bounds_annotations"] == []
    assert report["pixel_min"] == 0
    assert report["pixel_max"] == 35
    assert report["image_dtype_counts"] == {"uint8": 2}


def test_safe_extraction_rejects_normalized_path_aliases(tmp_path):
    archive = make_archive(tmp_path / "data.zip", [("a//b", b"one"), ("a/b", b"two")])
    with pytest.raises(ValueError, match="normalized"):
        safe_extract(archive, tmp_path / "output")
