"""WHU download and archive regressions; all network responses are mocked."""

import hashlib
import io
import json
import stat
import sys
import types
import zipfile

import pytest
from PIL import Image

from roof_pipeline.common import read_manifest
from roof_pipeline.prepare import import_whu
from scripts import download_whu as downloader


def make_archive(path, *, wrapped=True, extra=None):
    prefix = downloader.OFFICIAL_WRAPPER + "/" if wrapped else ""
    with zipfile.ZipFile(path, "w") as archive:
        for index, split in enumerate(("train", "val", "test")):
            for folder in ("image", "label"):
                image = Image.new(
                    "RGB" if folder == "image" else "L",
                    (32, 32),
                    index + 1 if folder == "image" else 255,
                )
                buffer = io.BytesIO()
                image.save(buffer, format="TIFF")
                archive.writestr(
                    prefix + split + "/" + folder + "/0.tif", buffer.getvalue()
                )
        if extra is not None:
            archive.writestr(*extra)
    return path


@pytest.mark.parametrize("wrapped", [True, False])
def test_official_layout_is_flattened_and_importable(tmp_path, wrapped):
    archive = make_archive(tmp_path / "dataset.zip", wrapped=wrapped)
    out = tmp_path / "source"
    counts = downloader.extract_official_archive(archive, out)
    assert counts == {"train": 1, "val": 1, "test": 1}
    assert (out / "train/image/0.tif").is_file()
    assert not (out / downloader.OFFICIAL_WRAPPER).exists()
    prepared = tmp_path / "prepared"
    import_whu(out, prepared)
    assert len(read_manifest(prepared)) == 3


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "/escape.txt",
        "C:/escape.txt",
        "..\\escape.txt",
        "train/image/../../escape.txt",
        "train/image/0.tif:payload",
        "train/image/NUL",
        "train/image/.. ",
    ],
)
def test_unsafe_archive_paths_are_rejected_before_any_writes(tmp_path, name):
    archive = make_archive(tmp_path / "dataset.zip", extra=(name, b"bad"))
    out = tmp_path / "source"
    with pytest.raises(ValueError, match="Unsafe ZIP entry"):
        downloader.extract_official_archive(archive, out)
    assert not out.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_zip_symlink_is_rejected_before_any_writes(tmp_path):
    symlink = zipfile.ZipInfo(downloader.OFFICIAL_WRAPPER + "/train/image/link.tif")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = make_archive(tmp_path / "dataset.zip", extra=(symlink, b"../../outside"))
    with pytest.raises(ValueError, match="Unsafe ZIP entry"):
        downloader.extract_official_archive(archive, tmp_path / "source")
    assert not (tmp_path / "source").exists()


def test_existing_unrelated_dataset_is_not_mixed_or_overwritten(tmp_path):
    archive = make_archive(tmp_path / "dataset.zip")
    out = tmp_path / "source"
    out.mkdir()
    (out / "mirror.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated files"):
        downloader.extract_official_archive(archive, out)
    assert (out / "mirror.txt").read_text(encoding="utf-8") == "keep"
    assert not (out / "train").exists()


def test_publisher_count_mismatch_fails_before_any_writes(tmp_path):
    archive = make_archive(tmp_path / "dataset.zip")
    with pytest.raises(ValueError, match="Unexpected publisher partition counts"):
        downloader.extract_official_archive(
            archive, tmp_path / "source", downloader.OFFICIAL_COUNTS
        )
    assert not (tmp_path / "source").exists()


def test_windows_case_alias_is_rejected_before_any_writes(tmp_path):
    alias = downloader.OFFICIAL_WRAPPER + "/TRAIN/image/0.tif"
    archive = make_archive(tmp_path / "dataset.zip", extra=(alias, b"bad"))
    with pytest.raises(ValueError, match="Duplicate ZIP destination"):
        downloader.extract_official_archive(archive, tmp_path / "source")
    assert not (tmp_path / "source").exists()


class Response(io.BytesIO):
    def __init__(self, content, *, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}


def test_interrupted_download_resumes_with_validator(tmp_path, monkeypatch):
    requests = []
    responses = iter(
        [
            Response(b"abc", headers={"Content-Length": "6", "ETag": '"version-1"'}),
            Response(
                b"def",
                status=206,
                headers={"Content-Range": "bytes 3-5/6", "ETag": '"version-1"'},
            ),
        ]
    )

    def open_response(request, **kwargs):
        requests.append(request)
        return next(responses)

    monkeypatch.setattr(downloader.urllib.request, "urlopen", open_response)
    archive = tmp_path / "dataset.zip"
    with pytest.raises(OSError, match="Incomplete download"):
        downloader.download_archive(archive)
    assert not archive.exists()
    assert archive.with_name("dataset.zip.part").read_bytes() == b"abc"
    downloader.download_archive(archive)
    assert archive.read_bytes() == b"abcdef"
    assert requests[1].get_header("Range") == "bytes=3-"
    assert requests[1].get_header("If-range") == '"version-1"'
    assert not archive.with_name("dataset.zip.part").exists()


def test_changed_server_response_restarts_instead_of_appending(tmp_path, monkeypatch):
    archive = tmp_path / "dataset.zip"
    archive.with_name("dataset.zip.part").write_bytes(b"old")
    archive.with_name("dataset.zip.part.json").write_text(
        json.dumps(
            {
                "url": downloader.OFFICIAL_URL,
                "etag": '"old"',
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda *a, **k: Response(
            b"new-data", headers={"Content-Length": "8", "ETag": '"new"'}
        ),
    )
    downloader.download_archive(archive)
    assert archive.read_bytes() == b"new-data"


def test_wrong_resume_range_preserves_partial_data(tmp_path, monkeypatch):
    archive = tmp_path / "dataset.zip"
    partial = archive.with_name("dataset.zip.part")
    partial.write_bytes(b"abc")
    archive.with_name("dataset.zip.part.json").write_text(
        json.dumps(
            {
                "url": downloader.OFFICIAL_URL,
                "etag": '"same"',
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda *a, **k: Response(
            b"bad", status=206, headers={"Content-Range": "bytes 2-4/6"}
        ),
    )
    with pytest.raises(ValueError, match="unexpected byte range"):
        downloader.download_archive(archive)
    assert partial.read_bytes() == b"abc"
    assert not archive.exists()


def test_completed_partial_is_promoted_without_network(tmp_path, monkeypatch):
    archive = tmp_path / "dataset.zip"
    archive.with_name("dataset.zip.part").write_bytes(b"complete")
    archive.with_name("dataset.zip.part.json").write_text(
        json.dumps(
            {
                "url": downloader.OFFICIAL_URL,
                "etag": '"same"',
                "content_length": 8,
            }
        ),
        encoding="utf-8",
    )

    def no_network(*args, **kwargs):
        raise AssertionError("A complete partial file requires no further download")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", no_network)
    downloader.download_archive(archive)
    assert archive.read_bytes() == b"complete"


def test_existing_archive_is_reused_with_head_metadata_only(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "dataset.zip")
    before = archive.read_bytes()
    requests = []

    def head_response(request, **kwargs):
        requests.append(request)
        return Response(b"", headers={"Last-Modified": "publisher-date"})

    monkeypatch.setattr(downloader.urllib.request, "urlopen", head_response)
    metadata = downloader.download_archive(archive)
    assert archive.read_bytes() == before
    assert [request.method for request in requests] == ["HEAD"]
    assert metadata["last_modified"] == "publisher-date"


def test_official_provenance_records_archive_hash_and_partitions(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "dataset.zip")
    monkeypatch.setattr(
        downloader, "OFFICIAL_COUNTS", {"train": 1, "val": 1, "test": 1}
    )
    monkeypatch.setattr(
        downloader,
        "download_archive",
        lambda path: {
            "etag": '"publisher"',
            "last_modified": "Tue, 03 Jul 2018 01:58:46 GMT",
        },
    )
    out = tmp_path / "source"
    downloader.download_official(out, archive)
    provenance = json.loads(
        (out / "download_provenance.json").read_text(encoding="utf-8")
    )
    assert (
        provenance["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert provenance["spatial_independence"] == "publisher_geographic_partitions"
    assert provenance["counts"] == {"train": 1, "val": 1, "test": 1}
    assert provenance["etag"] == '"publisher"'


def test_mirror_is_pinned_and_marks_spatial_independence_unverified(
    tmp_path, monkeypatch
):
    calls = []
    info = types.SimpleNamespace(id="giswqs/WHU-Building-Dataset", sha="fixed-commit")
    api = types.SimpleNamespace(dataset_info=lambda *a, **k: info)
    module = types.SimpleNamespace(
        HfApi=lambda: api, snapshot_download=lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    out = tmp_path / "mirror"
    downloader.download_mirror(out, "main", 2)
    provenance = json.loads(
        (out / "download_provenance.json").read_text(encoding="utf-8")
    )
    assert calls[0]["revision"] == "fixed-commit"
    assert provenance["revision"] == "fixed-commit"
    assert provenance["spatial_independence"] == "unverified"
