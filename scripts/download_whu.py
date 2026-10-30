"""Download WHU publisher geographic partitions, or explicitly choose its HF mirror.

The official ZIP is about 5.43 GB. Downloads resume from a validated .part file.
Use --archive to reuse an already completed official download without downloading it.
The third-party mirror has different partitions; its spatial independence is unverified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

OFFICIAL_URL = (
    "https://gpcv.whu.edu.cn/data/"
    "3.%20The%20cropped%20aerial%20image%20tiles%20and%20raster%20labels.zip"
)
PUBLISHER_PAGE = "https://gpcv.whu.edu.cn/data/building_dataset.html"
OFFICIAL_WRAPPER = "3. The cropped image tiles and raster labels"
OFFICIAL_COUNTS = {"train": 4736, "val": 1036, "test": 2416}
USER_AGENT = "RoofSegmentationDatasetDownloader/1.0"
BLOCK_SIZE = 1024 * 1024


def _headers(response) -> dict:
    return {
        "url": OFFICIAL_URL,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
    }


def download_archive(archive: Path) -> dict:
    """Stream the official ZIP, resuming only when a saved HTTP validator exists."""
    metadata_path = archive.with_name(archive.name + ".download.json")
    if archive.is_file():
        if metadata_path.is_file():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        # An archive fetched separately (for example with curl) has no saved headers.
        request = urllib.request.Request(
            OFFICIAL_URL, method="HEAD", headers={"User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return _headers(response)
        except (OSError, urllib.error.URLError):
            return {"url": OFFICIAL_URL, "etag": None, "last_modified": None}

    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".part")
    partial_metadata = archive.with_name(archive.name + ".part.json")
    saved = {}
    if partial_metadata.is_file():
        saved = json.loads(partial_metadata.read_text(encoding="utf-8"))
    validator = saved.get("etag") or saved.get("last_modified")
    offset = (
        partial.stat().st_size
        if partial.is_file() and saved.get("url") == OFFICIAL_URL and validator
        else 0
    )
    if offset and saved.get("content_length") == offset:
        # A prior process may have finished streaming but stopped before renaming.
        partial.replace(archive)
        partial_metadata.replace(metadata_path)
        return saved
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
    print(
        f"Downloading official WHU archive (resume offset: {offset} bytes)", flush=True
    )
    request = urllib.request.Request(OFFICIAL_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status == 206:
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if not match or int(match[1]) != offset:
                raise ValueError(
                    "Server returned an unexpected byte range; partial download retained"
                )
            expected_size = int(match[3])
        elif response.status == 200:
            # If-Range mismatch or server without range support: restart safely.
            offset = 0
            length = response.headers.get("Content-Length")
            expected_size = int(length) if length is not None else None
        else:
            raise ValueError(f"Unexpected download status {response.status}")
        metadata = _headers(response)
        metadata["content_length"] = expected_size
        partial_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with partial.open("ab" if offset else "wb") as handle:
            downloaded = offset
            progress_at = offset + 128 * BLOCK_SIZE
            while block := response.read(BLOCK_SIZE):
                handle.write(block)
                downloaded += len(block)
                if downloaded >= progress_at:
                    print(f"Downloaded {downloaded / BLOCK_SIZE:.0f} MiB", flush=True)
                    progress_at = downloaded + 128 * BLOCK_SIZE
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise OSError(
            "Incomplete download; rerun the same command to resume the .part file"
        )
    partial.replace(archive)
    partial_metadata.replace(metadata_path)
    return metadata


def _archive_entries(
    archive: zipfile.ZipFile,
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    entries = []
    for info in archive.infolist():
        raw = info.orig_filename.replace("\\", "/")
        path = PurePosixPath(raw)
        mode = info.external_attr >> 16
        if (
            "\x00" in raw
            or path.is_absolute()
            or ".." in path.parts
            or any(":" in part for part in path.parts)
            or any(
                PureWindowsPath(part).is_reserved() or part.endswith((" ", "."))
                for part in path.parts
            )
            or stat.S_ISLNK(mode)
            or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))
        ):
            raise ValueError(f"Unsafe ZIP entry: {info.orig_filename!r}")
        entries.append((info, path))
    return entries


def extract_official_archive(
    archive_path: Path, out: Path, expected_counts: dict | None = None
) -> dict[str, int]:
    """Validate every path before extraction and flatten the publisher's known wrapper."""
    root = out.resolve()
    if archive_path.resolve().is_relative_to(root):
        raise ValueError(
            "Keep the downloaded archive outside the extraction output directory"
        )
    with zipfile.ZipFile(archive_path) as archive:
        entries = _archive_entries(archive)
        files = [path for info, path in entries if not info.is_dir()]
        prefix = None
        counts = {}
        for candidate in (PurePosixPath("."), PurePosixPath(OFFICIAL_WRAPPER)):
            if not all(path.is_relative_to(candidate) for path in files):
                continue
            relative = [path.relative_to(candidate) for path in files]
            image_counts = {
                split: sum(path.parts[:2] == (split, "image") for path in relative)
                for split in OFFICIAL_COUNTS
            }
            label_counts = {
                split: sum(path.parts[:2] == (split, "label") for path in relative)
                for split in OFFICIAL_COUNTS
            }
            if all(image_counts.values()) and image_counts == label_counts:
                prefix, counts = candidate, image_counts
                break
        if prefix is None:
            raise ValueError("ZIP lacks the expected train/val/test image/label layout")
        if expected_counts is not None and counts != expected_counts:
            raise ValueError(
                f"Unexpected publisher partition counts: {counts}; expected {expected_counts}"
            )

        planned = []
        destinations = set()
        for info, path in entries:
            if info.is_dir():
                continue
            relative = path.relative_to(prefix)
            destination = root.joinpath(*relative.parts)
            if not destination.resolve().is_relative_to(root):
                raise ValueError(
                    f"ZIP destination escapes output directory: {relative}"
                )
            for ancestor in (destination, *destination.parents):
                if ancestor == root:
                    break
                if ancestor.is_symlink():
                    raise ValueError(f"ZIP destination contains a symlink: {relative}")
            # Case folding also rejects Windows path aliases before any writes.
            key = str(relative).casefold()
            if key in destinations:
                raise ValueError(f"Duplicate ZIP destination: {relative}")
            destinations.add(key)
            planned.append((info, destination))
        if out.exists():
            unrelated = [
                path
                for path in out.rglob("*")
                if path.is_file()
                and str(PurePosixPath(path.relative_to(out).as_posix())).casefold()
                not in destinations
                and path.name != "download_provenance.json"
            ]
            if unrelated:
                raise ValueError(
                    "Output contains unrelated files; choose an empty directory to avoid mixing partitions"
                )
        for info, destination in planned:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=BLOCK_SIZE)
    return counts


def download_official(out: Path, archive: Path) -> None:
    metadata = download_archive(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        while block := handle.read(BLOCK_SIZE):
            digest.update(block)
    counts = extract_official_archive(archive, out, expected_counts=OFFICIAL_COUNTS)
    provenance = {
        **metadata,
        "source": "official",
        "url": OFFICIAL_URL,
        "publisher_page": PUBLISHER_PAGE,
        "archive_sha256": digest.hexdigest(),
        "archive": str(archive.resolve()),
        "counts": counts,
        "partition_policy": "Publisher geographic training, validation and test areas preserved without reshuffling",
        "spatial_independence": "publisher_geographic_partitions",
    }
    (out / "download_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Extracted official WHU partitions into {out.resolve()}", flush=True)


def download_mirror(out: Path, revision: str, workers: int) -> None:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install huggingface_hub: python -m pip install huggingface_hub"
        ) from exc
    out.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info("giswqs/WHU-Building-Dataset", revision=revision)
    print(f"Downloading fixed mirror revision {info.sha}", flush=True)
    snapshot_download(
        repo_id="giswqs/WHU-Building-Dataset",
        repo_type="dataset",
        revision=info.sha,
        local_dir=str(out),
        max_workers=workers,
    )
    (out / "download_provenance.json").write_text(
        json.dumps(
            {
                "source": "mirror",
                "repo_id": info.id,
                "revision": info.sha,
                "url": "https://huggingface.co/datasets/giswqs/WHU-Building-Dataset",
                "partition_policy": "Preserve third-party mirror folders without reshuffling; these differ from publisher partitions",
                "spatial_independence": "unverified",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "Mirror spatial independence is unverified; do not report this as the publisher geographic benchmark."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("official", "mirror"), default="official")
    parser.add_argument("--out", type=Path, default=Path("data/raw/whu"))
    parser.add_argument(
        "--archive",
        type=Path,
        help="Reuse a completed official ZIP; keep it outside --out",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Mirror download workers"
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Mirror ref, resolved to a fixed commit before downloading",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.source == "official":
        if args.archive is not None and not args.archive.is_file():
            parser.error("--archive must point to an existing, completed ZIP")
        archive = args.archive or args.out.parent / "whu_official.zip"
        download_official(args.out, archive)
    else:
        if args.archive is not None:
            parser.error("--archive is only valid with --source official")
        download_mirror(args.out, args.revision, args.workers)
    print(
        f'python -m roof_pipeline.prepare whu --source "{args.out.resolve()}" --out data/whu_prepared'
    )


if __name__ == "__main__":
    main()
