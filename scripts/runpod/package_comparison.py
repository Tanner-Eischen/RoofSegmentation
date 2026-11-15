"""Bundle only registered source and exact prepared WHU bytes for a Linux Pod."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value):
    value = str(value)
    path = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or windows.drive
        or any(part in {"..", "."} for part in path.parts)
    ):
        raise ValueError(f"Unsafe archive path: {value!r}")
    return path.as_posix()


def dataset_members(data):
    """Manifest-driven allowlist excludes previews, arbitrary files and symlinks."""
    data = Path(data).resolve()
    names = {"manifest.csv", "dataset_meta.json"}
    with (data / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Cannot package an empty manifest")
    for row in rows:
        for field in ("image", "mask"):
            name = safe_relative(row[field])
            prefix = "images/" if field == "image" else "masks/"
            if not name.startswith(prefix) or not name.lower().endswith(".png"):
                raise ValueError(
                    "WHU payload must contain PNGs under images/ or masks/"
                )
            names.add(name)
    for name in sorted(names):
        path = data / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(data)
        ):
            raise ValueError(f"Missing or linked dataset file: {name}")
        yield name, path


def dataset_fingerprint(data, members):
    """Match roof_pipeline.common's byte contract using standard-library hashes."""
    data = Path(data)
    hashes = {name: sha256(path) for name, path in members}
    digest = hashlib.sha256()
    digest.update((data / "manifest.csv").read_bytes())
    digest.update((data / "dataset_meta.json").read_bytes())
    with (data / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        for field in ("image", "mask"):
            relative = row[field]
            digest.update(
                json.dumps(
                    [row["sample_id"], field, relative], ensure_ascii=True
                ).encode("utf-8")
            )
            digest.update(bytes.fromhex(hashes[safe_relative(relative)]))
    return digest.hexdigest(), hashes


class DigestReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        payload = self.stream.read(size)
        self.digest.update(payload)
        return payload


def add_bytes(archive, name, payload):
    item = tarfile.TarInfo(safe_relative(name))
    item.size = len(payload)
    item.mode = 0o644
    # ZipFile.write uses local time and refuses dates before 1980. A one-day
    # margin keeps the restored source valid even in timezones west of UTC.
    item.mtime = 315619200  # 1980-01-02 UTC
    archive.addfile(item, io.BytesIO(payload))


def write_code_archive(plan_path, code, tools_dir):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    snapshot = plan_path.parent / "source_snapshot.zip"
    if sha256(snapshot) != plan["source_snapshot_sha256"]:
        raise ValueError("Registered source snapshot changed")
    with tarfile.open(code.with_suffix(".part"), "w:gz") as archive:
        with zipfile.ZipFile(snapshot) as source:
            if set(source.namelist()) != set(plan["source_sha256"]):
                raise ValueError("Source snapshot members differ from registered plan")
            for name, expected in plan["source_sha256"].items():
                safe_relative(name)
                if not name.startswith("roof_pipeline/") or not name.endswith(".py"):
                    raise ValueError(
                        "Only registered pipeline Python modules may be shipped"
                    )
                payload = source.read(name)
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError(f"Source hash mismatch: {name}")
                add_bytes(archive, name, payload)
        add_bytes(archive, "runpod_parent_plan.json", plan_path.read_bytes())
        add_bytes(archive, "runpod_parent_source_snapshot.zip", snapshot.read_bytes())
        for name in ("remote_run.py", "bootstrap.sh"):
            add_bytes(
                archive, f"scripts/runpod/{name}", (tools_dir / name).read_bytes()
            )
        supervisor = tools_dir / "supervise.sh"
        if supervisor.is_file():
            add_bytes(archive, "scripts/runpod/supervise.sh", supervisor.read_bytes())
    code.with_suffix(".part").replace(code)


def write_transfer_metadata(out, result):
    (out / "transfer_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (out / "SHA256SUMS").write_text(
        "".join(
            f"{record['sha256']}  {name}\n" for name, record in result["files"].items()
        ),
        encoding="utf-8",
    )


def refresh_code(plan_path, out, *, tools_dir=None):
    """Rebuild only the small code archive after verifying the retained data tar."""
    plan_path, out = Path(plan_path).resolve(), Path(out).resolve()
    result = json.loads((out / "transfer_manifest.json").read_text(encoding="utf-8"))
    if result["parent_plan_sha256"] != sha256(plan_path):
        raise ValueError("Cannot refresh a different prepared plan")
    payload = out / "data.tar"
    expected = result["files"]["data.tar"]
    if (
        payload.stat().st_size != expected["bytes"]
        or sha256(payload) != expected["sha256"]
    ):
        raise ValueError("Retained data archive changed")
    code = out / "code.tar.gz"
    write_code_archive(plan_path, code, Path(tools_dir or Path(__file__).parent))
    result["files"]["code.tar.gz"] = {
        "sha256": sha256(code),
        "bytes": code.stat().st_size,
    }
    write_transfer_metadata(out, result)
    return result


def build_bundle(plan_path, out, *, tools_dir=None):
    plan_path, out = Path(plan_path).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a fresh transfer directory")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    snapshot = plan_path.parent / "source_snapshot.zip"
    if sha256(snapshot) != plan["source_snapshot_sha256"]:
        raise ValueError("Registered source snapshot changed")
    tools_dir = Path(tools_dir or Path(__file__).parent)
    data = Path(plan["dataset"]["path"])
    # Materialize only small path metadata; stream the multi-GB payload itself.
    members = list(dataset_members(data))
    if sha256(data / "manifest.csv") != plan["dataset"]["manifest_sha256"]:
        raise ValueError("Dataset manifest changed after plan preparation")
    actual, file_hashes = dataset_fingerprint(data, members)
    if actual != plan["dataset"]["content_sha256"]:
        raise ValueError("Dataset content changed after plan preparation")
    out.mkdir(parents=True, exist_ok=True)
    code = out / "code.tar.gz"
    write_code_archive(plan_path, code, tools_dir)
    payload = out / "data.tar"
    with tarfile.open(payload.with_suffix(".part"), "w") as archive:
        for name, path in members:
            info = archive.gettarinfo(
                str(path), arcname=f"data/whu_official_prepared/{name}"
            )
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                hashed = DigestReader(stream)
                archive.addfile(info, hashed)
                if hashed.digest.hexdigest() != file_hashes[name]:
                    raise ValueError(f"Dataset changed while packaging: {name}")
    payload.with_suffix(".part").replace(payload)
    result = {
        "schema_version": 1,
        "parent_plan_sha256": sha256(plan_path),
        "dataset_content_sha256": plan["dataset"]["content_sha256"],
        "dataset_manifest_sha256": plan["dataset"]["manifest_sha256"],
        "source_snapshot_sha256": plan["source_snapshot_sha256"],
        "dataset_files": len(members),
        "files": {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in (code, payload)
        },
        "scope": "Registered source, parent plan, deploy helpers, manifest/metadata and manifest-listed images/masks only. No account credentials or environment files.",
    }
    write_transfer_metadata(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--refresh-code", action="store_true")
    args = parser.parse_args()
    operation = refresh_code if args.refresh_code else build_bundle
    print(json.dumps(operation(args.plan, args.out), indent=2))


if __name__ == "__main__":
    main()
