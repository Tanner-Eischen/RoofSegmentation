"""Retrieve a Runpod result archive, verify every byte, then acknowledge safely.

The watcher never stops or deletes a Pod. A separate supervisor owns billing.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

MAX_BYTES = 4 * 1024**3
CHUNK = 1024**2


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def safe_member(name, job_name):
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != name
        or len(path.parts) < 2
        or path.parts[0] != job_name
    ):
        raise ValueError(f"Unsafe result archive path: {name!r}")
    return PurePosixPath(*path.parts[1:]).as_posix()


def verify_archive(path, job_name, *, max_bytes=MAX_BYTES):
    """Verify the complete member set and all manifest hashes before extracting."""
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError("Result archive exceeds the four-GiB bound")
    with tarfile.open(path, "r:*") as archive:
        members = {}
        total = 0
        for member in archive.getmembers():
            relative = safe_member(member.name, job_name)
            if not member.isfile() or relative in members or member.size < 0:
                raise ValueError("Only unique regular result files are allowed")
            total += member.size
            if total > max_bytes:
                raise ValueError("Unpacked result files exceed the four-GiB bound")
            members[relative] = member
        manifest_name = "remote_result_manifest.json"
        if manifest_name not in members or members[manifest_name].size > 2 * CHUNK:
            raise ValueError("Missing or oversized internal result manifest")
        manifest_payload = archive.extractfile(members[manifest_name]).read()
        expected = json.loads(manifest_payload)
        if not isinstance(expected, dict) or set(expected) != set(members) - {
            manifest_name
        }:
            raise ValueError("Internal manifest does not describe every result file")
        records = {}
        for relative, record in expected.items():
            safe_member(f"{job_name}/{relative}", job_name)
            if (
                not isinstance(record, dict)
                or type(record.get("bytes")) is not int
                or record["bytes"] != members[relative].size
                or not isinstance(record.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            ):
                raise ValueError("Invalid internal file size/hash record")
            digest = hashlib.sha256()
            count = 0
            with archive.extractfile(members[relative]) as stream:
                for chunk in iter(lambda: stream.read(CHUNK), b""):
                    digest.update(chunk)
                    count += len(chunk)
            if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Internal result checksum mismatch: {relative}")
            records[relative] = dict(record)
        records[manifest_name] = {
            "bytes": len(manifest_payload),
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
        }
    return {"job_name": job_name, "files": records, "payload_bytes": total}


def extract_verified(path, destination, verification):
    """No tar extraction API, links or conflicting overwrite are permitted."""
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Result destination cannot be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    staging = Path(tempfile.mkdtemp(prefix=".retrieval-", dir=root))
    try:
        with tarfile.open(path, "r:*") as archive:
            for member in archive.getmembers():
                relative = safe_member(member.name, verification["job_name"])
                record = verification["files"][relative]
                target = root / relative
                if target.is_symlink() or not target.resolve().is_relative_to(root):
                    raise ValueError("Result path escapes the extraction directory")
                if target.exists():
                    if (
                        not target.is_file()
                        or target.stat().st_size != record["bytes"]
                        or sha256(target) != record["sha256"]
                    ):
                        raise FileExistsError(f"Conflicting local result: {relative}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=staging, delete=False) as stream:
                    temporary = Path(stream.name)
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as source:
                        for chunk in iter(lambda: source.read(CHUNK), b""):
                            stream.write(chunk)
                            digest.update(chunk)
                if (
                    temporary.stat().st_size != record["bytes"]
                    or digest.hexdigest() != record["sha256"]
                ):
                    raise ValueError("Archive changed after verification")
                # Hard-link creation atomically refuses an existing target on
                # NTFS and Linux filesystems. Both paths are on the same volume.
                os.link(temporary, target)
                temporary.unlink()
    finally:
        # This only removes our newly-created directory when empty.
        if not any(staging.iterdir()):
            staging.rmdir()
    return root


def download_resume(sftp, remote, local, expected, *, max_bytes=MAX_BYTES):
    local = Path(local)
    partial = local.with_name(local.name + ".part")
    if local.is_symlink() or partial.is_symlink():
        raise ValueError("Archive download paths cannot be symlinks")
    size = sftp.stat(remote).st_size
    if not 0 < size <= max_bytes:
        raise ValueError("Remote archive has invalid or oversized length")
    if local.exists():
        if local.stat().st_size != size or sha256(local) != expected:
            raise FileExistsError("Existing local archive conflicts with remote result")
        return local
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError("Partial download is larger than the remote archive")
    with sftp.open(remote, "rb") as source, partial.open("ab") as target:
        source.seek(offset)
        remaining = size - offset
        while remaining:
            chunk = source.read(min(CHUNK, remaining))
            if not chunk:
                raise EOFError(
                    "Result transfer ended before the registered archive size"
                )
            target.write(chunk)
            remaining -= len(chunk)
    if partial.stat().st_size != size or sha256(partial) != expected:
        raise ValueError(
            "Outer result archive checksum mismatch; no acknowledgment sent"
        )
    os.link(partial, local)
    partial.unlink()
    return local


def download_http(
    sftp, remote, local, expected, connection_path, *, max_bytes=MAX_BYTES
):
    """Download through the existing HTTPS Jupyter route; SSH still binds size/ACK.

    A private JSON file supplies authentication. Tokens stay out of URLs,
    process arguments, progress records and transport error messages.
    """
    import requests

    local = Path(local)
    partial = local.with_name(local.name + ".part")
    if local.is_symlink() or partial.is_symlink():
        raise ValueError("Archive download paths cannot be symlinks")
    size = sftp.stat(remote).st_size
    if not 0 < size <= max_bytes:
        raise ValueError("Remote archive has invalid or oversized length")
    if local.exists():
        if local.stat().st_size != size or sha256(local) != expected:
            raise FileExistsError("Existing local archive conflicts with remote result")
        return local
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError("Partial download is larger than the remote archive")
    if offset < size:
        configuration = json.loads(Path(connection_path).read_text(encoding="utf-8"))
        base = configuration.get("base_url")
        token = configuration.get("token")
        if (
            not isinstance(base, str)
            or not isinstance(token, str)
            or not token
            or "\r" in token
            or "\n" in token
        ):
            raise ValueError("Invalid private Jupyter connection configuration")
        parsed = urlsplit(base)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not parsed.hostname.endswith(".proxy.runpod.net")
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Jupyter archive route must be the existing HTTPS Runpod proxy"
            )
        if not str(remote).startswith("/") or ".." in PurePosixPath(remote).parts:
            raise ValueError("Jupyter archive path must be absolute and safe")
        url = base.rstrip("/") + "/files" + quote(str(remote), safe="/")
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.Session() as session:
                session.headers.update({"Authorization": "token " + token})
                with session.get(
                    url,
                    headers=headers,
                    stream=True,
                    verify=True,
                    allow_redirects=False,
                    timeout=(15, 60),
                ) as response:
                    if 300 <= response.status_code < 400:
                        raise OSError(
                            "HTTP archive redirect rejected; no credentials forwarded"
                        )
                    if response.status_code not in (200, 206):
                        raise OSError(
                            f"HTTP archive request failed (status {response.status_code})"
                        )
                    if (
                        response.headers.get("Content-Encoding", "identity").lower()
                        != "identity"
                    ):
                        raise ValueError(
                            "HTTP archive content encoding must preserve raw bytes"
                        )
                    expected_remaining = size - offset
                    length = response.headers.get("Content-Length", "")
                    if not length.isdecimal() or int(length) != expected_remaining:
                        raise ValueError(
                            "HTTP archive Content-Length differs from registered remaining size"
                        )
                    if offset:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        if (
                            response.status_code != 206
                            or match is None
                            or tuple(map(int, match.groups()))
                            != (offset, size - 1, size)
                        ):
                            raise ValueError(
                                "HTTP resume response does not match the requested byte range"
                            )
                    elif response.status_code != 200 or response.headers.get(
                        "Content-Range"
                    ):
                        raise ValueError(
                            "Initial HTTP archive response must describe the full registered file"
                        )
                    remaining = expected_remaining
                    with partial.open("ab") as target:
                        for chunk in response.iter_content(chunk_size=CHUNK):
                            if not chunk:
                                continue
                            if len(chunk) > remaining:
                                raise ValueError(
                                    "HTTP archive body exceeds its registered size"
                                )
                            target.write(chunk)
                            remaining -= len(chunk)
                    if remaining:
                        raise OSError(
                            "HTTP archive transfer ended early; partial download preserved"
                        )
        except requests.RequestException:
            raise OSError(
                "HTTP archive transport failed; partial download preserved"
            ) from None
    if partial.stat().st_size != size or sha256(partial) != expected:
        raise ValueError(
            "Outer result archive checksum mismatch; no acknowledgment sent"
        )
    os.link(partial, local)
    partial.unlink()
    return local


def read_optional(sftp, path, *, limit=CHUNK):
    try:
        if sftp.stat(path).st_size > limit:
            raise ValueError("Remote status/sidecar exceeds expected size")
        with sftp.open(path, "rb") as stream:
            return stream.read(limit + 1)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno == 2:
            return None
        raise


def optional_json(sftp, path):
    payload = read_optional(sftp, path)
    if payload is None:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # Status writers can be observed between truncate and completion.
        return None


def snapshot(sftp, remote_out):
    job = optional_json(sftp, str(remote_out / "remote_status.json"))
    comparison = optional_json(sftp, str(remote_out / "status.json"))
    current = None
    if isinstance(comparison, dict):
        arch, seed = comparison.get("architecture"), comparison.get("seed")
        if (
            isinstance(arch, str)
            and re.fullmatch(r"[a-zA-Z0-9_-]+", arch)
            and type(seed) is int
            and 0 <= seed < 2**32
        ):
            current = optional_json(
                sftp, str(remote_out / "runs" / f"{arch}_seed{seed}" / "progress.json")
            )
    return {
        "remote_job": job,
        "comparison": comparison,
        "current_run_progress": current,
    }


def acknowledge(sftp, remote_out, payload):
    target = str(remote_out / "retrieval_ack.json")
    existing = optional_json(sftp, target)
    if existing is not None:
        if (
            existing.get("verified") is not True
            or existing.get("tar_sha256") != payload["tar_sha256"]
        ):
            raise ValueError(
                "Existing remote acknowledgment identifies another archive"
            )
        return
    temporary = target + f".tmp.{os.getpid()}"
    with sftp.open(temporary, "w") as stream:
        stream.write(json.dumps(payload, allow_nan=False) + "\n")
        stream.flush()
    sftp.rename(temporary, target)


def connect(args):
    import paramiko

    client = paramiko.SSHClient()
    client.load_host_keys(str(args.known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(
            args.host,
            port=args.port,
            username="root",
            key_filename=str(args.identity),
            allow_agent=False,
            look_for_keys=False,
            timeout=30,
            auth_timeout=30,
            banner_timeout=30,
        )
    except Exception:
        client.close()
        raise
    client.get_transport().set_keepalive(30)
    sftp = client.open_sftp()
    sftp.get_channel().settimeout(90)
    return client, sftp


def watch(args):
    import paramiko

    local = args.local_out.resolve()
    local.mkdir(parents=True, exist_ok=True)
    remote_out = PurePosixPath(args.remote_out)
    if (
        not remote_out.is_absolute()
        or ".." in remote_out.parts
        or re.fullmatch(r"[a-zA-Z0-9_.-]+", remote_out.name) is None
    ):
        raise ValueError("Remote output must be an absolute named job directory")
    archive_name = remote_out.name + "_results.tar"
    remote_archive = remote_out.parent / archive_name
    client = sftp = None
    failures = 0
    try:
        while True:
            try:
                if client is None:
                    client, sftp = connect(args)
                progress = snapshot(sftp, remote_out)
                write_json(
                    local / "cloud_progress.json",
                    {
                        "watcher_phase": "polling",
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        **progress,
                    },
                )
                sidecar = read_optional(
                    sftp, str(remote_archive) + ".sha256", limit=4096
                )
                if sidecar is not None:
                    pieces = sidecar.decode("utf-8").strip().split()
                    if (
                        len(pieces) != 2
                        or re.fullmatch(r"[0-9a-f]{64}", pieces[0]) is None
                        or pieces[1].lstrip("*") != archive_name
                    ):
                        raise ValueError("Invalid result checksum sidecar")
                    expected = pieces[0]
                    try:
                        sftp.stat(str(remote_archive))
                    except FileNotFoundError:
                        time.sleep(args.poll_seconds)
                        continue
                    write_json(
                        local / "cloud_progress.json",
                        {"watcher_phase": "retrieving", **progress},
                    )
                    connection_path = getattr(args, "jupyter_connection", None)
                    if connection_path is None:
                        archive = download_resume(
                            sftp, str(remote_archive), local / archive_name, expected
                        )
                    else:
                        archive = download_http(
                            sftp,
                            str(remote_archive),
                            local / archive_name,
                            expected,
                            connection_path,
                        )
                    verified = verify_archive(archive, remote_out.name)
                    result_root = extract_verified(archive, local / "results", verified)
                    remote_phase = (progress["remote_job"] or {}).get(
                        "phase", "unknown"
                    )
                    report = {
                        "verified": True,
                        "artifact_integrity_verified": True,
                        "training_complete": remote_phase == "complete",
                        "remote_job_phase": remote_phase,
                        "archive_path": str(archive),
                        "tar_sha256": expected,
                        "result_root": str(result_root),
                        "verified_files": len(verified["files"]),
                        "verified_payload_bytes": verified["payload_bytes"],
                        "ack_sent": False,
                        "archive_transport": "https_jupyter"
                        if connection_path
                        else "ssh_sftp",
                    }
                    write_json(local / "retrieval_verification.json", report)
                    acknowledge(
                        sftp,
                        remote_out,
                        {
                            "verified": True,
                            "tar_sha256": expected,
                            "archive_sha256": expected,
                            "archive_name": archive_name,
                            "verified_files": len(verified["files"]),
                            "verified_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    report["ack_sent"] = True
                    write_json(local / "retrieval_verification.json", report)
                    write_json(
                        local / "cloud_progress.json",
                        {"watcher_phase": "retrieved", **report, **progress},
                    )
                    print(json.dumps(report), flush=True)
                    return report
                failures = 0
                time.sleep(args.poll_seconds)
            except (OSError, EOFError, paramiko.SSHException) as error:
                failures += 1
                if sftp is not None:
                    with contextlib.suppress(OSError, EOFError, paramiko.SSHException):
                        sftp.close()
                if client is not None:
                    with contextlib.suppress(OSError, EOFError, paramiko.SSHException):
                        client.close()
                client = sftp = None
                write_json(
                    local / "cloud_progress.json",
                    {
                        "watcher_phase": "connection_retry",
                        "consecutive_failures": failures,
                        "error": str(error),
                    },
                )
                if failures >= 3:
                    raise RuntimeError(
                        "Three consecutive connection failures; watcher stopped without deleting or stopping the Pod"
                    ) from error
                time.sleep(min(5 * failures, args.poll_seconds))
    except Exception as error:
        write_json(
            local / "cloud_progress.json",
            {"watcher_phase": "failed", "error": str(error)},
        )
        raise
    finally:
        if sftp is not None:
            with contextlib.suppress(OSError, EOFError, paramiko.SSHException):
                sftp.close()
        if client is not None:
            with contextlib.suppress(OSError, EOFError, paramiko.SSHException):
                client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--remote-out", required=True)
    parser.add_argument("--local-out", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=45)
    parser.add_argument(
        "--jupyter-connection",
        type=Path,
        help="Optional private JSON with the existing HTTPS Jupyter base_url/token; SSH remains required for status and ACK",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 5 <= args.poll_seconds <= 300:
        raise ValueError("Invalid SSH port or poll interval")
    if not args.identity.is_file() or not args.known_hosts.is_file():
        raise ValueError("Existing identity and pinned known-hosts files are required")
    if args.jupyter_connection is not None and not args.jupyter_connection.is_file():
        raise ValueError("The private Jupyter connection file must exist")
    watch(args)


if __name__ == "__main__":
    main()
