"""Remote retrieval acknowledges only complete, verified, bounded artifacts."""

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "runpod_watch", Path(__file__).parents[1] / "scripts/runpod/watch_remote.py"
)
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


def make_archive(path, *, change_payload=False, extra=None, omit_record=False):
    payloads = {
        "best.pt": b"registered checkpoint bytes",
        "history.json": b'[{"epoch":40}]',
    }
    expected = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in payloads.items()
    }
    if omit_record:
        expected.pop("best.pt")
    payloads["remote_result_manifest.json"] = json.dumps(expected).encode()
    if change_payload:
        payloads["best.pt"] = b"tampered checkpoint bytes!!"
    with tarfile.open(path, "w") as archive:
        for name, data in payloads.items():
            item = tarfile.TarInfo("job/" + name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
        if extra is not None:
            archive.addfile(extra, io.BytesIO(b"x") if extra.isfile() else None)
    return payloads


def test_valid_archive_roundtrip_and_identical_reuse(tmp_path):
    archive = tmp_path / "results.tar"
    files = make_archive(archive)
    verification = watcher.verify_archive(archive, "job")
    result = watcher.extract_verified(archive, tmp_path / "results", verification)
    for name, data in files.items():
        assert (result / name).read_bytes() == data
    watcher.extract_verified(archive, result, verification)
    (result / "best.pt").write_bytes(b"do not overwrite this")
    with pytest.raises(FileExistsError, match="Conflicting"):
        watcher.extract_verified(archive, result, verification)
    assert (result / "best.pt").read_bytes() == b"do not overwrite this"


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "wrong/file",
        "job/../../escape",
        "job/a\\b",
        "job/C:secret",
    ],
)
def test_unsafe_tar_names_rejected_before_extraction(tmp_path, name):
    archive = tmp_path / "unsafe.tar"
    item = tarfile.TarInfo(name)
    item.size = 1
    make_archive(archive, extra=item)
    with pytest.raises(ValueError, match="Unsafe"):
        watcher.verify_archive(archive, "job")


def test_links_duplicates_missing_manifest_entries_and_bad_hashes_are_rejected(
    tmp_path,
):
    link = tarfile.TarInfo("job/linked")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../outside"
    archive = tmp_path / "link.tar"
    make_archive(archive, extra=link)
    with pytest.raises(ValueError, match="regular"):
        watcher.verify_archive(archive, "job")
    duplicate = tarfile.TarInfo("job/best.pt")
    duplicate.size = 1
    make_archive(archive, extra=duplicate)
    with pytest.raises(ValueError, match="unique"):
        watcher.verify_archive(archive, "job")
    make_archive(archive, omit_record=True)
    with pytest.raises(ValueError, match="every result"):
        watcher.verify_archive(archive, "job")
    make_archive(archive, change_payload=True)
    with pytest.raises(ValueError, match="size/hash|checksum"):
        watcher.verify_archive(archive, "job")


def test_compressed_bomb_is_bounded_by_unpacked_size(tmp_path):
    archive = tmp_path / "bomb.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        item = tarfile.TarInfo("job/large")
        item.size = 10000
        stream.addfile(item, io.BytesIO(b"0" * item.size))
    assert archive.stat().st_size < 1024
    with pytest.raises(ValueError, match="Unpacked"):
        watcher.verify_archive(archive, "job", max_bytes=1024)


class RemoteWriter(io.BytesIO):
    def __init__(self, owner, name):
        super().__init__()
        self.owner, self.name = owner, name

    def write(self, value):
        return super().write(value.encode() if isinstance(value, str) else value)

    def close(self):
        self.owner.files[self.name] = self.getvalue()
        super().close()


class FakeSFTP:
    def __init__(self, files):
        self.files = files
        self.seeks = []
        self.on_rename = None

    def stat(self, name):
        if name not in self.files:
            raise FileNotFoundError(2, "missing")
        return SimpleNamespace(st_size=len(self.files[name]))

    def open(self, name, mode):
        if "w" in mode:
            return RemoteWriter(self, name)
        self.stat(name)
        source = io.BytesIO(self.files[name])
        seek = source.seek

        def recorded_seek(offset, *args):
            self.seeks.append(offset)
            return seek(offset, *args)

        source.seek = recorded_seek
        return source

    def rename(self, old, new):
        if self.on_rename is not None:
            self.on_rename(old, new)
        self.files[new] = self.files.pop(old)

    def close(self):
        pass


def test_download_resumes_at_verified_final_offset_and_checks_outer_hash(tmp_path):
    payload = b"abcdef" * 200000
    remote = FakeSFTP({"archive": payload})
    target = tmp_path / "archive.tar"
    target.with_name(target.name + ".part").write_bytes(payload[:12345])
    watcher.download_resume(
        remote, "archive", target, hashlib.sha256(payload).hexdigest()
    )
    assert target.read_bytes() == payload and remote.seeks == [12345]
    other = tmp_path / "bad.tar"
    with pytest.raises(ValueError, match="Outer"):
        watcher.download_resume(remote, "archive", other, "0" * 64)
    assert not other.exists() and other.with_name(other.name + ".part").exists()


def test_ack_is_written_only_after_local_archive_and_all_members_verify(
    tmp_path, monkeypatch
):
    archive = tmp_path / "fixture.tar"
    make_archive(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    files = {
        "/runs/job_results.tar": archive.read_bytes(),
        "/runs/job_results.tar.sha256": f"{digest}  job_results.tar\n".encode(),
        "/runs/job/remote_status.json": b'{"phase":"complete"}',
        "/runs/job/status.json": b'{"phase":"complete","completed_runs":3}',
    }
    remote = FakeSFTP(files)
    local = tmp_path / "download"

    def before_ack(old, new):
        report = json.loads((local / "retrieval_verification.json").read_text())
        assert report["verified"] and report["ack_sent"] is False
        assert watcher.sha256(local / "job_results.tar") == digest
        assert (
            local / "results/best.pt"
        ).read_bytes() == b"registered checkpoint bytes"
        assert new == "/runs/job/retrieval_ack.json"

    remote.on_rename = before_ack
    monkeypatch.setattr(
        watcher, "connect", lambda _: (SimpleNamespace(close=lambda: None), remote)
    )
    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(SSHException=type("SSHError", (Exception,), {})),
    )
    report = watcher.watch(
        SimpleNamespace(local_out=local, remote_out="/runs/job", poll_seconds=45)
    )
    assert report["ack_sent"] and report["training_complete"]
    ack = json.loads(remote.files["/runs/job/retrieval_ack.json"])
    assert ack["verified"] and ack["tar_sha256"] == digest


def test_internal_integrity_failure_never_acknowledges(tmp_path, monkeypatch):
    archive = tmp_path / "fixture.tar"
    make_archive(archive, change_payload=True)
    digest = watcher.sha256(archive)
    remote = FakeSFTP(
        {
            "/runs/job_results.tar": archive.read_bytes(),
            "/runs/job_results.tar.sha256": f"{digest}  job_results.tar".encode(),
        }
    )
    monkeypatch.setattr(
        watcher, "connect", lambda _: (SimpleNamespace(close=lambda: None), remote)
    )
    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(SSHException=type("SSHError", (Exception,), {})),
    )
    with pytest.raises(ValueError):
        watcher.watch(
            SimpleNamespace(
                local_out=tmp_path / "download", remote_out="/runs/job", poll_seconds=45
            )
        )
    assert not any("retrieval_ack" in key for key in remote.files)


def test_connection_failures_are_bounded_without_touching_pod(tmp_path, monkeypatch):
    error = type("SSHError", (Exception,), {})
    calls = []

    def failed(_):
        calls.append(1)
        raise error("temporary connection failure")

    monkeypatch.setattr(watcher, "connect", failed)
    monkeypatch.setattr(watcher.time, "sleep", lambda _: None)
    monkeypatch.setitem(sys.modules, "paramiko", SimpleNamespace(SSHException=error))
    with pytest.raises(RuntimeError, match="Three consecutive"):
        watcher.watch(
            SimpleNamespace(
                local_out=tmp_path / "download", remote_out="/runs/job", poll_seconds=45
            )
        )
    assert len(calls) == 3


class MockHTTPRequestError(Exception):
    pass


class MockHTTPResponse:
    def __init__(self, payload, *, status=200, headers=None, pieces=None):
        self.status_code = status
        self.headers = headers or {"Content-Length": str(len(payload))}
        self.pieces = pieces if pieces is not None else [payload]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_content(self, chunk_size):
        assert chunk_size == watcher.CHUNK
        for piece in self.pieces:
            if isinstance(piece, Exception):
                raise piece
            yield piece


def install_mock_http(monkeypatch, response):
    observed = []

    class Session:
        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def get(self, url, **kwargs):
            observed.append(
                {
                    "url": url,
                    **kwargs,
                    "headers": {**self.headers, **kwargs.get("headers", {})},
                }
            )
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(Session=Session, RequestException=MockHTTPRequestError),
    )
    return observed


def private_http_config(tmp_path, base="https://job-8888.proxy.runpod.net"):
    config = tmp_path / "private_connection.json"
    config.write_text(json.dumps({"base_url": base, "token": "private-test-token"}))
    return config


def test_http_binary_download_keeps_auth_in_header_and_verifies_transport(
    tmp_path, monkeypatch
):
    payload = b"archive bytes\x00\xff"
    calls = install_mock_http(monkeypatch, MockHTTPResponse(payload))
    target = tmp_path / "result.tar"
    config = private_http_config(tmp_path)
    watcher.download_http(
        FakeSFTP({"/workspace/a file.tar": payload}),
        "/workspace/a file.tar",
        target,
        hashlib.sha256(payload).hexdigest(),
        config,
    )
    assert target.read_bytes() == payload
    assert (
        calls[0]["url"]
        == "https://job-8888.proxy.runpod.net/files/workspace/a%20file.tar"
    )
    assert "private-test-token" not in calls[0]["url"]
    assert calls[0]["headers"]["Authorization"] == "token private-test-token"
    assert calls[0]["headers"]["Accept-Encoding"] == "identity"
    assert calls[0]["verify"] is True and calls[0]["allow_redirects"] is False
    assert calls[0]["timeout"] == (15, 60)


def test_http_resume_requires_registered_content_range_and_correct_offset(
    tmp_path, monkeypatch
):
    payload = b"0123456789"
    calls = install_mock_http(
        monkeypatch,
        MockHTTPResponse(
            payload[5:],
            status=206,
            headers={"Content-Length": "5", "Content-Range": "bytes 5-9/10"},
        ),
    )
    target = tmp_path / "result.tar"
    target.with_name(target.name + ".part").write_bytes(payload[:5])
    watcher.download_http(
        FakeSFTP({"/result.tar": payload}),
        "/result.tar",
        target,
        hashlib.sha256(payload).hexdigest(),
        private_http_config(tmp_path),
    )
    assert calls[0]["headers"]["Range"] == "bytes=5-"
    assert target.read_bytes() == payload


@pytest.mark.parametrize(
    "status,headers",
    [
        (200, {"Content-Length": "5"}),
        (206, {"Content-Length": "5", "Content-Range": "bytes 4-8/10"}),
        (206, {"Content-Length": "5", "Content-Range": "bytes 5-9/11"}),
        (206, {"Content-Length": "4", "Content-Range": "bytes 5-9/10"}),
    ],
)
def test_http_resume_rejects_ignored_or_wrong_ranges(
    tmp_path, monkeypatch, status, headers
):
    payload = b"0123456789"
    install_mock_http(
        monkeypatch, MockHTTPResponse(payload[5:], status=status, headers=headers)
    )
    target = tmp_path / "result.tar"
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(payload[:5])
    with pytest.raises(ValueError, match="range|Content-Length"):
        watcher.download_http(
            FakeSFTP({"/result.tar": payload}),
            "/result.tar",
            target,
            hashlib.sha256(payload).hexdigest(),
            private_http_config(tmp_path),
        )
    assert not target.exists() and partial.read_bytes() == payload[:5]


def test_http_oversize_and_overlong_body_are_rejected(tmp_path, monkeypatch):
    payload = b"0123456789"
    calls = install_mock_http(monkeypatch, MockHTTPResponse(payload))
    target = tmp_path / "result.tar"
    config = private_http_config(tmp_path)
    remote = FakeSFTP({"/result.tar": payload})
    with pytest.raises(ValueError, match="oversized"):
        watcher.download_http(
            remote, "/result.tar", target, watcher.sha256(config), config, max_bytes=9
        )
    assert calls == []
    install_mock_http(
        monkeypatch,
        MockHTTPResponse(
            payload, headers={"Content-Length": "10"}, pieces=[payload + b"x"]
        ),
    )
    with pytest.raises(ValueError, match="body exceeds"):
        watcher.download_http(
            remote, "/result.tar", target, hashlib.sha256(payload).hexdigest(), config
        )
    assert not target.exists()


def test_http_truncation_and_transport_failures_preserve_resumable_bytes(
    tmp_path, monkeypatch
):
    payload = b"0123456789"
    remote = FakeSFTP({"/result.tar": payload})
    target = tmp_path / "result.tar"
    config = private_http_config(tmp_path)
    install_mock_http(monkeypatch, MockHTTPResponse(payload, pieces=[payload[:5]]))
    with pytest.raises(OSError, match="ended early"):
        watcher.download_http(
            remote, "/result.tar", target, hashlib.sha256(payload).hexdigest(), config
        )
    assert target.with_name(target.name + ".part").read_bytes() == payload[:5]
    install_mock_http(
        monkeypatch,
        MockHTTPRequestError("private-test-token must not appear in errors"),
    )
    with pytest.raises(OSError, match="transport failed") as error:
        watcher.download_http(
            remote, "/result.tar", target, hashlib.sha256(payload).hexdigest(), config
        )
    assert "private-test-token" not in str(error.value)


def test_http_hash_failure_cannot_finalize_archive(tmp_path, monkeypatch):
    payload = b"0123456789"
    install_mock_http(monkeypatch, MockHTTPResponse(payload))
    target = tmp_path / "result.tar"
    with pytest.raises(ValueError, match="Outer.*checksum"):
        watcher.download_http(
            FakeSFTP({"/result.tar": payload}),
            "/result.tar",
            target,
            "0" * 64,
            private_http_config(tmp_path),
        )
    assert not target.exists() and target.with_name(target.name + ".part").exists()


def test_http_rejects_redirects_without_forwarding_auth(tmp_path, monkeypatch):
    calls = install_mock_http(
        monkeypatch,
        MockHTTPResponse(
            b"", status=302, headers={"Location": "https://unrelated.invalid/"}
        ),
    )
    with pytest.raises(OSError, match="redirect rejected"):
        watcher.download_http(
            FakeSFTP({"/result.tar": b"x"}),
            "/result.tar",
            tmp_path / "result.tar",
            "0" * 64,
            private_http_config(tmp_path),
        )
    assert len(calls) == 1 and calls[0]["allow_redirects"] is False


@pytest.mark.parametrize(
    "base",
    [
        "http://job-8888.proxy.runpod.net",
        "https://unrelated.invalid",
        "https://user:pass@job-8888.proxy.runpod.net",
    ],
)
def test_http_route_cannot_send_credentials_to_an_unexpected_destination(
    tmp_path, monkeypatch, base
):
    calls = install_mock_http(monkeypatch, MockHTTPResponse(b"x"))
    with pytest.raises(ValueError, match="HTTPS Runpod proxy"):
        watcher.download_http(
            FakeSFTP({"/result.tar": b"x"}),
            "/result.tar",
            tmp_path / "result.tar",
            "0" * 64,
            private_http_config(tmp_path, base),
        )
    assert calls == []


def test_http_watcher_still_verifies_internal_archive_before_ssh_ack(
    tmp_path, monkeypatch
):
    archive = tmp_path / "fixture.tar"
    make_archive(archive)
    digest = watcher.sha256(archive)
    remote = FakeSFTP(
        {
            "/runs/job_results.tar": archive.read_bytes(),
            "/runs/job_results.tar.sha256": f"{digest}  job_results.tar\n".encode(),
            "/runs/job/remote_status.json": b'{"phase":"complete"}',
        }
    )
    monkeypatch.setattr(
        watcher, "connect", lambda _: (SimpleNamespace(close=lambda: None), remote)
    )
    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(SSHException=type("SSHError", (Exception,), {})),
    )
    install_mock_http(monkeypatch, MockHTTPResponse(archive.read_bytes()))
    report = watcher.watch(
        SimpleNamespace(
            local_out=tmp_path / "download",
            remote_out="/runs/job",
            poll_seconds=45,
            jupyter_connection=private_http_config(tmp_path),
        )
    )
    assert report["ack_sent"] and report["archive_transport"] == "https_jupyter"
    assert (
        tmp_path / "download/results/best.pt"
    ).read_bytes() == b"registered checkpoint bytes"
