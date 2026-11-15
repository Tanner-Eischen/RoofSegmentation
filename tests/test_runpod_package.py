"""Portable packages must preserve exact evidence while excluding unrelated data."""

import copy
import importlib.util
import json
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

from roof_pipeline.common import dataset_fingerprint
from scripts.make_smoke_dataset import make_fixture


def module(name, relative):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / relative
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


package = module("runpod_package", "scripts/runpod/package_comparison.py")
remote = module("runpod_remote", "scripts/runpod/remote_run.py")


def fixture_plan(tmp_path):
    data = make_fixture(tmp_path / "data", 1, 1, 1, 64)
    (data / "preview_train.png").write_bytes(b"must not ship")
    (data / ".env").write_bytes(b"must not ship")
    (tmp_path / "snapshot").mkdir()
    snapshot = tmp_path / "snapshot/source_snapshot.zip"
    source = b"# exact frozen source\r\n"
    with zipfile.ZipFile(snapshot, "w") as archive:
        archive.writestr("roof_pipeline/train.py", source)
    plan = {
        "source_snapshot_sha256": package.sha256(snapshot),
        "source_sha256": {
            "roof_pipeline/train.py": __import__("hashlib").sha256(source).hexdigest()
        },
        "dataset": {
            "path": str(data),
            "content_sha256": dataset_fingerprint(data),
            "manifest_sha256": package.sha256(data / "manifest.csv"),
        },
    }
    path = snapshot.parent / "plan.json"
    path.write_text(json.dumps(plan))
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("remote_run.py", "bootstrap.sh"):
        (tools / name).write_bytes(b"# fixture helper\n")
    return path, data, tools


def test_bundle_allowlist_preserves_data_source_and_manifest_bytes(tmp_path):
    plan, data, tools = fixture_plan(tmp_path)
    out = tmp_path / "out"
    result = package.build_bundle(plan, out, tools_dir=tools)
    assert result["dataset_files"] == 8
    with tarfile.open(out / "data.tar") as archive:
        names = archive.getnames()
        assert len(names) == 8 and not any(".env" in n or "preview" in n for n in names)
        for member in archive.getmembers():
            relative = Path(member.name).relative_to("data/whu_official_prepared")
            assert archive.extractfile(member).read() == (data / relative).read_bytes()
    with tarfile.open(out / "code.tar.gz") as archive:
        assert (
            archive.extractfile("roof_pipeline/train.py").read()
            == b"# exact frozen source\r\n"
        )
        assert not any(".git/" in n or ".env" in n for n in archive.getnames())
    assert result["files"]["data.tar"]["sha256"] == package.sha256(out / "data.tar")
    with pytest.raises(FileExistsError):
        package.build_bundle(plan, out, tools_dir=tools)


@pytest.mark.parametrize("path", ["../a", "/etc/passwd", "C:/secret", "images\\a.png"])
def test_unsafe_paths_are_rejected(path):
    with pytest.raises(ValueError, match="Unsafe"):
        package.safe_relative(path)


def test_tampered_frozen_source_is_rejected(tmp_path):
    plan, _, tools = fixture_plan(tmp_path)
    (plan.parent / "source_snapshot.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="snapshot changed"):
        package.build_bundle(plan, tmp_path / "out", tools_dir=tools)


def test_changed_image_is_rejected_before_bundle_creation(tmp_path):
    plan, data, tools = fixture_plan(tmp_path)
    image = next((data / "images").glob("*.png"))
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Dataset content changed"):
        package.build_bundle(plan, tmp_path / "out", tools_dir=tools)
    assert not (tmp_path / "out").exists()


def test_restored_tar_source_can_be_snapshotted_by_zipfile(tmp_path):
    plan, _, tools = fixture_plan(tmp_path)
    out = tmp_path / "bundle"
    package.build_bundle(plan, out, tools_dir=tools)
    restored = tmp_path / "restored"
    with tarfile.open(out / "code.tar.gz") as archive:
        item = archive.getmember("roof_pipeline/train.py")
        target = restored / item.name
        target.parent.mkdir(parents=True)
        target.write_bytes(archive.extractfile(item).read())
        os.utime(target, (item.mtime, item.mtime))
    snapshot = tmp_path / "remote_snapshot.zip"
    with zipfile.ZipFile(snapshot, "w") as archive:
        archive.write(target, "roof_pipeline/train.py")
    with zipfile.ZipFile(snapshot) as archive:
        assert archive.read("roof_pipeline/train.py") == target.read_bytes()


def test_derived_plan_allows_only_environment_and_path_changes():
    parent = {
        "architectures": ["a", "b", "c"],
        "seeds": [42],
        "settings": {"epochs": 40, "batch_size": 4},
        "initializations": {"a": {"encoder_sha256": "frozen"}},
        "source_sha256": {"x.py": "source"},
        "dataset": {
            "path": "C:/local",
            "content_sha256": "data",
            "manifest_sha256": "manifest",
            "split_counts": {"train": 4736},
        },
        "test_inference": False,
        "environment": {"python": "3.10"},
    }
    derived = copy.deepcopy(parent)
    derived["dataset"]["path"] = "/workspace/data"
    derived["environment"] = {"python": "3.12"}
    remote.same_controls(parent, derived)
    for key in ("settings", "seeds", "initializations", "source_sha256"):
        changed = copy.deepcopy(derived)
        changed[key] = {} if isinstance(changed[key], dict) else [1337]
        with pytest.raises(ValueError, match="scientific control"):
            remote.same_controls(parent, changed)
    changed = copy.deepcopy(derived)
    changed["dataset"]["content_sha256"] = "changed"
    with pytest.raises(ValueError, match="dataset identity"):
        remote.same_controls(parent, changed)


def test_execution_budget_rejects_known_truncation_before_training():
    estimate = {"total_seconds": {"upper": 7200}}
    assert remote.execution_budget(estimate, 0.281, 1.0, 0.1, 10800) == pytest.approx(
        0.662
    )
    with pytest.raises(ValueError, match="remaining job time budget"):
        remote.execution_budget(estimate, 0.281, 1.0, 0.1, 3600)
    with pytest.raises(ValueError, match="cost ceiling"):
        remote.execution_budget(estimate, 0.281, 0.5, 0.1, 10800)


def test_refresh_updates_helpers_without_rewriting_data_tar(tmp_path):
    plan, _, tools = fixture_plan(tmp_path)
    out = tmp_path / "bundle"
    original = package.build_bundle(plan, out, tools_dir=tools)
    old_stat = (out / "data.tar").stat()
    (tools / "remote_run.py").write_bytes(b"# refreshed helper\n")
    updated = package.refresh_code(plan, out, tools_dir=tools)
    assert updated["files"]["data.tar"] == original["files"]["data.tar"]
    assert (out / "data.tar").stat().st_mtime_ns == old_stat.st_mtime_ns
    assert (
        updated["files"]["code.tar.gz"]["sha256"]
        != original["files"]["code.tar.gz"]["sha256"]
    )
    with tarfile.open(out / "code.tar.gz") as archive:
        assert (
            archive.extractfile("scripts/runpod/remote_run.py").read()
            == b"# refreshed helper\n"
        )
