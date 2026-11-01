"""A supported-profile label must derive from installed versions and lock bytes."""

import importlib.metadata

from roof_pipeline.runtime import runtime_identity


def test_runtime_version_mismatch_cannot_look_supported(tmp_path):
    lock = tmp_path / "requirements-serving-windows.lock"
    lock.write_text("numpy==2.2.6\n", encoding="utf-8")
    matched = runtime_identity(
        tmp_path, platform_name="win32", version_lookup=lambda name: "2.2.6"
    )
    assert matched["profile_matches"]
    assert not matched["source_files_complete"]
    wrong = runtime_identity(
        tmp_path, platform_name="win32", version_lookup=lambda name: "1.0"
    )
    assert not wrong["profile_matches"]
    assert wrong["version_mismatches"]["numpy"]["installed"] == "1.0"
    lock.write_text("numpy==1.0\n", encoding="utf-8")
    changed = runtime_identity(
        tmp_path, platform_name="win32", version_lookup=lambda name: "1.0"
    )
    assert changed["profile_lock_sha256"] != matched["profile_lock_sha256"]


def test_missing_lock_or_distribution_is_explicit(tmp_path):
    missing = runtime_identity(tmp_path, platform_name="linux")
    assert not missing["profile_matches"] and missing["profile_lock_sha256"] is None
    (tmp_path / "requirements-release-linux.lock").write_text(
        "missing==1.0\n", encoding="utf-8"
    )

    def unavailable(name):
        raise importlib.metadata.PackageNotFoundError(name)

    result = runtime_identity(
        tmp_path, platform_name="linux", version_lookup=unavailable
    )
    assert not result["profile_matches"]
    assert result["version_mismatches"]["missing"]["installed"] is None
