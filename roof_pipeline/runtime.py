"""Report the installed CPU profile and exact startup source without credentials."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

BUILD_FILES = (
    "api/main.py",
    "api/inference.py",
    "roof_pipeline/inference_service.py",
    "roof_pipeline/release.py",
    "roof_pipeline/runtime.py",
    "roof_pipeline/models.py",
    "roof_pipeline/dataset.py",
    "roof_pipeline/predict.py",
)
PROFILES = {
    "win32": ("cpu-python312-windows-v1", "requirements-serving-windows.lock"),
    "linux": ("cpu-python312-linux-v1", "requirements-release-linux.lock"),
}


def runtime_identity(root=None, *, platform_name=None, version_lookup=None):
    """Version matching identifies a profile; it is not a security/quality verdict."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    platform_name = platform_name or sys.platform
    lookup = version_lookup or importlib.metadata.version
    profile_id, lock_name = PROFILES.get(platform_name, (None, None))
    lock = root / lock_name if lock_name else None
    expected = {}
    lock_hash = None
    if lock and lock.is_file():
        content = lock.read_bytes()
        lock_hash = hashlib.sha256(content).hexdigest()
        expected = {
            name.lower().replace("_", "-"): version
            for name, version in re.findall(
                r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", content.decode("utf-8"), re.M
            )
        }
    mismatches = {}
    for name, required in expected.items():
        try:
            actual = lookup(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != required:
            mismatches[name] = {"required": required, "installed": actual}
    source = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in BUILD_FILES
        if (root / name).is_file()
    }
    matched = bool(expected) and not mismatches and sys.version_info[:2] == (3, 12)
    identity = {
        "profile_id": profile_id,
        "profile_matches": matched,
        "profile_lock_sha256": lock_hash,
        "locked_package_count": len(expected),
        "version_mismatches": mismatches,
        "python": sys.version.split()[0],
        "platform": platform_name,
        "source_build_sha256": hashlib.sha256(
            json.dumps(source, sort_keys=True).encode()
        ).hexdigest(),
        "source_files_complete": len(source) == len(BUILD_FILES),
    }
    return identity
