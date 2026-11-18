"""Read-only audit of original WHU TIFF validity and prepared PNG parity.

No model loading, training or inference. Pixel colors never imply missingness.
Label nodata collisions with binary class codes are reported as ambiguous.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import warnings
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.errors import NotGeoreferencedWarning

COUNTS = {"train": 4736, "val": 1036, "test": 2416}
ARCHIVE_SHA = "62a4b4ef0fdfc98deee44b29fd0fe949554a928e1e9d38730fc45360668e2892"


def stream_sha(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024**2), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256(path):
    with Path(path).open("rb") as stream:
        return stream_sha(stream)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def inspect_raster(path, *, label=False):
    """Read real paths so GDAL applies sidecars; do not let label nodata redefine classes."""
    with rasterio.open(path) as source:
        indexes = (
            [1]
            if label
            else [
                i + 1
                for i, ci in enumerate(source.colorinterp)
                if ci in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            ]
        )
        if not label and len(indexes) != 3:
            if source.count < 3:
                raise ValueError(f"Expected RGB source: {path}")
            indexes = [1, 2, 3]
        values = source.read(indexes)
        masks = source.read_masks(indexes) > 0
        dataset_valid = source.dataset_mask() > 0
        finite = np.isfinite(values).all(axis=0)
        alpha_indexes = [
            i + 1 for i, ci in enumerate(source.colorinterp) if ci == ColorInterp.alpha
        ]
        alpha_valid = np.ones(source.shape, bool)
        for index in alpha_indexes:
            alpha_valid &= source.read(index) > 0
        declared_valid = masks.all(axis=0) & dataset_valid & alpha_valid
        validity = declared_valid & finite
        nodata = [
            float(value)
            if value is not None and math.isfinite(value)
            else str(value)
            if value is not None
            else None
            for value in source.nodatavals
        ]
        collision = label and any(
            value in (0, 1, 255) for value in source.nodatavals if value is not None
        )
        sidecars = [
            Path(name)
            for name in source.files
            if Path(name).resolve() != Path(path).resolve()
        ]
        metadata = {
            "width": source.width,
            "height": source.height,
            "bands": source.count,
            "dtypes": list(source.dtypes),
            "nodata": nodata,
            "mask_flags": [
                [flag.name for flag in flags] for flags in source.mask_flag_enums
            ],
            "colorinterp": [ci.name for ci in source.colorinterp],
            "alpha_bands": len(alpha_indexes),
            "alpha_invalid_pixels": int((~alpha_valid).sum()),
            "declared_invalid_pixels": int((~declared_valid).sum()),
            "nonfinite_pixels": int((~finite).sum()),
            "invalid_pixels": int((~validity).sum()),
            "nodata_class_collision": bool(collision),
            "palette": {
                str(key): list(value)
                for key, value in source.colormap(1).items()
                if key in np.unique(values)
            }
            if label and source.colorinterp[0] == ColorInterp.palette
            else None,
            "sidecars": [
                {"name": sidecar.name, "sha256": sha256(sidecar)}
                for sidecar in sidecars
            ],
        }
        return values, validity, metadata


def inspect_png(path, *, label=False):
    with Image.open(path) as image:
        alpha = np.asarray(image.convert("RGBA"))[..., 3] > 0
        values = np.asarray(
            image
            if label and image.mode == "P"
            else image.convert("L" if label else "RGB")
        ).copy()
    if label:
        if not set(np.unique(values)).issubset({0, 1, 255}):
            raise ValueError(f"Nonbinary prepared labels: {path}")
        values = np.where(values > 0, 255, 0).astype(np.uint8)
    return values, alpha


def original_label_pixels(path):
    """Reproduce the ORIGINAL importer: mode-P class indices, otherwise rendered L.

    WHU source TIFFs are 1-bit WhiteIsZero: GDAL index0 is white/building,
    index1 is black/background. PIL mode1 honors that photometric interpretation.
    """
    with Image.open(path) as image:
        if image.mode not in {"P", "1", "L"}:
            raise ValueError(f"Unexpected source label mode: {image.mode}")
        mode = image.mode
        photometric = image.tag_v2.get(262) if hasattr(image, "tag_v2") else None
        pixels = np.asarray(image if mode == "P" else image.convert("L")).copy()
    if not set(np.unique(pixels)).issubset({0, 1, 255}):
        raise ValueError("Unexpected rendered label codes")
    return np.where(pixels > 0, 255, 0).astype(np.uint8), mode, photometric


def audit_pair(source_image, source_label, prepared_image, prepared_label):
    rgb, image_valid, image_meta = inspect_raster(source_image)
    label, label_valid, label_meta = inspect_raster(source_label, label=True)
    if rgb.shape[1:] != label.shape[1:] or label.shape[0] != 1:
        raise ValueError("Source image/label geometry differs")
    if not set(np.unique(label)).issubset({0, 1, 255}):
        raise ValueError("Unexpected source label indices; resolve ontology explicitly")
    prepared_rgb, prepared_image_valid = inspect_png(prepared_image)
    prepared_mask, prepared_label_valid = inspect_png(prepared_label, label=True)
    rgb = np.moveaxis(rgb, 0, -1)
    normalized_label, label_mode, photometric = original_label_pixels(source_label)
    image_parity = rgb.shape == prepared_rgb.shape and np.array_equal(rgb, prepared_rgb)
    label_parity = normalized_label.shape == prepared_mask.shape and np.array_equal(
        normalized_label, prepared_mask
    )
    source_valid = image_valid & label_valid
    if source_valid.shape == prepared_image_valid.shape == prepared_label_valid.shape:
        lost = int((~source_valid & prepared_image_valid & prepared_label_valid).sum())
    else:
        lost = int((~source_valid).sum())
    return {
        "pixels": int(source_valid.size),
        "image_rgb_pixel_parity": image_parity,
        "normalized_label_pixel_parity": label_parity,
        "source_invalid_pixels": int((~source_valid).sum()),
        "prepared_lost_declared_or_nonfinite_invalid_pixels": lost,
        "source_label_values": json.dumps(np.unique(label).tolist()),
        "source_label_pil_mode": label_mode,
        "source_label_photometric": photometric,
        "source_label_rendered_values": json.dumps(
            np.unique(normalized_label).tolist()
        ),
        **{f"image_{key}": value for key, value in image_meta.items()},
        **{f"label_{key}": value for key, value in label_meta.items()},
    }


def fingerprint(rows, file_hashes, data):
    """Exact pre-existing dataset-content fingerprint, independently implemented."""
    digest = hashlib.sha256()
    for name in ("manifest.csv", "dataset_meta.json"):
        digest.update((data / name).read_bytes())
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        for field in ("image", "mask"):
            relative = row[field]
            digest.update(
                json.dumps(
                    [row["sample_id"], field, relative], ensure_ascii=True
                ).encode()
            )
            digest.update(bytes.fromhex(file_hashes[relative]))
    return digest.hexdigest()


def protected_files(repo, results):
    manifest_path = results / "remote_result_manifest.json"
    registered = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(registered) + 1 != 95:
        raise ValueError("Expected all 95 original registered training artifacts")
    files = {
        path.relative_to(repo).as_posix(): sha256(path)
        for path in [manifest_path, *[results / name for name in registered]]
    }
    for name, expected in registered.items():
        if files[(results / name).relative_to(repo).as_posix()] != expected["sha256"]:
            raise ValueError(
                f"Original training artifact differs from registration: {name}"
            )
    for packet in ("whu_architecture_test_2026-11-16", "whu_texture_errors_2026-11-16"):
        root = repo / "docs/results" / packet
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files[path.relative_to(repo).as_posix()] = sha256(path)
    return files


def run(source, archive, data, results, out, workers=4):
    source, archive, data, results, out = [
        Path(path).resolve() for path in (source, archive, data, results, out)
    ]
    repo = Path(__file__).resolve().parents[1]
    if workers < 1 or workers > 8:
        raise ValueError("Use 1–8 bounded CPU workers")
    if out.exists() or any(
        out == path or out.is_relative_to(path) for path in (source, data, results)
    ):
        raise ValueError(
            "Use a fresh audit directory outside original data and results"
        )
    metadata = json.loads((data / "dataset_meta.json").read_text(encoding="utf-8"))
    provenance_path = source / "download_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        metadata["source_provenance"]["archive_sha256"] != ARCHIVE_SHA
        or provenance["archive_sha256"] != ARCHIVE_SHA
        or provenance["source"] != "official"
        or sha256(archive) != ARCHIVE_SHA
    ):
        raise ValueError("Original publisher archive or provenance SHA differs")
    with (data / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if Counter(row["split"] for row in rows) != COUNTS or len(
        {row["sample_id"] for row in rows}
    ) != sum(COUNTS.values()):
        raise ValueError("Expected unchanged full publisher partitions")
    before = protected_files(repo, results)
    identity_files = [
        data / "manifest.csv",
        data / "dataset_meta.json",
        provenance_path,
    ]
    identity_hashes = {str(path): sha256(path) for path in identity_files}
    tasks = []
    for split, count in COUNTS.items():
        images = sorted((source / split / "image").glob("*.tif"))
        labels = {path.stem: path for path in (source / split / "label").glob("*.tif")}
        if len(images) != count or len(labels) != count:
            raise ValueError("Original TIFF counts differ")
        manifest = {row["sample_id"]: row for row in rows if row["split"] == split}
        for index, image in enumerate(images):
            ident = f"whu_{split}_{index:06d}"
            row = manifest[ident]
            tasks.append((ident, split, image, labels[image.stem], row))
    with zipfile.ZipFile(archive) as original:
        members = {}
        for member in original.infolist():
            if member.is_dir():
                continue
            relative = Path(*Path(member.filename).parts[1:]).as_posix()
            if relative in members:
                raise ValueError("Duplicate archive relative path")
            members[relative] = member.filename
        if len(members) != 2 * len(tasks):
            raise ValueError("Archive has unexpected payload files")

        def inspect(task):
            ident, split, image, label, row = task
            evidence = {
                "sample_id": ident,
                "split": split,
                "source_image": image.relative_to(source).as_posix(),
                "source_label": label.relative_to(source).as_posix(),
                "prepared_image": row["image"],
                "prepared_label": row["mask"],
            }
            for field, path in (("source_image", image), ("source_label", label)):
                evidence[f"{field}_sha256"] = sha256(path)
                with original.open(members[evidence[field]]) as stream:
                    archive_sha = stream_sha(stream)
                if archive_sha != evidence[f"{field}_sha256"]:
                    raise ValueError(f"Extracted source differs from archive: {path}")
            for field, relative in (
                ("prepared_image", row["image"]),
                ("prepared_label", row["mask"]),
            ):
                evidence[f"{field}_sha256"] = sha256(data / relative)
            return {
                **evidence,
                **audit_pair(image, label, data / row["image"], data / row["mask"]),
            }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            evidence = []
            for item in pool.map(inspect, tasks):
                evidence.append(item)
                if len(evidence) % 500 == 0:
                    print(
                        f"Audited {len(evidence)}/{len(tasks)} source/prepared pairs",
                        flush=True,
                    )
    original_hashes = {
        item[field]: item[f"{field}_sha256"]
        for item in evidence
        for field in ("source_image", "source_label")
    }
    for item in evidence:
        for kind in ("image", "label"):
            for sidecar in item[f"{kind}_sidecars"]:
                relative = Path(item[f"source_{kind}"]).parent / sidecar["name"]
                original_hashes[relative.as_posix()] = sidecar["sha256"]
    prepared_hashes = {
        item[field]: item[f"{field}_sha256"]
        for item in evidence
        for field in ("prepared_image", "prepared_label")
    }
    dataset_sha = fingerprint(rows, prepared_hashes, data)
    if (
        dataset_sha
        != "538ac8569dcfd817b8557bc23ed18977de5dbba8d4b19719c82ab5ee4d1a308d"
    ):
        raise ValueError("Prepared dataset differs from original study identity")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        checks = list(
            pool.map(
                lambda entry: sha256(entry[0]) == entry[1],
                [(source / name, value) for name, value in original_hashes.items()]
                + [(data / name, value) for name, value in prepared_hashes.items()],
            )
        )
    if (
        not all(checks)
        or protected_files(repo, results) != before
        or sha256(archive) != ARCHIVE_SHA
    ):
        raise ValueError(
            "Original source, data or scientific artifacts changed during audit"
        )
    if any(sha256(Path(path)) != value for path, value in identity_hashes.items()):
        raise ValueError("Dataset identity or source provenance changed during audit")
    summary = {}
    for split in COUNTS:
        subset = [item for item in evidence if item["split"] == split]
        summary[split] = {
            "pairs": len(subset),
            "pixels": sum(item["pixels"] for item in subset),
        }
        for key in (
            "source_invalid_pixels",
            "prepared_lost_declared_or_nonfinite_invalid_pixels",
            "image_declared_invalid_pixels",
            "label_declared_invalid_pixels",
            "image_nonfinite_pixels",
            "label_nonfinite_pixels",
            "image_alpha_invalid_pixels",
            "label_alpha_invalid_pixels",
        ):
            summary[split][key] = sum(item[key] for item in subset)
        for key in ("image_rgb_pixel_parity", "normalized_label_pixel_parity"):
            summary[split][f"{key}_failures"] = sum(not item[key] for item in subset)
        summary[split]["label_nodata_class_collisions"] = sum(
            item["label_nodata_class_collision"] for item in subset
        )
    unaffected = all(
        item["source_invalid_pixels"] == 0
        and item["image_rgb_pixel_parity"]
        and item["normalized_label_pixel_parity"]
        and not item["label_nodata_class_collision"]
        for item in evidence
    )
    report = {
        "schema_version": 1,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Original extracted TIFF metadata/masks/alpha/nonfinite audit; archive-member byte identity; importer-equivalent prepared pixel parity; no color heuristics",
        "label_policy": "Original importer preserves mode-P class indices but uses rendered L for mode1/L. WHU 1-bit WhiteIsZero TIFFs: raw0=white/building, raw1=black/background; PIL mode1 applies this photometric interpretation before normalization.",
        "source_archive_sha256": ARCHIVE_SHA,
        "dataset_content_sha256": dataset_sha,
        "prepared_metadata_sha256": sha256(data / "dataset_meta.json"),
        "source_provenance_sha256": sha256(provenance_path),
        "source_sidecar_files": len(original_hashes) - 2 * len(tasks),
        "split_summary": summary,
        "all_source_pairs_match_archive": True,
        "source_and_prepared_files_unchanged": True,
        "original_95_training_artifacts_unchanged": True,
        "original_test_and_texture_packets_unchanged": True,
        "protected_artifact_sha256": before,
        "historical_whu_scores_unaffected_by_declared_validity_bug": unaffected,
        "selected_agricultural_tiles": [
            item
            for item in evidence
            if item["sample_id"] in {"whu_test_000046", "whu_test_001948"}
        ],
        "metadata_profiles": {
            kind: dict(
                Counter(
                    json.dumps(
                        {
                            key: item[f"{kind}_{key}"]
                            for key in (
                                "nodata",
                                "mask_flags",
                                "colorinterp",
                                "dtypes",
                                "bands",
                            )
                        },
                        sort_keys=True,
                    )
                    for item in evidence
                )
            )
            for kind in ("image", "label")
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
        "limits": [
            "All-valid metadata cannot detect undocumented publisher missingness or incomplete labels.",
            "WHU background/class-zero remains valid; nodata collisions are flagged, not relabeled.",
            "No training, model loading, inference, metric recomputation or data mutation occurred.",
        ],
    }
    out.mkdir(parents=True)
    with (out / "pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evidence[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, allow_nan=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in item.items()
            }
            for item in evidence
        )
    write_json(out / "audit.json", report)
    lines = [
        "# Original WHU NoData/validity audit",
        "",
        report["method"],
        "",
        f"Audited {len(evidence):,} original image/label pairs against the publisher archive and unchanged prepared PNGs.",
        "",
        "| Split | Pairs | Source invalid pixels | Lost invalid pixels | RGB mismatches | Label mismatches | Label nodata/class collisions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, item in summary.items():
        lines.append(
            f"| {split} | {item['pairs']} | {item['source_invalid_pixels']} | {item['prepared_lost_declared_or_nonfinite_invalid_pixels']} | {item['image_rgb_pixel_parity_failures']} | {item['normalized_label_pixel_parity_failures']} | {item['label_nodata_class_collisions']} |"
        )
    lines += [
        "",
        "No WHU metric rerun is required for this declared-validity fix: every source pixel is valid under declared metadata, and prepared pixels match the original importer semantics."
        if unaffected
        else "Validity/parity concerns were found. Preserve old results; a corrected version and separate evaluation are required before drawing conclusions.",
        "",
        "The two previously inspected agricultural tiles are included with their original source filenames in audit.json. Inspect their source-validity counts alongside the original error records.",
        "",
        "Archive SHA256: `" + ARCHIVE_SHA + "`.",
        "Dataset content SHA256: `" + dataset_sha + "`.",
        "",
        "The original 95 registered training artifacts, existing test/texture packets, source TIFFs, publisher archive and prepared image/mask bytes passed before/after identity checks.",
        "",
        *["- " + limit for limit in report["limits"]],
        "",
        "[Full audit](audit.json) · [Per-pair evidence](pairs.csv) · [Checksums](checksums.json)",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        out / "checksums.json",
        {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()},
    )
    print(
        json.dumps(
            {"pairs": len(evidence), "unaffected": unaffected, "packet": str(out)}
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/whu_official"))
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/whu_official.zip")
    )
    parser.add_argument("--data", type=Path, default=Path("data/whu_official_prepared"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("runs/runpod_results_batch16_2026-11-15/results"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    run(args.source, args.archive, args.data, args.results, args.out, args.workers)


if __name__ == "__main__":
    main()
