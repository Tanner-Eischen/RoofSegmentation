#!/usr/bin/env python
"""
Fetch remote-sensing self-supervised pretrained encoder weights.

Default: SeCo (Seasonal Contrast, Mañas et al. ICCV 2021) ResNet-50 trained on
Sentinel-2 imagery. These are domain-matched to aerial/satellite input, which
typically beats ImageNet pretraining for building/roof segmentation.

Weights are PyTorch-Lightning .ckpt files hosted on Zenodo (record 4728033).
The prefix-remapping / loading into an SMP encoder happens in train_v2.py
(load_remote_weights); this script only downloads + md5-verifies + records a
manifest, so the (possibly large) download is decoupled from training.

Usage:
    python scripts/fetch_remote_weights.py                    # SeCo-1M R50 (default)
    python scripts/fetch_remote_weights.py --source seco100k  # SeCo-100K R50
    python scripts/fetch_remote_weights.py --out-dir pretrained

Sources / license:
    SeCo code:  https://github.com/ServiceNow/seasonal-contrast
    Paper:      https://arxiv.org/abs/2103.16607
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# Verified download URLs + md5sums from the SeCo repo README (Zenodo record 4728033).
SOURCES = {
    "seco": {
        "url": "https://zenodo.org/record/4728033/files/seco_resnet50_1m.ckpt?download=1",
        "filename": "seco_resnet50_1m.ckpt",
        "md5": "7b09c54aed33c0c988b425c54f4ef948",
        "desc": "SeCo-1M ResNet-50 (Sentinel-2 seasonal contrastive SSL)",
    },
    "seco100k": {
        "url": "https://zenodo.org/record/4728033/files/seco_resnet50_100k.ckpt?download=1",
        "filename": "seco_resnet50_100k.ckpt",
        "md5": "9672c303f6334ef816494c13b9d05753",
        "desc": "SeCo-100K ResNet-50 (Sentinel-2 seasonal contrastive SSL)",
    },
}

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "pretrained"
CHUNK = 1 << 20  # 1 MiB


def md5_file(path: Path, chunk: int = CHUNK) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` with a simple progress line. Resilient to redirects."""
    print(f"Downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "roof_detection/fetch_remote_weights"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as f:
        total = resp.length or 0
        done = 0
        while True:
            buf = resp.read(CHUNK)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total:
                pct = done * 100 / total
                print(f"\r  {done >> 20} / {total >> 20} MiB ({pct:5.1f}%)", end="", flush=True)
            else:
                print(f"\r  {done >> 20} MiB", end="", flush=True)
        print()
    tmp.replace(dest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=list(SOURCES), default="seco", help="Which RS-SSL checkpoint to fetch")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Directory to save weights + manifest")
    p.add_argument("--force", action="store_true", help="Re-download even if present & valid")
    args = p.parse_args()

    src = SOURCES[args.source]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / src["filename"]
    manifest_path = args.out_dir / "manifest.json"

    if dest.exists() and not args.force:
        actual = md5_file(dest)
        if actual == src["md5"]:
            print(f"Already present and valid (md5 OK): {dest}")
        else:
            print(f"Present but md5 mismatch (got {actual}); re-downloading...", file=sys.stderr)
            download(src["url"], dest)
    else:
        download(src["url"], dest)

    actual = md5_file(dest)
    ok = actual == src["md5"]
    manifest = {
        "source": args.source,
        "desc": src["desc"],
        "path": str(dest),
        "url": src["url"],
        "md5_expected": src["md5"],
        "md5_actual": actual,
        "valid": ok,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not ok:
        print(f"ERROR: md5 mismatch for {dest}: expected {src['md5']}, got {actual}", file=sys.stderr)
        print("The file may be corrupt or the host changed. Delete it and re-run, or download manually.",
              file=sys.stderr)
        return 1

    print(f"\nOK. Weights ready at: {dest}")
    print(f"Use in training via:  --encoder-init {dest}")
    print(f"Manifest written to:  {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
