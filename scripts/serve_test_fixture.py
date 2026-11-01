"""Run a temporary real Tiny checkpoint API for opt-in browser CI.

This is a generated, untrained integration fixture, never an accuracy/demo model.
No browser is launched here. Production startup remains api.main:app.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
import uvicorn

from api.main import Settings, create_app
from roof_pipeline.models import create_model
from roof_pipeline.release import prepare_release


def prepare_fixture(directory):
    directory = Path(directory)
    checkpoint = directory / "integration-only-tiny.pt"
    release = checkpoint.with_suffix(".release.json")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        state = {
            "state_dict": create_model("tiny").state_dict(),
            "config": {
                "arch": "tiny",
                "image_size": 32,
                "normalization": "imagenet_rgb_v1",
            },
            "threshold": 0.5,
        }
    torch.save(state, checkpoint)
    prepare_release(checkpoint, release)
    return Settings(str(checkpoint), model_release=str(release), demo_mode=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="roof-e2e-integration-") as directory:
        settings = prepare_fixture(directory)
        uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
