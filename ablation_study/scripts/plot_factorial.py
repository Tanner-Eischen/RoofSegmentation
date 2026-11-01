#!/usr/bin/env python3
"""Render completed-study curves and fixed-sample validation comparisons.

Uses Pillow rather than an additional plotting dependency. Qualitative examples
are a deterministic random sample, never selected by model score or appearance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__:
    from .analyze_factorial import ARMS, read_results, sha256
else:
    from analyze_factorial import ARMS, read_results, sha256


COLORS = ("#1775a8", "#bd530a", "#488c35", "#8e559c")
LABELS = ("Uniform / standard", "Hard / standard", "Uniform / robust", "Hard / robust")


def selected_indices(rows: list[dict], count: int, seed: int) -> list[int]:
    if count < 1 or count > len(rows):
        raise ValueError("examples must be positive and no larger than validation size")
    if any(row.get("split") != "val" for row in rows):
        raise ValueError("Qualitative sampling is validation-only")
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["sample_id"])
    return random.Random(seed).sample(ordered, count)


def error_overlay(
    rgb: Image.Image, prediction: np.ndarray, truth: np.ndarray
) -> Image.Image:
    """Green TP, red FP, blue FN, with unchanged true-negative background."""
    base = np.asarray(rgb.convert("RGB"), dtype=np.float32).copy()
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    if prediction.shape != truth.shape or prediction.shape != base.shape[:2]:
        raise ValueError("RGB, prediction and ground truth must have the same shape")
    for mask, color in (
        (prediction & truth, (40, 210, 70)),
        (prediction & ~truth, (245, 45, 45)),
        (~prediction & truth, (40, 110, 255)),
    ):
        base[mask] = 0.45 * base[mask] + 0.55 * np.asarray(color)
    return Image.fromarray(np.rint(base).astype(np.uint8))


def learning_curves(out: Path, reg: dict, destination: Path) -> Path:
    image = Image.new("RGB", (1800, 1070), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    title_font = ImageFont.load_default(size=28)
    draw.text(
        (60, 20),
        "Fixed-budget factorial study: all training seeds",
        fill="black",
        font=title_font,
    )
    draw.text(
        (60, 65),
        "Validation IoU uses threshold 0.50. Curves show epochs, not independent spatial replicates.",
        fill="#333333",
        font=font,
    )
    histories = {}
    for arm in ARMS:
        for seed in reg["seeds"]:
            path = out / "runs" / f"{arm}_seed{seed}" / "history.json"
            history = json.loads(path.read_text(encoding="utf-8"))
            if [entry["epoch"] for entry in history] != list(
                range(1, reg["epochs"] + 1)
            ):
                raise ValueError(f"Incomplete epoch history: {path}")
            for entry in history:
                for key in ("val_iou_0p5", "train_loss"):
                    value = float(entry[key])
                    if (
                        not math.isfinite(value)
                        or value < 0
                        or (key == "val_iou_0p5" and value > 1)
                    ):
                        raise ValueError(f"Invalid {key}: {path}")
            histories[arm, seed] = history
    for panel, (key, label) in enumerate(
        (
            ("val_iou_0p5", "Global validation IoU at 0.50"),
            ("train_loss", "Training BCE + Dice loss"),
        )
    ):
        left, top, right, bottom = 105 + panel * 870, 180, 865 + panel * 870, 850
        values = [
            float(entry[key]) for history in histories.values() for entry in history
        ]
        lower, upper = min(values), max(values)
        padding = max((upper - lower) * 0.1, 0.005)
        lower, upper = max(0.0, lower - padding), upper + padding
        if key == "val_iou_0p5":
            upper = min(1.0, upper)
        draw.text((left, top - 45), label, fill="black", font=font)
        for tick in range(6):
            y = bottom - (bottom - top) * tick / 5
            value = lower + (upper - lower) * tick / 5
            draw.line((left, y, right, y), fill="#dddddd", width=1)
            draw.text((left - 85, y - 10), f"{value:.3f}", fill="#333333", font=font)
        draw.line((left, top, left, bottom, right, bottom), fill="black", width=2)
        ticks = sorted({1, reg["epochs"], *range(5, reg["epochs"] + 1, 5)})
        for epoch in ticks:
            x = left + (right - left) * (epoch - 1) / max(1, reg["epochs"] - 1)
            draw.text((x - 10, bottom + 12), str(epoch), fill="black", font=font)
        draw.text(
            ((left + right) / 2 - 25, bottom + 45), "Epoch", fill="black", font=font
        )
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(reg["seeds"]):
                points = [
                    (
                        left
                        + (right - left)
                        * (entry["epoch"] - 1)
                        / max(1, reg["epochs"] - 1),
                        bottom
                        - (bottom - top)
                        * (float(entry[key]) - lower)
                        / (upper - lower),
                    )
                    for entry in histories[arm, seed]
                ]
                # Different markers identify seeds without relying on line colors.
                if len(points) > 1:
                    draw.line(points, fill=COLORS[arm_index], width=2)
                for x, y in points:
                    radius = 3
                    box = (x - radius, y - radius, x + radius, y + radius)
                    if seed_index % 3 == 0:
                        draw.ellipse(box, fill=COLORS[arm_index])
                    elif seed_index % 3 == 1:
                        draw.rectangle(box, outline=COLORS[arm_index], width=2)
                    else:
                        draw.polygon(
                            ((x, y - 4), (x - 4, y + 3), (x + 4, y + 3)),
                            outline=COLORS[arm_index],
                        )
    for index, label in enumerate(LABELS):
        draw.text((70 + 430 * index, 945), label, fill=COLORS[index], font=font)
    draw.text(
        (70, 990),
        "Seed markers: "
        + "; ".join(
            f"{seed} = {('circle', 'square', 'triangle')[i % 3]}"
            for i, seed in enumerate(reg["seeds"])
        ),
        fill="#333333",
        font=font,
    )
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "learning_curves.png"
    image.save(path)
    return path


def qualitative_examples(
    out: Path, reg: dict, destination: Path, seed: int, count: int, device: str
) -> Path:
    import torch

    # Direct execution from the repository does not otherwise add the repo root.
    repository = Path(reg["repository"]).resolve()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from roof_pipeline.common import dataset_fingerprint
    from roof_pipeline.dataset import RoofDataset
    from roof_pipeline.models import load_model

    if seed not in reg["seeds"]:
        raise ValueError(
            "Qualitative seed must be one of the registered training seeds"
        )
    data = Path(reg["data"])
    if dataset_fingerprint(data) != reg["dataset_content_sha256"]:
        raise ValueError("Dataset content differs from study registration")
    dataset = RoofDataset(data, "val", size=reg["image_size"], augment=False)
    indices = selected_indices(dataset.rows, count, seed)
    selection = {
        "split": "val",
        "seed": seed,
        "selection": "seeded random sample of sorted validation sample IDs; independent of scores",
        "sample_ids": [dataset.rows[index]["sample_id"] for index in indices],
        "study_registration_sha256": sha256(out / "registration.json"),
        "dataset_content_sha256": reg["dataset_content_sha256"],
        "threshold": 0.5,
    }
    destination.mkdir(parents=True, exist_ok=True)
    selection_path = destination / f"qualitative_seed{seed}_selection.json"
    if (
        selection_path.exists()
        and json.loads(selection_path.read_text(encoding="utf-8")) != selection
    ):
        raise ValueError(
            "Frozen qualitative sample selection differs; use its original options"
        )
    # Persist IDs before loading or running any model.
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    tile, gap, header = 240, 12, 100
    canvas = Image.new(
        "RGB", (6 * (tile + gap) + gap, header + count * (tile + 35) + 45), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    draw.text(
        (gap, 8),
        f"Validation examples / training seed {seed} / final epoch / threshold 0.50",
        fill="black",
        font=font,
    )
    draw.text(
        (gap, 33),
        "Overlay: green = true positive; red = false positive; blue = false negative. Random tiles, no performance selection.",
        fill="black",
        font=font,
    )
    for column, label in enumerate(("RGB", "Ground truth", *LABELS)):
        draw.text((gap + column * (tile + gap), 70), label, fill="black", font=font)
    samples = []
    for row_index, index in enumerate(indices):
        x, truth = dataset[index]
        with Image.open(data / dataset.rows[index]["image"]) as raw:
            rgb = raw.convert("RGB").resize(
                (dataset.size, dataset.size), Image.Resampling.BILINEAR
            )
        target = truth[0].numpy().astype(bool)
        y = header + row_index * (tile + 35)
        canvas.paste(rgb.resize((tile, tile)), (gap, y))
        canvas.paste(
            Image.fromarray(target.astype(np.uint8) * 255)
            .convert("RGB")
            .resize((tile, tile), Image.Resampling.NEAREST),
            (gap + tile + gap, y),
        )
        draw.text(
            (gap, y + tile + 5),
            dataset.rows[index]["sample_id"],
            fill="black",
            font=font,
        )
        samples.append((x, rgb, target))
    checkpoint_hashes = {}
    for column, arm in enumerate(ARMS, start=2):
        checkpoint = out / "runs" / f"{arm}_seed{seed}" / "final.pt"
        checkpoint_hashes[arm] = sha256(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = state["config"]
        if state.get("threshold") != 0.5 or config.get("epoch") != reg["epochs"]:
            raise ValueError(
                "Qualitative checkpoint is not the registered final endpoint"
            )
        if (
            any(
                config.get(key) != reg[key]
                for key in ("arch", "image_size", "dataset_content_sha256")
            )
            or config.get("seed") != seed
        ):
            raise ValueError(
                "Qualitative checkpoint configuration differs from registration"
            )
        model, _ = load_model(checkpoint, device)
        with torch.inference_mode():
            for row_index, (x, rgb, target) in enumerate(samples):
                probability = (
                    model(x.unsqueeze(0).to(device)).sigmoid()[0, 0].cpu().numpy()
                )
                if not np.all(np.isfinite(probability)):
                    raise ValueError("Nonfinite qualitative prediction")
                overlay = error_overlay(rgb, probability >= 0.5, target)
                canvas.paste(
                    overlay.resize((tile, tile)),
                    (gap + column * (tile + gap), header + row_index * (tile + 35)),
                )
        del state, model
    path = destination / f"qualitative_seed{seed}.png"
    canvas.save(path)
    (destination / f"qualitative_seed{seed}_checkpoints.json").write_text(
        json.dumps(checkpoint_hashes, indent=2) + "\n", encoding="utf-8"
    )
    return path


def plot_study(
    out: Path, device: str = "cpu", seed: int | None = None, examples: int = 4
) -> list[Path]:
    reg, _ = read_results(out)
    if seed is None:
        seed = reg["seeds"][0]
    if seed not in reg["seeds"] or examples < 1:
        raise ValueError(
            "Choose a registered training seed and a positive example count"
        )
    destination = out / "analysis" / "figures"
    paths = [learning_curves(out, reg, destination)]
    paths.append(qualitative_examples(out, reg, destination, seed, examples, device))
    (destination / "figure_manifest.json").write_text(
        json.dumps(
            {
                "study_registration_sha256": sha256(out / "registration.json"),
                "device": device,
                "qualitative_seed": seed,
                "artifacts": {path.name: sha256(path) for path in paths},
                "note": "Validation-only descriptive diagnostics; no test inference or parameter selection.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--seed", type=int, help="Registered seed to illustrate; defaults to first"
    )
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()
    for path in plot_study(args.out.resolve(), args.device, args.seed, args.examples):
        print(path)


if __name__ == "__main__":
    main()
