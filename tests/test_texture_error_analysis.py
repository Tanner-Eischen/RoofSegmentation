"""Count reconciliation and honest post-hoc error selection without model inference."""

import copy

import pytest

from scripts.analyze_architecture_comparison import ARCHITECTURES
from scripts.analyze_texture_errors import analyze_rows, mask_coverage, select_examples


def fixture_rows():
    control = [
        {"sample_id": "field_b", "tp": 0, "fp": 2, "fn": 0, "tn": 98},
        {"sample_id": "field_a", "tp": 0, "fp": 1, "fn": 0, "tn": 99},
        {"sample_id": "buildings", "tp": 20, "fp": 5, "fn": 10, "tn": 65},
    ]
    rows = {arch: copy.deepcopy(control) for arch in ARCHITECTURES}
    rows["unetplusplus-resnet34"][0].update(fp=32, tn=68)
    rows["unetplusplus-resnet34"][1].update(fp=31, tn=69)
    rows["unetplusplus-resnet34"][2].update(tp=25, fn=5, fp=10, tn=60)
    rows["segformer-b0"].reverse()  # Identity alignment must not rely on row order.
    return rows


def test_empty_and_building_groups_reconcile_global_counts_and_denominators():
    result = analyze_rows(fixture_rows())
    model = result["models"]["unetplusplus-resnet34"]
    empty, buildings = (
        model["groups"]["empty_gt"],
        model["groups"]["contains_buildings"],
    )
    assert empty["samples"] == 2 and buildings["samples"] == 1
    assert empty["iou"] is None  # Avoid a meaningless mean empty-tile IoU.
    assert empty["false_positive_pixel_fraction"] == 63 / 200
    assert empty["tiles_with_false_positives"] == 2
    assert buildings["iou"] == 25 / 40
    assert model["overall"]["iou"] == 25 / 103
    for key in ("tp", "fp", "fn", "tn"):
        assert empty[key] + buildings[key] == model["overall"][key]
    assert sum(group["pixels"] for group in model["groups"].values()) == 300


def test_posthoc_ranking_uses_fp_increase_and_stable_id_tie_break():
    comparison = analyze_rows(fixture_rows())["per_tile"]
    selected = select_examples(list(reversed(comparison)), 2)
    assert [row["sample_id"] for row in selected] == ["field_a", "field_b"]
    assert [row["unetplusplus_minus_unet_fp"] for row in selected] == [30, 30]
    assert all(row["gt_positive_pixels"] == 0 for row in selected)
    with pytest.raises(ValueError):
        select_examples(comparison, 0)


def test_coverage_decodes_masks_and_counts_empty_tiles_per_source_split(tmp_path):
    from PIL import Image

    manifest = []
    for index, (split, value) in enumerate(
        (("train", 0), ("train", 255), ("val", 255), ("test", 0))
    ):
        relative = f"mask{index}.png"
        Image.new("L", (4, 4), value).save(tmp_path / relative)
        manifest.append({"mask": relative, "split": split})
    result = mask_coverage(tmp_path, manifest)["splits"]
    assert result["train"] == {
        "samples": 2,
        "empty_gt_tiles": 1,
        "empty_gt_tile_fraction": 0.5,
    }
    assert result["val"]["empty_gt_tile_fraction"] == 0
    assert result["test"]["empty_gt_tile_fraction"] == 1


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "ground_truth", "pixels", "fractional", "negative"],
)
def test_inconsistent_row_coverage_or_truth_cannot_generate_a_comparison(mutation):
    rows = fixture_rows()
    target = rows["unetplusplus-resnet34"]
    if mutation == "missing":
        target.pop()
    elif mutation == "duplicate":
        target.append(dict(target[0]))
    elif mutation == "ground_truth":
        target[0].update(tp=1, tn=67)
    elif mutation == "pixels":
        target[0]["tn"] += 1
    elif mutation == "fractional":
        target[0]["fp"] = 3.5
    else:
        target[0]["fp"] = -1
    with pytest.raises(ValueError):
        analyze_rows(rows)
