# WHU annotation review — October 29, 2026

The first visual pass found concrete image/label mismatches. Annotation accuracy
is **not established** by the existing dimension and binary-encoding checks.
Do not correct labels automatically or treat this review as expert ground truth.

The [publisher's dataset description](https://gpcv.whu.edu.cn/data/building_dataset.html)
describes manually edited building vectors and rasterized labels. The description
does not specify inclusion rules for greenhouses, containers or trailers. Those
rules therefore remain unresolved in this review; visual differences alone do
not establish violations of an authoritative class policy.

## Sampling and evidence

The review packet is `runs/whu_annotation_audit_2026-10-29/`. It contains
`packet.json` with sample IDs, source hashes, sampling probabilities and population
counts; `review.csv` with findings for all 32 sampled tiles; native-resolution
RGB/GT-contour pairs in `panels/`; eight stratum contact sheets;
`targeted_review.json`; a separate `greenhouse_screen.json` with eight additional
thumbnail sheets and four native followups; a frozen-model trailer diagnostic;
and machine-readable `findings.json`.
Open the [case review CSV](../runs/whu_annotation_audit_2026-10-29/review.csv)
or the [sampling record](../runs/whu_annotation_audit_2026-10-29/packet.json).

Four tiles were sampled without replacement from each train/validation coverage
stratum, using seed **20261029**. All 32 paired panels were visually inspected.
GT contours are magenta; original RGB appears beside each overlay. Coverage is
GT-positive area per tile, not building size, greenhouse status or container type.

| Partition | Empty | Positive below 1% | 1% to below 10% | At least 10% |
| --- | ---: | ---: | ---: | ---: |
| Training population | 419 | 172 | 645 | 3,500 |
| Validation population | 299 | 106 | 201 | 430 |
| Reviewed per partition | 4 | 4 | 4 | 4 |

Equal allocation deliberately overrepresents sparse/empty tiles. Raw review
counts must not be reported as dataset prevalence. The sample is geographically
restricted to the supplied WHU partitions. This was one AI-assisted visual pass;
expert adjudication, repeat reviewers and inter-reviewer agreement are missing.
Test imagery/masks were not decoded, and no source labels were edited.

## Case findings

- **Local label/image mismatch:** [whu_train_000787 paired panel](../runs/whu_annotation_audit_2026-10-29/panels/whu_train_000787.png), a randomly sampled dense
  residential tile, contains a positive polygon over bare brown ground near
  original-image x=60–120, y=337–410. The adjacent long dark roof aligns. A stale
  footprint or temporal mismatch is possible, but its cause is unknown. This
  cannot be explained simply as a common translation of all outlines.
- **Large apparent omissions, diagnostic followup:** [whu_train_002862 paired panel](../runs/whu_annotation_audit_2026-10-29/panels/targeted_whu_train_002862.png) contains
  several large pale shed-like roofs at the left/middle, approximately x=0–185,
  y=90–440, with background masks. The large blue roof at lower right is labelled.
  This tile was selected from a separate screen of the 48 highest baseline
  training-error tiles and is **not part of the random sample**. The screen used
  the existing refinement run's `hard_examples.csv`; its hash and three native
  followups are recorded in `targeted_review.json`. Selection by model error is
  biased, so this demonstrates a case rather than an omission rate.
- **Small-structure uncertainty:** `whu_train_003668` has an unlabelled narrow
  pale object north of a labelled farm roof. `whu_val_000164` has red/pale
  rectangles in a field with an empty mask. These could be structures, equipment,
  debris or former slabs; neither is counted as a confirmed building omission.
- **Outlines:** no common translation was obvious across the reviewed positive
  tiles. Some attachments, tree-covered edges and small roofs remain ambiguous.
  This observation does not bound subpixel/pixel alignment error or prove that
  every boundary is accurate. Formal shift measurements need adjudicated outlines.
- **Containers/trailers:** `whu_train_002947` has labelled and unlabelled small
  rectangles in one equipment/storage yard. `whu_val_000778` excludes a narrow
  red/white/blue row beside a lane, and `whu_val_000666` excludes rows of small
  brown/orange rectangles beside labelled industrial roofs. Conversely, two
  isolated box/shed-like objects in `whu_val_000395` are positive. These are useful
  review cases, but RGB does not establish fixed shed versus mobile container.
  **Inconsistent treatment of known containers has not yet been established.**
- **Greenhouses:** no confidently identified greenhouse was encountered in the
  32-tile sample or the error-targeted followups. The green roofs in
  `whu_val_000054` resemble hangars/sheds; color is not a greenhouse label.
  An additional bounded search screened **256 RGB thumbnails**: 128 uniformly
  sampled training tiles with seed 20261030, and 128 validation tiles with seed
  20261031. This separate search does not change the original 32-tile sample;
  exact IDs, image hashes and overlap are recorded in `greenhouse_screen.json`.
  Four candidates were followed up at native resolution. The
  [nursery in whu_val_001022](../runs/whu_annotation_audit_2026-10-29/panels/horticulture_whu_val_001022.png)
  has open planting rows and roof/shade structures with different inclusion in
  the binary mask. The structures' enclosure and material cannot be established
  from RGB. Two other candidates resolved to residential roofs and a structure
  frame; the fourth resolved to caravan storage. No confidently identifiable
  enclosed greenhouse was established. **Greenhouse inclusion consistency remains
  unverified** and needs independently identified sites and an explicit policy
  for open shade structures. Thumbnail screening can miss small structures.

The targeted construction case `whu_train_002208` also shows that a building frame
under construction is positive. A written policy should distinguish this from
exposed foundations and demolished sites before any annotation revisions.

The frozen validation-winning U-Net was also checked on the targeted omission
case `whu_train_002862`, using CPU and threshold 0.50. Its
[prediction-disagreement panel](../runs/whu_annotation_audit_2026-10-29/omission_prediction_diagnostic.png)
shows predicted building regions on several visible, unlabelled roof areas.
In the approximate x=[0,185), y=[90,440) rectangle, 17,535 pixels count as false
positives against the supplied mask. That rectangle includes background and is
not an expert object annotation; the count cannot establish true roof accuracy.
This illustrates why a mask-score improvement alone cannot resolve the observed
annotation concerns. Counts and frozen-input hashes are in
`omission_prediction_diagnostic.json`. This remains a selected training case,
separate from representative review and validation metrics.

## Frozen U-Net diagnostic on recognizable trailers

The additional screen found rows of visually recognizable white caravans/trailers
in `whu_val_000251`. GT excludes the rows and includes the residence and shed.
The frozen validation-winning U-Net checkpoint
`runs/whu_refinement_2026-10-28/control/best.pt` was run on the native 512px tile
using CPU, four threads, and its unchanged validation-calibrated threshold 0.50.

| Region | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: |
| Full tile | 4,215 | 217 | 273 | 257,439 |
| Approximate trailer-row rectangle | 0 | 0 | 0 | 51,000 |

The rectangle is x=[0,300), y=[0,170) in original-image pixels and includes
trailers, gaps, grass, trees and shadows. It is **not an object annotation**.
Both GT and predicted positives are zero throughout this ROI. The
[RGB/GT/prediction panel](../runs/whu_annotation_audit_2026-10-29/trailer_val000251_diagnostic.png)
shows the correct exclusion at this site. Checkpoint/input hashes, threshold and
pixel counts are saved in `trailer_val000251_diagnostic.json`; probability and
binary prediction PNGs are retained. No model or threshold was changed.

This one-site diagnostic provides no evidence of substantial trailer confusion.
It does not prove performance on other trailer/container types or locations.

## Three-class pilot decision

Defer a production class change. The review establishes annotation concerns, but
does not establish substantial model confusion on **confirmed** containers or
trailers; the recognizable caravan site was correctly excluded. Have a domain
reviewer adjudicate the listed rectangles, identify
additional greenhouse sites, and compare frozen U-Net predictions on the same
train/validation cases. Aerial ambiguity should be recorded as unknown/ignore,
not forced into a semantic class.

If adjudication confirms repeated container/trailer false positives across
multiple independent sites, use a small separate pilot with background, fixed
building, and container/trailer classes. Define inclusion rules for sheds,
greenhouses, temporary structures, construction frames and demolition remnants
before labelling. Double-label a subset, track agreement and adjudication, split
by site, and compare binary building IoU plus container precision/recall against
the frozen binary baseline. Estimate object prevalence and confusion on the
pilot's representative portion; keep deliberately enriched hard cases separate.
Do not remap the current binary masks into three classes heuristically.

## Reproduce the packet

Use a fresh output directory; the generator refuses to overwrite reviewed data.
It generates pending review rows, not the visual findings recorded above.

```powershell
python -m roof_pipeline.annotation_audit --data data/whu_official_prepared --out runs/whu_annotation_audit_reproduction --per-stratum 4 --seed 20261029
python -m pytest tests/test_annotation_audit.py -q
```

Verification: all three audit tests passed, including reproducible sampling,
inclusion counts/probabilities, native panel creation, overwrite protection,
contour geometry and a corrupted test mask that proves test pixels are not read.
Ruff formatting and lint passed for the audit module and tests.
