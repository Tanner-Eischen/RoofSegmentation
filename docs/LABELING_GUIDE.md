# Create roof/building masks manually, without inventing labels

## Most efficient option: obtain existing human-labeled masks

WHU already supplies a matching mask for each image. If using WHU, **do not redraw
all roofs**. Import and audit them first. Manual annotation is for your own
independent imagery or for fixing genuinely incorrect labels.

## Labelme polygon workflow

The open-source desktop application is available at https://github.com/wkentaro/labelme
and its current instructions are at https://labelme.io/docs/annotate-image .

1. Use imagery you have rights to process and redistribute as labeled data.
   Record sensor, imagery date, source URL, resolution, and any CRS/transform.
2. Install with `python -m pip install labelme`, then run `labelme`.
3. Open one image, choose polygon mode, and trace the visible **building roof**.
   Label it `roof` or `building` (choose one convention for the entire dataset).
   The polygons should follow the *same target definition* as your benchmark.
4. Include all visible roof regions consistently, including attached extensions
   when they are part of the roof. Do not include ground, driveways, shadow,
   vegetation or unrelated pavement. Keep dormers and rooftop equipment within
   the parent roof if the task is roof extent, not rooftop-object detection.
5. Draw separate polygons for disconnected roof parts. Avoid self-intersecting
   polygons. Use `background` polygons only for genuinely excluded regions or
   internal holes. The `ignore` class is deliberately not implemented: decide
   before exporting whether a pixel belongs to the positive class or background.
6. Save `my_scene_01.json` beside `my_scene_01.jpg` or `.png`, and visually
   verify every saved image/JSON pair. Labelme stores points in image pixel
   coordinates and uses imageWidth/imageHeight metadata.
7. Create a `split_map.csv` with columns `stem,split,group`. Every geographic
   scene/location gets one stable group and only one split. Example:

```csv
stem,split,group
siteA_01,train,siteA
siteA_02,train,siteA
siteB_01,val,siteB
siteC_01,test,siteC
```

8. Convert polygons into label masks with the included script:

```bash
python -m roof_pipeline.prepare labelme --source data/my_annotations --split-map data/split_map.csv --out data/my_prepared
python -m roof_pipeline.prepare audit --root data/my_prepared
```

Generated masks are **single-channel** PNG with 0 for background and 255 for
roof/building. No anti-aliased intermediate gray levels. Image/mask size is
validated. `preview_train.png`, `preview_val.png` and `preview_test.png` make it
easy to spot polygon shifts and incorrectly assigned pixels.

The converter rejects unknown class names and non-polygon drawing types except
rectangles. It requires human-verified annotations; it will not infer masks
from an unlabeled photograph.

## Suggested review protocol

Two passes are advisable: first draw the polygons, then review them without the
annotation tool's fill obscuring the roof edges. Compare the generated mask
transparency overlay at 100% zoom. Inspect at least one roof-free tile and one
high-density tile. Check off-by-one crop boundaries. Compare annotation
interpretation across different labelers before merging groups.

## Pre-labeling from ML models (optional)

SAM/SAM2 or another segmentation model can propose masks that you correct
manually. Store those masks as **candidate labels** until an annotator accepts
and corrects them. Never train and evaluate the same model against its own
unedited pseudo-labels and report that as independent ground truth.
