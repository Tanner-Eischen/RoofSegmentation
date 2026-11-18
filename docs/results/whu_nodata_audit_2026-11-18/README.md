# Original WHU NoData/validity audit

Original extracted TIFF metadata/masks/alpha/nonfinite audit; archive-member byte identity; importer-equivalent prepared pixel parity; no color heuristics

Audited 8,188 original image/label pairs against the publisher archive and unchanged prepared PNGs.

| Split | Pairs | Source invalid pixels | Lost invalid pixels | RGB mismatches | Label mismatches | Label nodata/class collisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 4736 | 0 | 0 | 0 | 0 | 0 |
| val | 1036 | 0 | 0 | 0 | 0 | 0 |
| test | 2416 | 0 | 0 | 0 | 0 | 0 |

No WHU metric rerun is required for this declared-validity fix: every source pixel is valid under declared metadata, and prepared pixels match the original importer semantics.

The two previously inspected agricultural tiles are included with their original source filenames in audit.json. Inspect their source-validity counts alongside the original error records.

Archive SHA256: `62a4b4ef0fdfc98deee44b29fd0fe949554a928e1e9d38730fc45360668e2892`.
Dataset content SHA256: `538ac8569dcfd817b8557bc23ed18977de5dbba8d4b19719c82ab5ee4d1a308d`.

The original 95 registered training artifacts, existing test/texture packets, source TIFFs, publisher archive and prepared image/mask bytes passed before/after identity checks.

- All-valid metadata cannot detect undocumented publisher missingness or incomplete labels.
- WHU background/class-zero remains valid; nodata collisions are flagged, not relabeled.
- No training, model loading, inference, metric recomputation or data mutation occurred.

[Full audit](audit.json) · [Per-pair evidence](pairs.csv) · [Checksums](checksums.json)

## Source interpretation and agricultural examples

All 8,188 original RGB TIFFs are three-band uint8 with no declared NoData,
no alpha channel, and all-valid GDAL mask flags. All original label TIFFs are
one-bit WhiteIsZero (PhotometricInterpretation=0): raw index 0 renders white
(building), and raw index 1 renders black (background). Pillow opens these
as mode `1` and applies that photometric interpretation. The original importer
converts mode `1` to luminance before binary normalization; mode-`P` labels
instead retain their class indices. The corrected audit follows that original
contract. All 8,188 normalized source labels exactly match their prepared PNGs.
There are no source validity sidecars or label NoData/class collisions.

| Prepared ID | Original RGB TIFF | Original label TIFF | Valid source pixels |
| --- | --- | --- | ---: |
| `whu_test_000046` | `test/image/14.tif` | `test/label/14.tif` | 262144 |
| `whu_test_001948` | `test/image/32.tif` | `test/label/32.tif` | 262144 |

Both agricultural examples retain every source pixel. Their recorded errors
are unaffected by declared NoData handling. This conclusion applies to the
same frozen dataset lineage; it is not a new validation of model accuracy,
annotation completeness, or geographic generalization. Undeclared publisher
missingness cannot be inferred from all-valid metadata or from black/white RGB.

[Execution/test source fingerprints](verification.json)
