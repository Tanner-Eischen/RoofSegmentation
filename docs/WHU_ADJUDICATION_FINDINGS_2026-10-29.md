# WHU provisional adjudication — October 29, 2026

This is a second **AI-assisted visual pass**, not independent expert ground truth.
The proposed class policy and all decisions remain provisional until a domain
reviewer accepts them. No source dataset, trained checkpoint or test label was
changed. The purpose is to make the review concrete and to avoid silently
treating uncertain rectangles as established building or container classes.

## Scope and method

The pass reviewed all eight native-resolution contact sheets from the original
32-tile stratified train/validation sample, plus nine distinct native panels:
the seven targeted followups recorded in `targeted_review.json` and
`greenhouse_screen.json`, and `whu_train_000787`/`whu_val_000164`. The resulting
decision file covers **39 unique cases: 20 training and 19 validation**. The
original 32-tile sample is preserved; seven targeted cases remain a separate
cohort. Neither raw decision counts nor targeted cases estimate population error
rates. Pixel-level agreement and inter-reviewer reliability are not established.

All proposed geometry was traced from RGB and compared with the supplied mask.
Frozen U-Net predictions and probability maps were not used to draw corrections.
The bare-ground removal encloses one disconnected source-mask component after
checking its location in RGB; a bounding rectangle alone would have clipped the
neighbouring roof. Shed proposals follow visible roof/eave geometry and exclude
the adjoining dark cast shadows. Fine edges and tree-obscured eaves remain
approximate candidate geometry.

The [decision and polygon file](../runs/whu_adjudication_2026-10-29/proposals.json)
contains a rationale for every case and all polygon vertices in original
512 × 512 pixel coordinates. The 28-pixel panel header is not part of those
coordinates.

## Draft inclusion policy

- Include visible permanent roofed buildings and sheds, following roof/eave
  geometry rather than cast shadows.
- Exclude ground, exposed slabs, open nursery beds, pavement and cast shadows.
- Exclude recognizable mobile caravans/trailers.
- Preserve source labels and record uncertainty when fixed/mobile identity,
  greenhouse enclosure, open shade structures, storage tanks or construction
  stage cannot be resolved. These cases need explicit policy and object identity
  evidence before changing a mask.
- Treat tree-hidden and clipped edges as uncertain geometric details rather
  than claiming a pixel-exact corrected reference.

These are review proposals, not claims about the publisher's original class
policy. In particular, the source includes construction frames and a circular
tank-like roof; this review does not automatically erase those classes.

## Decisions

| Provisional decision | Cases | Meaning |
| --- | ---: | --- |
| Keep | 21 | No defensible binary change identified in this visual pass |
| Correct | 2 | Concrete candidate mask change supplied for review |
| Uncertain | 16 | Preserve binary mask; semantic/policy question remains |

No ignore regions were introduced. A broad ignore rectangle could remove
legitimate model errors as well as ambiguous objects. The 16 uncertain cases
therefore remain unresolved rather than receiving a forced class assignment.
The scoring workflow must keep them separate from an accepted corrected subset.

### Candidate corrections

| Case | Proposal | Changed pixels |
| --- | --- | ---: |
| `whu_train_000787` | Remove disconnected positive polygon on visibly bare ground | 2,219 removed; 0 added |
| `whu_train_002862` | Add eight visibly omitted shed-roof polygons | 26,844 added; 0 removed |

For `whu_train_000787`, the removed source component has bounding box
x=[53,113), y=[315,382). The adjacent long dark roof is retained. A source-date
or demolition mismatch is possible, but its cause is unknown. Inspect the
[proposal comparison](../runs/whu_adjudication_2026-10-29/whu_train_000787_proposal_qa.png)
and [coordinate crop](../runs/whu_adjudication_2026-10-29/whu_train_000787_detail_coordinates.png).

For `whu_train_002862`, roofs along the left/middle are visibly present while the
source labels include the lower-right industrial roof and some lower-edge
structures. Eight additions are supplied rather than redrawing the full tile.
Inspect the [proposal comparison](../runs/whu_adjudication_2026-10-29/whu_train_002862_proposal_qa.png)
and [coordinate crop](../runs/whu_adjudication_2026-10-29/whu_train_002862_detail_coordinates.png).
A reviewer must still check the proposed eave positions, tree occlusion and
tile-edge clipping. These candidate masks are not ready for confirmatory scoring
or training as independently verified labels.

The upper-right region around x=395–511, y=90–260 was explicitly rechecked on the
single RGB tile. It shows irregular quarry stockpiles and haul-road/gravel surfaces
above the railway, with no defensible roof/eave polygon. It remains background;
the [coordinate crop](../runs/whu_adjudication_2026-10-29/whu_train_002862_upper_right_retained_background.png)
records this decision. Side-by-side repeated panels should not be mistaken for
an extension of the original 512-pixel image.

### Remaining questions

- `whu_val_000164`: red/pale objects could be stored panels or remnants, slabs,
  or roofed structures. No building omission is confirmed.
- `whu_train_001516`, `whu_train_002212`, `whu_train_002947`,
  `whu_train_003668`, `whu_train_003741`, `whu_val_000395`,
  `whu_val_000778`, `whu_val_000258`, `whu_val_000666` and
  `whu_train_002069`: fixed shed versus mobile unit or equipment needs identity
  evidence. Visual similarity alone does not settle inconsistent labelling.
- `whu_val_000144`: the circular roof/tank is geometrically labelled; its intended
  target membership needs a storage-structure policy.
- `whu_train_002211`, `whu_train_002208` and `whu_train_001510`:
  construction frames, unfinished buildings and exposed foundations need
  consistent construction-stage rules.
- `whu_val_001022`: open nursery rows are background, while roof/shade structures
  have different treatment. Their enclosure and permanence are unresolved;
  greenhouse consistency remains unverified.

The recognizable caravan rows in `whu_val_000251` remain excluded, consistent
with the draft fixed-building rule. This one site does not establish behaviour
across other container/trailer sites.

## Consequences for scoring and the next experiment

Both candidate binary corrections are in the **training** partition. No validation
mask was changed merely to obtain an apparent metric improvement. On unchanged
validation masks with the same evaluated pixels, original-versus-review scores
must therefore be identical. Any reduction in the evaluated subset caused by
unresolved cases must be reported explicitly and cannot be compared directly with
the full validation score.

The appropriate next decision remains **domain adjudication**, not a three-class
training campaign. If the two proposed corrections are accepted, they justify a
versioned corrected-training pilot; the tiny, selected set alone cannot establish
a general performance gain. A useful validation-label sensitivity study requires
independently accepted validation changes or a newly annotated geographic review
set. Confirmed container false positives at multiple independent sites would be
needed to motivate the three-class pilot. None is established by this pass.

## Verification

The proposal JSON covers exactly the 32 original sample IDs and seven targeted
IDs without duplicates. Polygon coordinates are finite, in bounds and intended
for native-grid rasterization. Local QA masks contain only 0/255 values. The
bare-ground edit removes exactly its 2,219-pixel disconnected component and no
other original positive pixels. The shed edits add 26,844 pixels and remove no
original positives. All proposal image links and source inputs are local; no
browser or platform UI was used.
