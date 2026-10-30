# WHU adjudication and versioned review set — October 29, 2026

The user selected **AI-only provisional review**. That scope is complete: a second
AI visual pass supplied decisions, proposed corrections, a separately versioned
mask subset and frozen U-Net re-scoring. Human review is not a blocking step for
this provisional workflow. The labels and findings retain AI provenance; this
selection does not convert them into independently confirmed ground truth.

## Review deliverables

- [39-case native-image index](../data/whu_review_v1_draft/README.md): original RGB,
  source outlines and proposed outlines for every case.
- [Optional human reviewer worksheet](../runs/whu_adjudication_2026-10-29/reviewer_decisions.csv):
  proposed decisions and evidence notes, with blank decision/reviewer/rationale
  fields for the domain reviewer.
- [Visual findings and policy questions](WHU_ADJUDICATION_FINDINGS_2026-10-29.md).
- [Sealed provisional version](../data/whu_review_v1_provisional/review.json),
  identifier `whu-review-v1-provisional`.
- [Frozen-model validation comparison](../runs/whu_adjudication_2026-10-29/validation_comparison/report.md).

There are 39 unique cases: the unchanged 32-tile stratified sample plus seven
targeted followups, retaining 20 training and 19 validation cases. The second
pass proposes 21 keep decisions, 16 unresolved decisions and two corrections.
Both corrections are in training: remove 2,219 foreground pixels over bare ground
in `whu_train_000787`; add 26,844 foreground pixels across eight roof polygons in
`whu_train_002862`. Those are candidate geometries, not expert ground truth.
Uncertain object types and construction stages were not forced into classes.

## Frozen validation results

The frozen validation-winning U-Net uses its unchanged checkpoint and threshold
0.50. It was run once per validation image; the same predictions were compared
with the two label versions. No training or test inference was performed.

| Cohort | All validation cases | Original all-pixel IoU | Evaluable paired cases | Paired original IoU | Paired review IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Representative | 16 | 0.921062 | 10 | 0.919097 | 0.919097 |
| Targeted | 3 | 0.881180 | 2 | 0.912330 | 0.912330 |

All validation masks remained unchanged, so the paired label-change delta is
zero. Six representative cases and one targeted case have unlocalized semantic
uncertainty and are excluded from paired review scoring. They remain in the
original-label baseline. The different baseline and paired IoUs reflect different
case inclusion, not a model improvement. No ignore polygons were introduced to
remove errors. These raw subset scores are not the full-WHU validation score or
estimates of population annotation accuracy; equal stratum allocation and
targeted sampling must be preserved in interpretation.

The implementation also supports localized ignore masks: 255 means excluded in
a separate file; building masks retain binary 0/255. Both label versions are
scored on exactly the same remaining pixels. Fully ignored cases have no paired
score. Pending and unlocalized uncertain cases have original-only diagnostics.

## Optional later domain review

This section applies if an independently adjudicated version is desired later.
It is not required to finish the selected AI-only review.

Review the single native images and masks before looking at model predictions.
For each worksheet row, fill `decision`, `reviewer_id` and `rationale`. The
decision is `keep`, `correct` or `uncertain`. Check the two proposed polygons at
native resolution, especially eaves, tree occlusion and clipping. Identify
greenhouse enclosure, mobile versus fixed structures, storage tanks and
construction stages using domain evidence where RGB is insufficient.

The proposed class rules are in `data/whu_review_v1_draft/review.json` under
`policy.rules`. They need an explicit domain decision, including how unresolved
objects are handled. Importing human decisions records the named policy approver;
the software does not establish that person's expertise or independence.

Edit only draft `masks/` and `ignore_masks/` for any revised geometry. Preserve
`images/`, `original_masks/`, `source_selection.json`, split assignments and cohort
identities. A `keep` worksheet decision restores the original mask and clears
ignore pixels, explicitly rejecting a proposed correction. `uncertain` restores
the original binary mask and retains any manually localized ignore region.
`correct` requires an actual changed mask. Source images and originals are
hash-checked, and editable files may not alias another file through links.

Once a reviewer has supplied all decisions and approved the rules:

```powershell
.venv/Scripts/python.exe -m roof_pipeline.adjudication decisions --review data/whu_review_v1_draft --decisions runs/whu_adjudication_2026-10-29/reviewer_decisions.csv --policy-approved-by "ACTUAL REVIEWER NAME"
.venv/Scripts/python.exe -m roof_pipeline.adjudication seal --review data/whu_review_v1_draft --out data/whu_review_v1_adjudicated --version whu-review-v1-adjudicated
.venv/Scripts/python.exe -m roof_pipeline.review_score --review data/whu_review_v1_adjudicated --checkpoint runs/whu_refinement_2026-10-28/control/best.pt --out runs/whu_review_v1_adjudicated_comparison --device cpu
```

The supplied worksheet is intentionally blank in its human-decision columns.
Do not populate reviewer identity or approve a policy on a reviewer's behalf.
Sealed versions cannot be edited. Use a fresh output directory for each version
or comparison. The existing full-dataset evaluator and its fingerprint contract
are unchanged; the review scorer verifies the checkpoint's original parent
dataset fingerprint while separately binding the review version and its files.

## Reproduce the provisional workflow

These commands require fresh output paths; the current outputs already exist.

```powershell
.venv/Scripts/python.exe -m roof_pipeline.adjudication build --data data/whu_official_prepared --audit runs/whu_annotation_audit_2026-10-29 --checkpoint runs/whu_refinement_2026-10-28/control/best.pt --out data/whu_review_reproduction_draft --version whu-review-reproduction-draft
.venv/Scripts/python.exe -m roof_pipeline.adjudication propose --review data/whu_review_reproduction_draft --proposals runs/whu_adjudication_2026-10-29/proposals.json
.venv/Scripts/python.exe -m roof_pipeline.adjudication seal --review data/whu_review_reproduction_draft --out data/whu_review_reproduction_provisional --version whu-review-reproduction-provisional --allow-provisional
.venv/Scripts/python.exe -m roof_pipeline.review_score --review data/whu_review_reproduction_provisional --checkpoint runs/whu_refinement_2026-10-28/control/best.pt --out runs/whu_review_reproduction_comparison --device cpu --allow-provisional
```

## Experiment decision

The selected next experiment is an **exploratory binary label-quality ablation**:
original training labels versus a versioned training set containing the two
AI-proposed corrections, with matched U-Net initialization, recipe and budget,
and validation-only model selection. Preserve the provisional provenance in its
registration and results. Keep all reviewed validation cases out of training.
This tests the effect of those proposed training changes, rather than establishing
true annotation accuracy. Two selected corrected tiles alone cannot establish a
general benefit; wider representative review would assess systematic omissions.
The user subsequently requested testing the corrections. The
[matched correction experiment](WHU_CORRECTIONS_2026-10-29.md) completed original
and provisional-correction arms with unchanged validation targets and retained
final weights. Corrected training gained 0.000192 validation IoU over the matched
control, but remained below the incumbent U-Net, which stays selected. Its local
training effects are recorded separately from review-label rescoring.

A three-class experiment remains unsupported: greenhouse identity and rules are
unresolved, and the recognizable caravan site was correctly excluded. Multiple
independent sites with confirmed container/trailer confusion would change that
decision. A corrected-validation accuracy claim requires independently accepted
validation corrections or a new geographic reference set. None was manufactured
here to increase a score.

## Verification and review lessons

195 tests passed in each Python environment, including actual build/proposal/seal/
score integration, correct paired-count arithmetic, pending/uncertain exclusion,
immutable source identity, polygon validation, overwrite protection, explicit
human provenance, and link-alias rejection before writes. Ruff lint and changed
pipeline/tests formatting passed. The real subset's pixel changes, original
source hashes, checkpoint hash and CSV count aggregates were checked separately.

An [optional AGENTS.md rules proposal](AGENTS_REVIEW_RULES_PROPOSAL.md) captures
the coordinate and write-safety lessons. Those proposed rules have not been
installed or made a condition of the requested adjudication workflow.
