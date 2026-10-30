"""Versioned review subsets with preserved originals and explicit adjudication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .annotation_audit import coverage_stratum, mask_outline
from .common import dataset_fingerprint, load_binary_mask, read_manifest

SCHEMA = "roof_review_v1"
FILE_FIELDS = ("image", "original_mask", "mask", "ignore_mask")
IDENTITY_FIELDS = (
    "sample_id",
    "split",
    "group",
    "source",
    "cohort",
    "stratum",
    "inclusion_probability",
    "image",
    "original_mask",
    "mask",
    "ignore_mask",
    "image_sha256",
    "original_mask_sha256",
)
POLICY_RULES = {
    "target": "Visible permanent building roof extent; binary foreground. This is a proposed review policy, not an authoritative publisher policy.",
    "sheds": "Include visibly roofed permanent sheds and attached roof extensions.",
    "background": "Exclude ground, roads, shadows, open planting beds, exposed slabs and detached equipment.",
    "mobile_structures": "Exclude confidently identified mobile caravans/trailers. Unresolved container versus fixed shed identity requires adjudication or an explicit ignore polygon.",
    "greenhouses": "Enclosed fixed greenhouse and open shade structure inclusion requires domain confirmation. Preserve source labels pending identification; mark uncertainty explicitly.",
    "construction": "Frames, demolition remnants and roofless construction require domain confirmation; do not infer an intact roof.",
    "storage_tanks": "Tank-like roofs are present among source positives. Their intended building status needs domain confirmation; preserve or localize uncertainty until decided.",
    "alignment": "Trace visible roof extent on the native grid. Do not apply blanket shifts or copy model predictions into targets.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def safe_path(root: Path, relative: str) -> Path:
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError(
            "Review paths must be relative and remain inside the review set"
        )
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Review path resolves outside the review set")
    return resolved


def _version(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", value
    ):
        raise ValueError("Review version must be a short nonempty identifier")


def _content_hash(review):
    value = {key: item for key, item in review.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _write_origin(root, review):
    """Lock selection identities separately from editable adjudication decisions."""
    path = Path(root) / "source_selection.json"
    if path.exists():
        raise FileExistsError("Source selection is already recorded")
    value = {
        key: review[key]
        for key in (
            "source_dataset_content_sha256",
            "source_manifest_sha256",
            "checkpoint_sha256",
            "source_audit_packet_sha256",
        )
    }
    value["cases"] = [
        {key: case[key] for key in IDENTITY_FIELDS} for case in review["cases"]
    ]
    _write(path, value)
    review["source_selection_sha256"] = sha256(path)


def load_review(root: Path, *, require_sealed=False, allow_draft_edits=False):
    """Validate native masks, decision provenance, frozen files and a sealed digest."""
    root = Path(root)
    review = json.loads((root / "review.json").read_text(encoding="utf-8"))
    if review.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported review schema")
    _version(review.get("version"))
    if require_sealed and review.get("sealed") is not True:
        raise ValueError("Scoring requires a sealed review version")
    origin_path = root / "source_selection.json"
    if not origin_path.is_file() or sha256(origin_path) != review.get(
        "source_selection_sha256"
    ):
        raise ValueError("Review source selection hash mismatch")
    origin = json.loads(origin_path.read_text(encoding="utf-8"))
    if any(
        review.get(key) != value for key, value in origin.items() if key != "cases"
    ) or origin["cases"] != [
        {key: case.get(key) for key in IDENTITY_FIELDS}
        for case in review.get("cases", [])
    ]:
        raise ValueError("Review partition, cohort or original-source identity changed")
    for key in ("source_dataset_content_sha256", "checkpoint_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", review.get(key, "")):
            raise ValueError(f"Missing or invalid {key}")
    policy = review.get("policy", {})
    if (
        policy.get("status") not in {"draft", "approved"}
        or policy.get("reviewer_kind") not in {"unknown", "ai", "human"}
        or not policy.get("rules")
    ):
        raise ValueError("Review policy needs rules, status and reviewer provenance")
    if policy["status"] == "approved" and (
        policy["reviewer_kind"] != "human"
        or not str(policy.get("reviewer_id", "")).strip()
    ):
        raise ValueError("Approved policy requires an identified human adjudicator")
    cases, identifiers, groups = review.get("cases", []), set(), {}
    resolved_files, file_identities = set(), set()
    if not cases:
        raise ValueError("Review set is empty")
    for case in cases:
        for field in FILE_FIELDS:
            path = safe_path(root, case[field])
            info = path.stat()
            identity = (info.st_dev, info.st_ino)
            if path in resolved_files or (identity[1] and identity in file_identities):
                raise ValueError(
                    "Review inputs and editable masks must be distinct files, without aliases"
                )
            if field in {"mask", "ignore_mask"} and info.st_nlink > 1:
                raise ValueError(
                    "Editable review masks must not be hard-linked to other files"
                )
            resolved_files.add(path)
            if identity[1]:
                file_identities.add(identity)
        ident = case.get("sample_id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", ident) or ident in identifiers:
            raise ValueError("Invalid or duplicate review sample ID")
        identifiers.add(ident)
        if case.get("split") not in {"train", "val"} or not case.get("group"):
            raise ValueError("Review cases must retain train/val partition and group")
        if groups.setdefault(case["group"], case["split"]) != case["split"]:
            raise ValueError("Review group crosses train/validation partitions")
        probability = case.get("inclusion_probability")
        if case.get("cohort") == "representative":
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 < probability <= 1
            ):
                raise ValueError(
                    "Representative cases require a valid inclusion probability"
                )
        elif case.get("cohort") != "targeted" or probability is not None:
            raise ValueError(
                "Targeted cases have no representative inclusion probability"
            )
        if case.get("reviewer_kind") not in {"unknown", "ai", "human"}:
            raise ValueError("Case requires explicit reviewer provenance")
        decision = case.get("decision")
        if decision not in {"pending", "keep", "correct", "uncertain"}:
            raise ValueError("Unknown adjudication decision")
        if decision != "pending" and (
            not str(case.get("reviewer_id", "")).strip()
            or not str(case.get("rationale", "")).strip()
        ):
            raise ValueError("Decisions require an identified reviewer and rationale")
        with Image.open(safe_path(root, case["image"])) as image:
            size = image.size
            if image.mode != "RGB":
                raise ValueError("Review image must be native RGB")
        original = load_binary_mask(safe_path(root, case["original_mask"]), size)
        mask = load_binary_mask(safe_path(root, case["mask"]), size)
        ignored = load_binary_mask(safe_path(root, case["ignore_mask"]), size)
        changed = int((original != mask).sum())
        check_decision = review.get("sealed") or not allow_draft_edits
        if check_decision and (
            (decision in {"pending", "keep", "uncertain"} and changed)
            or (decision == "correct" and not changed)
        ):
            raise ValueError("Mask changes disagree with adjudication decision")
        if check_decision and decision in {"pending", "keep"} and ignored.any():
            raise ValueError(
                "Ignored pixels require an uncertain or correction decision"
            )
        # Originals are immutable even in a draft; proposed masks can be edited before sealing.
        for field in (
            FILE_FIELDS if review.get("sealed") else ("image", "original_mask")
        ):
            if sha256(safe_path(root, case[field])) != case.get(field + "_sha256"):
                raise ValueError(f"Review {field} hash mismatch for {ident}")
    if review.get("sealed") and review.get("content_sha256") != _content_hash(review):
        raise ValueError("Sealed review metadata/content hash mismatch")
    return review


def build_review(data, audit, checkpoint, out, version):
    data, audit, checkpoint, out = map(Path, (data, audit, checkpoint, out))
    _version(version)
    if out.exists():
        raise FileExistsError("Use a fresh review output directory")
    packet = json.loads((audit / "packet.json").read_text(encoding="utf-8"))
    if packet["manifest_sha256"] != sha256(data / "manifest.csv"):
        raise ValueError("Audit manifest differs from the source dataset")
    manifest = {r["sample_id"]: r for r in read_manifest(data)}
    with (audit / "review.csv").open(newline="", encoding="utf-8") as stream:
        prior = {r["sample_id"]: r for r in csv.DictReader(stream)}
    selection = {
        r["sample_id"]: dict(r, cohort="representative") for r in packet["samples"]
    }
    for filename in ("targeted_review.json", "greenhouse_screen.json"):
        source = audit / filename
        if source.exists():
            for row in json.loads(source.read_text(encoding="utf-8")).get(
                "native_followups", []
            ):
                selection.setdefault(
                    row["sample_id"],
                    dict(row, cohort="targeted", inclusion_probability=None),
                )
    source_sha = dataset_fingerprint(data)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["config"].get("dataset_content_sha256") != source_sha:
        raise ValueError("Frozen checkpoint source dataset differs")
    cases = []
    # Validate every audit reference before writing output files.
    for ident, selected in sorted(selection.items()):
        row = manifest[ident]
        if row["split"] not in {"train", "val"}:
            raise ValueError("Audit selection contains a test case")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
            raise ValueError("Invalid review sample ID")
        for field in ("image", "mask"):
            if sha256(data / row[field]) != selected[field + "_sha256"]:
                raise ValueError("Audit source content changed")
        case = {key: row[key] for key in ("sample_id", "split", "group", "source")}
        case.update(
            cohort=selected["cohort"],
            inclusion_probability=selected.get("inclusion_probability"),
            stratum=selected.get("stratum")
            or coverage_stratum(
                float((load_binary_mask(data / row["mask"]) > 0).mean())
            ),
            prior_observation=prior.get(ident, {}).get("notes")
            or selected.get("finding")
            or selected.get("notes", ""),
            decision="pending",
            reviewer_kind="unknown",
            reviewer_id="",
            rationale="",
        )
        case.update(
            image=f"images/{ident}.png",
            original_mask=f"original_masks/{ident}.png",
            mask=f"masks/{ident}.png",
            ignore_mask=f"ignore_masks/{ident}.png",
        )
        cases.append(case)
    for directory in ("images", "original_masks", "masks", "ignore_masks", "panels"):
        (out / directory).mkdir(parents=True, exist_ok=False)
    for case in cases:
        row = manifest[case["sample_id"]]
        shutil.copyfile(data / row["image"], out / case["image"])
        shutil.copyfile(data / row["mask"], out / case["original_mask"])
        shutil.copyfile(data / row["mask"], out / case["mask"])
        with Image.open(data / row["image"]) as image:
            Image.new("L", image.size, 0).save(out / case["ignore_mask"])
        for field in FILE_FIELDS:
            case[field + "_sha256"] = sha256(out / case[field])
    review = dict(
        schema_version=SCHEMA,
        version=version,
        sealed=False,
        source_dataset_content_sha256=source_sha,
        source_manifest_sha256=sha256(data / "manifest.csv"),
        checkpoint_sha256=sha256(checkpoint),
        checkpoint_path=str(checkpoint.resolve()),
        source_audit_packet_sha256=sha256(audit / "packet.json"),
        policy=dict(
            status="draft", reviewer_kind="unknown", reviewer_id="", rules=POLICY_RULES
        ),
        cases=cases,
    )
    _write_origin(out, review)
    _write(out / "review.json", review)
    load_review(out)
    render_review(out)
    return review


def apply_proposals(root, proposals):
    """Apply explicit native-grid polygons to an unsealed review draft only."""
    root, proposals = Path(root), Path(proposals)
    review = load_review(root)
    if review.get("sealed"):
        raise ValueError("A sealed version cannot be edited")
    if review["policy"]["status"] == "approved" or any(
        c["reviewer_kind"] == "human" for c in review["cases"]
    ):
        raise ValueError("AI proposals cannot replace recorded human adjudication")
    payload = json.loads(proposals.read_text(encoding="utf-8"))
    if (
        payload.get("reviewer_kind") != "ai"
        or not str(payload.get("reviewer_id", "")).strip()
    ):
        raise ValueError("AI proposals require explicit AI reviewer provenance")
    by_id = {r["sample_id"]: r for r in review["cases"]}
    updates, seen = [], set()
    from shapely.geometry import Polygon

    for proposal in payload["cases"]:
        ident = proposal["sample_id"]
        if ident not in by_id or ident in seen:
            raise ValueError("Unknown or duplicate proposal sample")
        seen.add(ident)
        case = dict(by_id[ident])
        mask = Image.fromarray(load_binary_mask(safe_path(root, case["original_mask"])))
        ignore = Image.new("L", mask.size, 0)
        for edit in proposal.get("edits", []):
            points = edit["points"]
            if len(points) < 3 or any(
                len(p) != 2
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    for v in p
                )
                or not 0 <= p[0] <= mask.width - 1
                or not 0 <= p[1] <= mask.height - 1
                for p in points
            ):
                raise ValueError(
                    "Proposal polygon must use finite in-bounds native coordinates"
                )
            polygon = Polygon(points)
            if not polygon.is_valid or polygon.area <= 0:
                raise ValueError("Proposal polygon is degenerate or self-intersecting")
            action = edit["action"]
            if (
                action not in {"add", "remove", "ignore"}
                or not str(edit.get("reason", "")).strip()
            ):
                raise ValueError("Polygon requires add/remove/ignore action and reason")
            ImageDraw.Draw(ignore if action == "ignore" else mask).polygon(
                [tuple(p) for p in points], fill=0 if action == "remove" else 255
            )
        case.update(
            decision=proposal["decision"],
            reviewer_kind="ai",
            reviewer_id=payload["reviewer_id"],
            rationale=proposal["rationale"],
            edits=proposal.get("edits", []),
        )
        original = load_binary_mask(safe_path(root, case["original_mask"]))
        changed, ignored = (
            int((np.asarray(mask) != original).sum()),
            int((np.asarray(ignore) > 0).sum()),
        )
        if (
            case["decision"] not in {"keep", "correct", "uncertain"}
            or not case["rationale"].strip()
            or (case["decision"] == "correct") != bool(changed)
            or (case["decision"] == "keep" and ignored)
        ):
            raise ValueError("Proposal pixels and decision disagree")
        updates.append((case, mask, ignore))
    for case, mask, ignore in updates:
        mask.save(safe_path(root, case["mask"]))
        ignore.save(safe_path(root, case["ignore_mask"]))
        by_id[case["sample_id"]] = case
    review["cases"] = list(by_id.values())
    review["policy"].update(reviewer_kind="ai", reviewer_id=payload["reviewer_id"])
    review["proposals_sha256"] = sha256(proposals)
    review["proposal_context"] = {
        key: value for key, value in payload.items() if key != "cases"
    }
    _write(root / "review.json", review)
    load_review(root)
    render_review(root)
    return review


def seal_review(root, out, version, *, allow_provisional=False):
    root, out = Path(root).resolve(), Path(out).resolve()
    _version(version)
    if out.exists() or out.is_relative_to(root) or root.is_relative_to(out):
        raise FileExistsError("Sealed output must be a fresh independent directory")
    review = load_review(root)
    if review.get("sealed"):
        raise ValueError("Create a new draft to revise a sealed version")
    human_ready = review["policy"]["status"] == "approved" and all(
        c["reviewer_kind"] == "human" and c["decision"] != "pending"
        for c in review["cases"]
    )
    if not human_ready and not allow_provisional:
        raise ValueError(
            "Human adjudication and approved policy required; use explicit provisional mode for AI drafts"
        )
    review.update(
        version=version,
        sealed=True,
        provisional=not human_ready,
        parent_review_sha256=sha256(root / "review.json"),
    )
    for case in review["cases"]:
        for field in FILE_FIELDS:
            case[field + "_sha256"] = sha256(safe_path(root, case[field]))
    review["content_sha256"] = _content_hash(review)
    shutil.copytree(root, out)
    _write(out / "review.json", review)
    load_review(out, require_sealed=True)
    render_review(out)
    return review


def export_decisions(root, out):
    """Write an editable human worksheet without overwriting existing decisions."""
    root, out = Path(root), Path(out)
    review = load_review(root)
    if out.exists():
        raise FileExistsError("Reviewer worksheet already exists")
    fields = (
        "sample_id",
        "split",
        "cohort",
        "proposed_decision",
        "proposed_rationale",
        "prior_observation",
        "decision",
        "reviewer_id",
        "rationale",
    )
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in review["cases"]:
            writer.writerow(
                {
                    "sample_id": case["sample_id"],
                    "split": case["split"],
                    "cohort": case["cohort"],
                    "proposed_decision": case["decision"],
                    "proposed_rationale": case["rationale"],
                    "prior_observation": case.get("prior_observation", ""),
                    "decision": "",
                    "reviewer_id": "",
                    "rationale": "",
                }
            )
    return out


def import_decisions(root, decisions, *, policy_approved_by):
    """Record explicitly supplied human decisions; never generate reviewer assent."""
    root, decisions = Path(root), Path(decisions)
    review = load_review(root, allow_draft_edits=True)
    if review.get("sealed"):
        raise ValueError("Human decisions must be applied to an unsealed draft")
    if not str(policy_approved_by).strip():
        raise ValueError("The policy approver must be identified")
    with decisions.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    identifiers = [r.get("sample_id") for r in rows]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != {
        c["sample_id"] for c in review["cases"]
    }:
        raise ValueError("Human worksheet must contain every case exactly once")
    by_id, updates = {r["sample_id"]: r for r in rows}, []
    for case in review["cases"]:
        row = by_id[case["sample_id"]]
        if row.get("split") != case["split"] or row.get("cohort") != case["cohort"]:
            raise ValueError("Human worksheet cannot change split or cohort")
        decision = row.get("decision", "").strip()
        if (
            decision not in {"keep", "correct", "uncertain"}
            or not row.get("reviewer_id", "").strip()
            or not row.get("rationale", "").strip()
        ):
            raise ValueError(
                "Every human decision needs keep/correct/uncertain, reviewer ID and rationale"
            )
        original = load_binary_mask(safe_path(root, case["original_mask"]))
        mask = load_binary_mask(safe_path(root, case["mask"]))
        ignore = load_binary_mask(safe_path(root, case["ignore_mask"]))
        if decision == "correct" and np.array_equal(mask, original):
            raise ValueError("A correction decision requires a changed proposed mask")
        if decision in {"keep", "uncertain"}:
            mask = original.copy()
        if decision == "keep":
            ignore = np.zeros_like(ignore)
        updated = dict(
            case,
            decision=decision,
            reviewer_kind="human",
            reviewer_id=row["reviewer_id"].strip(),
            rationale=row["rationale"].strip(),
            previous_proposal=dict(
                decision=case["decision"],
                reviewer_kind=case["reviewer_kind"],
                reviewer_id=case["reviewer_id"],
                rationale=case["rationale"],
            ),
        )
        updates.append((updated, mask, ignore))
    for case, mask, ignore in updates:
        Image.fromarray(mask).save(safe_path(root, case["mask"]))
        Image.fromarray(ignore).save(safe_path(root, case["ignore_mask"]))
    review["cases"] = [case for case, _, _ in updates]
    review["policy"].update(
        status="approved", reviewer_kind="human", reviewer_id=policy_approved_by.strip()
    )
    review["human_decisions_sha256"] = sha256(decisions)
    _write(root / "review.json", review)
    load_review(root)
    render_review(root)
    return review


def render_review(root):
    root = Path(root)
    review = load_review(root)
    (root / "panels").mkdir(exist_ok=True)
    lines = [
        f"# {review['version']}",
        "",
        f"Sealed: {review['sealed']}. Policy: {review['policy']['status']}. AI proposals require human adjudication before confirmed-correction claims.",
        "",
        "| Case | Split / cohort | Decision | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for case in review["cases"]:
        with Image.open(safe_path(root, case["image"])) as image:
            rgb = np.asarray(image).copy()
        original = load_binary_mask(safe_path(root, case["original_mask"]))
        mask = load_binary_mask(safe_path(root, case["mask"]))
        ignore = load_binary_mask(safe_path(root, case["ignore_mask"])) > 0
        source_overlay, proposed_overlay = rgb.copy(), rgb.copy()
        source_overlay[mask_outline(original)] = (255, 0, 255)
        proposed_overlay[mask_outline(mask)] = (0, 255, 180)
        proposed_overlay[ignore] = (
            0.5 * proposed_overlay[ignore] + np.array([255, 190, 0]) * 0.5
        ).astype(np.uint8)
        h, w = rgb.shape[:2]
        panel = Image.new("RGB", (w * 3, h + 30), (25, 25, 25))
        ImageDraw.Draw(panel).text(
            (6, 6),
            f"{case['sample_id']} | RGB / original magenta / proposed green; ignore amber | {case['decision']} ({case['reviewer_kind']})",
            fill="white",
        )
        for index, value in enumerate((rgb, source_overlay, proposed_overlay)):
            panel.paste(Image.fromarray(value), (index * w, 30))
        relative = f"panels/{case['sample_id']}.png"
        panel.save(root / relative)
        lines.append(
            f"| {case['sample_id']} | {case['split']} / {case['cohort']} | {case['decision']} | [native panel]({relative}) |"
        )
    lines += [
        "",
        "Decision fields and draft policy are in review.json. Original RGB/masks are immutable. Edit proposed masks and ignore masks only in an unsealed draft, with matching reviewer provenance and rationale. Seal into a new directory before scoring. Unknown pixels use a separate ignore mask (255 means excluded); binary masks remain 0/255.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    for name in ("data", "audit", "checkpoint", "out"):
        build.add_argument("--" + name, type=Path, required=True)
    build.add_argument("--version", required=True)
    apply = sub.add_parser("propose")
    apply.add_argument("--review", type=Path, required=True)
    apply.add_argument("--proposals", type=Path, required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--review", type=Path, required=True)
    seal.add_argument("--out", type=Path, required=True)
    seal.add_argument("--version", required=True)
    seal.add_argument("--allow-provisional", action="store_true")
    worksheet = sub.add_parser("worksheet")
    worksheet.add_argument("--review", type=Path, required=True)
    worksheet.add_argument("--out", type=Path, required=True)
    accept = sub.add_parser("decisions")
    accept.add_argument("--review", type=Path, required=True)
    accept.add_argument("--decisions", type=Path, required=True)
    accept.add_argument("--policy-approved-by", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_review(
            args.data, args.audit, args.checkpoint, args.out, args.version
        )
    elif args.command == "propose":
        result = apply_proposals(args.review, args.proposals)
    elif args.command == "worksheet":
        print(export_decisions(args.review, args.out))
        return
    elif args.command == "decisions":
        result = import_decisions(
            args.review, args.decisions, policy_approved_by=args.policy_approved_by
        )
    else:
        result = seal_review(
            args.review,
            args.out,
            args.version,
            allow_provisional=args.allow_provisional,
        )
    print(
        json.dumps(
            {
                "version": result["version"],
                "cases": len(result["cases"]),
                "sealed": result["sealed"],
            }
        )
    )


if __name__ == "__main__":
    main()
