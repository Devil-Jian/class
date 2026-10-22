"""Finalize the patient-approved 1183-case classification subset from prior image QC.

Never modify the original 1298 or 1195 source CSVs. Only remove exactly 12
patients flagged as missing images in the previous QC scan.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

from .locked_cohort import APPROVED_COHORTS, LABEL_TO_ID, THREE_WAY
from .manifest import write_json


def _load(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fingerprint(rows):
    entries = sorted(f"{r['patient_id'].strip()}|{r['主类别'].strip()}" for r in rows)
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def _write(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finalize_qc_subset(source, splits, qc, out_dir, original_1298=None,
                       expected_source_count=1195, expected_removals=12):
    """Derive and lock subset using observed QC; refuse unexpected exclusions."""
    rows, fields = _load(source)
    split_rows, split_fields = _load(splits)
    qc_rows, _ = _load(qc)
    if not {"patient_id", "主类别", "三分类标签", "配准等级"}.issubset(fields):
        raise ValueError("Approved source CSV is missing required label fields")
    if not {"patient_id", "split"}.issubset(split_fields):
        raise ValueError("Frozen patient split CSV missing expected fields")
    if len(rows) != expected_source_count:
        raise ValueError(f"Expected {expected_source_count} source cases, got {len(rows)}")
    ids = [r["patient_id"].strip() for r in rows]
    if len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("Duplicated/empty patient ID in approved source")
    approved = APPROVED_COHORTS.get(expected_source_count)
    if approved and _fingerprint(rows) != approved["patient_label_sha256"]:
        raise ValueError("Source patient IDs/labels differ from previously approved cohort")
    for row in rows:
        if (row["主类别"].strip() not in LABEL_TO_ID or
                THREE_WAY[row["主类别"].strip()] != row["三分类标签"].strip()):
            raise ValueError(f"Invalid disease labels for {row['patient_id']}")
    split_ids = [r["patient_id"].strip() for r in split_rows]
    if len(split_ids) != len(set(split_ids)) or set(split_ids) != set(ids):
        raise ValueError("Saved patient splits do not match the approved source cohort")
    split_lookup = {}
    for row in split_rows:
        split = row["split"].strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("Invalid frozen data split")
        split_lookup[row["patient_id"].strip()] = split
    qids = [r["patient_id"].strip() for r in qc_rows]
    if len(qids) != len(set(qids)) or not set(ids).issubset(qids):
        raise ValueError("QC scan does not cover every active source patient exactly once")
    # QC may have 1195 records OR the previous 1298 records before the 103
    # confirmed missing-directory patients were removed.
    excess = set(qids) - set(ids)
    if excess:
        if original_1298 is None:
            raise ValueError("QC has extra patients; provide --original-1298 to verify them")
        original, _ = _load(original_1298)
        if len(original) != 1298 or _fingerprint(original) != APPROVED_COHORTS[1298]["patient_label_sha256"]:
            raise ValueError("Original 1298 file differs from the approved cohort")
        original_ids = {r["patient_id"].strip() for r in original}
        if len(excess) != 103 or excess != original_ids - set(ids):
            raise ValueError("QC has extra cases other than the 103 previously approved exclusions")
        if set(qids) != original_ids:
            raise ValueError("Superset QC has missing or unexpected original patient IDs")
    by_qc = {r["patient_id"].strip(): r for r in qc_rows}
    removed_ids = set()
    for row in rows:
        pid = row["patient_id"].strip()
        outcome = by_qc[pid]["nifti_ready"].strip().lower()
        if outcome not in {"true", "false"}:
            raise ValueError(f"Missing/invalid nifti_ready status for {pid}")
        if outcome == "false":
            causes = [x.strip() for x in by_qc[pid].get("issues", "").split(";") if x.strip()]
            if not causes or not all(x.endswith("_missing") for x in causes):
                raise ValueError(f"{pid} QC failure is not exclusively missing images: {causes}")
            removed_ids.add(pid)
    if len(removed_ids) != expected_removals:
        raise ValueError(f"Expected exactly {expected_removals} missing-image cases, found {len(removed_ids)}")
    kept = [r for r in rows if r["patient_id"].strip() not in removed_ids]
    saved_splits = [r for r in split_rows if r["patient_id"].strip() not in removed_ids]
    if len(kept) != expected_source_count - expected_removals:
        raise ValueError("Unexpected final cohort size")
    count = Counter(r["主类别"].strip() for r in kept)
    per_split = Counter(split_lookup[r["patient_id"].strip()] for r in kept)
    out = Path(out_dir)
    n = len(kept)
    new_source = out / f"classification_main5_{n}_including_REJECT.csv"
    new_splits = out / f"classification_main5_{n}_locked_splits.csv"
    excluded = out / f"classification_excluded_missing_{expected_removals}_audit.csv"
    summary_path = out / f"classification_main5_{n}_lock_report.json"
    exclusion_rows = []
    for row in rows:
        pid = row["patient_id"].strip()
        if pid in removed_ids:
            exclusion_rows.append({
                "patient_id": pid, "主类别": row["主类别"],
                "三分类标签": row["三分类标签"],
                "配准等级": row["配准等级"],
                "original_split": split_lookup[pid],
                "qc_issues": by_qc[pid].get("issues", ""),
                "action": "excluded_by_explicit_user_approval",
            })
    _write(new_source, fields, kept)
    _write(new_splits, split_fields, saved_splits)
    _write(excluded, ["patient_id", "主类别", "三分类标签", "配准等级",
                      "original_split", "qc_issues", "action"], exclusion_rows)
    summary = {
        "cohort_name": f"GX_Main5_{n}_ACTIVE",
        "approved_source_cases": len(rows),
        "excluded_missing_images": len(removed_ids),
        "patients": n,
        "by_primary_label": dict(count),
        "by_split": dict(per_split),
        "patient_label_membership_sha256": _fingerprint(kept),
        "cohort_csv_sha256": _digest(new_source),
        "splits_csv_sha256": _digest(new_splits),
        "source_csv_sha256": _digest(source),
        "qc_csv_sha256": _digest(qc),
        "original_qc_patients": len(qc_rows),
        "paths": {
            "cohort": str(new_source), "splits": str(new_splits),
            "excluded_12": str(excluded),
        },
        "note": "1195 original -> 1183 kept. Exactly 12 missing-image patients removed "
                "from the active subset; previous 103 and all prior patient splits preserved. "
                "No actual scan was deleted. Re-run image QC for remaining cases.",
    }
    write_json(summary_path, summary)
    return summary
