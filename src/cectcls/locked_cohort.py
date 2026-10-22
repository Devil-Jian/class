"""Freeze the explicitly approved primary disease classification cohorts.

The original 1298 are retained as an audit source. Exactly 103 user-identified
absent-directory patients are excluded from the active 1195-case cohort.
No additional drops occur for REJECT, rare diagnoses or image quality.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

from .manifest import read_json, write_json

LABEL_ORDER = ["正常", "肝囊肿", "血管瘤", "HCC", "转移瘤"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_ORDER)}
THREE_WAY = {
    "正常": "正常", "肝囊肿": "良性", "血管瘤": "良性",
    "HCC": "恶性", "转移瘤": "恶性",
}
APPROVED_COHORTS = {
    1298: {
        "counts": {"HCC": 724, "转移瘤": 92, "肝囊肿": 188, "血管瘤": 56, "正常": 238},
        "patient_label_sha256": "815f884a755f3dd8317a2c11c53b58b8e71406288f9125f94129850b3c4e0e17",
    },
    1195: {
        "counts": {"HCC": 720, "转移瘤": 87, "肝囊肿": 163, "血管瘤": 52, "正常": 173},
        "patient_label_sha256": "1959ee5f0dac045abce43f2dabaee8b01d1864240d70a0aa45c082003b1f1896",
    },
}
PATH_COLS = ("ncct_path", "ap_path", "pvp_path")
SCAN_FILENAMES = {
    "ncct_path": "ncct_zcrop.nii.gz",
    "ap_path": "ap_registered.nii.gz",
    "pvp_path": "pvp_registered.nii.gz",
    "liver_mask_path": "ncct_mask_zcrop.nii.gz",
}


def _write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stratified_split(rows, seed, train_ratio=0.70, val_ratio=0.15):
    """Group by patient label and source prefix; no leakage across datasets."""
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Invalid split fractions")
    groups = defaultdict(list)
    for row in rows:
        prefix = row.get("id_prefix", "").strip() or row["patient_id"].split("_")[0]
        groups[(row["主类别"], prefix)].append(row["patient_id"])
    splits = {}
    for (label, prefix), ids in sorted(groups.items()):
        ids.sort()
        Random(f"{seed}:{label}:{prefix}").shuffle(ids)
        n = len(ids)
        # Small strata go into training; small disease class must still have
        # sufficient patients across the whole cohort for a valid held-out split.
        n_train = n if n < 4 else max(1, round(n * train_ratio))
        n_val = 0 if n < 4 else min(max(1, round(n * val_ratio)), n - n_train)
        for k, pid in enumerate(ids):
            splits[pid] = "train" if k < n_train else ("val" if k < n_train + n_val else "test")
    return splits


def lock_cohort(source, out, expected_count=1298, seed=42, generate_splits=True,
                image_root=None, splits_source=None, approved_report=None):
    """Preserve membership but rebuild image paths under image_root when supplied.

    splits_source copies and validates a previously frozen patient split exactly.
    """
    if splits_source is not None and not generate_splits:
        raise ValueError("--splits-source cannot be used with --without-splits")
    image_root = Path(image_root).expanduser().resolve() if image_root else None
    path = Path(source).resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"patient_id", "主类别", "三分类标签", "配准等级", *PATH_COLS}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing locked cohort columns: {sorted(missing_columns)}")
        entries = list(reader)
    if len(entries) != expected_count:
        raise ValueError(f"Expected {expected_count} candidate patients, got {len(entries)}; "
                         "refusing to silently change the approved cohort")
    seen = set()
    cases = []
    count = Counter()
    grade_count = Counter()
    qc = []
    for row in entries:
        pid = row["patient_id"].strip()
        label = row["主类别"].strip()
        three_way = row["三分类标签"].strip()
        if not pid or pid in seen:
            raise ValueError(f"Duplicate/missing patient ID in locked cohort: {pid!r}")
        seen.add(pid)
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unexpected main disease for patient {pid}: {label!r}")
        if THREE_WAY[label] != three_way:
            raise ValueError(f"Inconsistent main / 3-class label: patient {pid} {label} {three_way}")
        if image_root is not None:
            if "/" in pid or "\\" in pid or pid in {".", ".."}:
                raise ValueError(f"Invalid patient directory name: {pid!r}")
            paths = {name: str(image_root / pid / filename)
                     for name, filename in SCAN_FILENAMES.items() if name in PATH_COLS}
        else:
            paths = {name: row[name].strip() for name in PATH_COLS}
        if not all(paths.values()):
            raise ValueError(f"Missing image path in locked cohort for {pid}")
        grade = row["配准等级"].strip().upper()
        if grade not in {"GOLD", "SILVER", "REJECT"}:
            raise ValueError(f"Unexpected registration grade {grade} for {pid}")
        mask = (str(image_root / pid / SCAN_FILENAMES["liver_mask_path"]) if image_root is not None
                else (row.get("liver_mask_path_candidate") or row.get("liver_mask_path") or "").strip())
        cases.append({
            "patient_id": pid, "label": label, **paths,
            "liver_mask_path": mask,
        })
        qc.append({
            "patient_id": pid, "主类别": label, "三分类标签": three_way,
            "配准等级": grade, "id_prefix": row.get("id_prefix", "").strip(),
            "NCCT对应及Syn生成QC": row.get("NCCT对应及Syn生成QC", "").strip(),
            "训练图像质量": row.get("是否可用于图像训练（质量）", "").strip(),
            "图像质量": row.get("图像质量", "").strip(),
        })
        count[label] += 1
        grade_count[grade] += 1
    approved = APPROVED_COHORTS.get(expected_count)
    if approved and dict(count) != approved["counts"]:
        raise ValueError(f"Cohort class counts differ from the approved {expected_count} list: {dict(count)}")
    canonical_membership = "\n".join(
        sorted(f"{row['patient_id'].strip()}|{row['主类别'].strip()}" for row in entries)
    )
    membership_sha256 = hashlib.sha256(canonical_membership.encode("utf-8")).hexdigest()
    if approved and membership_sha256 != approved["patient_label_sha256"]:
        raise ValueError(f"The {expected_count}-case patient IDs or primary labels differ "
                         f"from the approved cohort: sha256={membership_sha256} "
                         f"(expected {approved['patient_label_sha256']})")
    if expected_count == 1183:
        if not approved_report:
            raise ValueError("1183 active cohort requires --approved-report generated by prune-qc")
        report = read_json(approved_report)
        if (report.get("cohort_name") != "GX_Main5_1183_ACTIVE" or
                report.get("patients") != 1183 or
                report.get("approved_source_cases") != 1195 or
                report.get("excluded_missing_images") != 12 or
                report.get("patient_label_membership_sha256") != membership_sha256 or
                report.get("cohort_csv_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError("1183 patient cohort does not match its approved exclusion report")
        if splits_source and report.get("splits_csv_sha256") != hashlib.sha256(
                Path(splits_source).read_bytes()).hexdigest():
            raise ValueError("1183 fixed train/val/test splits changed since approved exclusion")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "cases.csv",
               ["patient_id", "label", *PATH_COLS, "liver_mask_path"], cases)
    _write_csv(out / "cohort_qc.csv",
               ["patient_id", "主类别", "三分类标签", "配准等级", "id_prefix",
                "NCCT对应及Syn生成QC", "训练图像质量", "图像质量"], qc)
    write_json(out / "labels.json", LABEL_TO_ID)
    write_json(out / "three_class_mapping.json", {
        "primary_to_three": THREE_WAY,
        "three_to_id": {"正常": 0, "良性": 1, "恶性": 2},
    })
    splits = _stratified_split(entries, seed) if generate_splits and splits_source is None else {}
    if splits_source is not None:
        with Path(splits_source).open("r", encoding="utf-8-sig", newline="") as f:
            original_splits = list(csv.DictReader(f))
        if len(original_splits) != len(entries):
            raise ValueError("Frozen splits must have exactly the approved patient count")
        for row in original_splits:
            pid, split = row["patient_id"].strip(), row["split"].strip().lower()
            if pid in splits or split not in {"train", "val", "test"}:
                raise ValueError(f"Duplicate/invalid patient split: {row}")
            splits[pid] = split
        if set(splits) != seen:
            raise ValueError("Frozen splits membership differs from locked patients")
    if generate_splits:
        _write_csv(out / "splits.csv", ["patient_id", "split"], [
            {"patient_id": row["patient_id"], "split": splits[row["patient_id"]]}
            for row in entries
        ])
    report = {
        "cohort_name": f"GX_Main5_{expected_count}_{'ACTIVE' if expected_count in {1195, 1183} else 'LOCKED'}",
        "approved_count": expected_count,
        "patients": len(entries), "by_primary_label": dict(count),
        "by_registration_grade": dict(grade_count),
        "source_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "approved_report": str(Path(approved_report).resolve()) if approved_report else None,
        "image_root_override": str(image_root) if image_root else None,
        "splits_source": str(Path(splits_source).resolve()) if splits_source else None,
        "patient_label_membership_sha256": membership_sha256,
        "labels": LABEL_TO_ID,
        "splits_created": bool(generate_splits),
        "by_split": dict(Counter(splits.values())),
        "image_files_checked": False,
        "nifti_grid_checked": False,
        "note": "The 1195 active cohort explicitly excludes 103 absent-directory patients from "
                "the original 1298. Do not drop any of the remaining patients automatically. "
                "REJECT remains eligible for real CECT classification; run imaging QC separately.",
    }
    write_json(out / "cohort_lock_report.json", report)
    return report
