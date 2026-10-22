"""Freeze the user-approved 1298-patient primary disease classification cohort.

No automatic dropping of REJECT subjects, small classes, or image-quality flags.
Missing files / incompatible registration are separate QC findings, not cohort edits.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

from .manifest import write_json

LABEL_ORDER = ["正常", "肝囊肿", "血管瘤", "HCC", "转移瘤"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_ORDER)}
THREE_WAY = {
    "正常": "正常", "肝囊肿": "良性", "血管瘤": "良性",
    "HCC": "恶性", "转移瘤": "恶性",
}
EXPECTED_DISTRIBUTION = {"HCC": 724, "转移瘤": 92, "肝囊肿": 188, "血管瘤": 56, "正常": 238}
LOCKED_ID_LABEL_SHA256 = "815f884a755f3dd8317a2c11c53b58b8e71406288f9125f94129850b3c4e0e17"
PATH_COLS = ("ncct_path", "ap_path", "pvp_path")


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


def lock_cohort(source, out, expected_count=1298, seed=42, generate_splits=True):
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
        paths = {name: row[name].strip() for name in PATH_COLS}
        if not all(paths.values()):
            raise ValueError(f"Missing image path in locked cohort for {pid}")
        grade = row["配准等级"].strip().upper()
        if grade not in {"GOLD", "SILVER", "REJECT"}:
            raise ValueError(f"Unexpected registration grade {grade} for {pid}")
        mask = (row.get("liver_mask_path_candidate") or row.get("liver_mask_path") or "").strip()
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
    if expected_count == 1298 and dict(count) != EXPECTED_DISTRIBUTION:
        raise ValueError(f"Cohort class counts differ from the approved list: {dict(count)}")
    canonical_membership = "\n".join(
        sorted(f"{row['patient_id'].strip()}|{row['主类别'].strip()}" for row in entries)
    )
    membership_sha256 = hashlib.sha256(canonical_membership.encode("utf-8")).hexdigest()
    if expected_count == 1298 and membership_sha256 != LOCKED_ID_LABEL_SHA256:
        raise ValueError("The 1298-case patient IDs or primary labels differ from the approved "
                         f"cohort: sha256={membership_sha256} (expected {LOCKED_ID_LABEL_SHA256})")
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
    splits = _stratified_split(entries, seed) if generate_splits else {}
    if generate_splits:
        _write_csv(out / "splits.csv", ["patient_id", "split"], [
            {"patient_id": row["patient_id"], "split": splits[row["patient_id"]]}
            for row in entries
        ])
    report = {
        "cohort_name": "GX_Main5_1298_LOCKED",
        "approved_count": expected_count,
        "patients": len(entries), "by_primary_label": dict(count),
        "by_registration_grade": dict(grade_count),
        "source_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "patient_label_membership_sha256": membership_sha256,
        "labels": LABEL_TO_ID,
        "splits_created": bool(generate_splits),
        "by_split": dict(Counter(splits.values())),
        "image_files_checked": False,
        "nifti_grid_checked": False,
        "note": "Do not delete REJECT or any other locked patient automatically. "
                "Run separate filesystem, AP/PVP and NCCT matching QC before training/synthesis.",
    }
    write_json(out / "cohort_lock_report.json", report)
    return report
