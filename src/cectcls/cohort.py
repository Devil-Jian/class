"""Create an audited classification cohort from GX metadata and registered NIfTI directory."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
import random

from .manifest import write_json


REQUIRED = ("patient_id", "分类（暂定）", "三分类标签", "主类别", "配准等级")
INPUT_FILENAMES = {
    "ncct_path": "ncct_zcrop.nii.gz",
    "ap_path": "ap_registered.nii.gz",
    "pvp_path": "pvp_registered.nii.gz",
    "liver_mask_path": "ncct_mask_zcrop.nii.gz",
}


def _rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata missing columns: {sorted(missing)}")
        return list(reader)


def _write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _split_cases(cases, train_ratio, val_ratio, seed):
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than one")
    groups = defaultdict(list)
    for case in cases:
        groups[case["label"]].append(case["patient_id"])
    result, warnings = {}, []
    for label, patient_ids in sorted(groups.items()):
        ids = sorted(patient_ids)
        random.Random(f"{seed}:{label}").shuffle(ids)
        n = len(ids)
        if n < 3:
            n_train, n_val = n, 0
            warnings.append(f"Class {label} has {n} patients: all assigned to train, cannot evaluate this class")
        else:
            n_val = max(1, round(n * val_ratio))
            n_test = max(1, round(n * (1 - train_ratio - val_ratio)))
            while n - n_val - n_test < 1:
                if n_val > n_test:
                    n_val -= 1
                else:
                    n_test -= 1
            n_train = n - n_val - n_test
        for j, patient_id in enumerate(ids):
            result[patient_id] = "train" if j < n_train else ("val" if j < n_train + n_val else "test")
    return result, warnings


def build_cohort(metadata, image_root, out, label_column="三分类标签",
                 grades=("GOLD", "SILVER", "REJECT"), train_ratio=0.70,
                 val_ratio=0.15, seed=42, require_mask=True):
    """Filter classification=yes, valid labels & grades, existing inputs, then split patient-wise."""
    if label_column not in {"三分类标签", "主类别"}:
        raise ValueError("label_column must be 三分类标签 or 主类别")
    rows = _rows(metadata)
    root = Path(image_root).expanduser().resolve()
    allowed = {str(g).strip().upper() for g in grades}
    reject = Counter()
    grouped = defaultdict(list)
    for row in rows:
        if row["分类（暂定）"].strip() != "是":
            reject["not_classification"] += 1
            continue
        patient_id = row["patient_id"].strip()
        if not patient_id:
            reject["missing_patient_id"] += 1
            continue
        if row["配准等级"].strip().upper() not in allowed:
            reject["registration_grade"] += 1
            continue
        label = row[label_column].strip()
        if label not in {"正常", "良性", "恶性"} and label_column == "三分类标签":
            reject["invalid_or_missing_three_class_label"] += 1
            continue
        if not label:
            reject["missing_class_label"] += 1
            continue
        grouped[patient_id].append(row)
    cases = []
    for patient_id, duplicates in sorted(grouped.items()):
        labels = {r[label_column].strip() for r in duplicates}
        if len(labels) != 1:
            reject["conflicting_duplicate_labels"] += len(duplicates)
            continue
        row = duplicates[0]
        patient_dir = root / patient_id
        if not patient_dir.is_dir():
            reject["missing_patient_directory"] += 1
            continue
        paths = {col: patient_dir / basename for col, basename in INPUT_FILENAMES.items()}
        missing = [col for col in ("ncct_path", "ap_path", "pvp_path") if not paths[col].is_file()]
        if require_mask and not paths["liver_mask_path"].is_file():
            missing.append("liver_mask_path")
        if missing:
            for col in missing:
                reject["missing_" + col] += 1
            continue
        cases.append({
            "patient_id": patient_id, "label": next(iter(labels)),
            **{col: str(path) if path.is_file() else "" for col, path in paths.items()},
        })
        reject["duplicate_metadata_rows_collapsed"] += len(duplicates) - 1
    splits, warnings = _split_cases(cases, train_ratio, val_ratio, seed)
    out = Path(out)
    _write_csv(out / "cases.csv", list(cases[0]) if cases else
               ["patient_id", "label", *INPUT_FILENAMES], cases)
    _write_csv(out / "splits.csv", ["patient_id", "split"], [
        {"patient_id": case["patient_id"], "split": splits[case["patient_id"]]}
        for case in cases
    ])
    report = {
        "metadata_rows": len(rows),
        "flagged_classification_rows": sum(r["分类（暂定）"].strip() == "是" for r in rows),
        "eligible_patient_count": len(cases),
        "task_label": label_column, "allowed_registration_grades": sorted(allowed),
        "by_label": dict(Counter(case["label"] for case in cases)),
        "by_split": dict(Counter(splits.values())),
        "exclusion_counts": dict(reject),
        "warnings": warnings,
        "image_root": str(root),
        "mask_interpretation": ("ncct_mask_zcrop.nii.gz treated as a liver mask; verify anatomy"
                                if require_mask else "mask optional; no liver-only guarantee"),
        "note": "File presence verified; run cectcls prepare to validate NIfTI shape/affine",
    }
    write_json(out / "cohort_report.json", report)
    return report
