#!/usr/bin/env python3
"""Build reorganized classification dataset manifest and index files.

Cohort criteria specified by user (2026-10-28):
1. Source: /dataset/dataset/metadata/GX_metadata_all.csv with '分类（暂定） == 是'.
2. Include the 162 cases labeled as '其他良性'.
3. Include the cases previously excluded due to registration mismatch / absent directories.
4. For the 334 cases without scan directory in gx_all_registered, retrieve them from
   /workspace/class/manifest_1028/WizTree_20261028201157.csv and include matched cases.
5. Preserve frozen train/val/test splits for the 1,183 active cohort patients;
   deterministically stratify all newly incorporated cases.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
import pandas as pd


# Canonical label mappings
LABEL_ORDER = ["正常", "肝囊肿", "血管瘤", "HCC", "转移瘤", "其他良性"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_ORDER)}

THREE_CLASS_MAPPING = {
    "正常": "正常",
    "肝囊肿": "良性",
    "血管瘤": "良性",
    "其他良性": "良性",
    "HCC": "恶性",
    "转移瘤": "恶性",
    "ICC": "恶性",
    "FNH": "良性",
}
THREE_CLASS_TO_ID = {"正常": 0, "良性": 1, "恶性": 2}


def parse_wiztree_records(wiztree_csv_path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Parse WizTree CSV to extract patient files and root directory paths."""
    wiz_df = pd.read_csv(wiztree_csv_path, skiprows=1, low_memory=False)
    patient_files = defaultdict(dict)
    patient_dirs = {}

    for fn in wiz_df["文件名称"].dropna():
        clean = fn.replace("\\", "/")
        parts = clean.split("/")
        # Expected: D:, ct_nii, gx, nii, <patient_id>, <filename>
        if len(parts) >= 5 and parts[1] == "ct_nii" and parts[2] == "gx" and parts[3] == "nii":
            pid = parts[4].strip()
            if len(parts) >= 6 and parts[5].strip():
                fname = parts[5].strip().lower()
                patient_files[pid][fname] = fn
            else:
                patient_dirs[pid] = fn

    return patient_files, patient_dirs


def assign_stratified_splits(
    new_cases: list[dict], seed: int = 42, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> dict[str, str]:
    """Deterministically stratify new cases by (label, prefix)."""
    groups = defaultdict(list)
    for c in new_cases:
        groups[(c["primary_label"], c["id_prefix"])].append(c["patient_id"])

    splits = {}
    for (label, prefix), ids in sorted(groups.items()):
        ids.sort()
        Random(f"{seed}:{label}:{prefix}").shuffle(ids)
        n = len(ids)
        if n < 4:
            # Place small cohorts into train to maintain training representation
            for pid in ids:
                splits[pid] = "train"
        else:
            n_train = max(1, round(n * train_ratio))
            n_val = min(max(1, round(n * val_ratio)), n - n_train)
            for k, pid in enumerate(ids):
                if k < n_train:
                    splits[pid] = "train"
                elif k < n_train + n_val:
                    splits[pid] = "val"
                else:
                    splits[pid] = "test"
    return splits


def main():
    manifest_dir = Path("/workspace/class/manifest_1028")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path("/dataset/dataset/metadata/GX_metadata_all.csv")
    wiztree_path = manifest_dir / "WizTree_20261028201157.csv"
    reg_root = Path("/win_cect/shared/gx_all_registered")
    canonical_reg_root = Path("/dataset/dataset/gx_all_registered")

    # Load frozen 1183 active splits
    frozen_splits_path = Path("/workspace/class/input/classification_main5_1183_locked_splits.csv")
    df_frozen = pd.read_csv(frozen_splits_path)
    frozen_splits = dict(zip(df_frozen["patient_id"].str.strip(), df_frozen["split"].str.strip()))
    print(f"Loaded {len(frozen_splits)} frozen active splits.")

    # Parse metadata
    df_meta = pd.read_csv(metadata_path, low_memory=False)
    cls_df = df_meta[df_meta["分类（暂定）"] == "是"].drop_duplicates(subset=["patient_id"]).copy()
    print(f"Total unique patients in metadata with 分类（暂定） == 是: {len(cls_df)}")

    # Parse WizTree
    wiz_files, wiz_dirs = parse_wiztree_records(wiztree_path)
    print(f"Parsed {len(wiz_files)} patient records from WizTree.")

    cases = []
    wiztree_recovered = []
    new_cases_to_split = []

    for _, row in cls_df.iterrows():
        pid = str(row["patient_id"]).strip()
        main_label = str(row["主类别"]).strip() if pd.notna(row["主类别"]) else "NaN"
        three_label = str(row["三分类标签"]).strip() if pd.notna(row["三分类标签"]) else "NaN"
        prefix = str(row.get("id_prefix", "")).strip() or pid.split("_")[0]
        grade = str(row.get("配准等级", "")).strip() if pd.notna(row.get("配准等级")) else "UNREGISTERED"
        qc_score = row.get("配准得分")
        qc_detail = str(row.get("配准质控详情", "")).strip() if pd.notna(row.get("配准质控详情")) else ""

        # Check Linux registered folder
        p_dir = reg_root / pid
        in_reg = p_dir.is_dir()
        has_ncct = (p_dir / "ncct_zcrop.nii.gz").is_file() if in_reg else False
        has_ap = (p_dir / "ap_registered.nii.gz").is_file() if in_reg else False
        has_pvp = (p_dir / "pvp_registered.nii.gz").is_file() if in_reg else False
        has_mask = (p_dir / "ncct_mask_zcrop.nii.gz").is_file() if in_reg else False
        reg_complete = has_ncct and has_ap and has_pvp

        # Check WizTree
        in_wiz = pid in wiz_files
        wfiles = wiz_files.get(pid, {})
        wiz_has_ncct = "ncct.nii.gz" in wfiles
        wiz_has_ap = "ap.nii.gz" in wfiles
        wiz_has_pvp = "pvp.nii.gz" in wfiles
        wiz_has_roi = ("ap_roi.nii.gz" in wfiles) or ("pvp_roi.nii.gz" in wfiles)

        category = None
        storage_source = None
        include = False
        split = None

        if pid in frozen_splits:
            category = "original_active_1183"
            include = True
            storage_source = "registered_linux"
            split = frozen_splits[pid]
            ncct_path = str(canonical_reg_root / pid / "ncct_zcrop.nii.gz")
            ap_path = str(canonical_reg_root / pid / "ap_registered.nii.gz")
            pvp_path = str(canonical_reg_root / pid / "pvp_registered.nii.gz")
            mask_path = str(canonical_reg_root / pid / "ncct_mask_zcrop.nii.gz") if has_mask else ""

        elif main_label in {"HCC", "正常", "肝囊肿", "转移瘤", "血管瘤"}:
            if reg_complete:
                category = "recovered_registered_main5"
                include = True
                storage_source = "registered_linux"
                ncct_path = str(canonical_reg_root / pid / "ncct_zcrop.nii.gz")
                ap_path = str(canonical_reg_root / pid / "ap_registered.nii.gz")
                pvp_path = str(canonical_reg_root / pid / "pvp_registered.nii.gz")
                mask_path = str(canonical_reg_root / pid / "ncct_mask_zcrop.nii.gz") if has_mask else ""
            elif in_wiz and (wiz_has_ap and wiz_has_pvp):
                category = "recovered_wiztree_334"
                include = True
                storage_source = "wiztree_windows"
                ncct_path = wfiles.get("ncct.nii.gz", "")
                ap_path = wfiles.get("ap.nii.gz", "")
                pvp_path = wfiles.get("pvp.nii.gz", "")
                mask_path = wfiles.get("ap_roi.nii.gz", wfiles.get("pvp_roi.nii.gz", ""))
                wiztree_recovered.append({
                    "patient_id": pid,
                    "primary_label": main_label,
                    "id_prefix": prefix,
                    "has_ncct": wiz_has_ncct,
                    "has_ap": wiz_has_ap,
                    "has_pvp": wiz_has_pvp,
                    "has_roi": wiz_has_roi,
                    "windows_ncct_path": ncct_path,
                    "windows_ap_path": ap_path,
                    "windows_pvp_path": pvp_path,
                    "windows_roi_path": mask_path,
                    "all_wiztree_files": ";".join(sorted(wfiles.keys())),
                })

        elif main_label == "其他良性":
            if reg_complete:
                category = "recovered_benign_other_162"
                include = True
                storage_source = "registered_linux"
                ncct_path = str(canonical_reg_root / pid / "ncct_zcrop.nii.gz")
                ap_path = str(canonical_reg_root / pid / "ap_registered.nii.gz")
                pvp_path = str(canonical_reg_root / pid / "pvp_registered.nii.gz")
                mask_path = str(canonical_reg_root / pid / "ncct_mask_zcrop.nii.gz") if has_mask else ""
            elif in_wiz and (wiz_has_ap and wiz_has_pvp):
                category = "benign_other_wiztree"
                include = True
                storage_source = "wiztree_windows"
                ncct_path = wfiles.get("ncct.nii.gz", "")
                ap_path = wfiles.get("ap.nii.gz", "")
                pvp_path = wfiles.get("pvp.nii.gz", "")
                mask_path = wfiles.get("ap_roi.nii.gz", "")
                wiztree_recovered.append({
                    "patient_id": pid,
                    "primary_label": main_label,
                    "id_prefix": prefix,
                    "has_ncct": wiz_has_ncct,
                    "has_ap": wiz_has_ap,
                    "has_pvp": wiz_has_pvp,
                    "has_roi": wiz_has_roi,
                    "windows_ncct_path": ncct_path,
                    "windows_ap_path": ap_path,
                    "windows_pvp_path": pvp_path,
                    "windows_roi_path": mask_path,
                    "all_wiztree_files": ";".join(sorted(wfiles.keys())),
                })

        if include:
            case_info = {
                "patient_id": pid,
                "primary_label": main_label,
                "primary_label_id": LABEL_TO_ID.get(main_label, -1),
                "three_class_label": THREE_CLASS_MAPPING.get(main_label, three_label),
                "three_class_id": THREE_CLASS_TO_ID.get(THREE_CLASS_MAPPING.get(main_label, three_label), -1),
                "id_prefix": prefix,
                "split": split,
                "cohort_category": category,
                "storage_source": storage_source,
                "registration_grade": grade,
                "registration_score": qc_score if pd.notna(qc_score) else "",
                "registration_detail": qc_detail,
                "image_dir": str(canonical_reg_root / pid) if storage_source == "registered_linux" else str(wiz_dirs.get(pid, "")),
                "ncct_path": ncct_path,
                "ap_path": ap_path,
                "pvp_path": pvp_path,
                "liver_mask_path": mask_path,
                "has_ncct": bool(ncct_path),
                "has_ap": bool(ap_path),
                "has_pvp": bool(pvp_path),
                "has_mask": bool(mask_path),
                "is_complete_3phase": bool(ncct_path and ap_path and pvp_path),
            }
            cases.append(case_info)
            if split is None:
                new_cases_to_split.append(case_info)

    # Stratify new cases
    print(f"Stratifying {len(new_cases_to_split)} new cases across train/val/test...")
    assigned_splits = assign_stratified_splits(new_cases_to_split, seed=42)
    for c in new_cases_to_split:
        c["split"] = assigned_splits[c["patient_id"]]

    cases.sort(key=lambda x: (LABEL_TO_ID.get(x["primary_label"], 99), x["patient_id"]))

    # 1. Write cases.csv
    csv_fields = [
        "patient_id",
        "primary_label",
        "primary_label_id",
        "three_class_label",
        "three_class_id",
        "id_prefix",
        "split",
        "cohort_category",
        "storage_source",
        "registration_grade",
        "registration_score",
        "registration_detail",
        "image_dir",
        "ncct_path",
        "ap_path",
        "pvp_path",
        "liver_mask_path",
        "has_ncct",
        "has_ap",
        "has_pvp",
        "has_mask",
        "is_complete_3phase",
    ]
    cases_csv_path = manifest_dir / "cases.csv"
    with cases_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(cases)
    print(f"Wrote {len(cases)} cases to {cases_csv_path}")

    # 2. Write cases.jsonl
    cases_jsonl_path = manifest_dir / "cases.jsonl"
    with cases_jsonl_path.open("w", encoding="utf-8") as f:
        for c in cases:
            rec = {
                "patient_id": c["patient_id"],
                "split": c["split"],
                "label": c["primary_label_id"],
                "label_name": c["primary_label"],
                "three_label": c["three_class_label"],
                "three_label_id": c["three_class_id"],
                "volumes": {
                    "ncct": c["ncct_path"],
                    "ap": c["ap_path"],
                    "pvp": c["pvp_path"],
                },
                "liver_mask": c["liver_mask_path"],
                "storage_source": c["storage_source"],
                "cohort_category": c["cohort_category"],
                "registration_grade": c["registration_grade"],
                "is_complete_3phase": c["is_complete_3phase"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(cases)} cases to {cases_jsonl_path}")

    # 3. Write splits.csv
    splits_csv_path = manifest_dir / "splits.csv"
    with splits_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "split", "primary_label", "cohort_category"])
        writer.writeheader()
        for c in cases:
            writer.writerow({
                "patient_id": c["patient_id"],
                "split": c["split"],
                "primary_label": c["primary_label"],
                "cohort_category": c["cohort_category"],
            })
    print(f"Wrote splits to {splits_csv_path}")

    # 4. Write labels.json & three_class_mapping.json
    labels_json_path = manifest_dir / "labels.json"
    with labels_json_path.open("w", encoding="utf-8") as f:
        json.dump(LABEL_TO_ID, f, indent=2, ensure_ascii=False)

    three_map_path = manifest_dir / "three_class_mapping.json"
    with three_map_path.open("w", encoding="utf-8") as f:
        json.dump({
            "primary_to_three": THREE_CLASS_MAPPING,
            "three_to_id": THREE_CLASS_TO_ID,
        }, f, indent=2, ensure_ascii=False)

    # 5. Write wiztree_recovered_cases.csv
    wiztree_csv_path = manifest_dir / "wiztree_recovered_cases.csv"
    if wiztree_recovered:
        wiz_fields = [
            "patient_id",
            "primary_label",
            "id_prefix",
            "has_ncct",
            "has_ap",
            "has_pvp",
            "has_roi",
            "windows_ncct_path",
            "windows_ap_path",
            "windows_pvp_path",
            "windows_roi_path",
            "all_wiztree_files",
        ]
        with wiztree_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=wiz_fields)
            writer.writeheader()
            writer.writerows(wiztree_recovered)
        print(f"Wrote {len(wiztree_recovered)} WizTree recovered cases to {wiztree_csv_path}")

    # 6. Build reorganization_report.json
    label_counts = Counter(c["primary_label"] for c in cases)
    three_label_counts = Counter(c["three_class_label"] for c in cases)
    split_counts = Counter(c["split"] for c in cases)
    category_counts = Counter(c["cohort_category"] for c in cases)
    storage_counts = Counter(c["storage_source"] for c in cases)
    grade_counts = Counter(c["registration_grade"] for c in cases)

    report = {
        "manifest_name": "GX_Classification_Reorganized_1028",
        "created_at": "2026-10-28",
        "total_cases": len(cases),
        "by_primary_label": dict(label_counts),
        "by_three_class_label": dict(three_label_counts),
        "by_split": dict(split_counts),
        "by_cohort_category": dict(category_counts),
        "by_storage_source": dict(storage_counts),
        "by_registration_grade": dict(grade_counts),
        "wiztree_recovered_count": len(wiztree_recovered),
        "checksums": {
            "cases_csv_sha256": hashlib.sha256(cases_csv_path.read_bytes()).hexdigest(),
            "cases_jsonl_sha256": hashlib.sha256(cases_jsonl_path.read_bytes()).hexdigest(),
            "splits_csv_sha256": hashlib.sha256(splits_csv_path.read_bytes()).hexdigest(),
        },
        "description": "Comprehensive reorganization incorporating 1,183 active cohort, 134 registered Main 5 (including 103 mismatch cases), 116 其他良性 cases, and 36 WizTree recovered cases.",
    }

    report_path = manifest_dir / "reorganization_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote audit report to {report_path}")

    print("\n=== Final Cohort Summary ===")
    print(f"Total Included Cases: {len(cases)}")
    print("By Primary Label:", dict(label_counts))
    print("By Three-Class Label:", dict(three_label_counts))
    print("By Split:", dict(split_counts))
    print("By Category:", dict(category_counts))
    print("By Storage Source:", dict(storage_counts))


if __name__ == "__main__":
    main()
