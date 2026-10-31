#!/usr/bin/env python3
"""
Build Balanced Mini Cohort (Scheme 1: 245 cases, 175 Train, 35 Val, 35 Test).
Strict 1:1:1:1:1 class balance across all splits.
Prioritizes highest image registration quality (GOLD > SILVER > REJECT).
"""

import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["正常", "肝囊肿", "血管瘤", "HCC", "转移瘤"]
PLAN = {
    "train": 35,  # 35 * 5 = 175
    "val": 7,     # 7 * 5 = 35
    "test": 7     # 7 * 5 = 35
}


def build_balanced_mini_cohort(
    data_root: Path,
    out_manifest: Path,
    out_splits_csv: Path,
    seed: int = 42
):
    cases_jsonl_path = data_root / "manifests" / "cases.jsonl"
    qc_csv_path = data_root / "cohort" / "cohort_qc.csv"

    logger.info(f"Loading full cases from: {cases_jsonl_path}")
    with open(cases_jsonl_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f]

    logger.info(f"Total input cases in full manifest: {len(cases)}")

    # Load QC grades for quality prioritization
    grade_map = {}
    if qc_csv_path.exists():
        qc_df = pd.read_csv(qc_csv_path)
        col_grade = "配准等级" if "配准等级" in qc_df.columns else "registration_grade"
        for _, row in qc_df.iterrows():
            grade_map[str(row["patient_id"])] = str(row.get(col_grade, "UNKNOWN"))

    grade_rank = {"GOLD": 0, "SILVER": 1, "REJECT": 2, "UNKNOWN": 3}

    for c in cases:
        pid = c["patient_id"]
        c["grade"] = grade_map.get(pid, "UNKNOWN")
        c["grade_rank"] = grade_rank.get(c["grade"], 3)

    cases_df = pd.DataFrame(cases)

    selected_records = []
    selected_pids = set()
    summary_rows = []

    for split_name, target_per_class in PLAN.items():
        split_pool = cases_df[cases_df["split"] == split_name].copy()
        
        for cname in CLASS_NAMES:
            class_pool = split_pool[split_pool["label_name"] == cname].copy()
            # Deterministic, quality-first sorting:
            # 1) Higher quality first (GOLD -> SILVER -> REJECT)
            # 2) Consistent random shuffle with fixed seed for same-grade tie breaking
            shuffled = class_pool.sample(frac=1.0, random_state=seed)
            sorted_pool = shuffled.sort_values(by="grade_rank", kind="stable")

            picked = sorted_pool.head(target_per_class)
            if len(picked) < target_per_class:
                raise ValueError(
                    f"Not enough cases for {cname} in {split_name}: need {target_per_class}, got {len(picked)}"
                )

            records = picked.to_dict("records")
            selected_records.extend(records)
            for r in records:
                selected_pids.add(r["patient_id"])
                summary_rows.append({
                    "patient_id": r["patient_id"],
                    "split": split_name,
                    "label_name": cname,
                    "label_id": r["label"],
                    "grade": r["grade"]
                })

            grade_counts = picked["grade"].value_counts().to_dict()
            logger.info(
                f"[{split_name.upper():5s}] {cname:6s}: picked {len(picked)} | Quality: {grade_counts}"
            )

    # Export mini cases.jsonl (strip auxiliary sorting keys)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        for r in selected_records:
            clean_item = {k: v for k, v in r.items() if k not in {"grade", "grade_rank"}}
            f.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
    logger.info(f"Saved balanced mini manifest ({len(selected_records)} cases) to: {out_manifest}")

    # Export mini splits CSV
    summary_df = pd.DataFrame(summary_rows)
    out_splits_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df[["patient_id", "split", "label_name", "label_id", "grade"]].to_csv(
        out_splits_csv, index=False, encoding="utf-8-sig"
    )
    logger.info(f"Saved balanced mini splits summary to: {out_splits_csv}")

    # Print summary table
    print("\n" + "=" * 70)
    print(">> Balanced Mini Cohort Summary (Scheme 1: 245 Cases)")
    print("=" * 70)
    pivot = summary_df.pivot_table(
        index="label_name", columns="split", values="patient_id", aggfunc="count"
    )[["train", "val", "test"]]
    pivot["total"] = pivot.sum(axis=1)
    print(pivot)
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build Balanced Mini Cohort")
    parser.add_argument("--data-root", type=Path, default="/dataset/downstream_classification")
    parser.add_argument("--out-manifest", type=Path, default="/dataset/downstream_classification/manifests/cases_balanced_mini.jsonl")
    parser.add_argument("--out-splits", type=Path, default="/dataset/downstream_classification/cohort/splits_balanced_mini.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_balanced_mini_cohort(
        data_root=args.data_root,
        out_manifest=args.out_manifest,
        out_splits_csv=args.out_splits,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
