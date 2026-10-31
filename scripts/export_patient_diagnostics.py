#!/usr/bin/env python3
"""Export patient-level predictions and error diagnostics for Validation and Test cohorts.

Analyzes test_results.json / val_results.json from trained models to identify:
1. Hard / Misclassified cases across different models.
2. Prediction probabilities, rank-1 errors, and confidence margins.
3. Cross-reference with cohort_qc.csv (registration grade, QC notes).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cectcls.train import evaluate_checkpoint


LABEL_MAP = {0: "正常", 1: "肝囊肿", 2: "血管瘤", 3: "HCC", 4: "转移瘤"}
THREE_WAY_MAP = {0: "正常", 1: "良性", 2: "恶性"}
PRIMARY_TO_THREE = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}


def parse_args():
    parser = argparse.ArgumentParser(description="Export patient-level classification results for Val & Test.")
    default_work_dir = (
        r"R:\downstream_classification"
        if os.name == "nt" or os.path.exists(r"R:\downstream_classification")
        else "/dataset/downstream_classification"
    )
    parser.add_argument("--work-dir", type=str, default=default_work_dir,
                        help="Root dataset/experiments directory")
    parser.add_argument("--split", type=str, default="both", choices=["val", "test", "both"],
                        help="Split to evaluate/export (default: both)")
    parser.add_argument("--models", nargs="+", default=["resnet18_mil", "dinov2s_mil", "deit_mil", "resnet50_mil"],
                        help="Models to inspect")
    parser.add_argument("--groups", nargs="+", default=["cect_real", "cect_real_syn", "cect_syn", "ncct"],
                        help="Training groups to inspect")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (defaults to <work-dir>/diagnostics)")
    return parser.parse_args()


def load_cohort_qc(work_dir: Path) -> dict[str, dict]:
    """Load metadata and QC info from cohort_qc.csv if available."""
    qc_path = work_dir / "cohort" / "cohort_qc.csv"
    if not qc_path.exists():
        qc_path = work_dir / "manifests" / "cohort_qc.csv"
    qc_info = {}
    if qc_path.exists():
        with open(qc_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qc_info[row["patient_id"]] = {
                    "reg_grade": row.get("配准等级", "N/A"),
                    "qc_note": row.get("NCCT对应及Syn生成QC", ""),
                    "id_prefix": row.get("id_prefix", ""),
                }
    return qc_info


def get_or_run_predictions(work_dir: Path, model_name: str, group_name: str, split: str, args) -> list[dict]:
    """Load cached predictions or run evaluation on split."""
    exp_dir = work_dir / "experiments" / model_name / group_name
    result_file = exp_dir / f"{split}_results.json"

    # Check if already cached
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "predictions" in data and len(data["predictions"]) > 0:
                return data["predictions"]

    ckpt_path = exp_dir / "best.pt"
    if not ckpt_path.exists():
        return []

    manifest_path = work_dir / "manifests" / "cases.jsonl"
    labels_path = work_dir / "manifests" / "labels.json"
    selection_path = work_dir / "manifests" / "matched_bags.json"
    png_index_path = work_dir / "png_dataset" / "png_index.json"

    eval_args = SimpleNamespace(
        checkpoint=str(ckpt_path),
        split=split,
        device=args.device,
        out=str(result_file),
        manifest=str(manifest_path),
        labels=str(labels_path),
        selection=str(selection_path),
        png_index=str(png_index_path) if png_index_path.exists() else None,
        syn_index=None,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=42,
        image_size=224,
        uniform=16,
        random_count=16,
    )
    print(f">> Running [{split}] inference for {model_name}/{group_name}...")
    try:
        evaluate_checkpoint(eval_args)
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("predictions", [])
    except Exception as e:
        print(f"   [Error] Inference failed for {model_name}/{group_name} on {split}: {e}")
        return []


def main():
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else work_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    qc_info = load_cohort_qc(work_dir)

    print("=" * 85)
    print(">> Patient-Level Diagnostics and Error Analysis Exporter")
    print(f">> Work Directory: {work_dir}")
    print(f">> Output Directory: {out_dir}")
    print(f">> Target Splits: {splits}")
    print("=" * 85)

    for split in splits:
        print(f"\n>> Processing split: [{split.upper()}] ...")
        # Structure: patient_id -> { "label": int, "label_name": str, "models": { model_group_key: pred_record } }
        patients_dict: dict[str, dict] = {}

        # Collect predictions from all models & groups
        active_experiments = []
        for model in args.models:
            for group in args.groups:
                exp_key = f"{model}__{group}"
                preds = get_or_run_predictions(work_dir, model, group, split, args)
                if not preds:
                    continue
                active_experiments.append(exp_key)
                for p in preds:
                    pid = p["patient_id"]
                    if pid not in patients_dict:
                        true_label_id = p["label"]
                        patients_dict[pid] = {
                            "patient_id": pid,
                            "true_label_id": true_label_id,
                            "true_label_name": LABEL_MAP.get(true_label_id, str(true_label_id)),
                            "true_3class_id": PRIMARY_TO_THREE.get(true_label_id, 0),
                            "true_3class_name": THREE_WAY_MAP.get(PRIMARY_TO_THREE.get(true_label_id, 0), ""),
                            "predictions": {},
                        }
                    probs = np.array(p["probabilities"])
                    pred_label_id = int(np.argmax(probs))
                    pred_conf = float(probs[pred_label_id])
                    true_label_id = p["label"]
                    true_prob = float(probs[true_label_id])
                    is_correct_5 = (pred_label_id == true_label_id)

                    three_probs = p.get("three_class_probabilities")
                    if three_probs:
                        arr_3 = np.array(three_probs)
                        pred_3_id = int(np.argmax(arr_3))
                        true_3_id = PRIMARY_TO_THREE.get(true_label_id, 0)
                        is_correct_3 = (pred_3_id == true_3_id)
                    else:
                        pred_3_id = None
                        is_correct_3 = None

                    patients_dict[pid]["predictions"][exp_key] = {
                        "pred_label_id": pred_label_id,
                        "pred_label_name": LABEL_MAP.get(pred_label_id, str(pred_label_id)),
                        "pred_confidence": pred_conf,
                        "true_class_prob": true_prob,
                        "is_correct_5": is_correct_5,
                        "pred_3class_name": THREE_WAY_MAP.get(pred_3_id, "") if pred_3_id is not None else "",
                        "is_correct_3": is_correct_3,
                        "probabilities": [round(x, 4) for x in p["probabilities"]],
                    }

        if not patients_dict:
            print(f"[Warning] No predictions found or generated for split: {split}")
            continue

        # Compile detailed CSV
        csv_path = out_dir / f"patient_predictions_{split}.csv"
        hard_cases_path = out_dir / f"hard_cases_disagreement_{split}.csv"

        headers = [
            "patient_id", "split", "true_label", "true_3class",
            "reg_grade", "qc_note", "hospital_prefix",
            "total_models_evaluated", "correct_count_5class", "error_count_5class",
            "accuracy_ratio_5class", "correct_count_3class", "accuracy_ratio_3class",
            "is_consensus_hard_case"
        ]
        # Append per-experiment columns
        for exp in active_experiments:
            headers.extend([
                f"{exp}__pred",
                f"{exp}__conf",
                f"{exp}__is_correct",
                f"{exp}__true_prob",
            ])

        rows = []
        hard_rows = []

        for pid, pdata in sorted(patients_dict.items()):
            preds_map = pdata["predictions"]
            evaluated_count = len(preds_map)
            if evaluated_count == 0:
                continue

            correct_5 = sum(1 for m in preds_map.values() if m["is_correct_5"])
            correct_3 = sum(1 for m in preds_map.values() if m.get("is_correct_3", False))
            acc_ratio_5 = round(correct_5 / evaluated_count, 3)
            acc_ratio_3 = round(correct_3 / evaluated_count, 3)

            # Consensus Hard Case: Misclassified by >= 50% of the models
            is_hard = (acc_ratio_5 <= 0.5)

            meta = qc_info.get(pid, {})
            row = {
                "patient_id": pid,
                "split": split,
                "true_label": pdata["true_label_name"],
                "true_3class": pdata["true_3class_name"],
                "reg_grade": meta.get("reg_grade", "N/A"),
                "qc_note": meta.get("qc_note", ""),
                "hospital_prefix": meta.get("id_prefix", pid.split("_")[0]),
                "total_models_evaluated": evaluated_count,
                "correct_count_5class": correct_5,
                "error_count_5class": evaluated_count - correct_5,
                "accuracy_ratio_5class": acc_ratio_5,
                "correct_count_3class": correct_3,
                "accuracy_ratio_3class": acc_ratio_3,
                "is_consensus_hard_case": "YES" if is_hard else "NO",
            }

            for exp in active_experiments:
                if exp in preds_map:
                    item = preds_map[exp]
                    row[f"{exp}__pred"] = item["pred_label_name"]
                    row[f"{exp}__conf"] = round(item["pred_confidence"], 3)
                    row[f"{exp}__is_correct"] = 1 if item["is_correct_5"] else 0
                    row[f"{exp}__true_prob"] = round(item["true_class_prob"], 3)
                else:
                    row[f"{exp}__pred"] = ""
                    row[f"{exp}__conf"] = ""
                    row[f"{exp}__is_correct"] = ""
                    row[f"{exp}__true_prob"] = ""

            rows.append(row)
            if is_hard:
                hard_rows.append(row)

        # Write Full CSV
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        # Write Hard Cases CSV
        with open(hard_cases_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(hard_rows)

        # Generate Markdown Summary Report
        md_path = out_dir / f"error_diagnostic_report_{split}.md"
        total_p = len(rows)
        hard_count = len(hard_rows)

        # Group hard cases by true label
        hard_by_label = {}
        for r in hard_rows:
            label = r["true_label"]
            hard_by_label[label] = hard_by_label.get(label, 0) + 1

        # Hard cases by registration grade
        hard_by_reg = {}
        for r in hard_rows:
            reg = r["reg_grade"]
            hard_by_reg[reg] = hard_by_reg.get(reg, 0) + 1

        md_content = [
            f"# {split.upper()} Cohort Patient-Level Diagnostic & Error Analysis Report",
            "",
            f"- **Total Patients Evaluated**: {total_p}",
            f"- **Consensus Hard / Misclassified Cases**: {hard_count} ({hard_count / max(1, total_p) * 100:.1f}%)",
            f"- **Models Included**: {', '.join(active_experiments)}",
            "",
            "## 1. Hard Cases Distribution by True Disease Label",
            "",
            "| True Disease Label | Hard Cases Count | Total in Split | Misclassification Rate (%) |",
            "| :--- | :---: | :---: | :---: |",
        ]

        # Calculate class totals in split
        class_totals = {}
        for r in rows:
            lbl = r["true_label"]
            class_totals[lbl] = class_totals.get(lbl, 0) + 1

        for lbl, total_lbl in sorted(class_totals.items()):
            cnt = hard_by_label.get(lbl, 0)
            rate = cnt / total_lbl * 100
            md_content.append(f"| **{lbl}** | {cnt} | {total_lbl} | {rate:.1f}% |")

        md_content.extend([
            "",
            "## 2. Hard Cases vs Registration QC Grade",
            "",
            "| Registration Grade (配准等级) | Hard Cases Count | Note |",
            "| :--- | :---: | :--- |",
        ])
        for reg, cnt in sorted(hard_by_reg.items()):
            md_content.append(f"| **{reg}** | {cnt} | {'呼吸运动或切片未严格对齐' if reg == 'REJECT' else '标准或优质对齐'} |")

        md_content.extend([
            "",
            "## 3. Top 15 Most Disputed / Misclassified Patients",
            "",
            "| Patient ID | True Label | Reg Grade | 5-Class Accuracy Ratio | Models Disagreeing |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ])

        # Sort by accuracy ratio ascending (worst predicted first)
        sorted_worst = sorted(hard_rows, key=lambda x: (x["accuracy_ratio_5class"], x["accuracy_ratio_3class"]))
        for r in sorted_worst[:15]:
            md_content.append(
                f"| `{r['patient_id']}` | **{r['true_label']}** | `{r['reg_grade']}` | "
                f"{r['accuracy_ratio_5class'] * 100:.1f}% ({r['correct_count_5class']}/{r['total_models_evaluated']}) | "
                f"误诊为多模型不同类别 |"
            )

        md_content.extend([
            "",
            "---",
            f"📄 **Full Patient CSV**: [{csv_path.name}](./{csv_path.name})  ",
            f"⚠️ **Hard Cases Only CSV**: [{hard_cases_path.name}](./{hard_cases_path.name})  ",
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content) + "\n")

        print(f"   ✓ Full CSV saved: {csv_path}")
        print(f"   ✓ Hard cases saved: {hard_cases_path}")
        print(f"   ✓ Diagnostic report: {md_path}")
        print(f"   ✓ Found {hard_count}/{total_p} consensus hard cases.")

    print("\n" + "=" * 85)
    print(">> Diagnostics export complete! All files saved to:", out_dir)
    print("=" * 85)


if __name__ == "__main__":
    main()
