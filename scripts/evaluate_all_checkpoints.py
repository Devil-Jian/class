#!/usr/bin/env python3
"""Evaluate all saved best.pt models on the test set and generate publication-ready tables.

Traverses <work_dir>/experiments to locate each <model>/<group>/best.pt, evaluates
on the 167-patient held-out Test set (strictly real CECT / NCCT images), saves
patient-level probabilities and labels in test_results.json, and exports
paper_main_table.csv and paper_main_table.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

# Ensure cectcls package is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cectcls.train import evaluate_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all best.pt models on the Test Set.")
    default_work_dir = (
        r"R:\downstream_classification"
        if os.name == "nt" or os.path.exists(r"R:\downstream_classification")
        else "/dataset/downstream_classification"
    )
    parser.add_argument("--work-dir", type=str, default=default_work_dir,
                        help=f"Root directory containing manifests, png_dataset, and experiments (default: {default_work_dir})")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation (default: cuda:0 if available else cpu)")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size (default: 4)")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--bootstrap-rounds", type=int, default=1000,
                        help="Bootstrap rounds for 95%% Confidence Interval (default: 1000, set 0 to disable)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-evaluation even if test_results.json already exists")
    return parser.parse_args()


def compute_bootstrap_ci(y_true, y_probs, n_classes, rounds=1000, seed=42):
    """Compute 95% Confidence Interval for Macro-AUC via non-parametric bootstrapping."""
    if rounds <= 0 or len(y_true) == 0:
        return None
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n = len(y_true)
    boot_aucs = []

    for _ in range(rounds):
        indices = rng.choice(n, size=n, replace=True)
        sample_y = y_true[indices]
        sample_probs = y_probs[indices]
        if len(np.unique(sample_y)) < n_classes:
            continue
        try:
            if n_classes == 2:
                auc = roc_auc_score(sample_y, sample_probs[:, 1])
            else:
                auc = roc_auc_score(sample_y, sample_probs, multi_class="ovr",
                                    average="macro", labels=list(range(n_classes)))
            boot_aucs.append(float(auc))
        except ValueError:
            continue

    if len(boot_aucs) < 50:
        return None
    lower = np.percentile(boot_aucs, 2.5)
    upper = np.percentile(boot_aucs, 97.5)
    return lower, upper


def main():
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    exp_dir = work_dir / "experiments"

    manifest_path = work_dir / "manifests" / "cases.jsonl"
    labels_path = work_dir / "manifests" / "labels.json"
    selection_path = work_dir / "manifests" / "matched_bags.json"
    png_index_path = work_dir / "png_dataset" / "png_index.json"

    print("=" * 80)
    print(">> Automated Test Set Evaluation & Paper Results Generator")
    print(f">> Work Directory: {work_dir}")
    print(f">> Experiments:    {exp_dir}")
    print(f">> Device:         {args.device}")
    print(f">> Manifest:       {manifest_path}")
    print(f">> Test Split:     167 Held-out Patients (100% Real Evaluation)")
    print("=" * 80)

    if not exp_dir.is_dir():
        print(f"[Error] Experiments directory not found at: {exp_dir}")
        sys.exit(1)

    # Find all best.pt checkpoints
    best_checkpoints = sorted(list(exp_dir.glob("*/*/best.pt")))
    if not best_checkpoints:
        # Check one more depth level in case structure is slightly different
        best_checkpoints = sorted(list(exp_dir.glob("**/best.pt")))

    if not best_checkpoints:
        print(f"[Warning] No 'best.pt' files found in {exp_dir}.")
        print("Please ensure your models have finished at least 1 epoch of training.")
        sys.exit(0)

    print(f"Found {len(best_checkpoints)} trained checkpoint(s) ready for evaluation.\n")

    results_by_model = {}
    table_rows = []

    for ckpt_path in best_checkpoints:
        group_dir = ckpt_path.parent
        model_name = group_dir.parent.name
        group_name = group_dir.name
        test_out_json = group_dir / "test_results.json"

        print("-" * 80)
        print(f">> Model: [{model_name}] | Group: [{group_name}]")
        print(f"   Checkpoint: {ckpt_path}")

        # Check if already evaluated
        if test_out_json.exists() and not args.force:
            print(f"   Found existing test results in {test_out_json.name}. Loading...")
            with open(test_out_json, "r", encoding="utf-8") as f:
                res_data = json.load(f)
            metrics = res_data.get("metrics", {})
            predictions = res_data.get("predictions", [])
        else:
            # Construct evaluation arguments
            eval_args = SimpleNamespace(
                checkpoint=str(ckpt_path),
                split="test",
                device=args.device,
                out=str(test_out_json),
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
            print("   Running evaluation on 167 Test patients...")
            metrics = evaluate_checkpoint(eval_args)
            with open(test_out_json, "r", encoding="utf-8") as f:
                res_data = json.load(f)
            predictions = res_data.get("predictions", [])

        # Extract primary metrics
        acc = metrics.get("accuracy", 0.0)
        b_acc = metrics.get("balanced_accuracy", 0.0)
        f1 = metrics.get("macro_f1", 0.0)
        macro_auc = metrics.get("macro_auc", 0.0)

        three_metrics = metrics.get("three_class", {})
        three_auc = three_metrics.get("macro_auc", 0.0) if three_metrics else None

        # Calculate 95% Confidence Intervals if predictions exist
        ci_str = ""
        three_ci_str = ""
        if predictions and args.bootstrap_rounds > 0:
            y_true_5 = [p["label"] for p in predictions]
            y_prob_5 = [p["probabilities"] for p in predictions]
            ci_5 = compute_bootstrap_ci(y_true_5, y_prob_5, n_classes=len(y_prob_5[0]), rounds=args.bootstrap_rounds)
            if ci_5:
                ci_str = f" [{ci_5[0]:.4f}-{ci_5[1]:.4f}]"

            if "three_class_probabilities" in predictions[0]:
                primary_to_three = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
                y_true_3 = [primary_to_three.get(p["label"], 0) for p in predictions]
                y_prob_3 = [p["three_class_probabilities"] for p in predictions]
                ci_3 = compute_bootstrap_ci(y_true_3, y_prob_3, n_classes=3, rounds=args.bootstrap_rounds)
                if ci_3:
                    three_ci_str = f" [{ci_3[0]:.4f}-{ci_3[1]:.4f}]"

        auc_fmt = f"{macro_auc:.4f}{ci_str}" if macro_auc else "N/A"
        three_auc_fmt = f"{three_auc:.4f}{three_ci_str}" if three_auc else "N/A"

        print(f"   ✓ Macro-AUC (5-Class): {auc_fmt}")
        print(f"   ✓ 3-Class AUC (良恶性): {three_auc_fmt}")
        print(f"   ✓ Accuracy:            {acc:.4f} | Bal-Acc: {b_acc:.4f} | Macro-F1: {f1:.4f}")
        print(f"   ✓ Predictions Saved:   {test_out_json}")

        record = {
            "Model": model_name,
            "Training Group": group_name,
            "Macro-AUC (5-Class)": f"{macro_auc:.4f}" if macro_auc else "N/A",
            "5-Class 95% CI": ci_str.strip(" []") if ci_str else "N/A",
            "3-Class AUC": f"{three_auc:.4f}" if three_auc else "N/A",
            "3-Class 95% CI": three_ci_str.strip(" []") if three_ci_str else "N/A",
            "Accuracy": f"{acc:.4f}",
            "Balanced Acc": f"{b_acc:.4f}",
            "Macro-F1": f"{f1:.4f}",
        }
        table_rows.append(record)
        results_by_model.setdefault(model_name, []).append(record)

    # Export Paper Main Table
    csv_out = exp_dir / "paper_main_table.csv"
    md_out = exp_dir / "paper_main_table.md"

    headers = [
        "Model", "Training Group", "Macro-AUC (5-Class)", "5-Class 95% CI",
        "3-Class AUC", "3-Class 95% CI", "Accuracy", "Balanced Acc", "Macro-F1"
    ]

    # Write CSV
    import csv
    with open(csv_out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(table_rows)

    # Write Markdown
    md_lines = [
        "# Downstream Multi-Phase CECT Classification: Paper Main Results (Test Set)",
        "",
        "> **Cohort**: 167 Independent Held-out Patients (100% Real Scans Evaluated)",
        "> **Validation Metric**: Bootstrap 95% Confidence Intervals (N=1000)",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in table_rows:
        md_lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")

    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # Pretty print summary table to terminal
    print("\n" + "=" * 110)
    print("🏆 FINAL PAPER RESULTS SUMMARY TABLE (Held-out 167 Test Cases)")
    print("=" * 110)
    for model_name, rows in results_by_model.items():
        print(f"\n[Framework: {model_name}]")
        print(f"{'Training Group':<16} | {'Macro-AUC':<10} | {'3-Class AUC':<12} | {'Accuracy':<9} | {'Bal-Acc':<9} | {'Macro-F1':<9}")
        print("-" * 75)
        for r in rows:
            print(f"{r['Training Group']:<16} | {r['Macro-AUC (5-Class)']:<10} | {r['3-Class AUC']:<12} | {r['Accuracy']:<9} | {r['Balanced Acc']:<9} | {r['Macro-F1']:<9}")

    print("\n" + "=" * 110)
    print(f">> Full Results CSV saved to:      {csv_out}")
    print(f">> Publication Markdown saved to: {md_out}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
