#!/usr/bin/env python3
"""
Comprehensive Data Cleanliness & Mislabeling Audit Script.
Cross-references:
1. Dataset Ground Truth (cases.csv)
2. Phase Registration QC (cohort_qc.csv)
3. Multi-Model Consensus (ResNet18, ResNet50, DeiT, DINOv2)
4. BiomedCLIP Zero-Shot Pretrained Predictions (biomedclip_patient_predictions.csv)
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

def build_comprehensive_audit():
    data_root = Path("/dataset/downstream_classification")
    cohort_dir = data_root / "cohort"
    diag_dir = data_root / "diagnostics"
    
    # 1. Cases & Splits
    cases_df = pd.read_csv(cohort_dir / "cases.csv")
    splits_df = pd.read_csv(cohort_dir / "splits.csv")
    master_df = cases_df[["patient_id", "label"]].merge(splits_df, on="patient_id", how="left")
    master_df.rename(columns={"label": "true_label"}, inplace=True)
    
    # Extract hospital prefix
    master_df["hospital"] = master_df["patient_id"].apply(lambda x: x.split("_")[0] if "_" in str(x) else "Unknown")

    # 2. Registration QC
    qc_csv = cohort_dir / "cohort_qc.csv"
    if qc_csv.exists():
        qc_df = pd.read_csv(qc_csv)
        qc_df.rename(columns={"配准等级": "registration_grade", "训练图像质量": "qc_flags"}, inplace=True)
        cols = [c for c in ["patient_id", "registration_grade", "qc_flags"] if c in qc_df.columns]
        master_df = master_df.merge(qc_df[cols], on="patient_id", how="left")
    else:
        master_df["registration_grade"] = "UNKNOWN"
        master_df["qc_flags"] = ""

    # 3. Multi-Model Predictions from both Val and Test
    test_pred_csv = diag_dir / "patient_predictions_test.csv"
    val_pred_csv = diag_dir / "patient_predictions_val.csv"
    
    model_preds_dfs = []
    if test_pred_csv.exists():
        model_preds_dfs.append(pd.read_csv(test_pred_csv))
    if val_pred_csv.exists():
        model_preds_dfs.append(pd.read_csv(val_pred_csv))
        
    if model_preds_dfs:
        combined_model_df = pd.concat(model_preds_dfs, ignore_index=True)
        # Compute consensus prediction and consensus agreement across evaluated models
        pred_cols = [c for c in combined_model_df.columns if c.endswith("__pred")]
        
        def compute_mil_consensus(row):
            from collections import Counter
            preds = [row[c] for c in pred_cols if pd.notna(row[c])]
            if not preds:
                return pd.Series([np.nan, 0.0, False])
            cnt = Counter(preds)
            top_pred, top_count = cnt.most_common(1)[0]
            rate = top_count / len(preds)
            correct = (top_pred == row["true_label"])
            return pd.Series([top_pred, round(rate, 3), correct])
            
        combined_model_df[["mil_consensus_pred", "mil_consensus_rate", "mil_consensus_correct"]] = combined_model_df.apply(compute_mil_consensus, axis=1)
        master_df = master_df.merge(
            combined_model_df[["patient_id", "mil_consensus_pred", "mil_consensus_rate", "mil_consensus_correct"]],
            on="patient_id",
            how="left"
        )
    else:
        master_df["mil_consensus_pred"] = np.nan
        master_df["mil_consensus_rate"] = np.nan
        master_df["mil_consensus_correct"] = np.nan

    # 4. BiomedCLIP Zero-Shot Predictions
    biomed_csv = diag_dir / "biomedclip_patient_predictions.csv"
    if biomed_csv.exists():
        biomed_df = pd.read_csv(biomed_csv)
        b_cols = ["patient_id", "biomedclip_pred", "biomedclip_conf", "biomedclip_match",
                  "prob_正常", "prob_肝囊肿", "prob_血管瘤", "prob_HCC", "prob_转移瘤"]
        b_present = [c for c in b_cols if c in biomed_df.columns]
        master_df = master_df.merge(biomed_df[b_present], on="patient_id", how="left")

    # 5. Diagnostic Classification Category
    def categorize_case(row):
        is_reject = (str(row.get("registration_grade")).upper() == "REJECT")
        models_pred = row.get("mil_consensus_pred")
        true_lbl = row.get("true_label")
        biomed_pred = row.get("biomedclip_pred")
        agreement = row.get("mil_consensus_rate", 0.0)

        # Cases evaluated by our supervised MIL ensemble (Val + Test)
        if pd.notna(models_pred):
            if models_pred != true_lbl:
                if is_reject:
                    return "DIRTY_REGISTRATION_MISALIGN"
                if agreement >= 0.8:
                    if biomed_pred == models_pred:
                        return "HIGH_CONFIDENCE_MISLABEL_DUAL_CONFIRMED"
                    return "HIGH_CONFIDENCE_MISLABEL_MODEL_CONSENSUS"
                return "BORDERLINE_HARD_CASE"
            else:
                if is_reject:
                    return "CLEAN_LABEL_ROBUST_DESPITE_QC"
                return "CLEAN_CONFIRMED"
        else:
            # Training set cases (not evaluated by checkpoint ensembles)
            if is_reject:
                return "DIRTY_REGISTRATION_MISALIGN"
            if pd.notna(biomed_pred) and biomed_pred != true_lbl:
                return "POSSIBLE_SUSPECT_BIOMED_FLAG"
            return "CLEAN_CANDIDATE"

    master_df["audit_category"] = master_df.apply(categorize_case, axis=1)

    out_csv = diag_dir / "dataset_label_audit_master.csv"
    master_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Master audit exported to {out_csv}")
    print("\nAudit Category Distribution:")
    print(master_df["audit_category"].value_counts())

if __name__ == "__main__":
    build_comprehensive_audit()
