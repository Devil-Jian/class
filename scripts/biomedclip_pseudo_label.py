#!/usr/bin/env python3
"""
BiomedCLIP Zero-Shot Pseudo-Labeling & Feature Extraction Script for CECT Cohort.
Uses Microsoft's BiomedCLIP (PubMedBERT + ViT-B/16) to provide independent zero-shot
semantic predictions for liver lesion candidates.
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import open_clip

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["正常", "肝囊肿", "血管瘤", "HCC", "转移瘤"]

# Optimized prompt templates tailored for medical imaging / PubMed captions
PROMPT_TEMPLATES = {
    "正常": [
        "computed tomography scan showing normal liver parenchyma without lesion",
        "unremarkable abdominal CT scan of healthy liver",
        "normal liver CT scan"
    ],
    "肝囊肿": [
        "computed tomography scan showing simple benign hepatic cyst with fluid attenuation",
        "well-circumscribed hypodense liver cyst on CT",
        "benign hepatic cyst on abdominal CT scan"
    ],
    "血管瘤": [
        "computed tomography scan showing hepatic cavernous hemangioma with peripheral nodular enhancement",
        "liver hemangioma on contrast-enhanced CT",
        "benign cavernous hepatic hemangioma on CT"
    ],
    "HCC": [
        "computed tomography scan showing hepatocellular carcinoma HCC with arterial hyperenhancement",
        "hypervascular liver tumor hepatocellular carcinoma on CT",
        "malignant hepatocellular carcinoma on contrast-enhanced CT"
    ],
    "转移瘤": [
        "computed tomography scan showing multiple liver metastases with peripheral ring enhancement",
        "secondary metastatic liver tumors on contrast-enhanced CT",
        "gastrointestinal metastasis to the liver on CT scan"
    ]
}


def build_text_embeddings(model, tokenizer, device: str):
    text_embeds = []
    logger.info("Computing text embeddings for 5 liver disease classes...")
    with torch.no_grad():
        for cname in CLASS_NAMES:
            prompts = PROMPT_TEMPLATES[cname]
            toks = tokenizer(prompts).to(device)
            feats = model.encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            # Ensemble mean across templates
            class_feat = feats.mean(dim=0, keepdim=True)
            class_feat = class_feat / class_feat.norm(dim=-1, keepdim=True)
            text_embeds.append(class_feat)
    return torch.cat(text_embeds, dim=0)  # [5, 512]


def run_biomedclip_audit(
    cases_csv: Path,
    splits_csv: Path,
    selected_bags_json: Path,
    png_dir: Path,
    out_csv: Path,
    split_filter: str = "all",
    top_k_slices: int = 4,
    device: str = "cuda:0"
):
    logger.info(f"Loading cases from {cases_csv} and splits from {splits_csv}")
    cases_df = pd.read_csv(cases_csv)
    if splits_csv.exists():
        splits_df = pd.read_csv(splits_csv)
        cases_df = cases_df.merge(splits_df, on="patient_id", how="left")
    
    if split_filter != "all" and "split" in cases_df.columns:
        cases_df = cases_df[cases_df["split"] == split_filter].copy()
    logger.info(f"Target cases to process: {len(cases_df)} (filter={split_filter})")

    logger.info(f"Loading selected attention bags from {selected_bags_json}")
    with open(selected_bags_json, "r", encoding="utf-8") as f:
        bags_data = json.load(f)
    bags_by_pid = {item["patient_id"]: item for item in bags_data}

    # Load local BiomedCLIP model
    model_cache = Path("/root/.cache/biomedclip")
    logger.info(f"Loading BiomedCLIP from local cache: {model_cache} on {device}")
    model, _, val_transform = open_clip.create_model_and_transforms(f"local-dir:{model_cache}")
    tokenizer = open_clip.get_tokenizer(f"local-dir:{model_cache}")
    model = model.to(device).eval()

    text_embeds = build_text_embeddings(model, tokenizer, device)

    results = []
    logger.info(f"Evaluating patients using top-{top_k_slices} attention slices (AP + PVP)...")
    
    for _, row in tqdm(cases_df.iterrows(), total=len(cases_df), desc="BiomedCLIP Audit"):
        pid = row["patient_id"]
        true_label = row["label"]
        split = row.get("split", "unknown")
        
        bag = bags_by_pid.get(pid)
        if not bag or "selected_positions" not in bag:
            results.append({
                "patient_id": pid,
                "split": split,
                "true_label": true_label,
                "biomedclip_pred": "MISSING_BAG",
                "biomedclip_conf": 0.0,
                "prob_正常": 0.0,
                "prob_肝囊肿": 0.0,
                "prob_血管瘤": 0.0,
                "prob_HCC": 0.0,
                "prob_转移瘤": 0.0,
            })
            continue

        top_positions = bag["selected_positions"][:top_k_slices]
        slice_sims = []

        for pos in top_positions:
            s_id = pos["reference_slice_id"]
            # Evaluate AP and PVP phases
            for phase in ["ap", "pvp"]:
                png_path = png_dir / "real" / phase / pid / f"ref_{s_id:04d}.png"
                if not png_path.exists():
                    continue
                try:
                    img = Image.open(png_path).convert("RGB")
                    t_img = val_transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        f_img = model.encode_image(t_img)
                        f_img = f_img / f_img.norm(dim=-1, keepdim=True)
                        sim = (f_img @ text_embeds.T).squeeze(0)  # [5]
                        slice_sims.append(sim)
                except Exception as e:
                    logger.debug(f"Error reading slice {png_path}: {e}")

        if not slice_sims:
            results.append({
                "patient_id": pid,
                "split": split,
                "true_label": true_label,
                "biomedclip_pred": "NO_VALID_SLICES",
                "biomedclip_conf": 0.0,
                "prob_正常": 0.0,
                "prob_肝囊肿": 0.0,
                "prob_血管瘤": 0.0,
                "prob_HCC": 0.0,
                "prob_转移瘤": 0.0,
            })
            continue

        mean_sim = torch.stack(slice_sims).mean(dim=0)
        # Scaled softmax
        probs = (mean_sim * 40.0).softmax(dim=-1).cpu().numpy()
        pred_idx = probs.argmax()
        pred_label = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx])

        results.append({
            "patient_id": pid,
            "split": split,
            "true_label": true_label,
            "biomedclip_pred": pred_label,
            "biomedclip_conf": round(conf, 4),
            "biomedclip_match": (pred_label == true_label),
            "prob_正常": round(float(probs[0]), 4),
            "prob_肝囊肿": round(float(probs[1]), 4),
            "prob_血管瘤": round(float(probs[2]), 4),
            "prob_HCC": round(float(probs[3]), 4),
            "prob_转移瘤": round(float(probs[4]), 4),
        })

    out_df = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved BiomedCLIP predictions to {out_csv}")

    # Accuracy summary
    valid_df = out_df[out_df["biomedclip_match"].notna()]
    acc = valid_df["biomedclip_match"].mean() * 100
    logger.info(f"BiomedCLIP Zero-Shot Accuracy across {len(valid_df)} cases: {acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="BiomedCLIP Zero-Shot Audit")
    parser.add_argument("--cases", type=Path, default="/dataset/downstream_classification/cohort/cases.csv")
    parser.add_argument("--splits", type=Path, default="/dataset/downstream_classification/cohort/splits.csv")
    parser.add_argument("--bags", type=Path, default="/dataset/downstream_classification/manifests/selected_bags.json")
    parser.add_argument("--png-dir", type=Path, default="/dataset/downstream_classification/png_dataset")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default="/dataset/downstream_classification/diagnostics/biomedclip_patient_predictions.csv")
    args = parser.parse_args()

    run_biomedclip_audit(
        cases_csv=args.cases,
        splits_csv=args.splits,
        selected_bags_json=args.bags,
        png_dir=args.png_dir,
        out_csv=args.out,
        split_filter=args.split,
        top_k_slices=args.top_k,
        device=args.device
    )


if __name__ == "__main__":
    main()
