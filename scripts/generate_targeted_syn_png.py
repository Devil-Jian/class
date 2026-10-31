#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Targeted 5-slice NCCT -> Multi-phase CECT Synthesis & Lossless PNG Export.

Features:
  - Reads matched Top-16 slice indices from matched_bags.json.
  - Efficiently loads 5-slice NCCT context ([z-2, z-1, z, z+1, z+2]) from /dataset/h5_cache.
  - Stage 1: Residual Palette Diffusion coarse synthesis.
  - Stage 2: Multi-phase DDPM prior with SDEdit & Mode A spectral refinement.
  - Exports both Stage 1 and Stage 2 AP/PVP slices as windowed [-160, 240] HU 8-bit PNGs.
  - Exports real AP/PVP/NCCT PNGs for ground-truth pairing and rapid visual inspection.
  - Generates unified dataset_manifest.jsonl for zero-I/O downstream training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Ensure syn-final-1026 and cectcls in python path
SYN_ROOT = Path("/workspace/syn-final-1026").resolve()
CLASS_ROOT = Path("/workspace/class").resolve()
for p in (SYN_ROOT, CLASS_ROOT, CLASS_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from inference.pipeline_engine import (
    load_stage1_residual_pipeline,
    load_stage2_refiner,
    normalize_hu,
    PHASE_NAMES,
)
from src.prior_ddpm.spectral_refiner import apply_step1_spectral_liver_refinement, DEFAULT_STEP1_CONFIG
from src.residual import normalize_ncct_hu
from diffusers.utils.torch_utils import randn_tensor


def window_uint8(hu: np.ndarray, lo: float = -160.0, hi: float = 240.0) -> np.ndarray:
    """Window HU array to uint8 [0, 255] safely."""
    arr = np.asarray(hu, dtype=np.float32)
    clipped = np.clip(arr, lo, hi)
    return np.rint((clipped - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def apply_flip(arr: np.ndarray, mode: str) -> np.ndarray:
    """Apply spatial flip across last two dimensions (H, W)."""
    if mode == "both":
        return np.flip(arr, axis=(-2, -1)).copy()
    elif mode == "horizontal":
        return np.flip(arr, axis=-1).copy()
    elif mode == "vertical":
        return np.flip(arr, axis=-2).copy()
    elif mode == "none":
        return arr.copy()
    else:
        raise ValueError(f"Unknown flip mode: {mode}")


def get_patient_h5_data(h5_root: Path, patient_id: str) -> Optional[Dict[str, np.ndarray]]:
    """Load full volumes from H5 cache if available."""
    h5_path = h5_root / f"{patient_id}.h5"
    if not h5_path.is_file():
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            data = {}
            for k in ("ncct", "mask", "ap", "pvp"):
                if k in f:
                    data[k] = np.asarray(f[k], dtype=np.float32 if k != "mask" else np.uint8)
            return data
    except Exception as e:
        print(f"[Warning] Failed to read H5 for {patient_id}: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Targeted 5-slice CECT Synthesis & PNG Exporter")
    parser.add_argument("--matched", type=str, default="/dataset/downstream_classification/manifests/matched_bags.json",
                        help="Path to matched_bags.json")
    parser.add_argument("--manifest", type=str, default="/dataset/downstream_classification/manifests/cases.jsonl",
                        help="Path to cases.jsonl")
    parser.add_argument("--h5-root", type=str, default="/dataset/h5_cache",
                        help="Path to /dataset/h5_cache directory")
    parser.add_argument("--output-dir", type=str, default="/dataset/downstream_classification/png_dataset",
                        help="Output directory for PNG dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"],
                        help="Patient split to process (default: train)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Inference batch size across slices (default: 4)")
    parser.add_argument("--stage1-model", type=str,
                        default="/dataset/train_output/syn-final-1013/checkpoints/residual-palette-stage1-stride3-50k/best_model",
                        help="Path to Stage 1 model")
    parser.add_argument("--stage2-ckpt", type=str,
                        default="/dataset/train_output/syn-final-1013/checkpoints/ddpm_prior_multiphase/checkpoint-030000.pt",
                        help="Path to Stage 2 checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="GPU device (default: auto, picks GPU with most free VRAM)")
    parser.add_argument("--flip", type=str, default="both", choices=["both", "horizontal", "vertical", "none"],
                        help="Spatial flip mode: both (default, up-down + left-right), horizontal, vertical, none")
    parser.add_argument("--patient-id", type=str, default=None, help="Process single patient for testing")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of patients to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        if torch.cuda.is_available():
            best_dev, max_free = 0, -1
            for dev_idx in range(torch.cuda.device_count()):
                free_mem = torch.cuda.mem_get_info(dev_idx)[0]
                if free_mem > max_free:
                    max_free = free_mem
                    best_dev = dev_idx
            device = torch.device(f"cuda:{best_dev}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    h5_root = Path(args.h5_root)

    print("=" * 70)
    print(">> Targeted 5-slice CECT Synthesis & PNG Exporter")
    print(f">> Device: {device} | Split: {args.split} | Batch Size: {args.batch_size}")
    print(f">> Matched bags: {args.matched}")
    print(f">> Output root:  {out_root}")
    print("=" * 70)

    # 1. Load Manifest & Matched bags
    cases_map = {}
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                cases_map[item["patient_id"]] = item

    with open(args.matched, "r", encoding="utf-8") as f:
        patients_matched = json.load(f)

    # Filter patients
    targets = []
    for p in patients_matched:
        pid = p["patient_id"]
        if pid not in cases_map:
            continue
        p_split = cases_map[pid]["split"]
        if args.split != "all" and p_split != args.split:
            continue
        if args.patient_id and pid != args.patient_id:
            continue
        targets.append(p)

    if args.limit:
        targets = targets[:args.limit]

    print(f">> Total target patients to process: {len(targets)}")
    if not targets:
        print(">> No patients to process. Exiting.")
        return

    # 2. Load Models
    print("\n[Loading Models]")
    stage1_pipe = load_stage1_residual_pipeline(args.stage1_model, device)
    stage2_refiner = load_stage2_refiner(args.stage2_ckpt, device)

    # Calibration & Spectral configs
    stage1_calib = {
        "calibrate_nullspace": True,
        "target_soft_tissue_enh_hu": (8.0, 28.0, 19.0),
        "w_dc_high": 0.90,
        "w_dc_low": 0.35,
        "t_high": 700,
        "t_low": 250,
        "soft_tissue_hu_min": 35.0,
        "soft_tissue_hu_max": 75.0,
    }

    mode_cfg = {
        "sdedit_t0": 350,
        "low_frequency_weight": 0.0,
        "fft_cutoff": {"AP": 36, "PVP": 32, "DP": 28},
        "use_step1_spectral": True,
        "step1_spectral_cfg": DEFAULT_STEP1_CONFIG,
    }

    manifest_records = []
    manifest_path = out_root / "dataset_manifest.jsonl"
    existing_completed = set()
    if manifest_path.is_file() and not args.overwrite:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    existing_completed.add(rec["patient_id"])

    # 3. Processing loop
    pbar = tqdm(targets, desc="Processing Patients", dynamic_ncols=True)
    for p_idx, p_item in enumerate(pbar):
        pid = p_item["patient_id"]
        p_split = cases_map[pid]["split"]
        p_label = cases_map[pid]["label"]
        p_label_name = cases_map[pid]["label_name"]

        # Check completed
        patient_done_file = out_root / "syn_stage2" / "ap" / pid / ".done"
        if patient_done_file.is_file() and not args.overwrite:
            tqdm.write(f">> [Skip] Patient {pid} already completed. Use --overwrite to re-generate.")
            continue

        h5_data = get_patient_h5_data(h5_root, pid)
        if h5_data is None:
            tqdm.write(f"[Skip] H5 data not found for {pid}")
            continue

        ncct_vol = h5_data.get("ncct")
        mask_vol = h5_data.get("mask")
        real_ap_vol = h5_data.get("ap")
        real_pvp_vol = h5_data.get("pvp")
        d_ncct = len(ncct_vol)

        selected_positions = p_item.get("selected_positions", [])
        if not selected_positions:
            continue

        # Prepare 5-slice condition tensors for each selected slice
        n_slices = len(selected_positions)
        cond_tensors = []
        center_tensors = []
        center_ncct_hu_list = []
        center_mask_list = []
        real_ap_slices = []
        real_pvp_slices = []
        ref_ids = []

        for pos in selected_positions:
            ref_id = int(pos["reference_slice_id"])
            ncct_id = int(pos["matched_ncct_slice_id"])
            ap_id = int(pos.get("ap_registered_slice_id", ref_id))
            pvp_id = int(pos.get("pvp_registered_slice_id", ref_id))

            # 5-slice neighborhood with edge clamping
            z_indices = [int(np.clip(ncct_id + off, 0, d_ncct - 1)) for off in (-2, -1, 0, 1, 2)]
            raw_slices_5 = ncct_vol[z_indices].astype(np.float32)
            slices_5 = apply_flip(raw_slices_5, args.flip)
            center_slice = slices_5[2:3].copy()

            cond_tensor = torch.from_numpy(slices_5)
            cond_norm = normalize_ncct_hu(cond_tensor, -200.0, 600.0)

            cond_tensors.append(cond_norm)
            center_tensors.append(torch.from_numpy(center_slice))
            center_ncct_hu_list.append(slices_5[2])
            if mask_vol is not None:
                center_mask_list.append(apply_flip(mask_vol[ncct_id], args.flip))
            else:
                center_mask_list.append(np.ones((512, 512), dtype=np.uint8))

            if real_ap_vol is not None and ap_id < len(real_ap_vol):
                real_ap_slices.append(apply_flip(real_ap_vol[ap_id], args.flip))
            else:
                real_ap_slices.append(None)

            if real_pvp_vol is not None and pvp_id < len(real_pvp_vol):
                real_pvp_slices.append(apply_flip(real_pvp_vol[pvp_id], args.flip))
            else:
                real_pvp_slices.append(None)

            ref_ids.append(ref_id)

        # Batch inference
        stage1_coarse_all = np.zeros((n_slices, 3, 512, 512), dtype=np.float32)
        stage2_refined_all = np.zeros((n_slices, 3, 512, 512), dtype=np.float32)

        for b_start in range(0, n_slices, args.batch_size):
            b_end = min(b_start + args.batch_size, n_slices)
            b_size = b_end - b_start

            b_cond = torch.stack(cond_tensors[b_start:b_end]).to(device)
            b_center = torch.stack(center_tensors[b_start:b_end]).to(device)

            # Stage 1: Coarse synthesis
            with torch.inference_mode():
                b_coarse = stage1_pipe(
                    b_cond,
                    ncct_center_hu=b_center,
                    num_inference_steps=50,
                    eta=0.0,
                    output_type="numpy",
                    noise=randn_tensor(
                        (b_size, 3, 512, 512), device=device, dtype=torch.float32,
                        generator=[torch.Generator(device=device).manual_seed(42 + b_start + i) for i in range(b_size)]
                    ),
                    **stage1_calib,
                )
            stage1_coarse_all[b_start:b_end] = b_coarse

            # Stage 2: Refinement
            b_norm = normalize_hu(b_coarse, stage2_refiner.hu_min, stage2_refiner.hu_max)
            with torch.inference_mode():
                s2_out = stage2_refiner.refine_slice(
                    x_synth=torch.from_numpy(b_norm).to(device),
                    t0=350,
                    low_frequency_weight=0.0,
                    fft_cutoff_rc={"AP": 36, "PVP": 32, "DP": 28},
                    butterworth_order=2,
                    num_inference_steps=100,
                    eta=0.0,
                    seed=42 ^ 0x5DEECE66D,
                    slice_start=b_start,
                )
            candidate = s2_out["refined"].float().cpu().numpy()
            b_unanchored = b_coarse + (candidate - b_norm) * ((stage2_refiner.hu_max - stage2_refiner.hu_min) / 2.0)

            # Step 1 Spectral Refinement
            b_ncct = np.stack(center_ncct_hu_list[b_start:b_end])
            b_mask = np.stack(center_mask_list[b_start:b_end])
            b_refined = apply_step1_spectral_liver_refinement(
                coarse_hu=b_coarse,
                ddpm_unanchored_hu=b_unanchored,
                ncct_hu=b_ncct,
                liver_mask=b_mask,
                spacing_xyz=(0.75, 0.75, 5.0),
                cfg=DEFAULT_STEP1_CONFIG,
            )
            stage2_refined_all[b_start:b_end] = b_refined

        # 4. Save PNG files & construct manifest records
        patient_records = []
        for s_idx in range(n_slices):
            ref_id = ref_ids[s_idx]

            # Paths
            paths = {
                "syn_stage1_ap": out_root / "syn_stage1" / "ap" / pid / f"ref_{ref_id:04d}.png",
                "syn_stage1_pvp": out_root / "syn_stage1" / "pvp" / pid / f"ref_{ref_id:04d}.png",
                "syn_stage2_ap": out_root / "syn_stage2" / "ap" / pid / f"ref_{ref_id:04d}.png",
                "syn_stage2_pvp": out_root / "syn_stage2" / "pvp" / pid / f"ref_{ref_id:04d}.png",
                "real_ncct": out_root / "real" / "ncct" / pid / f"ref_{ref_id:04d}.png",
                "real_ap": out_root / "real" / "ap" / pid / f"ref_{ref_id:04d}.png",
                "real_pvp": out_root / "real" / "pvp" / pid / f"ref_{ref_id:04d}.png",
            }
            for p in paths.values():
                p.parent.mkdir(parents=True, exist_ok=True)

            # Stage 1 (0: AP, 1: PVP, 2: DP)
            Image.fromarray(window_uint8(stage1_coarse_all[s_idx, 0])).save(paths["syn_stage1_ap"])
            Image.fromarray(window_uint8(stage1_coarse_all[s_idx, 1])).save(paths["syn_stage1_pvp"])

            # Stage 2 (0: AP, 1: PVP, 2: DP)
            Image.fromarray(window_uint8(stage2_refined_all[s_idx, 0])).save(paths["syn_stage2_ap"])
            Image.fromarray(window_uint8(stage2_refined_all[s_idx, 1])).save(paths["syn_stage2_pvp"])

            # Real ground truths
            Image.fromarray(window_uint8(center_ncct_hu_list[s_idx])).save(paths["real_ncct"])
            if real_ap_slices[s_idx] is not None:
                Image.fromarray(window_uint8(real_ap_slices[s_idx])).save(paths["real_ap"])
            if real_pvp_slices[s_idx] is not None:
                Image.fromarray(window_uint8(real_pvp_slices[s_idx])).save(paths["real_pvp"])

            patient_records.append({
                "patient_id": pid,
                "split": p_split,
                "label": p_label,
                "label_name": p_label_name,
                "reference_slice_id": ref_id,
                "matched_ncct_slice_id": int(selected_positions[s_idx]["matched_ncct_slice_id"]),
                "syn_stage1_ap": str(paths["syn_stage1_ap"].relative_to(out_root)),
                "syn_stage1_pvp": str(paths["syn_stage1_pvp"].relative_to(out_root)),
                "syn_stage2_ap": str(paths["syn_stage2_ap"].relative_to(out_root)),
                "syn_stage2_pvp": str(paths["syn_stage2_pvp"].relative_to(out_root)),
                "real_ap": str(paths["real_ap"].relative_to(out_root)) if real_ap_slices[s_idx] is not None else None,
                "real_pvp": str(paths["real_pvp"].relative_to(out_root)) if real_pvp_slices[s_idx] is not None else None,
                "real_ncct": str(paths["real_ncct"].relative_to(out_root)),
            })

        # Append to manifest
        with open(manifest_path, "a", encoding="utf-8") as mf:
            for rec in patient_records:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Mark done
        patient_done_file.touch()

    print("\n" + "=" * 70)
    print(">> All synthesis and PNG export successfully completed!")
    print(f">> Dataset root:     {out_root}")
    print(f">> Manifest output:  {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
