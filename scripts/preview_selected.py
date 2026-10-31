#!/usr/bin/env python3
"""Batch export Top-16 preview visualizations for cohort patients.

All files are completely flattened inside two English top-level folders:
<out_dir>/
├── montage_4x4/
│   ├── A0_1004_top16_montage.png
│   ├── A0_1024_top16_montage.png
│   └── ... (所有患者的 4x4 概览图完全平铺于此)
│
└── triplet_slices/
    ├── A0_1004_rank01_slice110.png
    ├── A0_1004_rank02_slice114.png
    ├── ...
    ├── A0_1004_rank16_slice080.png
    ├── A0_1024_rank01_slice055.png
    └── ... (所有患者的三期相并排图完全平铺于此，无子文件夹)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from cectcls.imaging import read_hu


def window_to_uint8(arr: np.ndarray, lo=-160.0, hi=240.0) -> np.ndarray:
    return np.rint((np.clip(arr, lo, hi) - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def process_patient(task: dict) -> dict:
    pid = task["patient_id"]
    label = task["label"]
    positions = task["positions"]
    volumes = task["volumes"]
    montage_dir = Path(task["montage_dir"])
    triplet_dir = Path(task["triplet_dir"])
    force = task.get("force", False)

    montage_path = montage_dir / f"{pid}_top16_montage.png"

    # Fast skip check if already fully generated
    if not force and montage_path.exists():
        # Check first and last ranked slice
        first_ref = positions[0]["reference_slice_id"]
        last_ref = positions[-1]["reference_slice_id"]
        f1 = triplet_dir / f"{pid}_rank01_slice{first_ref:03d}.png"
        f16 = triplet_dir / f"{pid}_rank{len(positions):02d}_slice{last_ref:03d}.png"
        if f1.exists() and f16.exists():
            return {"patient_id": pid, "status": "skipped"}

    ap_vol = volumes["ap"]
    pvp_vol = volumes["pvp"]
    ncct_vol = volumes.get("ncct")

    montage_tiles = []

    for pos in positions:
        rank = pos["rank"]
        ref = pos["reference_slice_id"]
        ap_id = pos.get("ap_registered_slice_id", ref)
        pvp_id = pos.get("pvp_registered_slice_id", ref)
        score = pos.get("attention_score", 0.0)
        z_mm = pos.get("reference_z_mm", 0.0)

        # 1. Load 2D HU slices
        ap_hu = read_hu(ap_vol, ap_id)
        pvp_hu = read_hu(pvp_vol, pvp_id)
        ap_img = Image.fromarray(window_to_uint8(ap_hu)).convert("RGB")
        pvp_img = Image.fromarray(window_to_uint8(pvp_hu)).convert("RGB")

        panels = []
        titles = []
        if ncct_vol:
            ncct_hu = read_hu(ncct_vol, ref)
            ncct_img = Image.fromarray(window_to_uint8(ncct_hu)).convert("RGB")
            panels = [ncct_img, ap_img, pvp_img]
            titles = ["NCCT (Plain)", "AP (Arterial)", "PVP (Venous)"]
        else:
            panels = [ap_img, pvp_img]
            titles = ["AP (Arterial)", "PVP (Venous)"]

        w, h = panels[0].size
        combined = Image.new("RGB", (w * len(panels), h + 45), (25, 25, 25))
        draw = ImageDraw.Draw(combined)

        for idx, (panel, title) in enumerate(zip(panels, titles)):
            combined.paste(panel, (idx * w, 45))
            draw.text((idx * w + 12, 12), title, fill=(240, 240, 240))

        header_text = f"Patient: {pid} | Rank {rank:02d} | Slice: {ref} | Attn Score: {score:.4f} | Z: {z_mm:.1f}mm"
        draw.text((w * len(panels) - 490, 12), header_text, fill=(255, 215, 0))

        # Flattened directly into triplet_dir
        out_triplet_file = triplet_dir / f"{pid}_rank{rank:02d}_slice{ref:03d}.png"
        combined.save(out_triplet_file)

        # 2. Thumbnail for montage (using AP phase)
        tile = ap_img.resize((220, 220), Image.Resampling.BILINEAR)
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.rectangle([0, 0, 220, 24], fill=(0, 0, 0))
        tile_draw.text((6, 4), f"#{rank} Slc:{ref} ({score:.3f})", fill=(255, 255, 0))
        montage_tiles.append(tile)

    # 3. Build 4x4 montage
    cols, rows = 4, 4
    montage = Image.new("RGB", (220 * cols, 220 * rows + 45), (35, 35, 35))
    m_draw = ImageDraw.Draw(montage)
    m_draw.text((12, 12), f"Patient: {pid} | Label: {label} | Top-16 AP Attention Matrix (Rank 1 -> 16)", fill=(255, 255, 255))
    for idx, tile in enumerate(montage_tiles[:16]):
        r = idx // cols
        c = idx % cols
        montage.paste(tile, (c * 220, 45 + r * 220))

    montage.save(montage_path)
    return {"patient_id": pid, "status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="Export 4x4 Montages and Triplet Slices into flattened English folders.")
    parser.add_argument("--out-dir", type=str, default="/dataset/downstream_classification/top16_inspection",
                        help="Root directory where montage_4x4 and triplet_slices will be created")
    parser.add_argument("--patient", type=str, default="",
                        help="Specific patient ID (leave blank to process all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of patients to process (0 = all 1,183 patients)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (default: 4)")
    parser.add_argument("--force", action="store_true",
                        help="Force regenerate even if already exists")
    parser.add_argument("--selection", type=str,
                        default="/dataset/downstream_classification/manifests/selected_bags.json",
                        help="Path to selected_bags.json")
    parser.add_argument("--manifest", type=str,
                        default="/dataset/downstream_classification/manifests/cases.jsonl",
                        help="Path to cases.jsonl")
    args = parser.parse_args()

    out_root = Path(args.out_dir).resolve()
    montage_dir = out_root / "montage_4x4"
    triplet_dir = out_root / "triplet_slices"
    montage_dir.mkdir(parents=True, exist_ok=True)
    triplet_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.selection).open("r", encoding="utf-8") as f:
        all_selections = json.load(f)

    cases_map = {}
    with Path(args.manifest).open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cases_map[c["patient_id"]] = c

    if args.patient:
        targets = [p for p in all_selections if p["patient_id"] == args.patient]
        if not targets:
            sys.exit(f"Patient {args.patient} not found in {args.selection}")
    else:
        targets = all_selections
        if args.limit > 0:
            targets = targets[:args.limit]

    print("=" * 65)
    print("Top-16 Slice Visual Inspection Export")
    print(f"  - Montage directory: {montage_dir}")
    print(f"  - Triplet directory: {triplet_dir}")
    print(f"  - Target patients:   {len(targets)}")
    print(f"  - Workers:           {args.workers}")
    print("=" * 65)

    tasks = []
    for p in targets:
        pid = p["patient_id"]
        case = cases_map.get(pid, {})
        tasks.append({
            "patient_id": pid,
            "label": case.get("label", ""),
            "positions": p["selected_positions"],
            "volumes": case.get("volumes", {}),
            "montage_dir": str(montage_dir),
            "triplet_dir": str(triplet_dir),
            "force": args.force,
        })

    t0 = time.time()
    ok_count, skip_count, err_count = 0, 0, 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_patient, task): task["patient_id"] for task in tasks}
        with tqdm(total=len(tasks), desc="Exporting Previews", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res["status"] == "ok":
                        ok_count += 1
                    elif res["status"] == "skipped":
                        skip_count += 1
                except Exception as exc:
                    err_count += 1
                    print(f"\n[Error] {futures[future]}: {exc}", file=sys.stderr)
                pbar.update(1)

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print(f"Export Completed in {elapsed:.1f}s")
    print(f"  - Successfully written: {ok_count} patients")
    print(f"  - Skipped (existing):   {skip_count} patients")
    print(f"  - Errors:               {err_count} patients")
    print(f"  - Montage (4x4):        {montage_dir}")
    print(f"  - Triplets (flattened): {triplet_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
