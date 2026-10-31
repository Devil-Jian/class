#!/usr/bin/env python3
"""Export matched NCCT / AP / PVP alignment visualization images with 2-worker HDD optimization.

Features:
- Optimized data loading: utilizes volume_array caching and vectorized uint8 windowing.
- Fast I/O: uses optimized PNG compression (compress_level=1) for 3x faster writing.
- Controlled concurrency: defaults to 2 workers to avoid HDD seek contention.
- Fully flattened English directory structure:
  <out_dir>/
  ├── montage_4x4/
  │   ├── A0_1004_matched_top16_montage.png
  │   └── ... (all patients flattened directly)
  ├── matched_triplets/
  │   ├── A0_1004_rank01_ref110_ncct110_dz+0.0mm.png
  │   └── ... (all triplets flattened directly)
  └── matched_qc_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
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


def process_matched_patient(task: dict) -> dict:
    pid = task["patient_id"]
    label = task["label"]
    split = task["split"]
    positions = task["positions"]
    volumes = task["volumes"]
    montage_dir = Path(task["montage_dir"])
    triplet_dir = Path(task["triplet_dir"])
    force = task.get("force", False)

    montage_file = montage_dir / f"{pid}_matched_top16_montage.png"

    # Fast skip check
    if not force and montage_file.exists():
        # Check first and last triplet
        p_first = positions[0]
        p_last = positions[-1]
        f_first = triplet_dir / f"{pid}_rank{p_first['rank']:02d}_ref{p_first['reference_slice_id']:03d}_ncct{p_first['matched_ncct_slice_id']:03d}_dz{p_first.get('matching_delta_z_mm', 0.0):+.1f}mm.png"
        f_last = triplet_dir / f"{pid}_rank{p_last['rank']:02d}_ref{p_last['reference_slice_id']:03d}_ncct{p_last['matched_ncct_slice_id']:03d}_dz{p_last.get('matching_delta_z_mm', 0.0):+.1f}mm.png"
        if f_first.exists() and f_last.exists():
            return {"patient_id": pid, "status": "skipped", "alerts": []}

    ap_vol = volumes.get("ap")
    pvp_vol = volumes.get("pvp")
    ncct_vol = volumes.get("ncct")

    montage_tiles = []
    alerts = []

    for pos in positions:
        rank = pos["rank"]
        ref = pos["reference_slice_id"]
        ncct_id = pos["matched_ncct_slice_id"]
        ap_id = pos.get("ap_registered_slice_id", ref)
        pvp_id = pos.get("pvp_registered_slice_id", ref)
        score = pos.get("matching_score", 0.0)
        dz = pos.get("matching_delta_z_mm", 0.0)
        status = pos.get("matching_status", "provisional")
        attn = pos.get("attention_score", 0.0)
        metrics = pos.get("matching_metrics", {})
        nmi_ap = metrics.get("nmi_ap", 0.0)
        ncc_ap = metrics.get("ncc_ap", 0.0)

        is_alert = (status == "review_required" or abs(dz) > 10.0 or score < 0.10)
        if is_alert:
            alerts.append({
                "patient_id": pid, "rank": rank, "ref_slice": ref,
                "matched_ncct_slice": ncct_id, "dz_mm": dz,
                "matching_score": score, "matching_status": status,
            })

        # 1. Read HU slices (utilizes volume_array cache internally)
        ap_hu = read_hu(ap_vol, ap_id)
        pvp_hu = read_hu(pvp_vol, pvp_id)
        ncct_hu = read_hu(ncct_vol, ncct_id)

        # 2. Vectorized windowing to uint8
        ap_img = Image.fromarray(window_to_uint8(ap_hu)).convert("RGB")
        pvp_img = Image.fromarray(window_to_uint8(pvp_hu)).convert("RGB")
        ncct_img = Image.fromarray(window_to_uint8(ncct_hu)).convert("RGB")

        # 3. Create triplet image: [NCCT | AP | PVP]
        panels = [ncct_img, ap_img, pvp_img]
        titles = [
            f"平扫 NCCT (Matched Slice: {ncct_id})",
            f"动脉期 AP (Ref Slice: {ref})",
            f"门脉期 PVP (Ref Slice: {ref})",
        ]

        w, h = panels[0].size
        header_h = 50
        combined = Image.new("RGB", (w * 3, h + header_h), (25, 25, 25))
        draw = ImageDraw.Draw(combined)

        for idx, (panel, title) in enumerate(zip(panels, titles)):
            combined.paste(panel, (idx * w, header_h))
            draw.text((idx * w + 14, 15), title, fill=(240, 240, 240))

        # Header text
        status_color = (255, 75, 75) if is_alert else (80, 235, 120)
        status_text = f"[{'ALERT: review_required' if is_alert else 'OK: provisional'}]"
        info_text = f"患者: {pid} ({label}) | Rank {rank:02d} | Δz: {dz:+.1f}mm | Match Score: {score:.3f} (NMI:{nmi_ap:.2f}, NCC:{ncc_ap:.2f}) | Attn: {attn:.3f}"
        
        draw.text((w * 3 - 750, 15), info_text, fill=(255, 220, 90))
        draw.text((w * 3 - 220, 15), status_text, fill=status_color)

        # Draw red border around image if alert
        if is_alert:
            draw.rectangle([0, header_h, w * 3 - 1, h + header_h - 1], outline=(255, 60, 60), width=4)

        triplet_file = triplet_dir / f"{pid}_rank{rank:02d}_ref{ref:03d}_ncct{ncct_id:03d}_dz{dz:+.1f}mm.png"
        combined.save(triplet_file, compress_level=1)

        # 4. Thumbnail for 4x4 montage (showing AP with matching tags)
        tile = ap_img.resize((220, 220), Image.Resampling.BILINEAR)
        t_draw = ImageDraw.Draw(tile)
        t_draw.rectangle([0, 0, 220, 26], fill=(0, 0, 0))
        t_color = (255, 90, 90) if is_alert else (255, 220, 80)
        t_draw.text((6, 4), f"#{rank} Ref:{ref}→NC:{ncct_id} (Δz:{dz:+.1f})", fill=t_color)
        if is_alert:
            t_draw.rectangle([0, 0, 219, 219], outline=(255, 60, 60), width=3)
        montage_tiles.append(tile)

    # 5. Build 4x4 montage
    cols, rows = 4, 4
    m_header_h = 50
    montage = Image.new("RGB", (220 * cols, 220 * rows + m_header_h), (35, 35, 35))
    m_draw = ImageDraw.Draw(montage)
    title_str = f"患者: {pid} | 类别: {label} | 匹配 Top-16 对照矩阵 (Rank 1→16 从左到右) | 异常层: {len(alerts)}处"
    m_draw.text((15, 15), title_str, fill=(255, 255, 255))
    for idx, tile in enumerate(montage_tiles[:16]):
        r = idx // cols
        c = idx % cols
        montage.paste(tile, (c * 220, m_header_h + r * 220))

    montage.save(montage_file, compress_level=1)
    return {"patient_id": pid, "status": "ok", "alerts": alerts}


def main():
    parser = argparse.ArgumentParser(description="Export matched alignment inspection images with 2 workers.")
    parser.add_argument("--out-dir", type=str, default="/dataset/downstream_classification/matched_inspection",
                        help="Root output directory (montage_4x4 and matched_triplets will be created inside)")
    parser.add_argument("--matched", type=str,
                        default="/dataset/downstream_classification/manifests/matched_bags.json",
                        help="Path to matched_bags.json")
    parser.add_argument("--manifest", type=str,
                        default="/dataset/downstream_classification/manifests/cases.jsonl",
                        help="Path to cases.jsonl")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent processes (default: 2, optimized for HDD)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of patients to process (0 = all 1,183)")
    parser.add_argument("--patient", type=str, default="",
                        help="Process only a specific patient ID")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite existing images")
    args = parser.parse_args()

    out_root = Path(args.out_dir).resolve()
    montage_dir = out_root / "montage_4x4"
    triplet_dir = out_root / "matched_triplets"
    montage_dir.mkdir(parents=True, exist_ok=True)
    triplet_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.matched).open("r", encoding="utf-8") as f:
        all_matched = json.load(f)

    cases_map = {}
    with Path(args.manifest).open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cases_map[c["patient_id"]] = c

    if args.patient:
        targets = [p for p in all_matched if p["patient_id"] == args.patient]
        if not targets:
            sys.exit(f"Patient {args.patient} not found in {args.matched}")
    else:
        targets = all_matched
        if args.limit > 0:
            targets = targets[:args.limit]

    print("=" * 70)
    print("NCCT <-> CECT 匹配对齐可视化核对导出 (HDD 2-Worker 优化版)")
    print(f"  - 全景概览图路径: {montage_dir}")
    print(f"  - 三期并排图路径: {triplet_dir}")
    print(f"  - 目标患者总数:   {len(targets)}")
    print(f"  - 并发进程数:     {args.workers} (HDD 保护模式)")
    print("=" * 70)

    tasks = []
    for p in targets:
        pid = p["patient_id"]
        case = cases_map.get(pid, {})
        tasks.append({
            "patient_id": pid,
            "label": case.get("label", ""),
            "split": p.get("split", ""),
            "positions": p["selected_positions"],
            "volumes": case.get("volumes", {}),
            "montage_dir": str(montage_dir),
            "triplet_dir": str(triplet_dir),
            "force": args.force,
        })

    t0 = time.time()
    ok_count, skip_count, err_count = 0, 0, 0
    total_alerts = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_matched_patient, task): task["patient_id"] for task in tasks}
        with tqdm(total=len(tasks), desc="Exporting Matched Previews", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res["status"] == "ok":
                        ok_count += 1
                        total_alerts.extend(res.get("alerts", []))
                    elif res["status"] == "skipped":
                        skip_count += 1
                except Exception as exc:
                    err_count += 1
                    print(f"\n[Error] {futures[future]}: {exc}", file=sys.stderr)
                pbar.update(1)

    # Save QC summary report
    report_file = out_root / "matched_qc_summary.json"
    summary_report = {
        "total_patients_checked": len(targets),
        "patients_exported": ok_count,
        "patients_skipped": skip_count,
        "errors": err_count,
        "total_slices_evaluated": len(targets) * 16,
        "total_review_required_alerts": len(total_alerts),
        "alert_slices": total_alerts,
        "montage_dir": str(montage_dir),
        "triplet_dir": str(triplet_dir),
    }
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"匹配对比图导出完成! 总耗时: {elapsed:.1f}s")
    print(f"  - 成功写入: {ok_count} 位患者")
    print(f"  - 已存在跳过: {skip_count} 位患者")
    print(f"  - 失败异常: {err_count} 位患者")
    print(f"  - 质检警示切片数 (review_required): {len(total_alerts)} / {len(targets) * 16}")
    print(f"  - 质检详细报告: {report_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
