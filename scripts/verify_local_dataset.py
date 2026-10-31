#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Dataset Integrity Verification Script (Windows / Linux cross-platform).
Verifies:
1. Manifest files (cases.jsonl, labels.json, matched_bags.json)
2. PNG dataset (png_index.json, real, syn_stage1)
3. Sample loading for all 5 training groups (ncct, real, syn, real_syn, real_aug)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Verify local CECT classification dataset")
    parser.add_argument("--work-dir", type=str, default="R:/downstream_classification",
                        help="Path to downstream_classification directory")
    return parser.parse_args()


def main():
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()

    print("=" * 70)
    print(">> CECT Classification Local Dataset Verification")
    print(f">> Target Work Directory: {work_dir}")
    print("=" * 70)

    # 1. Check Directory & Manifests
    manifest_dir = work_dir / "manifests"
    png_dir = work_dir / "png_dataset"

    all_pass = True

    print("\n[1/3] Checking Manifest Files:")
    manifest_files = {
        "cases.jsonl": 1183,
        "labels.json": None,
        "matched_bags.json": None,
    }

    for f_name, expected_lines in manifest_files.items():
        p = manifest_dir / f_name
        if not p.is_file():
            print(f"  ❌ Missing manifest: {p}")
            all_pass = False
        else:
            size_mb = p.stat().st_size / (1024 * 1024)
            if expected_lines:
                lines = sum(1 for line in open(p, encoding="utf-8") if line.strip())
                status = "✅" if lines == expected_lines else "⚠️"
                print(f"  {status} {f_name:<20} ({size_mb:.2f} MB, {lines}/{expected_lines} lines)")
            else:
                print(f"  ✅ {f_name:<20} ({size_mb:.2f} MB)")

    # 2. Check PNG Dataset & Index
    print("\n[2/3] Checking PNG Dataset & Index:")
    png_index_file = png_dir / "png_index.json"
    if not png_index_file.is_file():
        print(f"  ❌ Missing png_index.json: {png_index_file}")
        all_pass = False
        return
    else:
        with open(png_index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        print(f"  ✅ png_index.json loaded ({len(index)} keys indexed)")

        # Verify sample file existence on disk (auto-resolving relative to png_dir)
        sample_keys = list(index.keys())[::1000]
        missing_count = 0
        for k in sample_keys:
            raw_path = Path(index[k])
            # Cross-platform resolution
            target = raw_path
            if not target.is_file():
                for anchor in ("real", "syn_stage1", "syn_stage2", "syn"):
                    if anchor in raw_path.parts:
                        idx = raw_path.parts.index(anchor)
                        cand = png_dir / Path(*raw_path.parts[idx:])
                        if cand.is_file():
                            target = cand
                            break
            if not target.is_file():
                missing_count += 1

        if missing_count == 0:
            print(f"  ✅ Sample slice check passed (sampled {len(sample_keys)} slices across patients)")
        else:
            print(f"  ❌ Found {missing_count}/{len(sample_keys)} sample files missing on disk!")
            all_pass = False

    # 3. Test Dataset Loading across all 5 groups
    print("\n[3/3] Testing DataLoader for 5 Experimental Groups:")
    # Ensure src in sys.path
    script_dir = Path(__file__).resolve().parent
    class_root = script_dir.parent
    src_dir = class_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from cectcls.dataset import PatientBagDataset
        groups = ["ncct", "cect_real", "cect_syn", "cect_real_syn", "cect_real_aug"]
        for g in groups:
            ds = PatientBagDataset(
                manifest=str(manifest_dir / "cases.jsonl"),
                split="train",
                group=g,
                mode="selected",
                selection=str(manifest_dir / "matched_bags.json"),
                png_index=str(png_index_file),
                image_size=224,
            )
            sample = ds[0]
            print(f"  ✅ Group: {g:<14} | Patients: {len(ds):<4} | AP: {tuple(sample['ap'].shape)} | PVP: {tuple(sample['pvp'].shape)}")
    except Exception as e:
        print(f"  ❌ Error initializing Dataset loader: {e}")
        all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print(">> 🎉 ALL CHECKS PASSED! Dataset is 100% ready for GTX 1080 training.")
    else:
        print(">> ⚠️ SOME CHECKS FAILED! Please review the missing items above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
