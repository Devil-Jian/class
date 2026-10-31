#!/usr/bin/env python3
"""Convert 3D NIfTI CT scans to per-patient HDF5 (.h5) cache files.

Features:
- Stores 3D arrays in (Z, H, W) layout: slice indexing `h5['ap'][i]` is instant and contiguous.
- Chunks by (1, H, W) so individual 2D slice reads take sub-milliseconds with zero gzip overhead.
- Multi-processing concurrent conversion with resume capability (skips already converted files).
- Preserves exact canonical orientation and voxel values identical to cectcls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


def _load_canonical_zhw(path: str, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Load NIfTI in closest canonical orientation and return (Z, H, W) array and affine."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    # nibabel data is (X, Y, Z). Transpose to (Z, Y, X) = (Z, H, W)
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=dtype)
    zhw = np.ascontiguousarray(np.transpose(data, (2, 1, 0)))
    return zhw, img.affine


def convert_single_patient(task: dict) -> dict:
    patient_id = task["patient_id"]
    out_file = Path(task["out_file"])
    dtype_str = task["dtype"]
    target_dtype = np.int16 if dtype_str == "int16" else np.float32
    compress = task.get("compress", None)  # None, "gzip", or "lzf"

    if out_file.exists() and out_file.stat().st_size > 1024 * 1024:
        return {"patient_id": patient_id, "status": "skipped", "msg": "Already exists"}

    tmp_out = out_file.with_suffix(".tmp.h5")
    try:
        # Load all 4 volumes
        ap_arr, affine = _load_canonical_zhw(task["ap_path"], dtype=target_dtype)
        pvp_arr, _ = _load_canonical_zhw(task["pvp_path"], dtype=target_dtype)
        ncct_arr, _ = _load_canonical_zhw(task["ncct_path"], dtype=target_dtype)
        
        # Mask is stored as uint8
        mask_arr, _ = _load_canonical_zhw(task["mask_path"], dtype=np.uint8)

        z_len, h, w = ap_arr.shape
        chunk_shape = (1, h, w)

        with h5py.File(tmp_out, "w") as h5:
            # Datasets
            kw = {"chunks": chunk_shape}
            if compress:
                kw["compression"] = compress

            h5.create_dataset("ap", data=ap_arr, dtype=target_dtype, **kw)
            h5.create_dataset("pvp", data=pvp_arr, dtype=target_dtype, **kw)
            h5.create_dataset("ncct", data=ncct_arr, dtype=target_dtype, **kw)
            h5.create_dataset("mask", data=(mask_arr > 0).astype(np.uint8), dtype=np.uint8, **kw)

            # Metadata attributes
            h5.attrs["patient_id"] = patient_id
            h5.attrs["label"] = task.get("label", "")
            h5.attrs["z_slices"] = z_len
            h5.attrs["height"] = h
            h5.attrs["width"] = w
            h5.attrs["affine"] = affine

        tmp_out.replace(out_file)
        return {"patient_id": patient_id, "status": "ok", "z_slices": z_len}
    except Exception as exc:
        if tmp_out.exists():
            tmp_out.unlink()
        return {"patient_id": patient_id, "status": "error", "msg": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Convert NIfTI dataset to per-patient HDF5 files.")
    parser.add_argument("--out-dir", required=True, type=str,
                        help="Destination directory for .h5 files (e.g. /dataset/h5_cache)")
    parser.add_argument("--cases", type=str, default="/dataset/downstream_classification/cohort/cases.csv",
                        help="Path to cases.csv or source CSV list")
    parser.add_argument("--image-root", type=str, default="/win_cect/shared/gx_all_registered",
                        help="Root directory of registered NIfTI CT scans")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--dtype", choices=["float32", "int16"], default="float32",
                        help="Data type for CT HU voxel values: float32 (default) or int16 (saves 50%% disk)")
    parser.add_argument("--compress", choices=["none", "lzf", "gzip"], default="none",
                        help="Compression filter for HDF5: none (fastest I/O, default), lzf, or gzip")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of patients to process (0 = all)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_path = Path(args.cases)
    if not cases_path.exists():
        fallback = Path("/workspace/class/input/classification_main5_1183_including_REJECT.csv")
        if fallback.exists():
            cases_path = fallback
        else:
            sys.exit(f"Cases CSV not found: {args.cases}")

    print(f"Reading cohort from: {cases_path}")
    print(f"Output directory:    {out_dir}")
    print(f"Workers:             {args.workers}")
    print(f"Data type:           {args.dtype}")
    print(f"Compression:         {args.compress}")

    tasks = []
    with cases_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"].strip()
            # Paths resolution
            image_root = Path(args.image_root)
            ap = row.get("ap_path") or str(image_root / pid / "ap_registered.nii.gz")
            pvp = row.get("pvp_path") or str(image_root / pid / "pvp_registered.nii.gz")
            ncct = row.get("ncct_path") or str(image_root / pid / "ncct_zcrop.nii.gz")
            mask = (row.get("liver_mask_path") or row.get("liver_mask_path_candidate") or
                    str(image_root / pid / "ncct_mask_zcrop.nii.gz"))

            tasks.append({
                "patient_id": pid,
                "label": row.get("label") or row.get("主类别") or "",
                "ap_path": ap,
                "pvp_path": pvp,
                "ncct_path": ncct,
                "mask_path": mask,
                "out_file": str(out_dir / f"{pid}.h5"),
                "dtype": args.dtype,
                "compress": None if args.compress == "none" else args.compress,
            })

    if args.limit > 0:
        tasks = tasks[:args.limit]

    total = len(tasks)
    print(f"Total patients to convert: {total}")

    t0 = time.time()
    ok_count, skip_count, err_count = 0, 0, 0
    errors = []

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="Converting to H5")
    except ImportError:
        pbar = None

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_single_patient, task): task["patient_id"] for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            status = res["status"]
            if status == "ok":
                ok_count += 1
            elif status == "skipped":
                skip_count += 1
            else:
                err_count += 1
                errors.append(res)
                print(f"\n[ERROR] {res['patient_id']}: {res['msg']}", file=sys.stderr)

            if pbar:
                pbar.update(1)
            else:
                done = ok_count + skip_count + err_count
                if done % 20 == 0 or done == total:
                    pct = done / total * 100
                    elapsed = time.time() - t0
                    speed = done / max(1.0, elapsed)
                    print(f"[{done}/{total} ({pct:.1f}%)] Converted: {ok_count}, Skipped: {skip_count}, Errors: {err_count} ({speed:.2f} patients/s)")

    if pbar:
        pbar.close()

    elapsed = time.time() - t0
    print("\n" + "=" * 50)
    print(f"H5 Conversion Completed in {elapsed:.1f}s")
    print(f"  - Successfully written: {ok_count}")
    print(f"  - Skipped (already exist): {skip_count}")
    print(f"  - Failed with errors:   {err_count}")
    print(f"  - Destination:          {out_dir}")
    print("=" * 50)

    if errors:
        err_file = out_dir / "conversion_errors.json"
        with err_file.open("w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(errors)} error details to: {err_file}")


if __name__ == "__main__":
    main()
