"""Build audited patient-level, NCCT-grid-aligned paired AP/PVP slice manifests."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _resolve(row, name, base, required=True):
    raw = (row.get(name + "_path") or row.get(name) or "").strip()
    if not raw:
        if required:
            raise ValueError(f"Missing {name}_path for patient {row.get('patient_id')}")
        return None
    p = Path(raw)
    p = p if p.is_absolute() else base / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return str(p)


def _image(path):
    image = nib.as_closest_canonical(nib.load(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI: {path}, shape={image.shape}")
    return image


def _assert_registered(fixed, moving, name, patient_id):
    if fixed.shape != moving.shape or not np.allclose(
        fixed.affine, moving.affine, rtol=0, atol=1e-3
    ):
        raise ValueError(
            f"{patient_id} {name} is not resampled onto the NCCT fixed grid; "
            "resample using the saved registration transform before prepare"
        )


def _cached_header(meta, path, pid, phase):
    """Revalidate file identity against the cheap header scan's cached stat."""
    p = Path(path).resolve()
    stat = p.stat()
    if (str(p) != meta["path"] or stat.st_size != meta["size"]
            or stat.st_mtime_ns != meta["mtime_ns"]):
        raise ValueError(
            f"{pid} {phase} source NIfTI changed after header audit; rerun prepare from audit"
        )
    return SimpleNamespace(shape=tuple(meta["shape"]),
                           affine=np.asarray(meta["affine"], dtype=float))


def _mask_signature(mask_path):
    p = Path(mask_path).resolve()
    stat = p.stat()
    return {"path": str(p), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _save_mask_cache(path, cache):
    """Write atomically every 25 patients, avoiding partial state on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(cache, stream, ensure_ascii=False)
    os.replace(temporary, path)


def prepare(cases_csv, splits_csv, out, label_map_path=None, header_cache_path=None,
            reuse_mask_cache=True):
    """Fail closed for split, shape, affine, ID, and label inconsistencies."""
    cases_csv, splits_csv, out = Path(cases_csv), Path(splits_csv), Path(out)
    with splits_csv.open(newline="", encoding="utf-8-sig") as f:
        split_rows = list(csv.DictReader(f))
    splits = {}
    for r in split_rows:
        pid, split = r["patient_id"].strip(), r["split"].strip().lower()
        split = {"valid": "val", "validation": "val"}.get(split, split)
        if pid in splits or split not in {"train", "val", "test"}:
            raise ValueError(f"Duplicate/invalid split: {r}")
        splits[pid] = split
    with cases_csv.open(newline="", encoding="utf-8-sig") as f:
        case_rows = list(csv.DictReader(f))
    if not case_rows:
        raise ValueError("Empty cases CSV")
    if label_map_path is not None:
        labels = read_json(label_map_path)
        if set(r["label"].strip() for r in case_rows) - set(labels):
            raise ValueError("Cases contain classes absent from fixed label_map")
        if sorted(labels.values()) != list(range(len(labels))):
            raise ValueError("Fixed label_map indices must be contiguous from zero")
    else:
        labels = {name: i for i, name in enumerate(sorted({r["label"].strip() for r in case_rows}))}
    started = time.monotonic()
    headers = None
    if header_cache_path:
        saved_headers = read_json(header_cache_path)
        sha = hashlib.sha256(cases_csv.read_bytes()).hexdigest()
        if saved_headers.get("schema") != 1 or saved_headers.get("cases_csv_sha256") != sha:
            raise ValueError("Cases changed after header QC: rerun audit-images first")
        headers = saved_headers["patients"]
        if set(headers) != {r["patient_id"].strip() for r in case_rows}:
            raise ValueError("Header QC patient list differs from cases")
    out.mkdir(parents=True, exist_ok=True)
    mask_cache_file = out / "mask_slice_cache.json"
    saved_mask = (read_json(mask_cache_file) if reuse_mask_cache and
                  mask_cache_file.is_file() else {})
    mask_changed, cache_hits, mask_decodes = 0, 0, 0
    records = []
    seen = set()
    for r in case_rows:
        pid = r["patient_id"].strip()
        if pid in seen or pid not in splits:
            raise ValueError(f"Duplicate patient or missing split: {pid}")
        seen.add(pid)
        base = cases_csv.parent
        volumes = {phase: _resolve(r, phase, base) for phase in ("ncct", "ap", "pvp")}
        mask_path = _resolve(r, "liver_mask", base, required=False)
        if headers is not None:
            patient_headers = headers[pid]
            for phase in ("ncct", "ap", "pvp"):
                if phase not in patient_headers:
                    raise ValueError(f"{pid} {phase} missing from audited header cache")
            fixed = _cached_header(patient_headers["ncct"], volumes["ncct"], pid, "ncct")
            for phase in ("ap", "pvp"):
                meta = _cached_header(patient_headers[phase], volumes[phase], pid, phase)
                _assert_registered(fixed, meta, phase, pid)
            if mask_path:
                if "liver_mask" not in patient_headers:
                    raise ValueError(f"{pid} mask missing from audited header cache")
                meta = _cached_header(patient_headers["liver_mask"], mask_path,
                                      pid, "liver_mask")
                _assert_registered(fixed, meta, "liver_mask", pid)
        else:
            fixed = _image(volumes["ncct"])
            for phase in ("ap", "pvp"):
                _assert_registered(fixed, _image(volumes[phase]), phase, pid)
        if mask_path:
            signature = _mask_signature(mask_path)
            old = saved_mask.get(pid, {})
            if (reuse_mask_cache and all(old.get(k) == v for k, v in signature.items())
                    and isinstance(old.get("valid_ids"), list)
                    and old.get("shape") == list(fixed.shape)):
                valid = old["valid_ids"]
                cache_hits += 1
            else:
                mask = _image(mask_path)
                _assert_registered(fixed, mask, "liver_mask", pid)
                valid = np.flatnonzero(
                    np.any(np.asarray(mask.dataobj) > 0, axis=(0, 1))
                ).astype(int).tolist()
                mask_decodes += 1
                if reuse_mask_cache:
                    saved_mask[pid] = {**signature, "shape": list(fixed.shape),
                                       "valid_ids": valid}
                    mask_changed += 1
                    if mask_changed % 25 == 0:
                        _save_mask_cache(mask_cache_file, saved_mask)
        else:
            valid = list(range(fixed.shape[2]))
        if not valid:
            raise ValueError(f"No valid slices for {pid}")
        center = ((fixed.shape[0] - 1) / 2.0, (fixed.shape[1] - 1) / 2.0)
        pos = {
            str(z): float((fixed.affine @ np.array([*center, z, 1.0]))[2])
            for z in valid
        }
        records.append({
            "patient_id": pid, "split": splits[pid],
            "label": labels[r["label"].strip()], "label_name": r["label"].strip(),
            "volumes": volumes, "liver_mask": mask_path,
            "valid_ids": valid, "reference_z_mm": pos,
            "shape": list(fixed.shape),
        })
        if len(records) % 100 == 0 or len(records) == len(case_rows):
            print(f"[prepare] {len(records)}/{len(case_rows)} patients; "
                  f"mask decoded={mask_decodes}, cache reused={cache_hits}; "
                  f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    if reuse_mask_cache and mask_changed:
        _save_mask_cache(mask_cache_file, saved_mask)
    if set(splits) != seen:
        raise ValueError(f"Splits contain patients absent from cases: {sorted(set(splits)-seen)[:10]}")
    write_jsonl(out / "cases.jsonl", records)
    write_json(out / "labels.json", labels)
    write_json(out / "qc.json", {
        "patients": len(records),
        "by_split": {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")},
        "label_map": labels,
        "header_cache_used": headers is not None,
        "mask_slices_reused": cache_hits,
        "masks_decompressed": mask_decodes,
        "elapsed_seconds": round(time.monotonic()-started, 2),
        "note": "AP/PVP/mask header grids match canonical NCCT; header check does NOT "
                "establish anatomical alignment or voxel integrity. Mask contents used "
                "only to identify nonzero z slices.",
    })
    return records
