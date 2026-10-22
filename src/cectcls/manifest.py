"""Build audited patient-level, NCCT-grid-aligned paired AP/PVP slice manifests."""
from __future__ import annotations

import csv
import json
from pathlib import Path

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


def prepare(cases_csv, splits_csv, out, label_map_path=None):
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
        fixed = _image(volumes["ncct"])
        for phase in ("ap", "pvp"):
            _assert_registered(fixed, _image(volumes[phase]), phase, pid)
        if mask_path:
            mask = _image(mask_path)
            _assert_registered(fixed, mask, "liver_mask", pid)
            valid = np.flatnonzero(
                np.any(np.asarray(mask.dataobj) > 0, axis=(0, 1))
            ).astype(int).tolist()
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
    if set(splits) != seen:
        raise ValueError(f"Splits contain patients absent from cases: {sorted(set(splits)-seen)[:10]}")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "cases.jsonl", records)
    write_json(out / "labels.json", labels)
    write_json(out / "qc.json", {
        "patients": len(records),
        "by_split": {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")},
        "label_map": labels,
        "note": "AP/PVP/mask verified on identical canonical NCCT grid; no affine-only registration assumed",
    })
    return records
