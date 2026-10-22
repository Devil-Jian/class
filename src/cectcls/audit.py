"""Fast, non-destructive NIfTI header audit for the exact locked cohort.

Use NIfTI headers ONLY. In particular, do not call as_closest_canonical on
image voxel data during audit: axis reorientation can decompress .nii.gz.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from .manifest import write_json

PHASES = (
    ("ncct", "ncct_path"), ("ap", "ap_path"),
    ("pvp", "pvp_path"), ("liver_mask", "liver_mask_path"),
)
FIELDS = (
    "patient_id", "label", "nifti_ready", "issues", "ncct_shape",
    "ap_shape", "pvp_shape", "mask_shape",
)


def _header(path):
    """Return canonical shape/affine from metadata without touching voxel array."""
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError("not_3d")
    orient = nib.orientations.io_orientation(image.affine)
    ras = np.array([[0., 1.], [1., 1.], [2., 1.]])
    if not np.all(np.isfinite(orient)):
        raise ValueError("bad_orientation")
    transform = nib.orientations.ornt_transform(orient, ras)
    shape = [int(image.shape[int(axis)]) for axis in np.argsort(transform[:, 0])]
    affine = image.affine @ nib.orientations.inv_ornt_aff(transform, image.shape)
    if not np.isfinite(affine).all():
        raise ValueError("bad_affine")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "shape": shape,
        "affine": np.asarray(affine, dtype=float).tolist(),
    }


def _inspect_one(case, strict_mask, prior_headers=None):
    pid = case["patient_id"].strip()
    notes, headers, reused = [], {}, 0
    prior_headers = prior_headers or {}
    for phase, col in PHASES:
        val = (case.get(col) or "").strip()
        if not val:
            if phase == "liver_mask" and not strict_mask:
                notes.append("mask_path_missing")
                continue
            notes.append(f"{phase}_missing")
            continue
        path = Path(val)
        if not path.is_file():
            notes.append(f"{phase}_missing")
            continue
        try:
            old = prior_headers.get(phase)
            if old:
                st = path.stat()
                if (old.get("path") == str(path.resolve())
                        and old.get("size") == st.st_size
                        and old.get("mtime_ns") == st.st_mtime_ns):
                    headers[phase] = old
                    reused += 1
                    continue
            headers[phase] = _header(path)
        except Exception as e:
            kind = "not_3d" if str(e) == "not_3d" else f"unreadable:{type(e).__name__}"
            notes.append(f"{phase}_{kind}")
    if "ncct" in headers:
        fixed = headers["ncct"]
        for phase in ("ap", "pvp", "liver_mask"):
            moving = headers.get(phase)
            if moving and (
                moving["shape"] != fixed["shape"] or
                not np.allclose(moving["affine"], fixed["affine"], atol=1e-3, rtol=0)
            ):
                notes.append(f"{phase}_grid_mismatch")
    critical = (
        "_missing", "_grid_mismatch", "_not_3d", "_unreadable", "_bad_orientation",
        "_bad_affine",
    )
    ready = not any(any(s in x for s in critical) for x in notes
                    if strict_mask or x != "mask_path_missing")
    record = {
        "patient_id": pid, "label": case["label"],
        "nifti_ready": ready, "issues": ";".join(notes),
        "ncct_shape": str(headers.get("ncct", {}).get("shape", "")),
        "ap_shape": str(headers.get("ap", {}).get("shape", "")),
        "pvp_shape": str(headers.get("pvp", {}).get("shape", "")),
        "mask_shape": str(headers.get("liver_mask", {}).get("shape", "")),
    }
    return record, headers, notes, reused


def audit_images(cases_csv, out, strict_mask=False, workers=2):
    """Scan all 1298; threads are deliberately limited for rotating HDD storage."""
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8; prefer 1-2 on HDD")
    started = time.monotonic()
    cases_csv = Path(cases_csv)
    with cases_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    pids = [case["patient_id"].strip() for case in cases]
    if not all(pids) or len(pids) != len(set(pids)):
        raise ValueError("Duplicate/empty patient ID in locked cases CSV")
    cache_file = Path(out).with_name("image_header_cache.json")
    prior = {}
    checksum = hashlib.sha256(cases_csv.read_bytes()).hexdigest()
    if cache_file.is_file():
        try:
            previous = json.loads(cache_file.read_text(encoding="utf-8"))
            if previous.get("schema") == 1 and previous.get("cases_csv_sha256") == checksum:
                prior = previous.get("patients", {})
        except (ValueError, OSError):
            prior = {}
    def inspect(case):
        return _inspect_one(case, strict_mask, prior.get(case["patient_id"].strip()))
    if workers == 1:
        inspected = [inspect(case) for case in cases]
    else:
        # ThreadPoolExecutor.map retains original CSV order, so QC is deterministic.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            inspected = list(pool.map(inspect, cases))
    stats = Counter()
    rows, headers = [], {}
    for row, patient_headers, notes, reused in inspected:
        stats["headers_reused"] += reused
        rows.append(row)
        headers[row["patient_id"]] = patient_headers
        stats["ready" if row["nifti_ready"] else "needs_review"] += 1
        for note in notes:
            stats[note] += 1

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(cache_file, {
        "schema": 1,
        "cases_csv_sha256": checksum,
        "patients": headers,
    })
    summary = {
        "locked_patients_checked": len(cases),
        "ready": stats["ready"], "needs_review": stats["needs_review"],
        "issue_counts": dict(stats),
        "out_csv": str(out), "header_cache": str(cache_file),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "workers": workers, "voxel_data_read": False,
        "headers_reused": stats["headers_reused"],
        "note": "Metadata-only shape/affine test, not registration accuracy or voxel-integrity QC. "
                "REJECT remains in cohort. Mask content and semantics are NOT checked.",
    }
    write_json(out.with_suffix(".json"), summary)
    return summary
