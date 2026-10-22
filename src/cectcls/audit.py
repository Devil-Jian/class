"""Audit every locked patient image WITHOUT deleting or altering the cohort."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from .manifest import write_json


def audit_images(cases_csv, out, strict_mask=False):
    cases_csv = Path(cases_csv)
    with cases_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        cases = list(csv.DictReader(handle))
    results, stats = [], Counter()
    seen = set()
    for case in cases:
        pid = case["patient_id"].strip()
        if pid in seen:
            raise ValueError(f"Duplicate case {pid}")
        seen.add(pid)
        notes, shapes, affines = [], {}, {}
        for phase, col in (("ncct", "ncct_path"), ("ap", "ap_path"),
                           ("pvp", "pvp_path"), ("liver_mask", "liver_mask_path")):
            val = (case.get(col) or "").strip()
            if not val and phase == "liver_mask" and not strict_mask:
                notes.append("mask_path_missing")
                continue
            if not val or not Path(val).is_file():
                notes.append(f"{phase}_missing")
                continue
            try:
                im = nib.as_closest_canonical(nib.load(val))
                if len(im.shape) != 3:
                    notes.append(f"{phase}_not_3d")
                    continue
                shapes[phase] = [int(x) for x in im.shape]
                affines[phase] = np.asarray(im.affine)
            except Exception as e:
                notes.append(f"{phase}_unreadable:{type(e).__name__}")
        if "ncct" in shapes:
            for phase in ("ap", "pvp", "liver_mask"):
                if phase in shapes and (
                    shapes[phase] != shapes["ncct"] or
                    not np.allclose(affines[phase], affines["ncct"], rtol=0, atol=1e-3)
                ):
                    notes.append(f"{phase}_grid_mismatch")
        ready = not any(
            x.endswith("_missing") or "_grid_mismatch" in x or "_not_3d" in x
            or "_unreadable" in x or x == "mask_path_missing" and strict_mask
            for x in notes
        )
        if ready:
            stats["ready"] += 1
        else:
            stats["needs_review"] += 1
        for note in notes:
            stats[note] += 1
        results.append({
            "patient_id": pid, "label": case["label"],
            "nifti_ready": ready, "issues": ";".join(notes),
            "ncct_shape": str(shapes.get("ncct", "")),
            "ap_shape": str(shapes.get("ap", "")),
            "pvp_shape": str(shapes.get("pvp", "")),
            "mask_shape": str(shapes.get("liver_mask", "")),
        })
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["patient_id", "label", "nifti_ready", "issues",
              "ncct_shape", "ap_shape", "pvp_shape", "mask_shape"]
    with out.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "locked_patients_checked": len(cases), "ready": stats["ready"],
        "needs_review": stats["needs_review"],
        "issue_counts": dict(stats),
        "out_csv": str(out),
        "note": "This is an image path/header audit only. REJECT remains eligible for CECT classification; "
                "anatomical registration and liver-mask semantics require separate image QC.",
    }
    write_json(out.with_suffix(".json"), summary)
    return summary
