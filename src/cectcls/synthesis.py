"""External generator boundary: emit requests, then validate HU-valued synth outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .manifest import read_json, read_jsonl, write_json


def export_requests(matched, output_root, out, manifest=None, split="train"):
    patients = read_json(matched)
    patient_cases = {row["patient_id"]: row for row in read_jsonl(manifest)} if manifest else {}
    if manifest and any(row["patient_id"] not in patient_cases for row in patients):
        raise ValueError("Synthesis request references patients absent from source manifest")
    root = Path(output_root).resolve()
    requests, keys = [], set()
    for patient in patients:
        if split != "all" and patient.get("split") != split:
            continue
        pid = patient["patient_id"]
        for pos in patient["selected_positions"]:
            ref, ncct = int(pos["reference_slice_id"]), int(pos["matched_ncct_slice_id"])
            for phase in ("ap", "pvp"):
                key = f"{pid}|{ref}|{phase}"
                if key in keys:
                    raise ValueError(f"Duplicate synthesis request {key}")
                keys.add(key)
                target = root / pid / phase / f"ref_{ref:04d}.npy"
                target.parent.mkdir(parents=True, exist_ok=True)
                requests.append({
                    "key": key, "patient_id": pid, "target_phase": phase,
                    "reference_slice_id": ref, "matched_ncct_slice_id": ncct,
                    "ncct_path": patient_cases[pid]["volumes"]["ncct"] if manifest else None,
                    "output_path": str(target),
                    "matching_status": pos.get("matching_status", "unknown"),
                    "note": "Generator must write 2D float32 HU as [H=y,W=x], consistent with Real NIfTI slice transpose",
                })
    write_json(out, requests)
    return requests


def register_outputs(requests_path, out):
    requests = read_json(requests_path)
    index = {}
    missing = []
    for req in requests:
        path = Path(req["output_path"])
        if not path.is_file():
            missing.append(str(path))
            continue
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"Invalid 2D finite HU output: {path} ({array.shape})")
        index[req["key"]] = str(path)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} synthesis outputs; first: {missing[:5]}")
    write_json(out, index)
    return index
