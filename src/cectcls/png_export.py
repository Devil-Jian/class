"""Export final selected HU slices as fixed-window, lossless 8-bit grayscale PNGs."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .imaging import read_hu
from .manifest import read_json, read_jsonl, write_json


def window_uint8(hu, lo=-160.0, hi=240.0):
    if hi <= lo:
        raise ValueError("Invalid window range")
    arr = np.asarray(hu, dtype=np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError("Expected finite 2D HU image")
    return np.rint((np.clip(arr, lo, hi) - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def export_png(manifest, selection, out, syn_index=None, window=(-160.0, 240.0)):
    cases = {r["patient_id"]: r for r in read_jsonl(manifest)}
    patients = read_json(selection)
    synthetic = read_json(syn_index) if syn_index else None
    root = Path(out).expanduser().resolve()
    index = {}
    count = Counter()
    for patient in patients:
        pid = patient["patient_id"]
        if pid not in cases:
            raise ValueError(f"Selection contains missing patient {pid}")
        case = cases[pid]
        for item in patient["selected_positions"]:
            ref = int(item["reference_slice_id"])
            if "matched_ncct_slice_id" not in item:
                raise ValueError(f"NCCT matching not done for {pid} ref={ref}")
            ncct = int(item["matched_ncct_slice_id"])
            ap = int(item.get("ap_registered_slice_id", ref))
            pvp = int(item.get("pvp_registered_slice_id", ref))
            entries = [
                ("real", "ap", case["volumes"]["ap"], ap),
                ("real", "pvp", case["volumes"]["pvp"], pvp),
                ("ncct", "ncct", case["volumes"]["ncct"], ncct),
            ]
            if synthetic is not None and patient.get("split") == "train":
                for phase in ("ap", "pvp"):
                    key = f"{pid}|{ref}|{phase}"
                    if key not in synthetic:
                        raise KeyError(f"SYN output not registered: {key}")
                    entries.append(("syn", phase, synthetic[key], None))
            for group, phase, source, index_id in entries:
                # index uses the target CECT anatomical slice ID, NOT the matched NCCT ID
                key = f"{pid}|{ref}|{group}|{phase}"
                if key in index:
                    raise ValueError(f"Duplicate PNG target key: {key}")
                dest = root / group / phase / pid / f"ref_{ref:04d}.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                array = window_uint8(read_hu(source, index_id), *window)
                Image.fromarray(array, mode="L").save(dest)
                index[key] = str(dest)
                count[group + "_" + phase] += 1
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "png_index.json", index)
    write_json(root / "png_export_summary.json", {
        "window_hu": [float(window[0]), float(window[1])],
        "format": "8-bit grayscale lossless PNG",
        "pixel_orientation": "rows=voxel y, columns=voxel x (after canonical NIfTI)",
        "selected_patients": len(patients),
        "counts": dict(count),
        "note": "Keep NIfTI HU source for matching and synthesis; PNG is for final downstream training",
    })
    return index
