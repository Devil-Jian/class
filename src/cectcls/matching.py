"""Anatomy-constrained NCCT axial slice matching with paired AP/PVP objectives."""
from __future__ import annotations

import math

import numpy as np

from .imaging import read_hu
from .manifest import read_json, read_jsonl, write_json


def ncc(a, b):
    x, y = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if x.size < 16 or np.std(x) < 1e-7 or np.std(y) < 1e-7:
        return 0.0
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1., 1.))


def nmi(a, b, bins=32, window=(-160., 240.)):
    """Normalized mutual information (H(A)+H(B))/H(A,B), approximately [1,2]."""
    x = np.clip(np.asarray(a).ravel(), *window)
    y = np.clip(np.asarray(b).ravel(), *window)
    if x.size < 16:
        return 1.
    hist, _, _ = np.histogram2d(x, y, bins=bins, range=[list(window), list(window)])
    p = hist / max(1., hist.sum())
    px, py = p.sum(axis=1), p.sum(axis=0)

    def entropy(z):
        z = z[z > 0]
        return -float(np.dot(z, np.log(z)))
    joint = entropy(p)
    return float((entropy(px) + entropy(py)) / joint) if joint > 1e-10 else 1.


def score_candidate(ap, pvp, ncct, roi, dz, radius):
    if roi.sum() < 64:
        return {"score": -1e6}
    a, p, n = [np.clip(x[roi], -160, 240) for x in (ap, pvp, ncct)]
    mi_ap, mi_pvp = nmi(a, n), nmi(p, n)
    cc_ap, cc_pvp = ncc(a, n), ncc(p, n)
    normalized_mi = np.mean([np.clip(mi_ap - 1, 0, 1), np.clip(mi_pvp - 1, 0, 1)])
    normalized_cc = np.mean([(cc_ap + 1) / 2, (cc_pvp + 1) / 2])
    score = 0.5 * normalized_mi + 0.35 * normalized_cc - 0.15 * abs(dz) / max(radius, 1e-6)
    return {"score": float(score), "nmi_ap": mi_ap, "nmi_pvp": mi_pvp,
            "ncc_ap": cc_ap, "ncc_pvp": cc_pvp}


def monotonic_path(candidate_rows):
    """DP maximize sum of similarity scores, requiring non-decreasing NCCT z-indices."""
    if not candidate_rows:
        return []
    dp = [{c["ncct_slice_id"]: (c["score"], None) for c in candidate_rows[0]}]
    for row in candidate_rows[1:]:
        prev = dp[-1]
        current = {}
        for candidate in row:
            j = candidate["ncct_slice_id"]
            options = [(value[0] - (0.03 if i == j else 0.), i)
                       for i, value in prev.items() if i <= j]
            if options:
                val, idx = max(options, key=lambda item: item[0])
                current[j] = (val + candidate["score"], idx)
        if not current:
            raise ValueError("No monotonic NCCT match path; inspect registration and search radius")
        dp.append(current)
    end = max(dp[-1], key=lambda j: dp[-1][j][0])
    chosen = [end]
    for step in range(len(dp) - 1, 0, -1):
        end = dp[step][end][1]
        chosen.append(end)
    chosen.reverse()
    return [next(c for c in row if c["ncct_slice_id"] == j)
            for row, j in zip(candidate_rows, chosen)]


def _downsample(array, stride):
    return np.asarray(array)[::stride, ::stride]


def match_ncct(manifest, selection, out, radius_mm=15., stride=4):
    if radius_mm <= 0 or stride < 1:
        raise ValueError("radius_mm > 0 and stride >= 1 required")
    cases = {r["patient_id"]: r for r in read_jsonl(manifest)}
    selected = read_json(selection)
    results = []
    for patient in selected:
        pid = patient["patient_id"]
        if pid not in cases:
            raise ValueError(f"Unknown patient: {pid}")
        case = cases[pid]
        targets = sorted(patient["selected_positions"], key=lambda p: int(p["reference_slice_id"]))
        candidates = []
        mask_volume = case["liver_mask"]
        for target in targets:
            ref = int(target["reference_slice_id"])
            z = float(case["reference_z_mm"][str(ref)])
            ap = _downsample(read_hu(case["volumes"]["ap"], target["ap_registered_slice_id"]), stride)
            pvp = _downsample(read_hu(case["volumes"]["pvp"], target["pvp_registered_slice_id"]), stride)
            mask_ref = (_downsample(read_hu(mask_volume, ref), stride) > 0 if mask_volume else None)
            rows = []
            for ncid in case["valid_ids"]:
                dz = float(case["reference_z_mm"][str(ncid)]) - z
                if abs(dz) > radius_mm + 1e-5:
                    continue
                ncct = _downsample(read_hu(case["volumes"]["ncct"], ncid), stride)
                if mask_ref is not None:
                    mask_candidate = _downsample(read_hu(mask_volume, ncid), stride) > 0
                    roi = mask_ref & mask_candidate
                else:
                    # Constrain to a central region when segmentation is unavailable.
                    h, w = ncct.shape
                    roi = np.zeros((h, w), dtype=bool)
                    roi[h // 5:4 * h // 5, w // 5:4 * w // 5] = True
                stats = score_candidate(ap, pvp, ncct, roi, dz, radius_mm)
                if stats["score"] > -1e5:
                    rows.append({"ncct_slice_id": int(ncid), "delta_z_mm": float(dz), **stats})
            if not rows:
                raise ValueError(f"No valid NCCT candidates for {pid} reference={ref}")
            candidates.append(rows)
        best = monotonic_path(candidates)
        mapped = []
        for item, choice in zip(targets, best):
            # These are similarity-based provisional matches, not verified anatomical truth.
            status = ("review_required" if choice["score"] < 0.10 or
                      abs(choice["delta_z_mm"]) > 10 else "provisional")
            mapped.append({**item, "matched_ncct_slice_id": choice["ncct_slice_id"],
                           "matching_score": choice["score"],
                           "matching_delta_z_mm": choice["delta_z_mm"],
                           "matching_status": status,
                           "matching_metrics": {k: v for k, v in choice.items()
                                               if k in {"nmi_ap", "nmi_pvp", "ncc_ap", "ncc_pvp"}}})
        results.append({**{k: v for k, v in patient.items() if k != "selected_positions"},
                        "matching_method": "paired_nmi_ncc_spatial_monotonic",
                        "search_radius_mm": radius_mm,
                        "selected_positions": mapped})
    write_json(out, results)
    return results
