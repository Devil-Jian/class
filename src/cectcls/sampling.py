"""Deterministic patient-level hybrid sampling with traceable source slice IDs."""
from __future__ import annotations

import hashlib
import random

import numpy as np


def _patient_seed(seed, epoch, patient_id):
    digest = hashlib.sha256(f"{seed}:{epoch}:{patient_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def mixed_sample(valid_ids, patient_id, epoch=0, seed=42, uniform=16, random_count=16):
    ids = sorted(set(int(i) for i in valid_ids))
    if not ids:
        raise ValueError(f"{patient_id} has no candidate slices")
    if uniform < 0 or random_count < 0 or uniform + random_count <= 0:
        raise ValueError("Invalid sampling counts")
    n = min(uniform, len(ids))
    uniform_ids = sorted({ids[int(k)] for k in np.linspace(0, len(ids) - 1, n).round().astype(int)})
    left = sorted(set(ids) - set(uniform_ids))
    count = min(random_count, len(left))
    random_ids = sorted(random.Random(_patient_seed(seed, epoch, patient_id)).sample(left, count))
    selected = sorted(set(uniform_ids + random_ids))
    return {
        "uniform_reference_ids": uniform_ids,
        "random_reference_ids": random_ids,
        "selected_reference_ids": selected,
        "selected_ap_ids": selected[:],
        "selected_pvp_ids": selected[:],
    }


def topk_attention(valid_ids, scores, k=16):
    if len(valid_ids) != len(scores) or k < 1:
        raise ValueError("Slice ids / scores mismatch or invalid k")
    order = sorted(range(len(valid_ids)), key=lambda i: (-float(scores[i]), int(valid_ids[i])))
    return [(int(valid_ids[i]), float(scores[i])) for i in order[:k]]
