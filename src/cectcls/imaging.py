"""Lazy canonical NIfTI axial slice loading and synchronised CT transforms."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


@lru_cache(maxsize=16)
def volume(path):
    return nib.as_closest_canonical(nib.load(str(path)))


def read_hu(path, slice_id=None):
    path = str(path)
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"Syn .npy image should be 2D HU, got: {arr.shape}")
        return np.asarray(arr, dtype=np.float32)
    if slice_id is None:
        raise ValueError(f"slice_id required for NIfTI: {path}")
    arr = np.asarray(volume(path).dataobj[:, :, int(slice_id)], dtype=np.float32)
    return arr.T.copy()  # image tensor: (H=y, W=x)


def image_tensor(path, slice_id, image_size=224, window=(-160., 240.)):
    data = read_hu(path, slice_id)
    lo, hi = window
    data = (np.clip(data, lo, hi) - lo) / (hi - lo)
    x = torch.from_numpy(np.ascontiguousarray(data)).float()[None, None, :, :]
    return F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)[0]


def augment_pair(ap, pvp, max_deg=10., scale=0.1):
    """Apply one shared affine warp to paired phases: each is [K,1,H,W]."""
    angle = (torch.rand(1).item() * 2 - 1) * max_deg * np.pi / 180
    factor = 1 + (torch.rand(1).item() * 2 - 1) * scale
    cosine, sine = np.cos(angle) / factor, np.sin(angle) / factor
    theta = ap.new_tensor([[cosine, -sine, 0], [sine, cosine, 0]])[None].expand(len(ap), -1, -1)
    grid = F.affine_grid(theta, ap.shape, align_corners=False)
    return (F.grid_sample(ap, grid, align_corners=False, padding_mode="border"),
            F.grid_sample(pvp, grid, align_corners=False, padding_mode="border"))
