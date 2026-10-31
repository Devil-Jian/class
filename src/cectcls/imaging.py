"""Lazy canonical NIfTI axial slice loading and synchronised CT transforms."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

try:
    import h5py
except ImportError:
    h5py = None
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


@lru_cache(maxsize=16)
def volume(path):
    return nib.as_closest_canonical(nib.load(str(path)))


@lru_cache(maxsize=16)
def volume_array(path):
    return np.asarray(volume(path).get_fdata(dtype=np.float32), dtype=np.float32)


@lru_cache(maxsize=64)
def _get_h5_file(h5_file_str: str):
    if h5py is None:
        return None
    return h5py.File(h5_file_str, "r")


def _read_from_h5(h5_root: str, path: str, slice_id: int):
    p = Path(path)
    h5_file = Path(h5_root) / f"{p.parent.name}.h5"
    if not h5_file.is_file():
        return None
    name = p.name.lower()
    if "ap" in name:
        key = "ap"
    elif "pvp" in name:
        key = "pvp"
    elif "mask" in name:
        key = "mask"
    elif "ncct" in name:
        key = "ncct"
    else:
        return None
    try:
        f = _get_h5_file(str(h5_file))
        if key in f:
            return np.asarray(f[key][int(slice_id)], dtype=np.float32)
    except Exception:
        pass
    return None


def read_hu(path, slice_id=None):
    path = str(path)
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"Syn .npy image should be 2D HU, got: {arr.shape}")
        return np.asarray(arr, dtype=np.float32)
    if slice_id is None:
        raise ValueError(f"slice_id required for NIfTI: {path}")

    h5_root = os.environ.get("H5_ROOT", "").strip()
    if not h5_root and os.path.isdir("/dataset/h5_cache"):
        h5_root = "/dataset/h5_cache"
    if h5_root:
        cached = _read_from_h5(h5_root, path, slice_id)
        if cached is not None:
            return cached

    arr = volume_array(path)[:, :, int(slice_id)]
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
