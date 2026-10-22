"""HDD QC: metadata-only inspection, canonical affine, and persistent mask caching."""
import csv
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from cectcls.audit import _header, audit_images
from cectcls.manifest import prepare, read_json


def _create_data(tmp_path):
    shape = (9, 7, 6)
    affine = np.array([[0, -2, 0, 10], [1, 0, 0, 20],
                       [0, 0, 3, 30], [0, 0, 0, 1]], dtype=float)
    arr = np.zeros(shape, dtype=np.float32)
    arr[2:7, 2:5, 2:4] = 30
    paths = {}
    for phase in ("ncct", "ap", "pvp"):
        p = tmp_path / f"{phase}.nii.gz"
        nib.save(nib.Nifti1Image(arr, affine), p)
        paths[phase + "_path"] = str(p)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1:8, 1:6, 2:4] = 1
    p = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), p)
    paths["liver_mask_path"] = str(p)
    fields = ["patient_id", "label", *paths]
    with (tmp_path / "cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerow({"patient_id": "P001", "label": "HCC", **paths})
    with (tmp_path / "splits.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=["patient_id", "split"])
        w.writeheader()
        w.writerow({"patient_id": "P001", "split": "train"})
    return paths


def test_header_only_qc_matches_canonical_without_voxels(tmp_path, monkeypatch):
    paths = _create_data(tmp_path)
    mask_img = nib.load(paths["liver_mask_path"])
    ref = nib.as_closest_canonical(mask_img)
    meta = _header(Path(paths["liver_mask_path"]))
    assert meta["shape"] == list(ref.shape)
    np.testing.assert_allclose(meta["affine"], ref.affine)
    monkeypatch.setattr(nib, "as_closest_canonical",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            AssertionError("Fast audit must not decode/reorient voxels")))
    report = audit_images(tmp_path / "cases.csv", tmp_path / "audit.csv",
                          strict_mask=True, workers=2)
    assert report["ready"] == 1
    assert report["voxel_data_read"] is False
    assert report["headers_reused"] == 0
    report = audit_images(tmp_path / "cases.csv", tmp_path / "audit.csv",
                          strict_mask=True, workers=1)
    assert report["headers_reused"] == 4


def test_prepare_reuses_cached_mask_slices_without_decompression(tmp_path, monkeypatch):
    _create_data(tmp_path)
    audit_images(tmp_path / "cases.csv", tmp_path / "audit.csv", strict_mask=True)
    cache = tmp_path / "image_header_cache.json"
    out = tmp_path / "manifests"
    rows = prepare(tmp_path / "cases.csv", tmp_path / "splits.csv", out,
                   header_cache_path=cache)
    assert rows[0]["valid_ids"] == [2, 3]
    first = read_json(out / "qc.json")
    assert first["masks_decompressed"] == 1
    import cectcls.manifest as m
    monkeypatch.setattr(m, "_image", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("Unchanged source must not be decompressed again")))
    repeat = prepare(tmp_path / "cases.csv", tmp_path / "splits.csv", out,
                     header_cache_path=cache)
    assert repeat[0]["valid_ids"] == rows[0]["valid_ids"]
    final = read_json(out / "qc.json")
    assert final["mask_slices_reused"] == 1
    assert final["masks_decompressed"] == 0


def test_file_change_requires_fresh_header_audit(tmp_path):
    paths = _create_data(tmp_path)
    audit_images(tmp_path / "cases.csv", tmp_path / "audit.csv", strict_mask=True)
    path = Path(paths["pvp_path"])
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    with pytest.raises(ValueError, match="changed after header audit"):
        prepare(tmp_path / "cases.csv", tmp_path / "splits.csv",
                tmp_path / "manifests", header_cache_path=tmp_path / "image_header_cache.json")
