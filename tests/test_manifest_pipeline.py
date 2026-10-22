import csv
import nibabel as nib
import numpy as np
import pytest

from cectcls.dataset import BalancedSourceSampler, PatientBagDataset
from cectcls.manifest import prepare


def csv_file(path, names, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def test_patient_manifest_on_ncct_grid(tmp_path):
    shape = (24, 24, 8)
    affine = np.diag([1., 1., 3., 1.])
    for phase in ("ncct", "ap", "pvp"):
        nib.save(nib.Nifti1Image(np.ones(shape, np.float32) * 40, affine),
                 tmp_path / f"{phase}.nii.gz")
    liver = np.zeros(shape, dtype=np.uint8)
    liver[4:20, 4:20, 2:6] = 1
    nib.save(nib.Nifti1Image(liver, affine), tmp_path / "mask.nii.gz")
    csv_file(tmp_path / "cases.csv",
             ["patient_id", "label", "ncct_path", "ap_path", "pvp_path", "liver_mask_path"],
             [{"patient_id": "P1", "label": "HCC", "ncct_path": "ncct.nii.gz",
               "ap_path": "ap.nii.gz", "pvp_path": "pvp.nii.gz",
               "liver_mask_path": "mask.nii.gz"}])
    csv_file(tmp_path / "splits.csv", ["patient_id", "split"],
             [{"patient_id": "P1", "split": "train"}])
    rows = prepare(tmp_path / "cases.csv", tmp_path / "splits.csv", tmp_path / "out")
    assert rows[0]["valid_ids"] == [2, 3, 4, 5]
    ds = PatientBagDataset(tmp_path / "out" / "cases.jsonl", "train",
                           mode="mixed", uniform=2, random_count=1, image_size=64)
    item = ds[0]
    assert item["ap"].shape == (3, 1, 64, 64)
    assert item["ap_slice_ids"] == item["pvp_slice_ids"]


def test_equal_training_steps_real_plus_synthetic():
    class Dummy:
        cases = list(range(12))
        group = "cect_real_syn"
        split = "train"
        epoch = 0
    ds = Dummy()
    sampled = list(BalancedSourceSampler(ds, seed=13))
    assert len(sampled) == 12
    assert len({i % 12 for i in sampled}) == 12
    assert sum(i >= 12 for i in sampled) == 6


def test_bad_grid_rejected(tmp_path):
    shape = (8, 8, 4)
    for p, s in (("ncct", 1.), ("ap", 1.), ("pvp", 2.)):
        nib.save(nib.Nifti1Image(np.zeros(shape, np.float32),
                                 np.diag([1., 1., s, 1.])),
                 tmp_path / f"{p}.nii.gz")
    csv_file(tmp_path / "cases.csv",
             ["patient_id", "label", "ncct_path", "ap_path", "pvp_path"],
             [{"patient_id": "P1", "label": "0", "ncct_path": "ncct.nii.gz",
               "ap_path": "ap.nii.gz", "pvp_path": "pvp.nii.gz"}])
    csv_file(tmp_path / "splits.csv", ["patient_id", "split"],
             [{"patient_id": "P1", "split": "train"}])
    with pytest.raises(ValueError, match="NCCT fixed grid"):
        prepare(tmp_path / "cases.csv", tmp_path / "splits.csv", tmp_path / "out")
