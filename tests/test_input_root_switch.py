"""Source volume mount changes must not change patient labels or saved split assignments."""
import csv
from pathlib import Path

from cectcls.locked_cohort import lock_cohort


def _write(path, fields, entries):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)


def _read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_two_ct_mounts_same_patients_labels_splits(tmp_path):
    src = tmp_path / "locked.csv"
    fields = ["patient_id", "主类别", "三分类标签", "配准等级",
              "ncct_path", "ap_path", "pvp_path"]
    rows = [
        dict(patient_id="P001", 主类别="HCC", 三分类标签="恶性", 配准等级="REJECT",
             ncct_path="/old/P001/ncct.nii.gz", ap_path="/old/P001/ap.nii.gz",
             pvp_path="/old/P001/pvp.nii.gz"),
        dict(patient_id="P002", 主类别="正常", 三分类标签="正常", 配准等级="GOLD",
             ncct_path="/old/P002/ncct.nii.gz", ap_path="/old/P002/ap.nii.gz",
             pvp_path="/old/P002/pvp.nii.gz"),
    ]
    _write(src, fields, rows)
    fixed_splits = tmp_path / "splits.csv"
    _write(fixed_splits, ["patient_id", "split"],
           [{"patient_id": "P001", "split": "test"},
            {"patient_id": "P002", "split": "train"}])
    a = tmp_path / "first"
    b = tmp_path / "second"
    lock_cohort(src, a, expected_count=2,
                image_root="/dataset/dataset/gx_all_registered",
                splits_source=fixed_splits)
    lock_cohort(src, b, expected_count=2,
                image_root="/win_cect/shared/gx_all_registered",
                splits_source=fixed_splits)
    first = _read(a / "cases.csv")
    second = _read(b / "cases.csv")
    assert [(r["patient_id"], r["label"]) for r in first] == [
        (r["patient_id"], r["label"]) for r in second
    ]
    assert _read(a / "splits.csv") == _read(b / "splits.csv")
    assert first[0]["ncct_path"].startswith("/dataset/dataset/gx_all_registered/")
    assert second[0]["ncct_path"].startswith("/win_cect/shared/gx_all_registered/")
    assert second[0]["liver_mask_path"].endswith("/P001/ncct_mask_zcrop.nii.gz")
