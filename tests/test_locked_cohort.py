"""Locked cohort membership tests use synthetic IDs only (no patient CSV in git)."""
import csv

from cectcls.locked_cohort import lock_cohort
from cectcls.manifest import read_json


def test_lock_cohort_preserves_reject_and_fixed_labels(tmp_path):
    source = tmp_path / "approved.csv"
    rows = [
        ("P001", "正常", "正常", "GOLD"),
        ("P002", "肝囊肿", "良性", "SILVER"),
        ("P003", "血管瘤", "良性", "REJECT"),
        ("P004", "HCC", "恶性", "REJECT"),
        ("P005", "转移瘤", "恶性", "REJECT"),
    ]
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "patient_id", "主类别", "三分类标签", "配准等级",
            "ncct_path", "ap_path", "pvp_path", "liver_mask_path_candidate"
        ])
        writer.writeheader()
        for pid, label, three, grade in rows:
            writer.writerow({
                "patient_id": pid, "主类别": label, "三分类标签": three,
                "配准等级": grade, "ncct_path": f"/dataset/{pid}/ncct.nii.gz",
                "ap_path": f"/dataset/{pid}/ap.nii.gz",
                "pvp_path": f"/dataset/{pid}/pvp.nii.gz",
            })
    out = tmp_path / "out"
    result = lock_cohort(source, out, expected_count=5, generate_splits=False)
    assert result["patients"] == 5
    assert result["by_registration_grade"]["REJECT"] == 3
    assert read_json(out / "labels.json") == {
        "正常": 0, "肝囊肿": 1, "血管瘤": 2, "HCC": 3, "转移瘤": 4
    }
    assert not (out / "splits.csv").exists()


def test_missing_patient_causes_lock_failure(tmp_path):
    source = tmp_path / "empty.csv"
    source.write_text("patient_id,主类别,三分类标签,配准等级,ncct_path,ap_path,pvp_path\n",
                      encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="Expected 1298"):
        lock_cohort(source, tmp_path / "out")
