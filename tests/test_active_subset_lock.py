"""Ensure an explicitly approved subset is immutable without storing patient data in git."""
import csv
import hashlib

import pytest

from cectcls import locked_cohort


def test_fixed_subset_hash_rejects_same_count_same_labels_wrong_members(tmp_path, monkeypatch):
    src = tmp_path / "cohort.csv"
    fields = [
        "patient_id", "主类别", "三分类标签", "配准等级",
        "ncct_path", "ap_path", "pvp_path",
    ]

    def write_rows(ids):
        rows = [
            {"patient_id": pid, "主类别": "HCC", "三分类标签": "恶性",
             "配准等级": "REJECT", "ncct_path": f"/scan/{pid}/ncct.nii.gz",
             "ap_path": f"/scan/{pid}/ap.nii.gz",
             "pvp_path": f"/scan/{pid}/pvp.nii.gz"}
            for pid in ids
        ]
        with src.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    approved_ids = ["P001", "P002"]
    sha = hashlib.sha256("\n".join(sorted(f"{pid}|HCC" for pid in approved_ids)).encode()).hexdigest()
    monkeypatch.setitem(locked_cohort.APPROVED_COHORTS, 2, {
        "counts": {"HCC": 2}, "patient_label_sha256": sha,
    })
    write_rows(approved_ids)
    locked_cohort.lock_cohort(src, tmp_path / "ok", expected_count=2,
                              generate_splits=False)

    write_rows(["P001", "P003"])
    with pytest.raises(ValueError, match="IDs or primary labels differ"):
        locked_cohort.lock_cohort(src, tmp_path / "wrong", expected_count=2,
                                  generate_splits=False)


def test_active_1195_public_manifest_contains_only_aggregate_constants():
    expected = locked_cohort.APPROVED_COHORTS[1195]
    assert sum(expected["counts"].values()) == 1195
    assert expected["counts"] == {
        "HCC": 720, "转移瘤": 87, "肝囊肿": 163, "血管瘤": 52, "正常": 173
    }
