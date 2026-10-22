"""Synthetic fixtures; no real patient identifiers are committed to git."""
import csv
from pathlib import Path

import pytest

from cectcls.qc_finalize import finalize_qc_subset
from cectcls.manifest import read_json


def _save(path, rows, cols):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


def _source_rows():
    return [
        {"patient_id": f"P{i:03}", "主类别": label, "三分类标签": three,
         "配准等级": grade, "ncct_path": f"/dummy/P{i:03}/ncct.nii.gz",
         "ap_path": f"/dummy/P{i:03}/ap.nii.gz",
         "pvp_path": f"/dummy/P{i:03}/pvp.nii.gz"}
        for i, label, three, grade in (
            (1, "HCC", "恶性", "REJECT"),
            (2, "正常", "正常", "GOLD"),
            (3, "血管瘤", "良性", "SILVER"),
            (4, "肝囊肿", "良性", "REJECT"),
        )
    ]


def _fixture(tmp_path, bad_issues="ncct_missing;ap_missing;pvp_missing;liver_mask_missing"):
    data = _source_rows()
    source = tmp_path / "source.csv"
    splits = tmp_path / "splits.csv"
    qc = tmp_path / "qc.csv"
    _save(source, data, list(data[0]))
    _save(splits, [
        {"patient_id": "P001", "split": "train"},
        {"patient_id": "P002", "split": "val"},
        {"patient_id": "P003", "split": "test"},
        {"patient_id": "P004", "split": "train"},
    ], ["patient_id", "split"])
    _save(qc, [
        {"patient_id": r["patient_id"],
         "nifti_ready": "False" if r["patient_id"] == "P004" else "True",
         "issues": bad_issues if r["patient_id"] == "P004" else ""}
        for r in data
    ], ["patient_id", "nifti_ready", "issues"])
    return source, splits, qc


def test_prune_only_failed_case_without_resplitting(tmp_path):
    src, splits, qc = _fixture(tmp_path)
    report = finalize_qc_subset(
        src, splits, qc, tmp_path / "out", expected_source_count=4,
        expected_removals=1
    )
    assert report["patients"] == 3
    assert report["excluded_missing_images"] == 1
    assert report["by_split"] == {"train": 1, "val": 1, "test": 1}
    final_splits = tmp_path / "out" / "classification_main5_3_locked_splits.csv"
    with final_splits.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["patient_id"], r["split"]) for r in rows] == [
        ("P001", "train"), ("P002", "val"), ("P003", "test")
    ]
    report_file = tmp_path / "out" / "classification_main5_3_lock_report.json"
    assert read_json(report_file)["patient_label_membership_sha256"] == report[
        "patient_label_membership_sha256"
    ]


def test_prune_refuses_mismatched_failure_count(tmp_path):
    src, splits, qc = _fixture(tmp_path)
    with pytest.raises(ValueError, match="Expected exactly 2"):
        finalize_qc_subset(src, splits, qc, tmp_path / "out",
                           expected_source_count=4, expected_removals=2)


def test_prune_refuses_non_missing_qc_failures(tmp_path):
    src, splits, qc = _fixture(tmp_path, bad_issues="pvp_grid_mismatch")
    with pytest.raises(ValueError, match="not exclusively missing images"):
        finalize_qc_subset(src, splits, qc, tmp_path / "out",
                           expected_source_count=4, expected_removals=1)
