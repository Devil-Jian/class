"""Small local synthetic cohort and PNG export integration checks."""
import csv

import nibabel as nib
import numpy as np
from PIL import Image

from cectcls.cohort import build_cohort
from cectcls.manifest import prepare, read_json, write_json
from cectcls.png_export import export_png, window_uint8
from cectcls.dataset import PatientBagDataset


def _csv(path, names, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=names)
        w.writeheader()
        w.writerows(rows)


def _make_case(root, pid, label, grade="GOLD"):
    p = root / pid
    p.mkdir(parents=True)
    affine = np.diag([1., 1., 3., 1.])
    arr = np.linspace(-200, 300, 16 * 16 * 5, dtype=np.float32).reshape((16, 16, 5))
    for filename in ("ncct_zcrop", "ap_registered", "pvp_registered"):
        nib.save(nib.Nifti1Image(arr, affine), p / f"{filename}.nii.gz")
    mask = np.zeros_like(arr)
    mask[2:14, 2:14, 1:4] = 1
    nib.save(nib.Nifti1Image(mask, affine), p / "ncct_mask_zcrop.nii.gz")
    return {
        "patient_id": pid, "三分类标签": label, "主类别": "HCC",
        "配准等级": grade, "分类（暂定）": "是",
    }


def test_metadata_to_png_pipeline(tmp_path):
    root = tmp_path / "gx_all_registered"
    rows = [
        _make_case(root, "A0_1", "恶性"),
        _make_case(root, "A0_2", "正常"),
        _make_case(root, "A0_3", "良性"),
        _make_case(root, "A0_4", "恶性", "REJECT"),
    ]
    rows.append(rows[0].copy())
    metadata = tmp_path / "GX_metadata_all.csv"
    _csv(metadata, list(rows[0]), rows)
    report = build_cohort(metadata, root, tmp_path / "cohort")
    # REJECT reflects poor registration quality, not classification ineligibility.
    assert report["eligible_patient_count"] == 4
    assert report["by_label"] == {"恶性": 2, "正常": 1, "良性": 1}
    assert report["exclusion_counts"].get("registration_grade", 0) == 0
    assert report["exclusion_counts"]["duplicate_metadata_rows_collapsed"] == 1
    cases = prepare(tmp_path / "cohort/cases.csv", tmp_path / "cohort/splits.csv",
                    tmp_path / "manifests")
    assert len(cases) == 4
    assert "A0_4" in {case["patient_id"] for case in cases}
    # Independently fix a tiny matched selection; no model is needed in this test.
    selected = [{
        "patient_id": case["patient_id"], "split": case["split"],
        "selected_positions": [{
            "reference_slice_id": 2, "ap_registered_slice_id": 2,
            "pvp_registered_slice_id": 2, "matched_ncct_slice_id": 2,
        }],
    } for case in cases]
    select = tmp_path / "matched.json"
    write_json(select, selected)
    train_rows = [x for x in selected if x["split"] == "train"]
    assert train_rows
    syn_index = {}
    for row in train_rows:
        for phase in ("ap", "pvp"):
            path = tmp_path / f"{row['patient_id']}_{phase}.npy"
            np.save(path, np.ones((16, 16), dtype=np.float32) * 100.)
            syn_index[f"{row['patient_id']}|2|{phase}"] = str(path)
    syn_path = tmp_path / "syn.json"
    write_json(syn_path, syn_index)
    exported = export_png(tmp_path / "manifests/cases.jsonl", select,
                          tmp_path / "png", syn_index=syn_path)
    assert len(exported) == len(cases) * 3 + len(train_rows) * 2
    with Image.open(next(iter(exported.values()))) as im:
        assert im.mode == "L"
    train_ds = PatientBagDataset(tmp_path / "manifests/cases.jsonl",
                                 "train", group="cect_syn", mode="selected",
                                 selection=select, png_index=tmp_path / "png/png_index.json",
                                 image_size=32)
    item = train_ds[0]
    assert item["ap"].shape == (1, 1, 32, 32)
    assert item["ap_slice_ids"] == [2]


def test_window_values():
    assert window_uint8(np.array([[-300., -160., 240., 400.]], dtype=np.float32)).tolist() == [[0, 0, 255, 255]]
