"""Patient-level paired slice bags; all training groups use identical target positions."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .imaging import augment_pair, image_tensor
from .manifest import read_json, read_jsonl
from .sampling import mixed_sample


class PatientBagDataset(Dataset):
    def __init__(self, manifest, split, group="cect_real", mode="mixed", selection=None,
                 syn_index=None, seed=42, image_size=224, uniform=16, random_count=16):
        self.cases = [r for r in read_jsonl(manifest) if r["split"] == split]
        self.split, self.group, self.mode = split, group, mode
        self.seed, self.epoch = seed, 0
        self.image_size = image_size
        self.uniform, self.random_count = uniform, random_count
        if not self.cases:
            raise ValueError(f"No cases in split={split}")
        if group not in {"ncct", "cect_real", "cect_syn", "cect_real_aug", "cect_real_syn"}:
            raise ValueError(f"Unknown group: {group}")
        if mode not in {"mixed", "full", "selected"}:
            raise ValueError(f"Unknown mode: {mode}")
        if mode == "selected" and not selection:
            raise ValueError("selected mode requires --selection manifest")
        self.selected = {r["patient_id"]: r for r in read_json(selection)} if selection else {}
        self.syn_index = read_json(syn_index) if syn_index else {}
        if any(r["patient_id"] not in self.selected for r in self.cases) and mode == "selected":
            raise ValueError("Selection missing patients in split")
        if group in {"cect_syn", "cect_real_syn"} and not syn_index:
            raise ValueError(f"{group} requires a registered --syn-index")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        repeated = self.split == "train" and self.mode == "selected" and self.group in {
            "cect_real_aug", "cect_real_syn",
        }
        return len(self.cases) * (2 if repeated else 1)

    def _positions(self, case):
        if self.mode == "full":
            ids = case["valid_ids"]
            return [{"reference_slice_id": i, "ap_registered_slice_id": i,
                     "pvp_registered_slice_id": i} for i in ids], None
        if self.mode == "mixed":
            sample = mixed_sample(case["valid_ids"], case["patient_id"],
                                  self.epoch if self.split == "train" else 0,
                                  self.seed, self.uniform, self.random_count)
            positions = [{"reference_slice_id": i, "ap_registered_slice_id": i,
                          "pvp_registered_slice_id": i} for i in sample["selected_reference_ids"]]
            return positions, sample
        return self.selected[case["patient_id"]]["selected_positions"], None

    @staticmethod
    def _syn_path(index, patient, refid, phase):
        key = f"{patient}|{refid}|{phase}"
        if key not in index:
            raise KeyError(f"Missing synthesized output for {key}; register-syn first")
        return index[key]

    def __getitem__(self, index):
        n = len(self.cases)
        case = self.cases[index % n]
        second = index >= n and self.split == "train" and self.mode == "selected"
        use_syn = self.group == "cect_syn" or (self.group == "cect_real_syn" and second)
        augmented = self.group == "cect_real_aug" and second
        positions, sample = self._positions(case)
        ap_images, pvp_images, ref_ids, ap_ids, pvp_ids = [], [], [], [], []
        for item in positions:
            ref = int(item["reference_slice_id"])
            apid = int(item.get("ap_registered_slice_id", ref))
            pvpid = int(item.get("pvp_registered_slice_id", ref))
            ncid = int(item.get("matched_ncct_slice_id", ref))
            if self.group == "ncct":
                a_path, a_id = case["volumes"]["ncct"], ncid
                p_path, p_id = a_path, a_id
            elif use_syn:
                a_path, a_id = self._syn_path(self.syn_index, case["patient_id"], ref, "ap"), None
                p_path, p_id = self._syn_path(self.syn_index, case["patient_id"], ref, "pvp"), None
            else:
                a_path, a_id = case["volumes"]["ap"], apid
                p_path, p_id = case["volumes"]["pvp"], pvpid
            ap_images.append(image_tensor(a_path, a_id, self.image_size))
            pvp_images.append(image_tensor(p_path, p_id, self.image_size))
            ref_ids.append(ref)
            ap_ids.append(apid)
            pvp_ids.append(pvpid)
        ap, pvp = torch.stack(ap_images), torch.stack(pvp_images)
        if augmented:
            ap, pvp = augment_pair(ap, pvp)
        return {
            "patient_id": case["patient_id"], "label": int(case["label"]),
            "ap": ap, "pvp": pvp,
            "reference_slice_ids": ref_ids,
            "ap_slice_ids": ap_ids, "pvp_slice_ids": pvp_ids,
            "sampling": sample, "source": "syn" if use_syn else ("real_aug" if augmented else self.group),
        }


def collate_bags(items):
    """Pad variable-length bags; padded slices never receive MIL attention."""
    n = len(items)
    max_k = max(len(i["reference_slice_ids"]) for i in items)
    _, channels, h, w = items[0]["ap"].shape
    ap = torch.zeros(n, max_k, channels, h, w)
    pvp = torch.zeros_like(ap)
    valid_mask = torch.zeros(n, max_k, dtype=torch.bool)
    for b, item in enumerate(items):
        k = len(item["reference_slice_ids"])
        ap[b, :k], pvp[b, :k], valid_mask[b, :k] = item["ap"], item["pvp"], True
    return {
        "ap": ap, "pvp": pvp, "valid_mask": valid_mask,
        "label": torch.tensor([i["label"] for i in items], dtype=torch.long),
        "records": [{
            "patient_id": i["patient_id"], "reference_slice_ids": i["reference_slice_ids"],
            "ap_slice_ids": i["ap_slice_ids"], "pvp_slice_ids": i["pvp_slice_ids"],
            "sampling": i["sampling"], "source": i["source"],
        } for i in items],
    }
