"""Export Top-K anatomical positions from the *same trained* Real CECT MIL model."""
from __future__ import annotations

from pathlib import Path

import torch

from .dataset import PatientBagDataset
from .manifest import read_jsonl, write_json
from .sampling import topk_attention
from .train import load_checkpoint


def export_selection(manifest, checkpoint, out, k=16, device="cpu", image_size=None):
    model, state = load_checkpoint(checkpoint, device)
    if state["stage"] != "selection" or state["group"] != "cect_real":
        raise ValueError("Select only with a frozen Real AP/PVP selection-stage checkpoint")
    size = image_size or state.get("image_size", 224)
    records = read_jsonl(manifest)
    by_split = {
        split: PatientBagDataset(manifest, split=split, group="cect_real",
                                 mode="full", image_size=size)
        for split in ("train", "val", "test") if any(r["split"] == split for r in records)
    }
    selected = []
    with torch.no_grad():
        for split, dataset in by_split.items():
            for j in range(len(dataset)):
                row = dataset[j]
                output = model(row["ap"][None].to(device), row["pvp"][None].to(device))
                scores = output["attention_scores"][0].cpu().tolist()
                chosen = topk_attention(row["reference_slice_ids"], scores, k)
                case = dataset.cases[j]
                selected.append({
                    "patient_id": row["patient_id"], "split": split,
                    "selection_method": "frozen_real_dual_phase_attention",
                    "selection_checkpoint": str(Path(checkpoint).resolve()),
                    "top_k": k,
                    "selected_positions": [{
                        "rank": rank, "reference_slice_id": slice_id,
                        "ap_registered_slice_id": slice_id, "pvp_registered_slice_id": slice_id,
                        "reference_z_mm": case["reference_z_mm"][str(slice_id)],
                        "attention_score": score,
                    } for rank, (slice_id, score) in enumerate(chosen, 1)],
                })
    write_json(out, selected)
    return selected
