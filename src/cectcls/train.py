"""Patient-disjoint MIL training with source-paired training groups and Real validation."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import PatientBagDataset, BalancedSourceSampler, collate_bags
from .manifest import read_json, write_json
from .models import DualPhaseMIL


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, probabilities, n_classes):
    predicted = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_auc": None,
    }
    try:
        if len(np.unique(y_true)) == n_classes:
            auc = (roc_auc_score(y_true, probabilities[:, 1]) if n_classes == 2
                   else roc_auc_score(y_true, probabilities, multi_class="ovr",
                                      average="macro", labels=list(range(n_classes))))
            result["macro_auc"] = float(auc)
    except ValueError:
        pass
    return result


PRIMARY_TO_THREE = {"正常": 0, "肝囊肿": 1, "血管瘤": 1, "HCC": 2, "转移瘤": 2}


def summarize_three_class(y_true, probabilities, label_map):
    """Derive normal/benign/malignant from five-disease patient probabilities."""
    if set(label_map) != set(PRIMARY_TO_THREE):
        return None
    inverse = {int(value): name for name, value in label_map.items()}
    y = np.asarray([PRIMARY_TO_THREE[inverse[int(k)]] for k in y_true], dtype=int)
    new_probs = np.zeros((len(probabilities), 3), dtype=float)
    for name, index in label_map.items():
        new_probs[:, PRIMARY_TO_THREE[name]] += probabilities[:, int(index)]
    return compute_metrics(y, new_probs, 3), new_probs


def sqrt_class_weights(cases, n_classes, device, cap=3.0):
    """Training-patient-only sqrt inverse-frequency weighting, capped and normalized."""
    counts = np.bincount([int(row["label"]) for row in cases], minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError(f"At least one class missing from training patients: {counts.tolist()}")
    weights = np.minimum(cap, np.sqrt(counts.max() / counts.astype(float)))
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def evaluate_model(model, loader, device, n_classes, return_predictions=False, label_map=None):
    model.eval()
    ys, probs, records = [], [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["ap"].to(device), batch["pvp"].to(device),
                            batch["valid_mask"].to(device))
            probability = torch.softmax(outputs["logits"], dim=1).cpu().numpy()
            ys.extend(batch["label"].tolist())
            probs.extend(probability.tolist())
            for record, label, pr in zip(batch["records"], batch["label"].tolist(), probability):
                records.append({
                    "patient_id": record["patient_id"], "label": int(label),
                    "probabilities": pr.tolist(), "reference_slice_ids": record["reference_slice_ids"],
                    "ap_slice_ids": record["ap_slice_ids"], "pvp_slice_ids": record["pvp_slice_ids"],
                })
    metrics = compute_metrics(np.asarray(ys), np.asarray(probs), n_classes)
    if label_map is not None:
        aggregate = summarize_three_class(np.asarray(ys), np.asarray(probs), label_map)
        if aggregate is not None:
            three_metrics, three_probs = aggregate
            metrics["three_class"] = three_metrics
            for record, three_pr in zip(records, three_probs):
                record["three_class_probabilities"] = three_pr.tolist()
    return (metrics, records) if return_predictions else metrics


def make_loader(args, split, group, mode, shuffle=False):
    dataset = PatientBagDataset(
        args.manifest, split=split, group=group, mode=mode,
        selection=getattr(args, "selection", None),
        syn_index=getattr(args, "syn_index", None),
        png_index=getattr(args, "png_index", None) if mode == "selected" else None,
        seed=args.seed, image_size=args.image_size,
        uniform=args.uniform, random_count=args.random_count,
    )
    source_sampler = (
        BalancedSourceSampler(dataset, seed=args.seed)
        if shuffle and mode == "selected" and group in {"cect_real_aug", "cect_real_syn"}
        else None
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=shuffle and source_sampler is None, sampler=source_sampler,
                        num_workers=args.workers, collate_fn=collate_bags)
    return dataset, loader


def fit(args):
    seed_everything(args.seed)
    labels = read_json(args.labels)
    n_classes = len(labels)
    if n_classes < 2:
        raise ValueError("Classification requires >=2 classes")
    if args.stage == "selection" and args.group != "cect_real":
        raise ValueError("The selection MIL must be trained on real AP/PVP only")
    mode = "mixed" if args.stage == "selection" else "selected"
    train_set, train_loader = make_loader(args, "train", args.group, mode, shuffle=True)
    class_weight_mode = getattr(args, "class_weight", "sqrt")
    if class_weight_mode not in {"sqrt", "none"}:
        raise ValueError("class_weight must be sqrt or none")
    weights = (sqrt_class_weights(train_set.cases, n_classes, args.device)
               if class_weight_mode == "sqrt" else None)
    validation_group = "ncct" if args.group == "ncct" else "cect_real"
    _, val_loader = make_loader(args, "val", validation_group, mode)
    phase_mode = "ncct" if args.group == "ncct" else "dual"
    model = DualPhaseMIL(args.model, n_classes, pretrained=args.pretrained,
                         phase_mode=phase_mode, freeze_encoder=args.freeze_encoder,
                         encode_chunk=args.encode_chunk,
                         imagenet_norm=args.pretrained,
                         image_size=args.image_size).to(args.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                   lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_sha = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    best_score = float("-inf")
    history = []
    sampling_log = out / "training_sampled_slices.jsonl"
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        model.train()
        total_loss, examples = 0., 0
        with sampling_log.open("a" if epoch else "w", encoding="utf-8") as log:
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch["ap"].to(args.device), batch["pvp"].to(args.device),
                                batch["valid_mask"].to(args.device))
                labels_batch = batch["label"].to(args.device)
                loss = F.cross_entropy(outputs["logits"], labels_batch, weight=weights)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(labels_batch)
                examples += len(labels_batch)
                for rec in batch["records"]:
                    log.write(json.dumps({"epoch": epoch, "patient_id": rec["patient_id"],
                                          "source": rec["source"],
                                          "selected_ap_ids": rec["ap_slice_ids"],
                                          "selected_pvp_ids": rec["pvp_slice_ids"],
                                          "sampling": rec["sampling"]}, ensure_ascii=False) + "\n")
        val_metrics = evaluate_model(model, val_loader, args.device, n_classes, label_map=labels)
        score = val_metrics["macro_auc"]
        if score is None or not math.isfinite(score):
            score = val_metrics["macro_f1"]
        row = {"epoch": epoch, "loss": total_loss / max(1, examples), **val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score = score
            ckpt = {
                "state_dict": model.state_dict(), "model_name": args.model,
                "num_classes": n_classes, "phase_mode": phase_mode,
                "freeze_encoder": args.freeze_encoder, "encode_chunk": args.encode_chunk,
                "imagenet_norm": args.pretrained,
                "stage": args.stage, "group": args.group, "epoch": epoch,
                "label_map": labels, "class_weight_mode": class_weight_mode,
                "class_weights": weights.cpu().tolist() if weights is not None else None,
                "manifest_sha256": source_sha,
                "image_size": args.image_size,
            }
            torch.save(ckpt, out / "best.pt")
            write_json(out / "best_val.json", row)
    write_json(out / "history.json", history)
    return out / "best.pt"


def load_checkpoint(path, device):
    # Checkpoints are Python pickle-based; only load artifacts from trusted authors.
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = DualPhaseMIL(state["model_name"], state["num_classes"],
                         pretrained=False, phase_mode=state["phase_mode"],
                         freeze_encoder=state.get("freeze_encoder", False),
                         encode_chunk=state.get("encode_chunk", 16),
                         imagenet_norm=state.get("imagenet_norm", False),
                         image_size=state.get("image_size", 224))
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), state


def evaluate_checkpoint(args):
    model, state = load_checkpoint(args.checkpoint, args.device)
    group = "ncct" if state["phase_mode"] == "ncct" else "cect_real"
    mode = "selected" if state["stage"] == "selected" else "mixed"
    _, loader = make_loader(args, args.split, group, mode)
    metrics, predictions = evaluate_model(model, loader, args.device,
                                           state["num_classes"], return_predictions=True,
                                           label_map=state.get("label_map"))
    write_json(args.out, {"checkpoint": str(args.checkpoint), "split": args.split,
                          "evaluation_group": group, "metrics": metrics,
                          "predictions": predictions})
    return metrics
