"""Patient-disjoint MIL training with source-paired training groups and Real validation."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    """Derive normal/benign/malignant directly from five-disease patient classification results.
    
    Clinical logic:
    1. A patient's 5-class predicted diagnosis (argmax) is directly mapped to 3-class:
       - 正常 -> 正常
       - 肝囊肿, 血管瘤 -> 良性
       - HCC, 转移瘤 -> 恶性
    2. Metrics (Accuracy, Balanced Acc, F1, Discrete AUC) reflect whether the model's
       primary diagnostic decision got the clinical super-category correct.
    """
    if set(label_map) != set(PRIMARY_TO_THREE):
        return None
    inverse = {int(value): name for name, value in label_map.items()}
    y_true_3 = np.asarray([PRIMARY_TO_THREE[inverse[int(k)]] for k in y_true], dtype=int)
    
    # 1. 直接获取 5 分类的最终预测决策 (argmax)
    pred_5 = probabilities.argmax(axis=1)
    # 2. 将最终分类结果直接映射为 3 分类标签 (正常=0, 良性=1, 恶性=2)
    pred_3 = np.asarray([PRIMARY_TO_THREE[inverse[int(k)]] for k in pred_5], dtype=int)
    
    # 3. 构造 3 分类 One-Hot 决策矩阵，直接由分类结果计算指标
    n_samples = len(y_true)
    pred_3_one_hot = np.zeros((n_samples, 3), dtype=float)
    pred_3_one_hot[np.arange(n_samples), pred_3] = 1.0
    
    metrics = compute_metrics(y_true_3, pred_3_one_hot, 3)
    
    # 保留连续加和概率以供需要时输出三分类概率分布
    soft_probs = np.zeros((n_samples, 3), dtype=float)
    for name, index in label_map.items():
        soft_probs[:, PRIMARY_TO_THREE[name]] += probabilities[:, int(index)]
        
    return metrics, soft_probs


def sqrt_class_weights(cases, n_classes, device, cap=3.0):
    """Training-patient-only sqrt inverse-frequency weighting, capped and normalized."""
    counts = np.bincount([int(row["label"]) for row in cases], minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError(f"At least one class missing from training patients: {counts.tolist()}")
    weights = np.minimum(cap, np.sqrt(counts.max() / counts.astype(float)))
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def evaluate_model(model, loader, device, n_classes, return_predictions=False, label_map=None, desc=None):
    model.eval()
    ys, probs, records = [], [], []
    loader_iter = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False) if desc else loader
    with torch.no_grad():
        for batch in loader_iter:
            outputs = model(batch["ap"].to(device, non_blocking=True),
                            batch["pvp"].to(device, non_blocking=True),
                            batch["valid_mask"].to(device, non_blocking=True))
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
    mix_mode = getattr(args, "mix_mode", "full")
    source_sampler = (
        BalancedSourceSampler(dataset, seed=args.seed)
        if shuffle and mode == "selected" and group in {"cect_real_aug", "cect_real_syn"} and mix_mode == "balanced"
        else None
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle and source_sampler is None,
        "sampler": source_sampler,
        "num_workers": args.workers,
        "collate_fn": collate_bags,
        "pin_memory": (args.device != "cpu" and torch.cuda.is_available()),
    }
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    return dataset, loader


def fit(args):
    seed_everything(args.seed)
    if getattr(args, "png_index", None) and os.path.isfile(args.png_index):
        print(f">> [DataLoader] Using PNG dataset from {args.png_index} (Zero-I/O Fast Mode)")
    else:
        h5_root = os.environ.get("H5_ROOT", "").strip()
        if not h5_root and os.path.isdir("/dataset/h5_cache"):
            h5_root = "/dataset/h5_cache"
            os.environ["H5_ROOT"] = h5_root
        if h5_root and os.path.isdir(h5_root):
            print(f">> [DataLoader] Using H5 cache at {h5_root} (High-Speed I/O Mode enabled)")
        else:
            print(f">> [DataLoader] Reading raw NIfTI files.")
    labels = read_json(args.labels)
    n_classes = len(labels)
    if n_classes < 2:
        raise ValueError("Classification requires >=2 classes")
    print(f">> [Experiment] Model: {args.model} | Group: {args.group} | Output: {args.out}")
    if args.stage == "selection" and args.group != "cect_real":
        raise ValueError("The selection MIL must be trained on real AP/PVP only")
    mode = "mixed" if args.stage == "selection" else "selected"
    train_set, train_loader = make_loader(args, "train", args.group, mode, shuffle=True)
    if args.group in {"cect_real_syn", "cect_real_aug"}:
        mix_mode = getattr(args, "mix_mode", "full")
        if mix_mode == "full":
            print(f">> [Dataset] 1:1 Full Mixture Mode: {len(train_set)} samples/epoch ({len(train_set.cases)} Real + {len(train_set.cases)} {args.group.split('_')[-1].upper()})")
        else:
            print(f">> [Dataset] 50/50 Subsampled Balanced Mode: {len(train_set.cases)} samples/epoch")
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
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{args.epochs:02d} [Train]",
                         dynamic_ncols=True, leave=False)
        with sampling_log.open("a" if epoch else "w", encoding="utf-8") as log:
            for batch in train_bar:
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch["ap"].to(args.device, non_blocking=True),
                                batch["pvp"].to(args.device, non_blocking=True),
                                batch["valid_mask"].to(args.device, non_blocking=True))
                labels_batch = batch["label"].to(args.device, non_blocking=True)
                loss = F.cross_entropy(outputs["logits"], labels_batch, weight=weights)
                loss.backward()
                optimizer.step()
                batch_loss = float(loss.item())
                total_loss += batch_loss * len(labels_batch)
                examples += len(labels_batch)
                train_bar.set_postfix({"loss": f"{batch_loss:.4f}",
                                       "avg_loss": f"{total_loss / max(1, examples):.4f}"})
                for rec in batch["records"]:
                    log.write(json.dumps({"epoch": epoch, "patient_id": rec["patient_id"],
                                          "source": rec["source"],
                                          "selected_ap_ids": rec["ap_slice_ids"],
                                          "selected_pvp_ids": rec["pvp_slice_ids"],
                                          "sampling": rec["sampling"]}, ensure_ascii=False) + "\n")
        val_metrics = evaluate_model(model, val_loader, args.device, n_classes, label_map=labels,
                                     desc=f"Epoch {epoch + 1:02d}/{args.epochs:02d} [Val]")
        score = val_metrics["macro_auc"]
        if score is None or not math.isfinite(score):
            score = val_metrics["macro_f1"]
        row = {"epoch": epoch, "loss": total_loss / max(1, examples), **val_metrics}
        history.append(row)

        auc_str = f"{row['macro_auc']:.4f}" if row.get("macro_auc") is not None else "N/A"
        three_acc = row.get("three_class", {}).get("accuracy")
        three_acc_str = f"{three_acc:.4f}" if three_acc is not None else "N/A"
        three_auc = row.get("three_class", {}).get("macro_auc")
        three_str = f"{three_auc:.4f}" if three_auc is not None else "N/A"
        is_best = score > best_score
        marker = " ★ [New Best]" if is_best else ""
        print(f"Epoch {epoch + 1:02d}/{args.epochs:02d} | "
              f"Train Loss: {row['loss']:.4f} | "
              f"Val Acc: {row['accuracy']:.4f} (3-Class: {three_acc_str}) | "
              f"Macro-AUC: {auc_str} | "
              f"3-Class AUC: {three_str}{marker}", flush=True)

        if is_best:
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

    best_path = out / "best.pt"
    if best_path.exists():
        try:
            import copy
            test_args = copy.copy(args)
            test_args.checkpoint = best_path
            test_args.split = "test"
            test_args.out = out / "test_results.json"
            print(f"\n>> [Evaluation] Automatically evaluating best model on [TEST] split...")
            test_metrics = evaluate_checkpoint(test_args)
            t_acc = test_metrics["accuracy"]
            t_3acc = test_metrics.get("three_class", {}).get("accuracy")
            t_3acc_str = f"{t_3acc:.4f}" if t_3acc is not None else "N/A"
            t_auc = test_metrics.get("macro_auc")
            t_auc_str = f"{t_auc:.4f}" if t_auc is not None else "N/A"
            t_3auc = test_metrics.get("three_class", {}).get("macro_auc")
            t_3auc_str = f"{t_3auc:.4f}" if t_3auc is not None else "N/A"
            print(f">> [TEST RESULTS] Acc: {t_acc:.4f} (3-Class: {t_3acc_str}) | "
                  f"Macro-AUC: {t_auc_str} | 3-Class AUC: {t_3auc_str}\n", flush=True)
        except Exception as e:
            print(f">> [Evaluation Warning] Automatic test evaluation skipped: {e}\n", flush=True)

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
                                           label_map=state.get("label_map"),
                                           desc=f"Evaluating [{args.split}]")
    write_json(args.out, {"checkpoint": str(args.checkpoint), "split": args.split,
                          "evaluation_group": group, "metrics": metrics,
                          "predictions": predictions})
    return metrics
