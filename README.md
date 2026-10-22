# AP/PVP Paired CECT Classification

Patient-level dual-phase CT classification with mixed slice sampling, attention MIL, auditable slice IDs, and targeted NCCT-to-CECT synthesis requests.

**Research protocol**: Real AP/PVP training -> paired uniform/random sampling -> Gated Attention MIL -> freeze best Real checkpoint -> full-liver attention Top-16 -> NCCT local anatomical similarity search -> synthesis requests -> five paired training configurations (NCCT, REAL, SYN, REAL-AUG, REAL+SYN 1:1). No lesion-detector slices are inserted.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+, PyTorch, and image volumes in **NIfTI HU**. The AP/PVP registered volumes must be resampled onto the **same NCCT fixed-image voxel grid** before indexing; mismatch is an error, not silently interpreted as a slice correspondence.

## Input CSV

`cases.csv`: `patient_id,label,ncct_path,ap_path,pvp_path,liver_mask_path` (mask optional; relative paths resolved relative to cases.csv). `splits.csv`: `patient_id,split` where split is train/val/test. Assign all scans of a patient to one split. Labels are encoded in `labels.json`. Hold out test patients from both the generator's and classifier's model development.

## Workflow

```bash
cectcls prepare --cases data/cases.csv --splits data/splits.csv --out runs/class/manifests
cectcls train --manifest runs/class/manifests/cases.jsonl --labels runs/class/manifests/labels.json --stage selection --model dinov2s_mil --epochs 30 --out runs/class/selection
cectcls export-selection --manifest runs/class/manifests/cases.jsonl --checkpoint runs/class/selection/best.pt --top-k 16 --out runs/class/manifests/selected_bags.json
cectcls match-ncct --manifest runs/class/manifests/cases.jsonl --selection runs/class/manifests/selected_bags.json --radius-mm 15 --out runs/class/manifests/matched_bags.json
cectcls export-syn-requests --manifest runs/class/manifests/cases.jsonl --matched runs/class/manifests/matched_bags.json --out-root runs/class/syn --out runs/class/manifests/synthetic_requests.json
# External NCCT->AP/PVP generator: read synthetic_requests.json, write HU-valued float32 .npy images to the prescribed output paths.
cectcls register-syn --requests runs/class/manifests/synthetic_requests.json --out runs/class/manifests/syn_index.json
cectcls train --manifest runs/class/manifests/cases.jsonl --labels runs/class/manifests/labels.json --stage selected --selection runs/class/manifests/matched_bags.json --syn-index runs/class/manifests/syn_index.json --group cect_real_syn --model dinov2s_mil --out runs/class/real_syn
cectcls evaluate --manifest runs/class/manifests/cases.jsonl --labels runs/class/manifests/labels.json --selection runs/class/manifests/matched_bags.json --checkpoint runs/class/real_syn/best.pt --split test --out runs/class/real_syn/test.json
```

`train --stage selection` uses 16 uniform + 16 stochastic paired anatomical positions per training epoch. Validation uses deterministic positions. `export-selection` scores **all liver-containing paired slices with the same trained model** and writes 16 fixed anatomical positions per patient, separately including AP/PVP slice IDs and attention scores. Phase fusion precedes MIL attention.

`match-ncct` searches near the fixed-space reference plane with NMI, NCC and a physical-distance penalty, plus a nondecreasing sequence constraint. A similarity score alone is not evidence of anatomical correspondence; inspect `matching_status` and QC before synthesis. AP and PVP are synthesized from the **same matched NCCT slice**, while the original Real target IDs are kept unchanged.

Use `--model resnet18_mil|deit_mil|dinov2s_mil`, `--group ncct|cect_real|cect_syn|cect_real_aug|cect_real_syn`. For real_aug and real_syn, a balanced patient sampler assigns approximately half the patients to the original view and half to the augmented/synthetic view each epoch (exact 1:1 when the train patient count is even). Each patient contributes one example per epoch, keeping training steps comparable to REAL. Training/validation/test always split by patient. For contrast-enhanced groups validation and test inputs are Real CECT; NCCT group is evaluated on NCCT.

**Pretrained backbones**: set `--pretrained` to enable timm weights (may download); for offline experiments pre-download weights or supply a model-cache. Default is uninitialized weights; report this choice in the paper. The external synthesis implementation and clinical truth annotation are not included. The synthesis request JSON explicitly includes the NCCT NIfTI path, matched axial slice ID, target phase and expected output path. Matching statuses are provisional until checked against anatomical landmarks; do not treat a high correlation alone as validated alignment.

Tests: `pytest -q`.
