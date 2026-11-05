# Lesion-ROI Offline MIL V1 — Implementation Plan

> Branch: `feat/lesion-roi-offline-mil-v1`  
> Repository: `Devil-Jian/class`  
> Status: **Planning only — no training implementation merged**  
> Scope: CT lesion detector → patient-level AP/PVP ROI bag → REAL-supervised feature encoder adaptation → frozen offline feature extraction → REAL / SYN / REAL_AUG / REAL_SYN MIL.

## 1. Objective and hypotheses

Build a reproducible patient-level liver lesion classification pipeline. The same lesion ROI manifests and patient splits are used by ResNet18, DeiT-Tiny/16, DINOv2-S/14, with optional LFM-DL and LFM-Seq ViT-S/8 benchmarks. Feature extraction is decoupled from the downstream MIL optimizer.

Research questions:

1. Does detector-guided lesion ROI sampling improve classification compared with the repository's original full-liver slice bags?
2. Under fixed detector proposals, does REAL-supervised encoder adaptation improve over frozen off-the-shelf pretrained encoders?
3. Under **the same fold-specific, frozen REAL-adapted encoder**, do SYN, REAL_AUG, and REAL_SYN improve patient-level classification relative to REAL?

Keep the current `cectcls` pipeline executable and unchanged unless a backward-compatible adapter is necessary.

## 2. Existing repository baseline (audited)

- `src/cectcls/models.py`: `DualPhaseMIL`, shared 2D timm encoder, concatenated AP/PVP features, projection, gated attention, patient classifier.
- `src/cectcls/dataset.py`: patient-level paired slice bags (mixed, full, selected), REAL/SYN/REAL_AUG/REAL_SYN image regimes.
- `src/cectcls/selection.py`: Real-only selector MIL and Top-K whole-liver slice selection.
- `src/cectcls/train.py`: patient-level training/evaluation, class weights, Macro-AUC, 3-class derived metrics.
- **Correction**: current alias `deit_mil` resolves to `deit_small_patch16_224`, **not DeiT-Tiny**. Add a distinct `deit_tiny_patch16_224.fb_in1k` model ID instead of silently changing the old alias.
- **Correction**: DINOv2-S/14 and LFM ViT-S/8 do **not** share compatible full checkpoint shapes. LFM will be treated as a separate encoder family, not loaded into the DINOv2-S/14 model.
- Existing `PRIMARY_TO_THREE` currently handles only the current five-class label set; a new disease taxonomy must be explicitly defined and verified before any 6-class/other-benign experiments.

## 3. Data contracts and provenance

### 3.1 Input
- Frozen cohort manifest with patient ID, source/center, disease label, image modality (REAL vs SYN), AP/PVP volumes, liver mask if available, geometry/registration QC, and patient-level split/fold membership.
- **Frozen upstream lesion detector** checkpoint and config; predictions on REAL AP/PVP (or approved ROI manifest). Detector confidence, bbox and phase recorded.
- Optional annotated tumor mask for detector QC or separately identified oracle-ROI ablation. Never silently use ground-truth boxes at test time.
- SYN volumes mapped to the same patient, with explicit NCCT/CECT or target grid transforms where needed.

### 3.2 Patient bag
- One bag per patient and image regime (not one folder per patch).
- Instances: unique lesion identifier, source image, phase, z index and physical z, bbox in original pixel coordinates, confidence, ROI kind (`lesion` or `liver_background`), detector/model hash, extraction settings.
- Preset V1 defaults (tune on development folds only): bbox expansion 20%, minimum 96-pixel ROI, up to 32 instances/patient, preserve all lesions by stratifying available slices, optional background liver ROIs for **both** tumor and normal cohorts.
- **Normal/zero-detection cases** must still have usable liver-parenchyma instances; no empty bag or trivial has-detection shortcut. Record false-negative/zero-detection rates.
- AP/PVP matching: use verified registration or lesion-level match with explicit QC. Same pixel coordinates are **not** assumed equivalent if phase geometry differs.
- For controlled REAL–SYN experiments, use a *fixed Real-derived ROI manifest* mapped into each image domain so image type does not change which lesion was sampled. Document this as an ROI-controlled analysis, not end-to-end detection.
- Preserve 3D lesion identity: correlate 2D proposals across neighboring z slices (3D CC on masks or detector association). Record unmatched, overlapping and low-confidence lesions.

### 3.3 Instance images
- Read original HU when available and apply each encoder's specified window, image size, and normalization *before* encoder inference.
- ImageNet encoders: 3-channel replicated CT with the original pretraining normalization; verify transform against checkpoint.
- LFM: standalone 1-channel ViT-S/8 with its own checkpoint loader; confirm checkpoint state-dict keys and preprocessing before use.
- Do not reconstruct a different HU window from already clipped 8-bit PNG.
- REAL_AUG: perform and cache seeded **image-space** augmentations (not post-hoc geometric transformations of feature vectors).
- Multiple image regimes from the same patient always retain the original fold membership.

## 4. Encoders and training arms

| ID | Encoder | Initial weights | Embedding | Input (default) | Role |
|---|---|---|---:|---|---|
| E1 | ResNet18 | ImageNet-1K | 512 | 224² RGB-replicated CT | CNN baseline |
| E2 | DeiT-Tiny/16 | ImageNet-1K | 192 | 224² RGB-replicated CT | lightweight ViT baseline |
| E3 | DINOv2-S/14 | official DINOv2 | 384 | 224² RGB-replicated CT | main ViT baseline |
| E4 (optional) | LFM-DL ViT-S/8 | DeepLesion self-supervision | 384 | 128² single-channel CT | CT baseline |
| E5 (optional) | LFM-Seq ViT-S/8 | DeepLesion → HCC self-supervision | 384 | 128² single-channel CT | HCC-adapted CT baseline |

For each encoder, benchmark **frozen pretrained** vs **REAL patient-level MIL-supervised fine-tuned**:

- Stage A: only training patients from the relevant fold, REAL AP/PVP ROI bag, dual-phase gated-MIL head; optimize patient-level labels. Freeze/full or partial unfreeze controls by architecture (ResNet18: candidate full; DeiT-Tiny: staged unfreeze/full; DINOv2: final 2–4 blocks, optionally LoRA later). Explicitly test `model.eval()` for frozen encoders and restore training mode only where appropriate.
- Choose training epoch/hyperparameters using that fold's validation patients only; retain the chosen encoder **without the temporary MIL head**.
- Stage B: freeze the encoder; extract all required train/validation/test REAL/SYN/AUG image features **for that fold**.
- Stage C: train a **fresh** offline-MIL head per regime and seed using the corresponding cached features.
- Pretrained-only + SYN-MIL is a separate control if making claims about performance without real-label supervision. A REAL-adapted encoder used in SYN-MIL is **not** a SYN-only end-to-end classifier.

## 5. Four independently addressable offline feature datasets

| Dataset ID | Patient-level feature-bag content | Source images | Fixed evaluation |
|---|---|---|---|
| `real` | One REAL AP/PVP bag per patient | Real CECT | REAL val/test |
| `syn` | One SYN AP/PVP bag per patient | Synthetic CECT | REAL val/test |
| `real_aug` | REAL and independently image-augmented REAL bags for each patient | Real CECT + seeded augmentations | REAL val/test |
| `real_syn` | REAL and SYN bags for each patient | Real CECT + matched synthetic CECT | REAL val/test |

**Binding requirement:** these are FOUR named, separately indexed datasets, not just four MIL runtime switches. Each dataset MUST be materialized as a complete `dataset_manifest.jsonl` of patient/view-to-feature references and validated independently. `real_aug` and `real_syn` contain multiple views of the same patient under the SAME patient label; the MIL training loader can sample views in balanced or full mode. They are not new patients and must not be treated as independent patient-level observations. Source feature tensors may be deduplicated on disk using immutable references, but each of the four dataset manifests must be standalone usable with explicit path resolution and checksums.

- Data views must not change underlying patient counts or fold assignments.
- Two mixture schedules: **balanced** (one view per patient/epoch, deterministic alternating or seeded sampling) as the primary controlled experiment, and **full** (both views, doubled training inputs) as sensitivity analysis.
- For all comparisons, freeze the SAME fold-specific REAL-adapted encoder; training recipe, ROI manifest, MIL architecture and validation protocol held constant.
- Explicitly track missing SYN/phase cases and exclusions. Do not silently skip patients in one arm while retaining them in another; create a matched-cohort analysis set and audit it.

## 6. Splits, leakage controls and statistics

**Mandatory:**
- Patient-level stratified 5-fold CV; all visits, phases, real/synthetic counterparts, augmentation variants and ROIs for each patient remain in one fold. Guard against duplicate raw patient IDs spanning centers.
- In each fold, encoder fine-tuning, MIL head fitting, normalization statistics and hyperparameter search use training patients only, with validation used for model selection. Held-out test fold is evaluated **once**.
- Upstream detector: if trained using tumor labels from the same evaluation cohort, produce out-of-fold detector predictions for validation/test patients; otherwise use an independent externally trained detector. Document detector train IDs and assert no leakage.
- If a single held-out external test center exists, never use it for encoder/detector/MIL training, selection, or threshold tuning. Training/validation and external testing are separate from inner CV.
- Real and synthetic data from the same patient cannot cross splits; no manual lesion annotations provided as model inference inputs for the primary test.
- No selecting Top-K with a model trained on held-out patient labels.
- A fair frozen-encoder comparison must report the extra REAL-supervised encoder adaptation stage.
- Independent patient or site bootstrap CIs; patient-level predictions only; Macro-AUC primary, Macro-F1, balanced accuracy, class-wise AUC/sensitivity, confusion matrix, 3-class metrics secondary. For paired regimes, use paired patient resampling for performance differences.
- Use continuous summed/aggregated disease probabilities for 3-class AUC, not argmax-derived one-hot scores.

## 7. Offline data format

Default root:

```text
outputs/lesion_mil_v1/
  roi_manifests/
    folds.json
    fold_{k}/patients.jsonl
    fold_{k}/instances.jsonl
    fold_{k}/qc.json
  encoders/
    fold_{k}/{encoder}/best_encoder.safetensors
    fold_{k}/{encoder}/finetune_metrics.json
  features/
    fold_{k}/{encoder}/
      base_real/{patient_id}.h5
      base_syn/{patient_id}.h5
      base_real_aug/{patient_id}_seed_{seed}.h5
      datasets/
        real/dataset_manifest.jsonl
        syn/dataset_manifest.jsonl
        real_aug/dataset_manifest.jsonl
        real_syn/dataset_manifest.jsonl
        real/dataset_qc.json
        syn/dataset_qc.json
        real_aug/dataset_qc.json
        real_syn/dataset_qc.json
  mil/
    fold_{k}/{encoder}/{regime}/{seed}/best.pt
    fold_{k}/{encoder}/{regime}/{seed}/predictions.jsonl
  results/
    per_fold.csv
    pooled_predictions.csv
    metrics_with_95ci.json
  qc/
    roi_overlays/
    duplicate_patient_ids.json
    split_integrity.json
```

Source-view feature HDF5 (one patient per encoder/fold/source view; each of the four datasets references one or more source views):

- `ap_features: [N,D]`, `pvp_features: [N,D]` (both float16 on disk; float32 in MIL).
- `slice_ids: [N]`, `lesion_ids: [N]`, `roi_types: [N]`, `ap_boxes: [N,4]`, `pvp_boxes: [N,4]`.
- `meta` serialized JSON with patient/source/label/split, geometry, ROI/detector hash, encoder name/checkpoint SHA256, window/normalization/resize, augmentation seed, input checksum, software version.
- If a phase has unmatched ROIs, fail or use an explicitly tested missing-phase fusion path; never fabricate a matching counterpart.
- Cache keys include fold, encoder checkpoint hash, ROI manifest hash, regime and transform/augmentation recipe to avoid silent stale-cache reuse.

## 8. Implementation phases and acceptance criteria

### P0 — Contracts and tests (first)
**Files:** `src/cectcls/offline/contracts.py`, `src/cectcls/offline/validation.py`, tests.

- Define versioned patient/instance/feature schemas, fold manifest and strict input validation.
- Implement zero-duplicate ID check, patient split integrity, ROI bounds/phase matching, checksum validation.
- Acceptance: small synthetic fixtures cover missing CT, missing SYN, empty detection, mismatched geometry, wrong class labels, duplicate IDs, test leakage.

### P1 — Detector predictions → patient bags
**Files:** `offline/lesion_bags.py`, `offline/roi_extract.py`, `scripts/build_lesion_bags.py`.

- Read detector outputs/optional approved masks, match AP/PVP, group per lesion across z, sample lesion + matched liver background patches.
- Export per-fold instance manifests and QC overlays, count tumor/normal/empty/false negatives, preserve provenance.
- Acceptance: deterministic ROI manifest; every included patient has nonempty bag; no input masked annotation used to make primary inference ROI unless explicitly declared oracle ablation; visualization checked on a small hand-reviewed pilot.

### P2 — Encoder registry & REAL fine-tuning
**Files:** `offline/encoders.py`, `offline/finetune.py`, `scripts/finetune_real_encoder.py`.

- Implement ResNet18, explicitly named DeiT-Tiny, DINOv2-S/14; optional LFM-DL/LFM-Seq later.
- Handle real CT HU/channel/resize/norm transforms per encoder; patient-level loss and dual-phase MIL.
- Acceptance: pretrained weights load with audited missing/unexpected keys; 1-bag overfit smoke test; no patient overlap; separately saved fold-specific encoder weights.

### P3 — Frozen offline features
**Files:** `offline/features.py`, `scripts/extract_offline_features.py`.

- Construct four explicit offline feature datasets: `real`, `syn`, `real_aug`, `real_syn`. For EACH dataset emit its own complete `dataset_manifest.jsonl` and QC report. Extract REAL, SYN, and seeded image-augmented REAL features with frozen fold-specific encoders. `real_aug` manifests point to REAL + AUG views; `real_syn` manifests point to REAL + SYN views. Deduplicate physical source-view HDF5 files when possible; never collapse the four dataset identities into implicit training flags.
- Acceptance: all four datasets exist as independently loadable validated manifests; cached [N,D] features match online frozen encoder output within tolerance; two phases aligned; same patients and labels are enforced; augmentation provenance and feature references are checked; resumes after interruption.

### P4 — Offline MIL training & evaluation
**Files:** `offline/mil.py`, `offline/train_mil.py`, `scripts/train_offline_mil.py`.

- Reuse original dual-phase projection + gated-attention design on cached embeddings, with mean-pooling ablation.
- Handle variable N/padding and lesion/background flags; train independently from four dataset manifests (`real`, `syn`, `real_aug`, `real_syn`) and real-only validation/test.
- Acceptance: CPU/GPU smoke; deterministic seed runs; same cohort and encoder across four regimes; no encoder gradient or imaging I/O inside MIL epoch; correct patient-level metrics and 95% CI.

### P5 — Experiment matrix, audits and documentation
**Files:** `scripts/run_offline_matrix.py`, `scripts/audit_offline_results.py`, `docs/lesion_roi_offline_mil.md`.

- Run E1–E3 × {pretrained only, real-adapted} × 4 regimes × folds. Optional E4–E5 after checkpoint validation.
- Compare original whole-liver bag vs ROI bag at fixed encoder/MIL, pretrained vs REAL-tuned encoders, and 4 data regimes under fixed encoder.
- Acceptance: one-command pilot then reproducible fold matrix; patient-level bootstrap CI and paired regime deltas; publishable CSV tables; current `cectcls train` unchanged.

## 9. Proposed CLI

```bash
# No data modifications in planning stage. Commands below are future interfaces.
python scripts/build_lesion_bags.py --cases ... --detector-predictions ... --splits ... --out ...
python scripts/finetune_real_encoder.py --bags ... --encoder dinov2_s14 --fold 0 --out ...
python scripts/extract_offline_features.py --bags ... --encoder-weights ... --regimes real,syn,real_aug,real_syn --out ...
python scripts/train_offline_mil.py --features ... --regime real_syn --mix balanced --fold 0 --out ...
python scripts/run_offline_matrix.py --config configs/lesion_mil_v1.yaml --pilot
```

## 10. Scope exclusions / decisions before full training

- No change to the existing 1,183-patient locked main-five cohort or its split metadata without an explicit approved new cohort version.
- Disease label sets (five-class vs three-class vs extended six-class) remain dataset configuration, never inferred from folder names.
- No promised performance threshold or claim that LFM/REAL_AUG/SYN must outperform baseline.
- Do not treat detector-conditioned ROI MIL as fully end-to-end classification at inference unless the detector is actually run on evaluation scans.
- Confirm detector prediction file schema, source REAL/SYN registration details and the authoritative cohort/folds before integrating production data. Provide adapters rather than hardcoding private local paths.
- **First deliverable of this branch is this plan**; P0–P5 are implementation follow-ups.
