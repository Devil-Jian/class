# CECT Classification — AP/PVP paired Attention-MIL

Classification pipeline for registered liver CT. **No tumor-mask/lesion-detector slices are added**. Real CECT AP/PVP are jointly used to train the same task-related MIL model that later exports its Top-16 influential *anatomical positions*. NCCT matching and synthetic contrast generation follow selection.

## 1. Input data

Expected directory on the GPU server:

```text
/dataset/dataset/gx_all_registered/
  A0_1/
    ncct_zcrop.nii.gz           # fixed NCCT, HU
    ap_registered.nii.gz        # registered AP, HU
    pvp_registered.nii.gz       # registered PVP, HU
    ncct_mask_zcrop.nii.gz      # MUST be verified to mean liver mask
```

Metadata CSV: `GX_metadata_all.csv`. The cohort scanner uses `patient_id`, `分类（暂定）` (`是`), `三分类标签` (default) or `主类别`, and `配准等级` (default GOLD/SILVER). Duplicate patients with conflicting classification labels are excluded. **REJECT/unregistered records are not automatically sent into the model.** A valid image file and mask must exist unless you explicitly use `--allow-missing-mask`. Actual NIfTI **shape AND affine grid** are strictly checked at `prepare`. 512x512 in-plane size alone does not prove grid alignment.

## 2. Install

Python >=3.10, PyTorch, CUDA recommended:

```bash
git clone https://github.com/Devil-Jian/class.git
cd class
pip install -e ".[dev]"
pytest -q
```

## 3. Full training sequence

Copy the uploaded CSV onto the training server first, e.g. to `/dataset/dataset/GX_metadata_all.csv`. Customize `CSV` below to the actual location.

```bash
export CSV=/dataset/dataset/GX_metadata_all.csv
export ROOT=/dataset/dataset/gx_all_registered
export WORK=/dataset/downstream_classification
mkdir -p "$WORK"
```

### Step A — Create patient manifests and splits (CPU)

```bash
cectcls prepare-cohort \
  --metadata "$CSV" \
  --image-root "$ROOT" \
  --label-column 三分类标签 \
  --grades GOLD SILVER \
  --out "$WORK/cohort"

cat "$WORK/cohort/cohort_report.json"

cectcls prepare \
  --cases "$WORK/cohort/cases.csv" \
  --splits "$WORK/cohort/splits.csv" \
  --out "$WORK/manifests"
```

Artifacts: `cohort/cases.csv`, `cohort/splits.csv`, `cohort/cohort_report.json`; `manifests/cases.jsonl`, `labels.json`, `qc.json`. The 70/15/15 split is patient-level and stratified by label. If a class contains too few patients, the split report warns that a separate test estimate is unavailable. **Review label distribution and all exclusions before training.** For a secondary disease-type experiment use `--label-column 主类别`, with a separate `--out` to prevent overwrites.

### Step B — Train Real AP+PVP Attention-MIL for task-related selection (GPU)

```bash
cectcls train \
  --manifest "$WORK/manifests/cases.jsonl" \
  --labels "$WORK/manifests/labels.json" \
  --stage selection \
  --group cect_real \
  --model dinov2s_mil \
  --uniform 16 --random-count 16 \
  --image-size 224 --batch-size 2 \
  --pretrained --freeze-encoder \
  --epochs 30 --device cuda \
  --out "$WORK/selection"
```

Every epoch: 16 uniformly spaced + 16 random, same physical positions for AP and PVP. Outputs: `selection/best.pt`, validation results, and `training_sampled_slices.jsonl` with separate AP/PVP slice IDs. **The selection-stage model consumes full HU NIfTI, windowed to [-160,240] on loading.** No separate selector model is trained.

### Step C — Use the frozen Real classification MIL to export Top-16 (GPU)

```bash
cectcls export-selection \
  --manifest "$WORK/manifests/cases.jsonl" \
  --checkpoint "$WORK/selection/best.pt" \
  --top-k 16 --device cuda \
  --out "$WORK/manifests/selected_bags.json"
```

This scores all valid liver-containing registered paired slices (not merely the epoch's sampled 32). Select 16 paired positions. Stores `reference_slice_id`, `ap_registered_slice_id`, `pvp_registered_slice_id`, attention score and patient split. Selection model is trained on train patients and tuned on val patients only; never trained on held-out test patients.

### Step D — Local NCCT slice matching (CPU)

```bash
cectcls match-ncct \
  --manifest "$WORK/manifests/cases.jsonl" \
  --selection "$WORK/manifests/selected_bags.json" \
  --radius-mm 15 \
  --out "$WORK/manifests/matched_bags.json"
```

Within NCCT fixed-space neighborhoods, score candidate planes using liver-ROI NMI, NCC, z-distance and a monotonic dynamic-programming constraint. One matched NCCT location yields both synthetic AP and PVP. Check match status and anatomy; **similarity metrics do not establish anatomical ground truth**. Never silently remove mismatch patients from the locked test set.

### Step E — Targeted synthesis: TRAIN patients only

```bash
cectcls export-syn-requests \
  --manifest "$WORK/manifests/cases.jsonl" \
  --matched "$WORK/manifests/matched_bags.json" \
  --split train \
  --out-root "$WORK/synthetic_hu" \
  --out "$WORK/manifests/synthetic_requests.json"
```

The external NCCT->CECT generator (not included here) reads `synthetic_requests.json`. Every request supplies `ncct_path`, `matched_ncct_slice_id`, `target_phase` (ap/pvp), `reference_slice_id`, and `output_path`. It writes **2D finite float32 HU `.npy` files**, oriented as rows=y, columns=x. Then:

```bash
cectcls register-syn \
  --requests "$WORK/manifests/synthetic_requests.json" \
  --out "$WORK/manifests/syn_index.json"
```

Do not generate test-patient Syn data for the main **Real-test** training-utility study.

### Step F — Export final selected PNGs (CPU)

```bash
cectcls export-png \
  --manifest "$WORK/manifests/cases.jsonl" \
  --selection "$WORK/manifests/matched_bags.json" \
  --syn-index "$WORK/manifests/syn_index.json" \
  --out "$WORK/png"
```

Stores **lossless 8-bit grayscale PNG** with fixed HU window [-160,240] (PNG is not HU). Structure:

```text
$WORK/png/
  real/ap/<patient_id>/ref_0042.png
  real/pvp/<patient_id>/ref_0042.png
  ncct/ncct/<patient_id>/ref_0042.png
  syn/ap/<patient_id>/ref_0042.png
  syn/pvp/<patient_id>/ref_0042.png
  png_index.json
  png_export_summary.json
```

The `ref_0042` filename names the target anatomical reference plane, **not** the matched NCCT index. True source slice IDs remain in `matched_bags.json`. Real + NCCT PNGs are exported for train/val/test; synthetic PNGs are exported **for train only**.

### Step G — Five formal PNG training groups (GPU)

Quick REAL group smoke test:

```bash
cectcls train \
  --manifest "$WORK/manifests/cases.jsonl" \
  --labels "$WORK/manifests/labels.json" \
  --stage selected --selection "$WORK/manifests/matched_bags.json" \
  --png-index "$WORK/png/png_index.json" \
  --model resnet18_mil --group cect_real \
  --pretrained --freeze-encoder --epochs 3 --batch-size 2 \
  --device cuda --out "$WORK/experiments/smoke_real"
```

Then train all 3 backbones x 5 groups = **15 formal configurations** (not including initial selection model):

```bash
cectcls train-grid \
  --manifest "$WORK/manifests/cases.jsonl" \
  --labels "$WORK/manifests/labels.json" \
  --selection "$WORK/manifests/matched_bags.json" \
  --png-index "$WORK/png/png_index.json" \
  --pretrained --freeze-encoder --epochs 30 --batch-size 2 \
  --device cuda --out "$WORK/experiments"
```

Data groups:
- `ncct` — NCCT, trained and tested on NCCT.
- `cect_real` — REAL AP+PVP; trained/tested on Real CECT.
- `cect_syn` — SYN AP+PVP; trained on Syn and evaluated on Real.
- `cect_real_aug` — REAL-AUG: approximately 1:1 Real : affine-augmented Real training views; Real evaluation.
- `cect_real_syn` — REAL+SYN: approximately 1:1 Real : Syn paired training views; Real evaluation.

The balanced-source sampler keeps **one example per training patient per epoch**, making optimizer steps comparable rather than doubling steps for augmentation groups. No geometric augmentation is used in val/test.

### Step H — Independent real-image test evaluation

```bash
cectcls evaluate \
  --manifest "$WORK/manifests/cases.jsonl" \
  --labels "$WORK/manifests/labels.json" \
  --selection "$WORK/manifests/matched_bags.json" \
  --png-index "$WORK/png/png_index.json" \
  --checkpoint "$WORK/experiments/dinov2s_mil/cect_syn/best.pt" \
  --split test --device cuda \
  --out "$WORK/experiments/dinov2s_mil/cect_syn/test.json"
```

Outputs include Macro-AUC (when evaluable), Macro-F1, balanced accuracy, and patient-level probabilities with selected AP/PVP IDs.

## 4. Important experiment boundaries

- Using *Real CECT attention* to select locations makes this a **CECT-guided paired-slice training-utility evaluation**, not an end-to-end NCCT-only clinical selection workflow.
- PNG quantizes [-160,240] HU to 256 levels. Keep source NIfTI in HU for matching and generation.
- `ncct_mask_zcrop.nii.gz` must be verified to represent **liver anatomy**, not a tumor-only mask. Otherwise `--allow-missing-mask` disables liver-mask sampling (all axial planes become candidates) until a true liver mask is supplied.
- The image-generation model and downstream held-out test set must be disjoint at patient level. Patients from generation training must not be represented as an independent synthesis holdout.
- `--pretrained` requests timm weights; allow downloads or pre-cache in restricted environments. Without the flag, encoders are randomly initialized.
- For clinical datasets, inspect `cohort_report.json`, split label counts, matching QC, and selected slice distributions before the full run.
