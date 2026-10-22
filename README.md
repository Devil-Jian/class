# CECT Classification — AP/PVP paired Attention-MIL

Classification pipeline for registered liver CT. **No tumor-mask/lesion-detector slices are added**. Real CECT AP/PVP are jointly used to train the same task-related MIL model that later exports its Top-16 influential *anatomical positions*. NCCT matching and synthetic contrast generation follow selection.

## 1. Input data

Expected directory on the GPU server:

```text
/win_cect/shared/gx_all_registered/
  A0_1/
    ncct_zcrop.nii.gz           # fixed NCCT, HU
    ap_registered.nii.gz        # registered AP, HU
    pvp_registered.nii.gz       # registered PVP, HU
    ncct_mask_zcrop.nii.gz      # MUST be verified to mean liver mask
```

Metadata CSV: `GX_metadata_all.csv`. The cohort scanner uses `patient_id`, `分类（暂定）` (`是`), `三分类标签` (default) or `主类别`, and `配准等级` (default GOLD/SILVER/REJECT for classification; REJECT requires extra cross-phase matching QC). Duplicate patients with conflicting classification labels are excluded. **REJECT is eligible for REAL CECT classification** because it marks poor NCCT/CECT registration; **waiting/unregistered records remain excluded** unless separately verified. REJECT patients need extra AP/PVP pairing QC and NCCT slice matching review before synthesis. A valid image file and mask must exist unless you explicitly use `--allow-missing-mask`. Actual NIfTI **shape AND affine grid** are strictly checked at `prepare`. 512x512 in-plane size alone does not prove grid alignment.

## Active classification cohort: 1,195 patients (2026-10-22)

The user explicitly confirmed **103 patient directories are absent** from the source
CT mount. The former 1,298-case cohort remains an audit snapshot; the active
five-disease classification cohort is now **1,195 patients**. No additional
patients are silently dropped. The original patient-level splits are only
filtered, not regenerated:

| Split | Active patients |
|---|---:|
| train | 844 |
| val | 180 |
| test | 171 |

The five-disease distribution is HCC=720, metastases=87, cyst=163,
hemangioma=52, normal=173. The new cohort's exact patient ID/label hash is
validated by `lock-cohort` as well as the count and distribution.

Download and place these **local-only** input files from the conversation:

```text
/workspace/class/input/
    classification_main5_1195_including_REJECT.csv
    classification_main5_1195_locked_splits.csv
```

The old 1298 files can remain in the same ignored `input/` directory for
provenance. The 103 excluded patient IDs are recorded separately in
`classification_excluded_missing_103_audit.csv`, not committed to GitHub.

```bash
cd /workspace/class
python -m pip install -e ".[dev]"
python -m pytest -q
export INPUT_ROOT=/win_cect/shared/gx_all_registered
export QC_WORKERS=2
time bash scripts/run_classification.sh qc
```

**Expected next QC outcome, based on the previous full scan**: up to 1183/1195
passes and **12 remaining incomplete patients**. Those 12 cases are **not**
approved for removal, so `qc` still stops; investigate them before
`prepare`. This is not a failed data migration.

All generated outputs remain under `/dataset/downstream_classification`.
The `/workspace/class` directory contains code and only lightweight local
files; `/win_cect/shared/gx_all_registered` is the read-only image input root.


## HDD-optimized QC (64 GB RAM)

Use `qc` for a fast **header-only** pass before creating MIL slice manifests.
It checks the unchanged 1,298-patient locked cohort, all source file paths,
3D shapes, and canonical-grid affines **without decompressing CT voxel arrays**.
It does **not** assess voxel-level image integrity, whether the mask is liver,
or anatomical registration accuracy. `REJECT` patients are never silently removed.

```bash
cd /workspace/class
export INPUT_ROOT=/win_cect/shared/gx_all_registered
export QC_WORKERS=2  # HDD: test 1 or 2, avoid 8-32 concurrent disk seeks

time bash scripts/run_classification.sh qc
cat /dataset/downstream_classification/cohort/image_audit.json

# After resolving QC problems (do not delete locked patients):
time bash scripts/run_classification.sh prepare
cat /dataset/downstream_classification/manifests/qc.json
```

The fast QC creates `cohort/image_header_cache.json`. The `prepare` step
reuses unchanged shape/affine metadata instead of reading 5,192 NIfTI headers
again. Only actual **mask contents** are decompressed to obtain nonzero
liver-slice indices; those are cached in `manifests/mask_slice_cache.json`
and reused on subsequent runs. File path, size, and nanosecond mtime are
checked before cache reuse. Change detection is metadata-based, not a
cryptographic proof that voxel content is identical. Re-running `prepare`
after no source changes should show `masks_decompressed: 0` in
`manifests/qc.json`.

`QC_WORKERS` controls short metadata reads and defaults to **2** for
rotating HDD; `WORKERS` controls training DataLoader workers independently.
Do not set QC threads to 64 merely because 64 GB of RAM is available:
HDD seek contention can make the scan **slower**. The initial full
mask-index creation still requires one sequential decompression per patient.

If an image changes while `prepare` is running, it stops with a
re-audit message rather than using a stale header. Cached index files are
stored only in `/dataset/downstream_classification`, not in the public repo.
This optimization does not change the frozen patient or disease label lists.

## 2. Install

Python >=3.10, PyTorch, CUDA recommended:

```bash
git clone https://github.com/Devil-Jian/class.git
cd class
pip install -e ".[dev]"
pytest -q
```

## 3. Historical 1,298-patient workflow (superseded by 1,195 active cohort above)

Copy the uploaded CSV onto the training server first, e.g. to `/dataset/dataset/GX_metadata_all.csv`. Customize `CSV` below to the actual location.

```bash
export CSV=/dataset/dataset/GX_metadata_all.csv
export ROOT=/dataset/dataset/gx_all_registered
export WORK=/dataset/downstream_classification
mkdir -p "$WORK"
```

### Step 0 — Lock the approved 1,298-case primary classification cohort (CPU)

The authoritative list is `classification_main5_1298_including_REJECT.csv` downloaded from the conversation. Copy it to the training server. This step **does not** reselect patients from the whole GX metadata file; it verifies the exact 1,298 approved patient IDs + disease labels against a stable SHA-256 fingerprint. It keeps all `REJECT` patients, while reporting their registration/synthesis QC separately.

```bash
export LOCKED=/dataset/dataset/classification_main5_1298_including_REJECT.csv
export WORK=/dataset/downstream_classification

cectcls lock-cohort \
  --source "$LOCKED" \
  --out "$WORK/cohort" \
  --seed 42

cat "$WORK/cohort/cohort_lock_report.json"

cectcls prepare \
  --cases "$WORK/cohort/cases.csv" \
  --splits "$WORK/cohort/splits.csv" \
  --label-map "$WORK/cohort/labels.json" \
  --out "$WORK/manifests"
```

Expected distribution: HCC=724, metastases=92, cyst=188, hemangioma=56, normal=238; `GOLD=543, SILVER=19, REJECT=736`. The canonical primary class indices are `正常=0`, `肝囊肿=1`, `血管瘤=2`, `HCC=3`, `转移瘤=4`.

**Important**: `prepare` checks actual file existence and common NIfTI shape+affine. A failed QC is not an instruction to silently discard a locked patient. Report and resolve the discrepancy with source images/registration before finalizing the main Real/Syn cohort. Do not confuse patient membership with successful NCCT-to-CECT slice matching. Patient ID prefixes are explicitly used as a stratification axis, but cannot eliminate disease/source confounding when a disease is absent from a source.

### Step A — Alternative: construct a NEW cohort from raw GX metadata (not for the approved 1,298)



```bash
cectcls prepare-cohort \
  --metadata "$CSV" \
  --image-root "$ROOT" \
  --label-column 三分类标签 \
  --grades GOLD SILVER REJECT \
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

**Class imbalance and evaluation:** default `--class-weight sqrt` computes square-root inverse-frequency CE weights **only from training patients**, caps them at 3 before normalization, and keeps the original 1,298 patients. Use `--class-weight none` for unweighted CE comparison. Five-disease Macro-AUC/Macro-F1 remain the primary endpoints. Normal/benign/malignant probabilities are derived by **summing the five-disease probabilities** in `three_class` metrics and `three_class_probabilities` patient records; no separate three-class model is trained.

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
