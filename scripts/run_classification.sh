#!/usr/bin/env bash
# Stepwise, patient-locked CECT classification; input root can change, workdir cannot.
set -euo pipefail

STAGE="${1:-help}"
INPUT_ROOT="${INPUT_ROOT:-/dataset/dataset/gx_all_registered}"
WORK="/dataset/downstream_classification"
INPUT_DIR="$WORK/input"
LOCKED_CSV="${LOCKED_CSV:-$INPUT_DIR/classification_main5_1298_including_REJECT.csv}"
FROZEN_SPLITS_CSV="${FROZEN_SPLITS_CSV:-$INPUT_DIR/classification_main5_1298_locked_splits.csv}"
DEVICE="${DEVICE:-cuda}"
SELECTION_EPOCHS="${SELECTION_EPOCHS:-30}"
EXPERIMENT_EPOCHS="${EXPERIMENT_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORKERS="${WORKERS:-0}"
mkdir -p "$WORK" "$INPUT_DIR"

PRETRAINED=()
if [[ "${USE_PRETRAINED:-1}" == "1" ]]; then PRETRAINED=(--pretrained); fi
FREEZE=()
if [[ "${FREEZE_ENCODER:-1}" == "1" ]]; then FREEZE=(--freeze-encoder); fi

COMMON=(
  --manifest "$WORK/manifests/cases.jsonl"
  --labels "$WORK/manifests/labels.json"
  --batch-size "$BATCH_SIZE"
  --workers "$WORKERS"
  --image-size 224
  --device "$DEVICE"
  --class-weight sqrt
  "${PRETRAINED[@]}"
  "${FREEZE[@]}"
)

case "$STAGE" in
  prepare)
    [[ -d "$INPUT_ROOT" ]] || { echo "Missing INPUT_ROOT directory: $INPUT_ROOT" >&2; exit 2; }
    [[ -f "$LOCKED_CSV" ]] || { echo "Missing approved patient CSV: $LOCKED_CSV" >&2; exit 2; }
    [[ -f "$FROZEN_SPLITS_CSV" ]] || { echo "Missing frozen patient splits: $FROZEN_SPLITS_CSV" >&2; exit 2; }
    cectcls lock-cohort \
      --source "$LOCKED_CSV" \
      --splits-source "$FROZEN_SPLITS_CSV" \
      --image-root "$INPUT_ROOT" \
      --out "$WORK/cohort"
    cectcls audit-images \
      --cases "$WORK/cohort/cases.csv" \
      --strict-mask \
      --out "$WORK/cohort/image_audit.csv"
    python - "$WORK/cohort/image_audit.json" <<'PY'
import json, sys
from pathlib import Path
a=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(f"Image audit: {a['ready']}/{a['locked_patients_checked']} ready")
if a['ready'] != a['locked_patients_checked']:
    raise SystemExit("STOP: some of the locked 1298 scans/masks require repair; nothing was deleted.")
PY
    cectcls prepare \
      --cases "$WORK/cohort/cases.csv" \
      --splits "$WORK/cohort/splits.csv" \
      --label-map "$WORK/cohort/labels.json" \
      --out "$WORK/manifests"
    ;;
  train-selector)
    cectcls train "${COMMON[@]}" \
      --stage selection --group cect_real --model dinov2s_mil \
      --uniform 16 --random-count 16 \
      --epochs "$SELECTION_EPOCHS" --encode-chunk 8 \
      --out "$WORK/selection"
    ;;
  top16)
    cectcls export-selection \
      --manifest "$WORK/manifests/cases.jsonl" \
      --checkpoint "$WORK/selection/best.pt" \
      --top-k 16 --device "$DEVICE" \
      --out "$WORK/manifests/selected_bags.json"
    ;;
  match)
    cectcls match-ncct \
      --manifest "$WORK/manifests/cases.jsonl" \
      --selection "$WORK/manifests/selected_bags.json" \
      --radius-mm 15 \
      --out "$WORK/manifests/matched_bags.json"
    ;;
  request-syn)
    cectcls export-syn-requests \
      --manifest "$WORK/manifests/cases.jsonl" \
      --matched "$WORK/manifests/matched_bags.json" \
      --split train \
      --out-root "$WORK/synthetic_hu" \
      --out "$WORK/manifests/synthetic_requests.json"
    echo "STOP: call your separate NCCT->CECT generator using the requests JSON."
    echo "The generator must write each 2D HU .npy output_path before register-syn."
    ;;
  register-syn)
    cectcls register-syn \
      --requests "$WORK/manifests/synthetic_requests.json" \
      --out "$WORK/manifests/syn_index.json"
    ;;
  export-png)
    cectcls export-png \
      --manifest "$WORK/manifests/cases.jsonl" \
      --selection "$WORK/manifests/matched_bags.json" \
      --syn-index "$WORK/manifests/syn_index.json" \
      --out "$WORK/png"
    ;;
  smoke)
    cectcls train "${COMMON[@]}" \
      --stage selected \
      --selection "$WORK/manifests/matched_bags.json" \
      --png-index "$WORK/png/png_index.json" \
      --model resnet18_mil --group cect_real \
      --epochs 2 --out "$WORK/experiments/smoke_real"
    ;;
  train-grid)
    cectcls train-grid "${COMMON[@]}" \
      --stage selected \
      --selection "$WORK/manifests/matched_bags.json" \
      --png-index "$WORK/png/png_index.json" \
      --epochs "$EXPERIMENT_EPOCHS" --encode-chunk 8 \
      --out "$WORK/experiments"
    ;;
  evaluate)
    for model in resnet18_mil deit_mil dinov2s_mil; do
      for group in ncct cect_real cect_syn cect_real_aug cect_real_syn; do
        target="$WORK/experiments/$model/$group"
        cectcls evaluate "${COMMON[@]}" \
          --selection "$WORK/manifests/matched_bags.json" \
          --png-index "$WORK/png/png_index.json" \
          --checkpoint "$target/best.pt" --split test \
          --out "$target/test.json"
      done
    done
    ;;
  help|*)
    cat <<HELP
Usage: INPUT_ROOT=/win_cect/shared/gx_all_registered bash scripts/run_classification.sh STAGE

STAGEs: prepare, train-selector, top16, match, request-syn,
        register-syn, export-png, smoke, train-grid, evaluate

INPUT_ROOT (only scan location): $INPUT_ROOT
WORK (always fixed):           $WORK
LOCKED_CSV:                    $LOCKED_CSV
FROZEN_SPLITS_CSV:             $FROZEN_SPLITS_CSV

Variables: DEVICE=cuda, SELECTION_EPOCHS=30, EXPERIMENT_EPOCHS=30,
           BATCH_SIZE=2, WORKERS=0, USE_PRETRAINED=1, FREEZE_ENCODER=1

Stages deliberately do not skip a failed image/registration/synthesis audit.
No patients are auto-removed from the fixed 1298-patient cohort.
HELP
    ;;
esac
