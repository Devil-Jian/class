#!/usr/bin/env bash
# Run Balanced-Mini (245 cases, 175 Train, 35 Val, 35 Test) Training on Server (Linux)
set -euo pipefail

MODEL="${1:-dinov2s_mil}"
GROUP="${2:-cect_real}"
DEVICE="${3:-cuda:0}"
EPOCHS="${4:-30}"
BATCH_SIZE="${5:-4}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="/dataset/downstream_classification"
MANIFEST="$PROJECT_ROOT/manifests_mini/cases_balanced_mini.jsonl"
if [[ ! -f "$MANIFEST" ]]; then
  MANIFEST="$WORK/manifests/cases_balanced_mini.jsonl"
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OUT_DIR="$WORK/experiments_mini/${MODEL}/${GROUP}"
mkdir -p "$OUT_DIR"

echo "======================================================================"
echo ">> Server Balanced-Mini (245 Cases, 5 Classes 1:1:1:1:1) Runner"
echo ">> Model: $MODEL | Group: $GROUP | Device: $DEVICE"
echo ">> Epochs: $EPOCHS | Batch Size: $BATCH_SIZE | Out: $OUT_DIR"
echo ">> Manifest: $MANIFEST"
echo "======================================================================"

python3 -m cectcls.cli train \
  --manifest "$MANIFEST" \
  --labels "$WORK/manifests/labels.json" \
  --selection "$WORK/manifests/matched_bags.json" \
  --png-index "$WORK/png_dataset/png_index.json" \
  --stage selected \
  --group "$GROUP" \
  --model "$MODEL" \
  --pretrained \
  --freeze-encoder \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --encode-chunk 16 \
  --workers 4 \
  --device "$DEVICE" \
  --class-weight none \
  --out "$OUT_DIR"

echo ">> [Finished] Mini experiment completed! Results saved to: $OUT_DIR"
