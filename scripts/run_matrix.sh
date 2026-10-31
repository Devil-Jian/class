#!/usr/bin/env bash
# Matrix Training Runner: 3 Models x 4 Groups with Automatic Best-Checkpoint Test Evaluation
# Usage:
#   ./scripts/run_matrix.sh [model|all] [group|all] [device] [epochs] [batch_size]
# Examples:
#   ./scripts/run_matrix.sh all all cuda:0 30 4          # Run full 3x4 = 12 experiments
#   ./scripts/run_matrix.sh dinov2s_mil all cuda:0 30 4  # Run DINOv2 on all 4 groups
#   ./scripts/run_matrix.sh resnet50_mil cect_real cuda:0 30 4
set -euo pipefail

MODEL_ARG="${1:-all}"
GROUP_ARG="${2:-all}"
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

MODELS=()
if [[ "$MODEL_ARG" == "all" ]]; then
  MODELS=("dinov2s_mil" "resnet50_mil" "deit_mil")
else
  MODELS=("$MODEL_ARG")
fi

GROUPS=()
if [[ "$GROUP_ARG" == "all" ]]; then
  GROUPS=("cect_real" "cect_syn" "cect_real_syn" "cect_real_aug")
else
  # Normalize cect_aug to cect_real_aug
  if [[ "$GROUP_ARG" == "cect_aug" ]]; then
    GROUPS=("cect_real_aug")
  else
    GROUPS=("$GROUP_ARG")
  fi
fi

echo "=========================================================================="
echo ">> Matrix Training Runner with Auto Test Evaluation"
echo ">> Models: ${MODELS[*]}"
echo ">> Groups: ${GROUPS[*]}"
echo ">> Device: $DEVICE | Epochs: $EPOCHS | Batch Size: $BATCH_SIZE"
echo ">> Manifest: $MANIFEST"
echo "=========================================================================="

for M in "${MODELS[@]}"; do
  for G in "${GROUPS[@]}"; do
    OUT_DIR="$WORK/experiments_mini/${M}/${G}"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "=========================================================================="
    echo ">> [Starting Experiment] Model: $M | Group: $G on $DEVICE"
    echo ">> Output Directory: $OUT_DIR"
    echo "=========================================================================="

    python3 -m cectcls.cli train \
      --manifest "$MANIFEST" \
      --labels "$WORK/manifests/labels.json" \
      --selection "$WORK/manifests/matched_bags.json" \
      --png-index "$WORK/png_dataset/png_index.json" \
      --stage selected \
      --group "$G" \
      --model "$M" \
      --pretrained \
      --freeze-encoder \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --encode-chunk 16 \
      --workers 4 \
      --device "$DEVICE" \
      --class-weight none \
      --mix-mode full \
      --out "$OUT_DIR"

    echo ">> [Completed] Experiment ($M / $G) finished successfully!"
  done
done

echo ""
echo "=========================================================================="
echo ">> ALL MATRIX EXPERIMENTS COMPLETED!"
echo ">> Summary of generated test results:"
find "$WORK/experiments_mini" -name "test_results.json" | sort
echo "=========================================================================="
