#!/usr/bin/env bash
# Verify top-2 configs from grid_precise on two fresh seeds (1337, 2026).
# Seed 42 results already live under lightning_logs/grid_precise/ — reuse them.
#
# Configs:
#   A: lmf_margin=0.30  lmf_gamma=2.0  label_smoothing=0.00  gambler_weight=0.1
#   B: lmf_margin=0.30  lmf_gamma=2.0  label_smoothing=0.05  gambler_weight=0.0
#
# Output: lightning_logs/grid_precise_verify/<run>/...
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
PYTHON="${PYTHON:-/var/home/damo/.conda/envs/dali_testing/bin/python}"
OUT_DIR="$REPO_ROOT/lightning_logs/grid_precise_verify"
mkdir -p "$OUT_DIR"

BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-60}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-8}"
PATIENCE="${PATIENCE:-15}"
LR="${LR:-3e-4}"
DENOISE="${DENOISE:-off}"
PRECISION_FLAG="${PRECISION_FLAG:-bf16-mixed}"

declare -a CONFIGS=(
  "A:0.30:2.0:0.00:0.1"
  "B:0.30:2.0:0.05:0.0"
)
SEEDS=(${SEEDS:-1337 2026})

i=0
TOTAL=$((${#CONFIGS[@]} * ${#SEEDS[@]}))
for cfg in "${CONFIGS[@]}"; do
  IFS=':' read -r LABEL M G S GW <<< "$cfg"
  for SEED in "${SEEDS[@]}"; do
    i=$((i + 1))
    RUN="verify_${LABEL}_m${M}_g${G}_s${S}_gw${GW}_seed${SEED}"
    LOG_FILE="$OUT_DIR/${RUN}.log"
    echo "=== [$i/$TOTAL] $RUN ==="
    "$PYTHON" training/train_precise.py \
        --run_name "grid_precise_verify/$RUN" \
        --data_dir "$DATA_DIR" \
        --batch_size "$BATCH_SIZE" \
        --max_epochs "$MAX_EPOCHS" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        --patience "$PATIENCE" \
        --denoise "$DENOISE" \
        --lmf_margin "$M" \
        --lmf_gamma "$G" \
        --label_smoothing "$S" \
        --gambler_weight "$GW" \
        --lr "$LR" \
        --seed "$SEED" \
        --precision "$PRECISION_FLAG" \
        --target_coverage 0.85 \
        > "$LOG_FILE" 2>&1 || {
      echo "!! FAILED: $RUN (see $LOG_FILE)"
      continue
    }
    echo "  done → $LOG_FILE"
  done
done
echo ""
echo "=== verification grid done ==="
