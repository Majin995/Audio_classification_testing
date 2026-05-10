#!/usr/bin/env bash
# Re-launch the 3 verification trials missing after run 1 completed.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="/var/home/damo/.conda/envs/dali_testing/bin/python"
DATA_DIR="/run/media/damo/Lexar M2/Data/Classifier_Dataset"
OUT_DIR="$REPO_ROOT/lightning_logs/grid_precise_verify"
mkdir -p "$OUT_DIR"

# label : margin : gamma : smoothing : gambler_w : seed
declare -a TRIALS=(
  "A:0.30:2.0:0.00:0.1:2026"
  "B:0.30:2.0:0.05:0.0:1337"
  "B:0.30:2.0:0.05:0.0:2026"
)

i=0
TOTAL=${#TRIALS[@]}
for spec in "${TRIALS[@]}"; do
  IFS=':' read -r LABEL M G S GW SEED <<< "$spec"
  i=$((i + 1))
  RUN="verify_${LABEL}_m${M}_g${G}_s${S}_gw${GW}_seed${SEED}"
  LOG="$OUT_DIR/${RUN}.log"
  echo "=== [$i/$TOTAL] $RUN ==="
  "$PYTHON" training/train_precise.py \
      --run_name "grid_precise_verify/$RUN" \
      --data_dir "$DATA_DIR" \
      --batch_size 64 --max_epochs 60 --warmup_epochs 8 --patience 15 \
      --denoise off \
      --lmf_margin "$M" --lmf_gamma "$G" \
      --label_smoothing "$S" --gambler_weight "$GW" \
      --lr 3e-4 --seed "$SEED" --precision bf16-mixed --target_coverage 0.85 \
      > "$LOG" 2>&1 || {
    echo "!! FAILED: $RUN (see $LOG)"
    continue
  }
  echo "  done"
done
echo ""
echo "=== verification remaining done ==="
