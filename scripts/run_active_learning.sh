#!/usr/bin/env bash
# Run the HydroPrecise active learning pipeline with the verify_B preset
# (best v1 grid config: m=0.30, g=2.0, s=0.05, gw=0.0).
#
# Defaults reproduce a typical 5-round AL run with entropy acquisition.
# Override via env vars, e.g.:
#   STRATEGY=bald INIT=2000 QUERY=2000 ROUNDS=4 ./scripts/run_active_learning.sh
#
# Output lands under lightning_logs/<RUN_NAME>/round_NN/ and
#                    lightning_logs/<RUN_NAME>/summary.{csv,json}
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
PYTHON="${PYTHON:-/var/home/damo/.conda/envs/dali_testing/bin/python}"

STRATEGY="${STRATEGY:-entropy}"
INIT="${INIT:-1000}"
QUERY="${QUERY:-1000}"
ROUNDS="${ROUNDS:-5}"
SEED="${SEED:-2026}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SCORE_BS="${SCORE_BS:-128}"
MAX_EPOCHS="${MAX_EPOCHS:-60}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-8}"
PATIENCE="${PATIENCE:-15}"
LR="${LR:-3e-4}"
PRECISION_FLAG="${PRECISION_FLAG:-bf16-mixed}"
RUN_NAME="${RUN_NAME:-al_${STRATEGY}_init${INIT}_q${QUERY}_r${ROUNDS}_seed${SEED}}"

EXTRA_ARGS=()
if [[ "${WARM_START:-0}" == "1" ]]; then EXTRA_ARGS+=(--warm_start); fi
if [[ "${STRATIFIED_QUERY:-0}" == "1" ]]; then EXTRA_ARGS+=(--stratified_query); fi
if [[ "${FINAL_CALIBRATION:-1}" == "1" ]]; then EXTRA_ARGS+=(--final_calibration); fi

LOG_DIR="$REPO_ROOT/lightning_logs/$RUN_NAME"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"

echo "=== AL run: $RUN_NAME ==="
echo "  strategy = $STRATEGY"
echo "  init = $INIT, query = $QUERY, rounds = $ROUNDS"
echo "  data_dir = $DATA_DIR"
echo "  log = $LOG_FILE"

"$PYTHON" -u training/train_active.py \
  --run_name "$RUN_NAME" \
  --data_dir "$DATA_DIR" \
  --strategy "$STRATEGY" \
  --init_size "$INIT" \
  --query_size "$QUERY" \
  --n_rounds "$ROUNDS" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --score_batch_size "$SCORE_BS" \
  --max_epochs "$MAX_EPOCHS" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --patience "$PATIENCE" \
  --lr "$LR" \
  --precision "$PRECISION_FLAG" \
  --target_coverage 0.85 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
echo "=== AL run done: see $LOG_DIR/summary.csv ==="
