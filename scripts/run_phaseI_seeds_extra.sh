#!/usr/bin/env bash
# Phase I — extend multi-seed ensemble from N=3 to N=5.
#
# 3-seed ensemble (s1337, s42, s2026) shipped as new selective-PR baseline
# with MP@cov0.85 = 0.7926. Adding seeds 7 and 12345 to:
#  1. tighten the test/μP variance estimate (N=3 std=±0.10 is itself noisy)
#  2. potentially lift the ensemble selective-PR further via more
#     independent predictions averaged.

set -uo pipefail
DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseI_console}"
mkdir -p "$LOG_ROOT"
cd "$(dirname "$0")/.."

R1=(
  --data_dir "$DATA_DIR"
  --use_gabor --use_scattering --use_sincnet --use_tdsbe
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 --label_smoothing 0.05
  --gambler_weight 0.1 --gambler_o 0.3
  --max_epochs 80 --patience 15
  --batch_size 64 --num_threads 8
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)

run() {
  local name="$1"; shift
  local logf="$LOG_ROOT/$name.log"
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE I SEEDS-EXTRA — starting: $name  (started $(date -Is))"
  echo "════════════════════════════════════════════════════════════════"
  python training/train_hydra.py "$@" --run_name "$name" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE I SEEDS-EXTRA — finished: $name  (exit=$rc; $(date -Is))"
  echo "════════════════════════════════════════════════════════════════"
  return 0
}

run phaseI_R1_s7     "${R1[@]}" --seed 7
run phaseI_R1_s12345 "${R1[@]}" --seed 12345

echo "════════════════════════════════════════════════════════════════"
echo " PHASE I SEEDS-EXTRA — completed."
echo "════════════════════════════════════════════════════════════════"
