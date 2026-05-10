#!/usr/bin/env bash
# Phase I — multi-seed R1 verification.
#
# Phase I sweep (5 runs) + I2b LPC retry showed every architectural lever
# fails on test/μP. The val/test gap is now diagnosed as a Lexar Train↔Test
# distribution shift compounded by very small Test set (110 source files,
# Tug=2). Before declaring R1 the ceiling, re-run R1 at 2 additional seeds
# (s42, s2026) to bound the gap variance and produce soup-eligible ckpts.
#
# Existing R1 = seed 1337 → val 0.6908 / test 0.6620.

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
  echo " PHASE I MULTI-SEED — starting run: $name"
  echo " log: $logf"
  echo " started: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  python training/train_hydra.py "$@" --run_name "$name" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE I MULTI-SEED — finished:   $name  (exit=$rc)"
  echo " finished: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  return 0
}

run phaseI_R1_s42   "${R1[@]}" --seed 42
run phaseI_R1_s2026 "${R1[@]}" --seed 2026

echo "════════════════════════════════════════════════════════════════"
echo " PHASE I MULTI-SEED — completed."
echo "════════════════════════════════════════════════════════════════"
