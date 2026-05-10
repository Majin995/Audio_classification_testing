#!/usr/bin/env bash
# Specialist v3 — LDAM-DRW teachers.
#
# v1/v2 use LMF (large-margin focal). v3 swaps the loss to LDAM-DRW
# (Cao et al., NeurIPS'19), which enforces label-distribution-aware
# margins that scale with class frequency. For a binary {pos, neg} with
# 25/75 imbalance, LDAM gives a wider decision margin to the positive
# class, producing different decision boundary geometry from LMF.
#
# Why this helps distillation: LDAM and LMF teachers tend to disagree
# on samples near the decision boundary — exactly the samples where the
# student benefits most from teacher consensus. Adds genuine geometric
# diversity to the teacher ensemble.

set -uo pipefail
DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseJ_console}"
mkdir -p "$LOG_ROOT"
cd "$(dirname "$0")/.."

V3=(
  --data_dir "$DATA_DIR"
  --use_gabor --use_scattering --use_sincnet --use_tdsbe
  --loss ldam --ldam_max_m 0.5 --ldam_s 30
  --ldam_drw_epoch 15 --ldam_drw_beta 0.99999
  --label_smoothing 0.05
  --gambler_weight 0.1 --gambler_o 0.3
  --max_epochs 40 --patience 8
  --batch_size 64 --num_threads 8
  --seed 1337
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)

run() {
  local name="$1"; shift
  local logf="$LOG_ROOT/$name.log"
  echo "════════════════════════════════════════════════════════════════"
  echo " SPECIALIST v3 (LDAM-DRW) — starting: $name  ($(date -Is))"
  echo "════════════════════════════════════════════════════════════════"
  python training/train_hydra.py "$@" --run_name "$name" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " SPECIALIST v3 — finished: $name  (exit=$rc; $(date -Is))"
  echo "════════════════════════════════════════════════════════════════"
  return 0
}

run phaseJ_specialist_v3_cargo     "${V3[@]}" --positive_class Cargo
run phaseJ_specialist_v3_passenger "${V3[@]}" --positive_class Passenger
run phaseJ_specialist_v3_tanker    "${V3[@]}" --positive_class Tanker
run phaseJ_specialist_v3_tug       "${V3[@]}" --positive_class Tug

echo "v3 specialists done."
