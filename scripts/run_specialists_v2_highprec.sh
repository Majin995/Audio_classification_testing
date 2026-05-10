#!/usr/bin/env bash
# Specialist v2 — HIGH-PRECISION teachers.
#
# v1 teachers use R1 defaults (LMF m=0.7, gambler_o=0.3): they fire
# liberally and abstain only on edge cases. v2 inverts this — bigger
# margin (m=0.9) + smaller gambler_o (0.1) makes each teacher
# *aggressively abstain* unless it is very confident the sample is its
# class. The student sees a stronger "I don't know" signal during
# distillation, making the abstain pathway more meaningful in
# production (helps the recall_floor problem).
#
# Pairs cleanly with v1 teachers in a 8-teacher distillation: v1 votes
# liberally, v2 votes selectively, the student averages.

set -uo pipefail
DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseJ_console}"
mkdir -p "$LOG_ROOT"
cd "$(dirname "$0")/.."

V2=(
  --data_dir "$DATA_DIR"
  --use_gabor --use_scattering --use_sincnet --use_tdsbe
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.9 --label_smoothing 0.1
  --gambler_weight 0.2 --gambler_o 0.1
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
  echo " SPECIALIST v2 (high-prec) — starting: $name  ($(date -Is))"
  echo "════════════════════════════════════════════════════════════════"
  python training/train_hydra.py "$@" --run_name "$name" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " SPECIALIST v2 — finished: $name  (exit=$rc; $(date -Is))"
  echo "════════════════════════════════════════════════════════════════"
  return 0
}

run phaseJ_specialist_v2_cargo     "${V2[@]}" --positive_class Cargo
run phaseJ_specialist_v2_passenger "${V2[@]}" --positive_class Passenger
run phaseJ_specialist_v2_tanker    "${V2[@]}" --positive_class Tanker
run phaseJ_specialist_v2_tug       "${V2[@]}" --positive_class Tug

echo "v2 specialists done."
