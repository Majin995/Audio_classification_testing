#!/usr/bin/env bash
# Specialist binary teachers — one per class, R1 backbone + Gamblers abstain.
# Each teacher emits 3 outputs: {pos_class, neg_class, abstain}. After all
# 4 teachers train, train_hydra_student.py distills them into a 5-output
# student (4-way + abstain).

set -uo pipefail
DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseJ_console}"
mkdir -p "$LOG_ROOT"
cd "$(dirname "$0")/.."

R1=(
  --data_dir "$DATA_DIR"
  --use_gabor --use_scattering --use_sincnet --use_tdsbe
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 --label_smoothing 0.05
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
  echo " SPECIALIST — starting: $name  (started $(date -Is))"
  echo "════════════════════════════════════════════════════════════════"
  python training/train_hydra.py "$@" --run_name "$name" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " SPECIALIST — finished: $name  (exit=$rc; $(date -Is))"
  echo "════════════════════════════════════════════════════════════════"
  return 0
}

run phaseJ_specialist_cargo     "${R1[@]}" --positive_class Cargo
run phaseJ_specialist_passenger "${R1[@]}" --positive_class Passenger
run phaseJ_specialist_tanker    "${R1[@]}" --positive_class Tanker
run phaseJ_specialist_tug       "${R1[@]}" --positive_class Tug

echo "════════════════════════════════════════════════════════════════"
echo " SPECIALIST — all 4 teachers trained."
echo "════════════════════════════════════════════════════════════════"
