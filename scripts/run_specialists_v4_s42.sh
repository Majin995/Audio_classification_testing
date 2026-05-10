#!/usr/bin/env bash
# Specialist v4 — second-seed teachers (s42 instead of s1337).
#
# v1-v3 vary the *loss / regularisation* dimension of the teacher
# ensemble. v4 varies the seed dimension at fixed config = v1 R1.
# Phase I established that seed-to-seed test/μP variance on Lexar is
# ±0.09 — the largest single source of disagreement we have observed.
# Adding s42 teachers gives the cleanest variance-reduction signal
# in the teacher ensemble.
#
# Pair with v1 in 8-teacher distillation: 2 specialists per class
# differing only by seed. Pure variance averaging at the teacher level
# before the student even runs.

set -uo pipefail
DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseJ_console}"
mkdir -p "$LOG_ROOT"
cd "$(dirname "$0")/.."

V4=(
  --data_dir "$DATA_DIR"
  --use_gabor --use_scattering --use_sincnet --use_tdsbe
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 --label_smoothing 0.05
  --gambler_weight 0.1 --gambler_o 0.3
  --max_epochs 40 --patience 8
  --batch_size 64 --num_threads 8
  --seed 42
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)

run() {
  local name="$1"; shift
  local logf="$LOG_ROOT/$name.log"
  echo "════════════════════════════════════════════════════════════════"
  echo " SPECIALIST v4 (s42) — starting: $name  ($(date -Is))"
  echo "════════════════════════════════════════════════════════════════"
  python training/train_hydra.py "$@" --run_name "$name" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " SPECIALIST v4 — finished: $name  (exit=$rc; $(date -Is))"
  echo "════════════════════════════════════════════════════════════════"
  return 0
}

run phaseJ_specialist_v4_cargo     "${V4[@]}" --positive_class Cargo
run phaseJ_specialist_v4_passenger "${V4[@]}" --positive_class Passenger
run phaseJ_specialist_v4_tanker    "${V4[@]}" --positive_class Tanker
run phaseJ_specialist_v4_tug       "${V4[@]}" --positive_class Tug

echo "v4 specialists done."
