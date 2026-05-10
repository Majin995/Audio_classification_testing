#!/usr/bin/env bash
# Phase I follow-up: rerun SWA experiments with fixed timing.
#
# The original I1/I1b had ``--swa_epoch_start 0.75 --patience 15``. SWA's
# averaging window opens at epoch 60 (75% × 80) but EarlyStopping fires at
# best+15 epochs — which for R1-class trajectories is around epoch 30-40.
# Result: SWA's window never opens, callback is a no-op.
#
# This follow-up uses ``--swa_epoch_start 0.30`` (epoch 24) and
# ``--patience 50`` so SWA actually fires. Two configurations:
#   I1c  SWA early + patience 50
#   I1d  SWA early + drop_path 0.10 + patience 50

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
  --max_epochs 80 --patience 50
  --batch_size 64 --num_threads 8
  --seed 1337
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)

run() {
  local name="$1"; shift
  local logf="$LOG_ROOT/$name.log"
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE I SWA-FIX — starting run: $name"
  echo " log: $logf"
  echo " started: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  python training/train_hydra.py "$@" --run_name "$name" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE I SWA-FIX — finished:   $name  (exit=$rc)"
  echo " finished: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  return 0
}

# ── I1c: SWA early + patience 50 ────────────────────────────────────────
run phaseI_I1c_swa_early "${R1[@]}" \
  --swa --swa_lrs 1e-4 --swa_epoch_start 0.30 --swa_anneal_epochs 10

# ── I1d: SWA early + drop_path + patience 50 ────────────────────────────
run phaseI_I1d_swa_early_droppath "${R1[@]}" \
  --swa --swa_lrs 1e-4 --swa_epoch_start 0.30 --swa_anneal_epochs 10 \
  --drop_path 0.10

echo "════════════════════════════════════════════════════════════════"
echo " PHASE I SWA-FIX — completed."
echo "════════════════════════════════════════════════════════════════"
