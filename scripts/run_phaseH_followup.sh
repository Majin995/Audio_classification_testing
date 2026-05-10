#!/usr/bin/env bash
# Phase H follow-up: retry runs to address known limitations of the main sweep.
#
# H1b: LDAM-DRW with realistic deferral (DRW epoch 15, patience 25).
#      The original H1 had ldam_drw_epoch=40 + patience=15, which means DRW
#      almost certainly never fired (early-stop hits at ~epoch 4+15=19).
#
# H7b: Stacked P0 with the same realistic LDAM-DRW timing (depends on the
#      H1 lesson — only run if H1b shows DRW lift).

set -uo pipefail
DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseH_console}"
mkdir -p "$LOG_ROOT"
cd "$(dirname "$0")/.."

COMMON=(
  --data_dir "$DATA_DIR"
  --use_gabor --use_scattering --use_sincnet --use_tdsbe
  --max_epochs 80
  --batch_size 64 --num_threads 8
  --seed 1337
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)

run() {
  local name="$1"; shift
  local logf="$LOG_ROOT/$name.log"
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE H FOLLOWUP — starting run: $name"
  echo " log: $logf"
  echo " started: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  python training/train_hydra.py "$@" --run_name "$name" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE H FOLLOWUP — finished:   $name  (exit=$rc)"
  echo " finished: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  return 0
}

# ── H1b: LDAM-DRW retry with realistic deferral ─────────────────────────
run phaseH_H1b_ldam_drw_early "${COMMON[@]}" \
  --loss ldam --ldam_max_m 0.5 --ldam_s 30.0 --ldam_drw_epoch 15 \
  --label_smoothing 0.05 --gambler_weight 0.0 \
  --patience 25

# ── H7b: Stacked P0 with the same DRW timing (gated on H1b being a win) ─
run phaseH_H7b_stacked_drw15 "${COMMON[@]}" \
  --loss ldam --ldam_max_m 0.5 --ldam_s 30.0 --ldam_drw_epoch 15 \
  --label_smoothing 0.05 --gambler_weight 0.0 \
  --use_jtfs \
  --use_global_attn --global_attn_heads 4 \
  --patience 25

echo
echo "════════════════════════════════════════════════════════════════"
echo " PHASE H FOLLOWUP — completed."
echo "════════════════════════════════════════════════════════════════"
