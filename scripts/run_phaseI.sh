#!/usr/bin/env bash
# Phase I — close the val→test gap and add genuinely orthogonal waveform
# branches. Built on Phase G R1's ship config:
#   --use_gabor --use_scattering --use_sincnet --use_tdsbe
#   --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 --label_smoothing 0.05
#   --gambler_weight 0.1 --gambler_o 0.3
#
# Sub-task α — free generalization wins
#   I1  α.2+α.3 SWA + drop_path                           (1 fresh run)
#   I1b α.2 vanilla SWA, no drop_path                     (ablation)
# Sub-task γ — net-new waveform branches
#   I2  γ.1 LPC residual added                            (1 fresh run)
#   I3  γ.2 Recurrence-plot branch added                  (1 fresh run)
#   I4  γ.1 + γ.2 stacked on top of R1 baseline           (1 fresh run)
#
# DART-MT (β.1) and CSC (γ.3) and CPC (δ.1) live in separate scripts.

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
  --seed 1337
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)

run() {
  local name="$1"; shift
  local logf="$LOG_ROOT/$name.log"
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE I — starting run: $name"
  echo " log: $logf"
  echo " started: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  python training/train_hydra.py "$@" --run_name "$name" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE I — finished:   $name  (exit=$rc)"
  echo " finished: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  return 0
}

# ── I1: SWA + stochastic depth (α.2 + α.3) ──────────────────────────────
# Targets val/test gap directly. swa_epoch_start as fraction → epoch 60/80.
run phaseI_I1_swa_droppath "${R1[@]}" \
  --swa --swa_lrs 1e-4 --swa_epoch_start 0.75 --swa_anneal_epochs 10 \
  --drop_path 0.10

# ── I1b: SWA only, no drop_path (ablation, isolates SWA effect) ─────────
run phaseI_I1b_swa_only "${R1[@]}" \
  --swa --swa_lrs 1e-4 --swa_epoch_start 0.75 --swa_anneal_epochs 10

# ── I2: γ.1 LPC residual branch added ───────────────────────────────────
run phaseI_I2_lpc "${R1[@]}" --use_lpc

# ── I3: γ.2 Recurrence-plot branch added ────────────────────────────────
# Memory hot path — RP downsamples to 1024 internally so the (1024,1024)
# recurrence map is ~1 MB at uint8 per item; at batch=64 that's ~64 MB.
run phaseI_I3_rp "${R1[@]}" --use_rp

# ── I4: γ.1 + γ.2 stacked on top of R1 ──────────────────────────────────
run phaseI_I4_lpc_rp "${R1[@]}" --use_lpc --use_rp

# ── I5: combo I1 (SWA+drop_path) + I4 (LPC+RP) ──────────────────────────
# Only worth running if I1 and one of {I2,I3,I4} both clear test ≥ 0.6700.
# Comment out unless gates pass.
# run phaseI_I5_full "${R1[@]}" \
#   --swa --swa_lrs 1e-4 --swa_epoch_start 0.75 --swa_anneal_epochs 10 \
#   --drop_path 0.10 --use_lpc --use_rp

echo
echo "════════════════════════════════════════════════════════════════"
echo " PHASE I — sweep complete (5 runs / 4 ran + 1 gated)."
echo "════════════════════════════════════════════════════════════════"
