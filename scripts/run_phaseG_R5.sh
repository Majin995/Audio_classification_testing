#!/usr/bin/env bash
# Phase G — R5: α + β-lean. Tests the plan's fallback hypothesis that the
# β regression was driven by over-aggressive aug intensities. Drops pitch
# entirely; lowers corpus/rir/branch_drop to V2 Phase F levels. Same MLP+
# Gamblers head as R1 so head and aug effects don't entangle.

DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseG_console}"
mkdir -p "$LOG_ROOT"

cd "$(dirname "$0")/.."

NAME=phaseG_R5_alpha_betalean
LOGF="$LOG_ROOT/$NAME.log"

echo "════════════════════════════════════════════════════════════════"
echo " PHASE G — starting run: $NAME"
echo " log: $LOGF"
echo " started: $(date -Is)"
echo "════════════════════════════════════════════════════════════════"

python training/train_hydra.py \
  --data_dir "$DATA_DIR" \
  --use_gabor --use_scattering \
  --use_sincnet --use_tdsbe \
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 --label_smoothing 0.05 \
  --max_epochs 80 --patience 15 \
  --batch_size 64 --num_threads 8 \
  --seed 1337 \
  --corpus_noise_prob 0.3 --corpus_noise_snr_min -3 --corpus_noise_snr_max 15 \
  --noise_pool_max 4096 --noise_pool_quantile 0.25 \
  --rir_prob 0.2 --branch_dropout_p 0.10 \
  --recall_floor 0.6 \
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95 \
  --run_name "$NAME" \
  > "$LOGF" 2>&1
RC=$?

echo "════════════════════════════════════════════════════════════════"
echo " PHASE G — finished:   $NAME  (exit=$RC)"
echo " finished: $(date -Is)"
echo "════════════════════════════════════════════════════════════════"
