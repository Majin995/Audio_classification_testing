#!/usr/bin/env bash
# Phase G — R6: α + γ-only (no β). Isolates the ArcFace effect from the
# rejected aug pack. Plan-prescribed sweep: arcface_scale=15 (the lower
# end), since scale=30 in R3 was confounded by β collapse.

DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseG_console}"
mkdir -p "$LOG_ROOT"

cd "$(dirname "$0")/.."

NAME=phaseG_R6_alpha_arcface15
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
  --head_type arcface --feature_norm layernorm_l2 \
  --arcface_margin 0.2 --arcface_scale 15.0 \
  --gambler_weight 0.0 \
  --recall_floor 0.6 \
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95 \
  --run_name "$NAME" \
  > "$LOGF" 2>&1
RC=$?

echo "════════════════════════════════════════════════════════════════"
echo " PHASE G — finished:   $NAME  (exit=$RC)"
echo " finished: $(date -Is)"
echo "════════════════════════════════════════════════════════════════"
