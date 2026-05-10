#!/usr/bin/env bash
# Resume Phase F by running ONLY the α+β+γ stacked configuration.
# α/β/γ are already complete — re-running them would waste compute.

DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseF_console}"
mkdir -p "$LOG_ROOT"

cd "$(dirname "$0")/.."

NAME=phaseF_stacked
LOGF="$LOG_ROOT/$NAME.log"

# If RESUME_CKPT is set in env, pass --resume_from_ckpt for true Lightning
# resume (optimizer + scheduler + epoch counter restored).
RESUME_FLAGS=()
if [ -n "${RESUME_CKPT:-}" ]; then
  RESUME_FLAGS=(--resume_from_ckpt "$RESUME_CKPT")
fi

echo "════════════════════════════════════════════════════════════════"
echo " PHASE F — resuming run: $NAME"
echo " log: $LOGF"
echo " started: $(date -Is)"
[ -n "${RESUME_CKPT:-}" ] && echo " resume_from_ckpt: $RESUME_CKPT"
echo "════════════════════════════════════════════════════════════════"

python -m training.train_precise_v2 \
  --data_dir "$DATA_DIR" \
  --use_lofar_branch \
  --loss focal --focal_gamma 2.0 --label_smoothing 0.05 --logit_adjust_tau 0.5 \
  --fusion_dim 128 --n_heads 4 --n_attn_blocks 1 \
  --dropout 0.10 --drop_path 0.20 \
  --no_pcen_on_cqt --no_spec_aug_all --no_boundary_attn \
  --feature_norm layernorm_l2 \
  --max_epochs 80 --patience 18 \
  --monitor val/macro_precision \
  --batch_size 64 --num_threads 8 \
  --use_pretrained_branch --pretrained_model facebook/wav2vec2-base \
  --pretrained_ch 128 \
  --head_type arcface --arcface_margin 0.3 --arcface_scale 30 \
  --mixup_alpha 0.1 \
  --corpus_noise_prob 0.3 --corpus_noise_snr_min -3 --corpus_noise_snr_max 15 \
  --noise_pool_max 4096 --noise_pool_quantile 0.25 \
  --rir_prob 0.2 --pitch_prob 0.2 --branch_dropout_p 0.1 \
  --rms_normalize --target_rms 1.0 \
  --hpf_hz 20.0 --hpf_order 4 \
  --run_name "$NAME" \
  --recall_floor 0.6 \
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95 \
  "${RESUME_FLAGS[@]}" \
  >> "$LOGF" 2>&1
RC=$?

echo "════════════════════════════════════════════════════════════════"
echo " PHASE F — finished:   $NAME  (exit=$RC)"
echo " finished: $(date -Is)"
echo "════════════════════════════════════════════════════════════════"
