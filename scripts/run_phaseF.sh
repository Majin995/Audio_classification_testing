#!/usr/bin/env bash
# Phase F — precision-focused improvements on the Lexar dataset.
# Runs α, β, γ, then the stacked α+β+γ in sequence on a single GPU.
# Each run writes its own lightning_logs/<name>/ tree with checkpoints,
# temperature.pt, thresholds.json, and selective_pr.md.
#
# Logging: each run's stdout+stderr is written directly to
#   $LOG_ROOT/<name>.log  via simple `>` redirect — no `tee`, no `pipefail`,
#   no buffering surprises. A run failure is recorded but does not abort
#   subsequent runs, so a partial queue still produces useful comparison
#   data. The master log records start/finish timestamps and exit codes.

DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseF_console}"
mkdir -p "$LOG_ROOT"

cd "$(dirname "$0")/.."

# Common Phase E base flags (same recipe that produced val/μP=0.752 on the
# previous dataset — preserves apples-to-apples comparability).
COMMON_BASE=(
  --data_dir "$DATA_DIR"
  --use_lofar_branch
  --loss focal --focal_gamma 2.0 --label_smoothing 0.05 --logit_adjust_tau 0.5
  --fusion_dim 128 --n_heads 4 --n_attn_blocks 1
  --dropout 0.10 --drop_path 0.20
  --no_pcen_on_cqt --no_spec_aug_all --no_boundary_attn
  --feature_norm layernorm_l2
  --max_epochs 80 --patience 18
  --monitor val/macro_precision
  --batch_size 64 --num_threads 8
)

# δ post-cal flags (applied to every run automatically).
DELTA_FLAGS=(
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)

run() {
  local name="$1"; shift
  local logf="$LOG_ROOT/$name.log"
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE F — starting run: $name"
  echo " log: $logf"
  echo " started: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  python -m training.train_precise_v2 "$@" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE F — finished:   $name  (exit=$rc)"
  echo " finished: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  return 0   # do not abort the queue on a single failure
}

# ── α: augmentation pack ─────────────────────────────────────────────────
run phaseF_alpha "${COMMON_BASE[@]}" \
  --head_type cosine \
  --corpus_noise_prob 0.3 --corpus_noise_snr_min -3 --corpus_noise_snr_max 15 \
  --noise_pool_max 4096 --noise_pool_quantile 0.25 \
  --rir_prob 0.2 --pitch_prob 0.2 --branch_dropout_p 0.1 \
  --run_name phaseF_alpha "${DELTA_FLAGS[@]}"

# ── β: preprocessing ─────────────────────────────────────────────────────
run phaseF_beta "${COMMON_BASE[@]}" \
  --head_type cosine \
  --rms_normalize --target_rms 0.1 \
  --hpf_hz 20.0 --hpf_order 4 \
  --run_name phaseF_beta "${DELTA_FLAGS[@]}"

# ── γ: pretrained wav2vec2 + ArcFace ─────────────────────────────────────
run phaseF_gamma "${COMMON_BASE[@]}" \
  --use_pretrained_branch --pretrained_model facebook/wav2vec2-base \
  --pretrained_ch 128 \
  --head_type arcface --arcface_margin 0.3 --arcface_scale 30 \
  --mixup_alpha 0.1 \
  --run_name phaseF_gamma "${DELTA_FLAGS[@]}"

# ── α+β+γ stacked (target_rms=1.0 because wav2vec2 expects ~unit-RMS) ────
run phaseF_stacked "${COMMON_BASE[@]}" \
  --use_pretrained_branch --pretrained_model facebook/wav2vec2-base \
  --pretrained_ch 128 \
  --head_type arcface --arcface_margin 0.3 --arcface_scale 30 \
  --mixup_alpha 0.1 \
  --corpus_noise_prob 0.3 --corpus_noise_snr_min -3 --corpus_noise_snr_max 15 \
  --noise_pool_max 4096 --noise_pool_quantile 0.25 \
  --rir_prob 0.2 --pitch_prob 0.2 --branch_dropout_p 0.1 \
  --rms_normalize --target_rms 1.0 \
  --hpf_hz 20.0 --hpf_order 4 \
  --run_name phaseF_stacked "${DELTA_FLAGS[@]}"

echo
echo "════════════════════════════════════════════════════════════════"
echo " PHASE F — all four runs completed."
echo " logs: $LOG_ROOT/"
echo " ckpts/cal: lightning_logs/phaseF_{alpha,beta,gamma,stacked}/version_*/checkpoints/"
echo "════════════════════════════════════════════════════════════════"
