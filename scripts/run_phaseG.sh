#!/usr/bin/env bash
# Phase G — HydroHydra waveform-only (no-spectrogram) precision improvements.
# Runs R1 → R2 → R3 → R4 sequentially on a single GPU.
# Each run writes its own lightning_logs/<name>/ tree with checkpoints,
# temperature.pt, thresholds.json, and selective_pr.md.
#
# Logging: each run's stdout+stderr is redirected directly to
#   $LOG_ROOT/<name>.log  via simple `>` redirect (no tee, no pipefail).
# A run failure is recorded but does not abort subsequent runs.

DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseG_console}"
mkdir -p "$LOG_ROOT"

cd "$(dirname "$0")/.."

# Common base flags. Margin 0.7 is the HydroHydra baseline grid winner.
# fusion_dim=192 / s4_n_blocks=2 are the historical defaults.
COMMON_BASE=(
  --data_dir "$DATA_DIR"
  --use_gabor --use_scattering
  --use_sincnet --use_tdsbe
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 --label_smoothing 0.05
  --max_epochs 80 --patience 15
  --batch_size 64 --num_threads 8
  --seed 1337
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)

# β augmentation pack flags.
BETA_FLAGS=(
  --corpus_noise_prob 0.5 --corpus_noise_snr_min -3 --corpus_noise_snr_max 15
  --noise_pool_max 4096 --noise_pool_quantile 0.25
  --rir_prob 0.3 --pitch_prob 0.3 --branch_dropout_p 0.15
)

# γ ArcFace head flags.
GAMMA_FLAGS=(
  --head_type arcface --feature_norm layernorm_l2
  --arcface_margin 0.2 --arcface_scale 30.0
  --gambler_weight 0.0
)

run() {
  local name="$1"; shift
  local logf="$LOG_ROOT/$name.log"
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE G — starting run: $name"
  echo " log: $logf"
  echo " started: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  python training/train_hydra.py "$@" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE G — finished:   $name  (exit=$rc)"
  echo " finished: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  return 0
}

# ── R1: α only (sincnet + tdsbe) — branches delta vs baseline ────────────
run phaseG_R1_alpha "${COMMON_BASE[@]}" \
  --run_name phaseG_R1_alpha

# ── R2: α + β (augmentation pack) ────────────────────────────────────────
run phaseG_R2_alphabeta "${COMMON_BASE[@]}" \
  "${BETA_FLAGS[@]}" \
  --run_name phaseG_R2_alphabeta

# ── R3: α + β + γ (ArcFace, candidate ship config) ───────────────────────
run phaseG_R3_full "${COMMON_BASE[@]}" \
  "${BETA_FLAGS[@]}" "${GAMMA_FLAGS[@]}" \
  --run_name phaseG_R3_full

# ── R4: R3 + wav2vec2 conv front-end ─────────────────────────────────────
run phaseG_R4_w2v "${COMMON_BASE[@]}" \
  "${BETA_FLAGS[@]}" "${GAMMA_FLAGS[@]}" \
  --use_w2v --w2v_ch 128 --w2v_target_sr 16000 \
  --run_name phaseG_R4_w2v

echo
echo "════════════════════════════════════════════════════════════════"
echo " PHASE G — all four runs completed."
echo " logs: $LOG_ROOT/"
echo " ckpts/cal: lightning_logs/phaseG_{R1_alpha,R2_alphabeta,R3_full,R4_w2v}/version_*/checkpoints/"
echo "════════════════════════════════════════════════════════════════"
