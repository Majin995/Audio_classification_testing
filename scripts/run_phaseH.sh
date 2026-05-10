#!/usr/bin/env bash
# Phase H — waveform-only research-driven extensions on top of Phase G R1.
#
# R1 baseline: --use_gabor --use_scattering --use_sincnet --use_tdsbe with
# LMF loss, MLP+Gamblers head. Best val/μP=0.6908, test/μP=0.6620.
#
# Phase H renditions add one lever each (P0/P1 from the literature scan):
#   H1  LDAM-DRW loss        (P0 — tanker class-imbalance lift)
#   H2  JTFS-lite scattering (P0 — Δ/ΔΔ modulation features)
#   H3  Global attention     (P0 — HELIX-lite hybrid SSM+attn)
#   H4  Sub-Center ArcFace   (P1 — heterogeneous-class-aware angular margin)
#   H5  DEMON-MoE head       (P1 — expert routing for tanker/cargo)
#   H6  Manifold mixup       (P1 — hidden-state regularizer)
#   H7  Stacked P0 combo     (LDAM + JTFS + global attn together)
#
# Sequential, one GPU, ~6h each → ~42h wall.

set -uo pipefail
DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
LOG_ROOT="${LOG_ROOT:-/var/home/damo/Documents/Git/Audio_classification_testing/lightning_logs/phaseH_console}"
mkdir -p "$LOG_ROOT"
cd "$(dirname "$0")/.."

COMMON=(
  --data_dir "$DATA_DIR"
  --use_gabor --use_scattering --use_sincnet --use_tdsbe
  --max_epochs 80 --patience 15
  --batch_size 64 --num_threads 8
  --seed 1337
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)

LMF_FLAGS=(
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 --label_smoothing 0.05
  --gambler_weight 0.1 --gambler_o 0.3
)

run() {
  local name="$1"; shift
  local logf="$LOG_ROOT/$name.log"
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE H — starting run: $name"
  echo " log: $logf"
  echo " started: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  python training/train_hydra.py "$@" --run_name "$name" > "$logf" 2>&1
  local rc=$?
  echo "════════════════════════════════════════════════════════════════"
  echo " PHASE H — finished:   $name  (exit=$rc)"
  echo " finished: $(date -Is)"
  echo "════════════════════════════════════════════════════════════════"
  return 0
}

# ── H1: LDAM-DRW loss ────────────────────────────────────────────────────
# Replaces LMF entirely; DRW activates at epoch 40.
run phaseH_H1_ldam_drw "${COMMON[@]}" \
  --loss ldam --ldam_max_m 0.5 --ldam_s 30.0 --ldam_drw_epoch 40 \
  --label_smoothing 0.05 \
  --gambler_weight 0.0

# ── H2: JTFS-lite (Δ + ΔΔ on scattering coeffs) ──────────────────────────
run phaseH_H2_jtfs "${COMMON[@]}" "${LMF_FLAGS[@]}" --use_jtfs

# ── H3: Global attention block (HELIX-lite) ──────────────────────────────
run phaseH_H3_globalattn "${COMMON[@]}" "${LMF_FLAGS[@]}" \
  --use_global_attn --global_attn_heads 4

# ── H4: Sub-Center ArcFace K=2 ───────────────────────────────────────────
# ArcFace zeroes gamblers internally; scale=15 to avoid Phase G's regression.
run phaseH_H4_subcenter_arcface "${COMMON[@]}" \
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.5 --label_smoothing 0.05 \
  --head_type subcenter_arcface --arcface_subcenters 2 \
  --feature_norm layernorm_l2 --arcface_margin 0.2 --arcface_scale 15.0

# ── H5: DEMON-MoE 4-expert head ──────────────────────────────────────────
run phaseH_H5_demon_moe "${COMMON[@]}" \
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 --label_smoothing 0.05 \
  --gambler_weight 0.0 \
  --head_type demon_moe --moe_n_experts 4 --moe_aux_weight 0.05

# ── H6: Manifold mixup ───────────────────────────────────────────────────
run phaseH_H6_manifold_mixup "${COMMON[@]}" "${LMF_FLAGS[@]}" \
  --manifold_mixup_alpha 0.4 --manifold_mixup_prob 0.5

# ── H7: Stacked P0 combo (LDAM + JTFS + global attn) ────────────────────
run phaseH_H7_stacked_p0 "${COMMON[@]}" \
  --loss ldam --ldam_max_m 0.5 --ldam_s 30.0 --ldam_drw_epoch 40 \
  --label_smoothing 0.05 --gambler_weight 0.0 \
  --use_jtfs \
  --use_global_attn --global_attn_heads 4

echo
echo "════════════════════════════════════════════════════════════════"
echo " PHASE H — all 7 renditions completed."
echo "════════════════════════════════════════════════════════════════"
