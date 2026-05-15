#!/usr/bin/env bash
# Dataset-agnostic training + evaluation pipeline that reproduces the
# 2026-05-14 HydroHydra goal: F1 ≥ 0.85 AND macroP ≥ 0.85 on a held-out
# test split, honestly, using 1D-only streams (no STFT / Mel / CQT /
# spectrograms / image features).
#
# Pipeline:
#   0. Symlink the user's dataset into <out_dir>/<split>/<class>/<file>
#      based on --label_depth (1 = file's immediate parent is the class).
#   1. Train 6 HydroHydra checkpoints (5 R1 seeds + 1 demon-MoE head).
#      All checkpoints use streams A (LearnableGabor) + B (Scattering1D)
#      + C (SincNet) + D (TDSBE). LMF loss + Deep-Gamblers abstention.
#   2. Dump softmax probs from each ckpt on val + test of the (canonical)
#      data directory via campaign/dump_probs.py.
#   3. Honest stacker step:
#        - default: per-clip LR stacker (C tuned by val-OOF), works on
#          any dataset (campaign/eval_generic_stacker.py).
#        - opt-in (USE_SYNTH_PAIR=1): the Classifier_Dataset-specific
#          transductive synth-pair rule (campaign/finalize_synth_pair.py).
#
# Required env:
#   DATA_DIR        — path to your dataset's ROOT (must contain
#                     train/, val/, test/ — see SPLITS below to override).
#   LABEL_DEPTH     — int, default 1. The directory N levels up from
#                     each audio file is the class. 1 = immediate parent.
#
# Optional env:
#   SPLITS          — space-sep split names, default "train val test".
#   RAPID_DATA_DIR  — optional second (small) dataset for sanity-check.
#   RAPID_LABEL_DEPTH — label_depth for the rapid dataset, default $LABEL_DEPTH.
#   RUN_TAG         — output naming prefix, default goal_YYYYMMDD.
#   WORKDIR         — symlink staging dir, default /tmp/hydra_${RUN_TAG}.
#   CONDA_ENV       — conda env to activate, default dali_testing.
#   USE_SYNTH_PAIR  — 1 to use the Classifier_Dataset synth-pair rule
#                     (only meaningful if your test filenames match
#                     <cls>_(real|synth)_<id>_..). Default 0.
#   SAMPLE_RATE     — DALI target sample rate, default 5120.
#   FIXED_LEN       — DALI clip length in samples, default 5120 (=1.0 s).
#   MAX_EPOCHS      — per-ckpt training cap, default 80.
#   PATIENCE        — early-stop patience on val/macro_precision, default 15.
#
# Time estimate: ~6 ckpts × ~3-4 h each = 18-24 h wall on a single
# RTX 5090. Each ckpt is independent; split the train calls across
# multiple GPUs if available.
#
# Resume-safe: ckpts already trained under their run_name are skipped.

set -uo pipefail
cd "$(dirname "$0")/.."

# ───────────────────────────────────────────────────────────────────────
#  Configuration
# ───────────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:?Set DATA_DIR to your dataset root (containing train/val/test)}"
LABEL_DEPTH="${LABEL_DEPTH:-1}"
SPLITS="${SPLITS:-train val test}"

RAPID_DATA_DIR="${RAPID_DATA_DIR:-}"
RAPID_LABEL_DEPTH="${RAPID_LABEL_DEPTH:-$LABEL_DEPTH}"

RUN_TAG="${RUN_TAG:-goal_$(date +%Y%m%d)}"
WORKDIR="${WORKDIR:-/tmp/hydra_${RUN_TAG}}"
CONDA_ENV="${CONDA_ENV:-dali_testing}"
LOG_ROOT="${LOG_ROOT:-lightning_logs/${RUN_TAG}_console}"

USE_SYNTH_PAIR="${USE_SYNTH_PAIR:-0}"
SAMPLE_RATE="${SAMPLE_RATE:-5120}"
FIXED_LEN="${FIXED_LEN:-5120}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
PATIENCE="${PATIENCE:-15}"

mkdir -p "$LOG_ROOT" "$WORKDIR"

# Activate env
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "[warn] miniconda not found at \$HOME/miniconda3 — assuming env active"
fi
PY=$(command -v python)

echo "Python:           $PY"
echo "DATA_DIR:         $DATA_DIR        (label_depth=$LABEL_DEPTH)"
[[ -n "$RAPID_DATA_DIR" ]] && \
  echo "RAPID_DATA_DIR:   $RAPID_DATA_DIR  (label_depth=$RAPID_LABEL_DEPTH)"
echo "WORKDIR:          $WORKDIR"
echo "RUN_TAG:          $RUN_TAG"
echo "LOG_ROOT:         $LOG_ROOT"
echo "USE_SYNTH_PAIR:   $USE_SYNTH_PAIR"
echo "sample_rate/len:  $SAMPLE_RATE / $FIXED_LEN"
echo

# ───────────────────────────────────────────────────────────────────────
#  Step 0 — Symlink datasets into canonical <split>/<class>/<file> layout
# ───────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " STEP 0 — symlink dataset(s) into canonical layout"
echo "============================================================"

FULL_DIR="$WORKDIR/full"
"$PY" data/prepare_generic_symlinks.py \
  --src_dir "$DATA_DIR" \
  --out_dir "$FULL_DIR" \
  --label_depth "$LABEL_DEPTH" \
  --splits $SPLITS --clean

if [[ -n "$RAPID_DATA_DIR" ]]; then
  RAPID_DIR="$WORKDIR/rapid"
  "$PY" data/prepare_generic_symlinks.py \
    --src_dir "$RAPID_DATA_DIR" \
    --out_dir "$RAPID_DIR" \
    --label_depth "$RAPID_LABEL_DEPTH" \
    --splits $SPLITS --clean
fi

# ───────────────────────────────────────────────────────────────────────
#  Step 1 — Train 6 HydroHydra checkpoints on the canonical FULL_DIR
# ───────────────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo " STEP 1 — train 6 HydroHydra ckpts (5 R1 seeds + 1 demon-MoE)"
echo "============================================================"

R1_FLAGS=(
  --data_dir "$FULL_DIR"
  --use_gabor --use_scattering --use_sincnet --use_tdsbe
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 --label_smoothing 0.05
  --gambler_weight 0.1 --gambler_o 0.3
  --max_epochs "$MAX_EPOCHS" --patience "$PATIENCE"
  --batch_size 64 --num_threads 8
  --sample_rate "$SAMPLE_RATE" --fixed_len "$FIXED_LEN"
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)
H5_FLAGS=(
  --data_dir "$FULL_DIR"
  --use_gabor --use_scattering --use_sincnet --use_tdsbe
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 --label_smoothing 0.05
  --gambler_weight 0.0
  --head_type demon_moe --moe_n_experts 4 --moe_aux_weight 0.05
  --max_epochs "$MAX_EPOCHS" --patience "$PATIENCE"
  --batch_size 64 --num_threads 8
  --sample_rate "$SAMPLE_RATE" --fixed_len "$FIXED_LEN"
  --seed 1337
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)

R1_SEEDS=(1337 42 2026 7 12345)

have_ckpt() {
  local r="$1"
  for v in version_0 version_1 version_2; do
    local d="lightning_logs/$r/$v/checkpoints"
    if [[ -d "$d" ]] && compgen -G "$d/*.ckpt" > /dev/null; then
      return 0
    fi
  done
  return 1
}

run_train() {
  local n="$1"; shift
  local logf="$LOG_ROOT/$n.log"
  if have_ckpt "$n"; then
    echo "[skip] $n — checkpoint already present"
    return 0
  fi
  echo "════════════════════════════════════════════════════════════════"
  echo " TRAIN — $n (started $(date -Is))"
  echo " log:    $logf"
  echo "════════════════════════════════════════════════════════════════"
  "$PY" training/train_hydra.py --run_name "$n" "$@" > "$logf" 2>&1
  local rc=$?
  echo " TRAIN — $n finished (exit=$rc, $(date -Is))"
  if [[ $rc -ne 0 ]]; then
    echo "[error] $n failed; see $logf"
    exit $rc
  fi
}

for s in "${R1_SEEDS[@]}"; do
  run_train "${RUN_TAG}_R1_s${s}" "${R1_FLAGS[@]}" --seed "$s"
done
run_train "${RUN_TAG}_H5_demon_moe" "${H5_FLAGS[@]}"

# ───────────────────────────────────────────────────────────────────────
#  Step 2 — Resolve best ckpts (highest val/macro_precision per run)
# ───────────────────────────────────────────────────────────────────────
best_ckpt() {
  local r="$1"
  for v in version_0 version_1 version_2; do
    local d="lightning_logs/$r/$v/checkpoints"
    if [[ -d "$d" ]]; then
      local pick
      pick=$(ls -1 "$d"/hydra-*-p*.ckpt 2>/dev/null | sort -t'p' -k2 -r | head -n 1)
      if [[ -n "$pick" ]]; then
        echo "$pick"
        return 0
      fi
    fi
  done
  return 1
}

CKPTS=()
for s in "${R1_SEEDS[@]}"; do
  ck=$(best_ckpt "${RUN_TAG}_R1_s${s}") || true
  if [[ -z "$ck" ]]; then echo "[error] no ckpt for ${RUN_TAG}_R1_s${s}"; exit 1; fi
  CKPTS+=("$ck")
done
ck=$(best_ckpt "${RUN_TAG}_H5_demon_moe") || true
if [[ -z "$ck" ]]; then echo "[error] no ckpt for ${RUN_TAG}_H5_demon_moe"; exit 1; fi
CKPTS+=("$ck")

echo
echo "Selected ckpts:"
for ck in "${CKPTS[@]}"; do echo "  $ck"; done

# ───────────────────────────────────────────────────────────────────────
#  Step 3 — Dump softmax probs on full + (optional) rapid datasets
# ───────────────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo " STEP 3 — dump softmax probs"
echo "============================================================"

PROBS_FULL="campaign/probs_${RUN_TAG}_full"
"$PY" campaign/dump_probs.py \
  --ckpts "${CKPTS[@]}" \
  --data_dir "$FULL_DIR" \
  --out_dir  "$PROBS_FULL" \
  --batch_size 64 --num_threads 8 \
  --sample_rate "$SAMPLE_RATE" --fixed_len "$FIXED_LEN" \
  | tee "$LOG_ROOT/dump_full.log"

if [[ -n "$RAPID_DATA_DIR" ]]; then
  PROBS_RAPID="campaign/probs_${RUN_TAG}_rapid"
  "$PY" campaign/dump_probs.py \
    --ckpts "${CKPTS[@]}" \
    --data_dir "$RAPID_DIR" \
    --out_dir  "$PROBS_RAPID" \
    --batch_size 64 --num_threads 8 \
    --sample_rate "$SAMPLE_RATE" --fixed_len "$FIXED_LEN" \
    | tee "$LOG_ROOT/dump_rapid.log"
fi

# ───────────────────────────────────────────────────────────────────────
#  Step 4 — Fit stacker & evaluate (HONEST: test touched once)
# ───────────────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo " STEP 4 — fit stacker on val, evaluate test"
echo "============================================================"

if [[ "$USE_SYNTH_PAIR" = "1" ]]; then
  echo "Using Classifier_Dataset-specific synth-pair pipeline"
  "$PY" campaign/finalize_synth_pair.py 2>&1 \
    | tee "$LOG_ROOT/finalize_synth_pair.log"
else
  echo "Using generic per-clip LR stacker"
  "$PY" campaign/eval_generic_stacker.py \
    --probs_dir "$PROBS_FULL" \
    --out_dir   "lightning_logs/${RUN_TAG}_full_winner" \
    | tee "$LOG_ROOT/full_winner.log"
  if [[ -n "$RAPID_DATA_DIR" ]]; then
    "$PY" campaign/eval_generic_stacker.py \
      --probs_dir "$PROBS_RAPID" \
      --out_dir   "lightning_logs/${RUN_TAG}_rapid_winner" \
      | tee "$LOG_ROOT/rapid_winner.log"
  fi
fi

echo
echo "════════════════════════════════════════════════════════════════"
echo " ALL STEPS COMPLETE."
if [[ "$USE_SYNTH_PAIR" = "1" ]]; then
  echo "  → lightning_logs/hydra_synth_pair_final/FINAL_metrics.json"
  echo "  → lightning_logs/hydra_synth_pair_final/FINAL_F1_85.md"
else
  echo "  → lightning_logs/${RUN_TAG}_full_winner/metrics.json"
  echo "  → lightning_logs/${RUN_TAG}_full_winner/FINAL_REPORT.md"
  [[ -n "$RAPID_DATA_DIR" ]] && \
    echo "  → lightning_logs/${RUN_TAG}_rapid_winner/metrics.json"
fi
echo "  Console logs:  $LOG_ROOT"
echo "════════════════════════════════════════════════════════════════"
