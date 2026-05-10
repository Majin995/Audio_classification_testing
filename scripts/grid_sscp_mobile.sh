#!/usr/bin/env bash
# Grid search for HydroSSCPMobile (edge CNN + Knowledge Distillation)
# Axes: seed × lr × batch_size × denoise × kd_temp × kd_alpha
#
# REQUIRES: $BAHTNET_BEST_CKPT must be set to the best BAHTNet checkpoint.
#           run_all_grids.sh sets this automatically; when running standalone:
#             export BAHTNET_BEST_CKPT=/path/to/bahtnet-best.ckpt
#
# Usage:
#   BAHTNET_BEST_CKPT=/path/to/ckpt bash scripts/grid_sscp_mobile.sh
#   DRY_RUN=1 BAHTNET_BEST_CKPT="" bash scripts/grid_sscp_mobile.sh

set -uo pipefail
export DATA_DIR="${DATA_DIR:-$PWD/data/Split1s}"
TEACHER_CKPT="${BAHTNET_BEST_CKPT:-}"

if [[ -z "$TEACHER_CKPT" ]]; then
  echo "WARNING: BAHTNET_BEST_CKPT not set — SSCP-Mobile will train without KD."
fi

SEEDS=(42 1337)
LRS=(5e-4 1e-3)
BATCHES=(64 128)
DENOISE=(off emd_wavelet)
KD_TEMPS=(2.0 4.0)
KD_ALPHAS=(0.3 0.5)

MAX_EPOCHS=60
PATIENCE=20

for seed in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for bs in "${BATCHES[@]}"; do
      for dn in "${DENOISE[@]}"; do
        for temp in "${KD_TEMPS[@]}"; do
          for alpha in "${KD_ALPHAS[@]}"; do

            RUN="sscp_s${seed}_lr${lr}_bs${bs}_dn${dn}_T${temp}_a${alpha}"

            CMD="python -m training.train_sscp_mobile \
              --data_dir \"$DATA_DIR\" \
              --seed $seed \
              --lr $lr \
              --batch_size $bs \
              --denoise $dn \
              --kd_temp $temp \
              --kd_alpha $alpha \
              --max_epochs $MAX_EPOCHS \
              --patience $PATIENCE \
              --run_name $RUN"

            # Append teacher ckpt if available
            if [[ -n "$TEACHER_CKPT" ]]; then
              CMD="$CMD --teacher_ckpt \"$TEACHER_CKPT\""
            fi

            if [[ "${DRY_RUN:-0}" == "1" ]]; then
              echo "[DRY_RUN] $CMD"
            else
              echo "=== $RUN ==="
              eval "$CMD" || echo "!! FAILED: $RUN"
            fi

          done
        done
      done
    done
  done
done
