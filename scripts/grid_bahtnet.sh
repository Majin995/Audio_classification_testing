#!/usr/bin/env bash
# Grid search for HydroBAHTNet (Boundary-Aware Hybrid Transformer)
# Axes: seed × lr × batch_size × denoise × loss_type × lmf_margin
# Total runs: 2 × 2 × 2 × 2 × 2 × 3 = 96  (use DRY_RUN to review first)
#
# IMPORTANT: BAHTNet checkpoints are used as the teacher for SSCP-Mobile.
#            The run_all_grids.sh driver picks the best BAHTNet ckpt automatically.
#
# Usage:
#   bash scripts/grid_bahtnet.sh
#   DRY_RUN=1 bash scripts/grid_bahtnet.sh

set -uo pipefail
export DATA_DIR="${DATA_DIR:-$PWD/data/Split1s}"

SEEDS=(42 1337)
LRS=(1e-4 3e-4)
BATCHES=(32 64)
DENOISE=(off emd_wavelet)
LOSSES=(focal lmf)
LMF_MARGINS=(0.20 0.35 0.50)

MAX_EPOCHS=60
PATIENCE=15

for seed in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for bs in "${BATCHES[@]}"; do
      for dn in "${DENOISE[@]}"; do
        for loss in "${LOSSES[@]}"; do

          # Sweep margins only for LMF; focal has no margin
          if [[ "$loss" == "focal" ]]; then
            margins=("0.00")
          else
            margins=("${LMF_MARGINS[@]}")
          fi

          for margin in "${margins[@]}"; do

            RUN="bahtnet_s${seed}_lr${lr}_bs${bs}_dn${dn}_${loss}_m${margin}"

            CMD="python -m training.train_bahtnet \
              --data_dir \"$DATA_DIR\" \
              --seed $seed \
              --lr $lr \
              --batch_size $bs \
              --denoise $dn \
              --loss $loss \
              --lmf_margin $margin \
              --max_epochs $MAX_EPOCHS \
              --patience $PATIENCE \
              --run_name $RUN"

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
