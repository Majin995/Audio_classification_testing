#!/usr/bin/env bash
# Grid search for HydroDCN (Deep Complex Network)
# Axes: seed × lr × batch_size × denoise × dcmf_templates × complex_depth
# Total runs: 2 × 2 × 2 × 2 × 2 × 2 = 64  (reduce with DRY_RUN first)
#
# Usage:
#   bash scripts/grid_dcn.sh
#   DRY_RUN=1 bash scripts/grid_dcn.sh

set -uo pipefail
export DATA_DIR="${DATA_DIR:-$PWD/data/Split1s}"

SEEDS=(42 1337)
LRS=(3e-4 1e-3)
BATCHES=(32 64)
DENOISE=(off emd_wavelet)
DCMF_TEMPLATES=(16 32)
COMPLEX_DEPTHS=(3 4)

MAX_EPOCHS=60
PATIENCE=15

for seed in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for bs in "${BATCHES[@]}"; do
      for dn in "${DENOISE[@]}"; do
        for tmpl in "${DCMF_TEMPLATES[@]}"; do
          for depth in "${COMPLEX_DEPTHS[@]}"; do

            RUN="dcn_s${seed}_lr${lr}_bs${bs}_dn${dn}_t${tmpl}_d${depth}"

            CMD="python -m training.train_dcn \
              --data_dir \"$DATA_DIR\" \
              --seed $seed \
              --lr $lr \
              --batch_size $bs \
              --denoise $dn \
              --dcmf_templates $tmpl \
              --complex_depth $depth \
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
