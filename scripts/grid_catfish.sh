#!/usr/bin/env bash
# Grid search for HydroCATFISH
# Axes: seed × lr × batch_size × denoise × gabor_n_filters
# Total runs: 2 × 2 × 2 × 2 × 2 = 32  (exhaustive initial sweep)
#
# Usage:
#   bash scripts/grid_catfish.sh
#   DRY_RUN=1 bash scripts/grid_catfish.sh   # echo commands only

set -uo pipefail
export DATA_DIR="${DATA_DIR:-$PWD/data/Split1s}"

SEEDS=(42 1337)
LRS=(3e-4 1e-3)
BATCHES=(32 64)
DENOISE=(off emd_wavelet)
GABOR_FILTERS=(32 64)

MAX_EPOCHS=60
PATIENCE=15

for seed in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for bs in "${BATCHES[@]}"; do
      for dn in "${DENOISE[@]}"; do
        for gf in "${GABOR_FILTERS[@]}"; do

          RUN="catfish_s${seed}_lr${lr}_bs${bs}_dn${dn}_gf${gf}"

          CMD="python -m training.train_catfish \
            --data_dir \"$DATA_DIR\" \
            --seed $seed \
            --lr $lr \
            --batch_size $bs \
            --denoise $dn \
            --gabor_n_filters $gf \
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
