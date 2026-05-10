#!/usr/bin/env bash
# Grid search for HydroALSI (Wav2Vec2 + CQT-ResNet dual-stream)
# Axes: seed × lr × denoise × freeze_wav2vec × fusion_heads
# Total runs: 2 × 2 × 2 × 2 × 2 = 32
#
# NOTE: ALSI uses batch_size=32 by default due to Wav2Vec2 memory.
#       Run on a ≥16 GB GPU or reduce batch further if OOM.
#
# Usage:
#   bash scripts/grid_alsi.sh
#   DRY_RUN=1 bash scripts/grid_alsi.sh

set -uo pipefail
export DATA_DIR="${DATA_DIR:-$PWD/data/Split1s}"

SEEDS=(42 1337)
LRS=(1e-4 3e-4)
DENOISE=(off emd_wavelet)
FREEZE_W2V=(true false)
FUSION_HEADS=(4 8)

MAX_EPOCHS=60
PATIENCE=15

for seed in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for dn in "${DENOISE[@]}"; do
      for fw in "${FREEZE_W2V[@]}"; do
        for fh in "${FUSION_HEADS[@]}"; do

          RUN="alsi_s${seed}_lr${lr}_dn${dn}_fw${fw}_fh${fh}"

          CMD="python -m training.train_alsi \
            --data_dir \"$DATA_DIR\" \
            --seed $seed \
            --lr $lr \
            --batch_size 32 \
            --denoise $dn \
            --freeze_wav2vec $fw \
            --fusion_heads $fh \
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
