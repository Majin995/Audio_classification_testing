#!/usr/bin/env bash
# Optuna TPE sweep wrapper for I2HOFI in LOFAR mode.
# Search axes (defined in scripts/optuna_sweep.py::_objective_i2hofi):
#   - hz_per_grid       in [0.5, 5.0]   (drives n_fft + grid_h)
#   - dropout_appnp     in [0.0, 0.4]
#   - dropout_gat       in [0.0, 0.4]
#   - dropout_classifier in [0.0, 0.5]
#   - lr                in [1e-4, 1e-3] (log)
#   - batch_size        in {16, 32}
#   - backbone          in {resnet18, resnet34}
#
# optuna-dashboard auto-launches at http://localhost:8080.
#
# Usage:
#   bash scripts/grid_i2hofi.sh
#   N_TRIALS=80 bash scripts/grid_i2hofi.sh
#   DRY_RUN=1   bash scripts/grid_i2hofi.sh

set -uo pipefail
export DATA_DIR="${DATA_DIR:-$PWD/data/Split1s}"

N_TRIALS="${N_TRIALS:-40}"

CMD=(python scripts/optuna_sweep.py
     --model     i2hofi
     --data_dir  "$DATA_DIR"
     --n_trials  "$N_TRIALS")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry_run)
fi

echo "=== i2hofi (LOFAR) Optuna sweep — n_trials=$N_TRIALS ==="
exec "${CMD[@]}"
