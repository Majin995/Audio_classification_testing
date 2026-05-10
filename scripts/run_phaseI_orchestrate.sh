#!/usr/bin/env bash
# Orchestrator: wait for the running Phase H followup to finish, then run
# Phase I in order:
#   1. Model soup over Phase G/H best ckpts
#   2. Main Phase I sweep (I1, I1b, I2, I3, I4)
#   3. Compile Phase I summary
#
# Launch once and walk away. Designed for unattended overnight execution.

set -uo pipefail
ROOT=/var/home/damo/Documents/Git/Audio_classification_testing
LOG_ROOT="$ROOT/lightning_logs/phaseI_console"
mkdir -p "$LOG_ROOT"
cd "$ROOT"

ORCH_LOG="$LOG_ROOT/orchestrator.log"
exec >> "$ORCH_LOG" 2>&1

echo "════════════════════════════════════════════════════════════════"
echo " PHASE I ORCHESTRATOR  $(date -Is)"
echo "════════════════════════════════════════════════════════════════"

# 1. Wait for any running train_hydra.py to finish.
echo " waiting for any running train_hydra.py to exit …"
while pgrep -f 'train_hydra.py' > /dev/null; do
  sleep 60
done
echo " no train_hydra running.  $(date -Is)"

# 2. Model soup over the strongest MLP-head ckpts
echo "════════════════════════════════════════════════════════════════"
echo " STEP 1 — model soup  $(date -Is)"
echo "════════════════════════════════════════════════════════════════"
SOUP_DIR="$ROOT/lightning_logs/phaseI_soup_g_h"
mkdir -p "$SOUP_DIR"
# Soup-eligible candidates: same architecture (MLP+Gambler head), R1 config.
# H1 / H1b use a different loss (LDAM vs LMF) — same architecture so
# state_dicts ARE compatible; soup tolerates loss-config differences.
# Skip H7 (different feature flags), H4/H5 (different head).
CANDIDATES=(
  "lightning_logs/phaseG_R1_alpha/version_0/checkpoints/hydra-026-p0.6908.ckpt"
  "lightning_logs/phaseH_H1_ldam_drw/version_0/checkpoints/hydra-032-p0.7035.ckpt"
  "lightning_logs/phaseH_H1b_ldam_drw_early/version_0/checkpoints/hydra-050-p0.7123.ckpt"
)
EXISTING=()
for c in "${CANDIDATES[@]}"; do
  [ -f "$c" ] && EXISTING+=("$c")
done
if [ "${#EXISTING[@]}" -lt 2 ]; then
  echo " not enough soup candidates — skipping soup step"
else
  python -m scripts.model_soup --ckpts "${EXISTING[@]}" \
    --data_dir "${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}" \
    --out_dir "$SOUP_DIR"
  echo " soup done  $(date -Is)"
fi

# 3. Main Phase I sweep
echo "════════════════════════════════════════════════════════════════"
echo " STEP 2 — Phase I sweep  $(date -Is)"
echo "════════════════════════════════════════════════════════════════"
bash scripts/run_phaseI.sh

# 4. Compile results
echo "════════════════════════════════════════════════════════════════"
echo " STEP 3 — compile Phase I summary  $(date -Is)"
echo "════════════════════════════════════════════════════════════════"
python scripts/compile_phaseI_results.py

echo "════════════════════════════════════════════════════════════════"
echo " PHASE I ORCHESTRATOR done  $(date -Is)"
echo "════════════════════════════════════════════════════════════════"
