#!/usr/bin/env bash
# Resume overnight chain: v4 specialists (all 4 classes, fresh) + final N-teacher student.
# v1/v2/v3 specialists and v1 student are already complete — skip them.

set -uo pipefail
DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$PROJ_ROOT/lightning_logs/phaseJ_console}"
mkdir -p "$LOG_ROOT"
cd "$PROJ_ROOT"

# ── Helpers ───────────────────────────────────────────────────────────

# Find the best ckpt across ALL versions of a run directory.
# Handles cases where a prior partial run left version_0 and the fresh
# re-run lands in version_1.
find_best_ckpt_run() {
    local run_dir="$1"
    [ -d "$run_dir" ] || { echo ""; return; }
    find "$run_dir" -name "hydra-*-p*.ckpt" 2>/dev/null \
        | sed -E "s|.*/hydra-[0-9]+-p([0-9.]+)\.ckpt$|\1\t&|" \
        | sort -t$'\t' -k1,1 -gr \
        | head -1 \
        | cut -f2-
}

R1_FLAGS=(
  --data_dir "$DATA_DIR"
  --use_gabor --use_scattering --use_sincnet --use_tdsbe
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 --label_smoothing 0.05
  --gambler_weight 0.1 --gambler_o 0.3
  --max_epochs 80 --patience 15
  --batch_size 64 --num_threads 8
  --seed 1337
  --recall_floor 0.6
  --selective_coverages 0.70,0.75,0.80,0.85,0.90,0.95
)
KD_FLAGS=( --kd_alpha 0.7 --kd_temperature 4.0 )

# ── Step 1: v4 specialists (all 4 classes, fresh run) ─────────────────
echo "[resume $(date -Is)] starting v4 specialists ..."
bash scripts/run_specialists_v4_s42.sh
echo "[resume $(date -Is)] v4 specialists exited."

# ── Step 2: collect best ckpts from every variant ─────────────────────
collect_class() {
    local cls_lower="$1"
    local out=()
    for v in "" "_v2" "_v3" "_v4"; do
        local run_dir="lightning_logs/phaseJ_specialist${v}_${cls_lower}"
        local ck
        ck=$(find_best_ckpt_run "$run_dir")
        [ -n "$ck" ] && out+=("$ck")
    done
    printf '%s\n' "${out[@]}"
}

CARGO_GROUP=(     $(collect_class cargo)     )
PASSENGER_GROUP=( $(collect_class passenger) )
TANKER_GROUP=(    $(collect_class tanker)    )
TUG_GROUP=(       $(collect_class tug)       )

echo "[resume] final N-teach groups:"
echo "  Cargo     (${#CARGO_GROUP[@]}):     ${CARGO_GROUP[*]}"
echo "  Passenger (${#PASSENGER_GROUP[@]}): ${PASSENGER_GROUP[*]}"
echo "  Tanker    (${#TANKER_GROUP[@]}):    ${TANKER_GROUP[*]}"
echo "  Tug       (${#TUG_GROUP[@]}):       ${TUG_GROUP[*]}"

# ── Step 3: final N-teacher student ───────────────────────────────────
if [ ${#CARGO_GROUP[@]} -gt 0 ] && [ ${#PASSENGER_GROUP[@]} -gt 0 ] \
   && [ ${#TANKER_GROUP[@]} -gt 0 ] && [ ${#TUG_GROUP[@]} -gt 0 ]; then
    NAME="phaseJ_student_final"
    LOG="$LOG_ROOT/$NAME.log"
    echo "[resume $(date -Is)] starting final student → $LOG"
    python -m training.train_hydra_student \
        --teacher_cargo     "${CARGO_GROUP[@]}" \
        --teacher_passenger "${PASSENGER_GROUP[@]}" \
        --teacher_tanker    "${TANKER_GROUP[@]}" \
        --teacher_tug       "${TUG_GROUP[@]}" \
        "${R1_FLAGS[@]}" "${KD_FLAGS[@]}" \
        --run_name "$NAME" \
        > "$LOG" 2>&1
    echo "[resume $(date -Is)] final student finished (exit=$?)."
else
    echo "[resume] WARNING: at least one class has no teacher — skipping final student."
fi

echo "[resume $(date -Is)] RESUME CHAIN COMPLETE."
