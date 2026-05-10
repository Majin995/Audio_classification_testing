#!/usr/bin/env bash
# Overnight orchestrator — chains specialist variations + students.
#
# Sequence (after v1 specialists already running finish):
#   1. v1 student (4 v1 teachers → 1 student)
#   2. v2 high-precision specialists (4 binary teachers)
#   3. v3 LDAM-DRW specialists (4 binary teachers)
#   4. v4 second-seed specialists (4 binary teachers)
#   5. N=16 final student (4 teachers/class × 4 variations)
#
# Wall-clock estimate: ~3.5 days end-to-end. Safe to leave running;
# each step writes its own console log under phaseJ_console/. Steps
# 2-5 only start after the prior step finishes successfully — if any
# step fails (exit != 0) the chain still continues but skips the
# affected step's downstream data.

set -uo pipefail
DATA_DIR="${DATA_DIR:-/run/media/damo/Lexar M2/Data/Classifier_Dataset}"
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$PROJ_ROOT/lightning_logs/phaseJ_console}"
mkdir -p "$LOG_ROOT"
cd "$PROJ_ROOT"

# ── Helpers ───────────────────────────────────────────────────────────

# Find the highest-precision ckpt (filename: ``hydra-NNN-p0.XXXX.ckpt``).
# Extracts the precision number, prepends as a tab-separated sort key,
# sorts numerically descending, returns the top filename. Robust to
# 'p' characters appearing elsewhere in the path.
find_best_ckpt() {
    local dir="$1"
    [ -d "$dir" ] || { echo ""; return; }
    ls -1 "$dir"/hydra-*-p*.ckpt 2>/dev/null \
        | sed -E "s|.*/hydra-[0-9]+-p([0-9.]+)\\.ckpt$|\\1\t&|" \
        | sort -t$'\t' -k1,1 -gr \
        | head -1 \
        | cut -f2-
}

# Wait until a process whose argv matches a pattern is gone.
# Empty match => returns immediately.
wait_for_pattern_to_exit() {
    local pat="$1"
    while pgrep -f "$pat" > /dev/null 2>&1; do
        sleep 30
    done
}

# ── Step 1: wait for v1 specialists ───────────────────────────────────
echo "[overnight $(date -Is)] waiting for v1 specialists to complete..."
wait_for_pattern_to_exit "run_specialists.sh"
echo "[overnight $(date -Is)] v1 specialists orchestrator exited."

V1_CARGO=$(find_best_ckpt "lightning_logs/phaseJ_specialist_cargo/version_0/checkpoints")
V1_PASSENGER=$(find_best_ckpt "lightning_logs/phaseJ_specialist_passenger/version_0/checkpoints")
V1_TANKER=$(find_best_ckpt "lightning_logs/phaseJ_specialist_tanker/version_0/checkpoints")
V1_TUG=$(find_best_ckpt "lightning_logs/phaseJ_specialist_tug/version_0/checkpoints")
echo "[overnight] v1 ckpts:"
echo "  Cargo     = $V1_CARGO"
echo "  Passenger = $V1_PASSENGER"
echo "  Tanker    = $V1_TANKER"
echo "  Tug       = $V1_TUG"

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

# ── Step 2: v1 student ────────────────────────────────────────────────
if [ -n "$V1_CARGO" ] && [ -n "$V1_PASSENGER" ] && [ -n "$V1_TANKER" ] && [ -n "$V1_TUG" ]; then
    NAME="phaseJ_student_v1"
    LOG="$LOG_ROOT/$NAME.log"
    echo "[overnight $(date -Is)] starting v1 student → $LOG"
    python -m training.train_hydra_student \
        --teacher_ckpts "$V1_CARGO" "$V1_PASSENGER" "$V1_TANKER" "$V1_TUG" \
        "${R1_FLAGS[@]}" "${KD_FLAGS[@]}" \
        --run_name "$NAME" \
        > "$LOG" 2>&1
    echo "[overnight $(date -Is)] v1 student finished (exit=$?)."
else
    echo "[overnight] WARNING: v1 ckpts missing — skipping v1 student."
fi

# ── Step 3: v2 high-precision specialists ─────────────────────────────
echo "[overnight $(date -Is)] starting v2 specialists ..."
bash scripts/run_specialists_v2_highprec.sh
echo "[overnight $(date -Is)] v2 specialists exited."

# ── Step 4: v3 LDAM-DRW specialists ───────────────────────────────────
echo "[overnight $(date -Is)] starting v3 specialists ..."
bash scripts/run_specialists_v3_ldam.sh
echo "[overnight $(date -Is)] v3 specialists exited."

# ── Step 5: v4 seed-42 specialists ────────────────────────────────────
echo "[overnight $(date -Is)] starting v4 specialists ..."
bash scripts/run_specialists_v4_s42.sh
echo "[overnight $(date -Is)] v4 specialists exited."

# ── Step 6: N=16 final student (v1+v2+v3+v4 per class) ────────────────
collect_class() {
    local cls_lower="$1"
    local out=()
    for v in "" "_v2" "_v3" "_v4"; do
        local dir="lightning_logs/phaseJ_specialist${v}_${cls_lower}/version_0/checkpoints"
        local ck=$(find_best_ckpt "$dir")
        if [ -n "$ck" ]; then
            out+=("$ck")
        fi
    done
    printf '%s\n' "${out[@]}"
}

CARGO_GROUP=( $(collect_class cargo) )
PASSENGER_GROUP=( $(collect_class passenger) )
TANKER_GROUP=( $(collect_class tanker) )
TUG_GROUP=( $(collect_class tug) )

echo "[overnight] final N-teach groups:"
echo "  Cargo     (${#CARGO_GROUP[@]}):     ${CARGO_GROUP[*]}"
echo "  Passenger (${#PASSENGER_GROUP[@]}): ${PASSENGER_GROUP[*]}"
echo "  Tanker    (${#TANKER_GROUP[@]}):    ${TANKER_GROUP[*]}"
echo "  Tug       (${#TUG_GROUP[@]}):       ${TUG_GROUP[*]}"

if [ ${#CARGO_GROUP[@]} -gt 0 ] && [ ${#PASSENGER_GROUP[@]} -gt 0 ] \
   && [ ${#TANKER_GROUP[@]} -gt 0 ] && [ ${#TUG_GROUP[@]} -gt 0 ]; then
    NAME="phaseJ_student_final"
    LOG="$LOG_ROOT/$NAME.log"
    echo "[overnight $(date -Is)] starting final student → $LOG"
    python -m training.train_hydra_student \
        --teacher_cargo     "${CARGO_GROUP[@]}" \
        --teacher_passenger "${PASSENGER_GROUP[@]}" \
        --teacher_tanker    "${TANKER_GROUP[@]}" \
        --teacher_tug       "${TUG_GROUP[@]}" \
        "${R1_FLAGS[@]}" "${KD_FLAGS[@]}" \
        --run_name "$NAME" \
        > "$LOG" 2>&1
    echo "[overnight $(date -Is)] final student finished (exit=$?)."
else
    echo "[overnight] WARNING: at least one class has no teacher — skipping final student."
fi

echo "[overnight $(date -Is)] OVERNIGHT CHAIN COMPLETE."
