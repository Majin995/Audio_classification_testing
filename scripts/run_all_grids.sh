#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_all_grids.sh — Sequential Optuna TPE sweep driver for all 5 UATR models
#
# Order is fixed:
#   catfish → alsi → dcn → bahtnet → sscp_mobile
#
# BAHTNet MUST finish before SSCP-Mobile because SSCP-Mobile uses the best
# BAHTNet checkpoint as its KD teacher.  The driver reads the ckpt path from
# optuna_bahtnet_best.txt, which optuna_sweep.py writes after BAHTNet completes.
#
# Usage:
#   conda activate dali_testing
#   export DATA_DIR=/path/to/Split1s
#   bash scripts/run_all_grids.sh
#
# Dry-run (print trial params, no training):
#   DRY_RUN=1 bash scripts/run_all_grids.sh
#
# Run a subset:
#   MODELS="catfish dcn" bash scripts/run_all_grids.sh
#
# Custom trial budget:
#   N_TRIALS=20 bash scripts/run_all_grids.sh
#
# Skip dashboard launch (e.g. in headless CI):
#   NO_DASHBOARD=1 bash scripts/run_all_grids.sh
#
# Logging:
#   All output is tee'd to grid_logs/<timestamp>/<model>.log
# ═══════════════════════════════════════════════════════════════════════════════

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/Split1s}"
N_TRIALS="${N_TRIALS:-40}"
LOG_DIR="$REPO_ROOT/grid_logs/$(date +%Y%m%d_%H%M%S)"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

mkdir -p "$LOG_DIR"

# ── Resolve Python interpreter ────────────────────────────────────────────────
# Using the conda env's Python binary directly avoids `conda run` wrapper bugs
# (conda run --no-capture-output mis-reports exit codes and truncates stderr).
# Preference order:
#   1. $PYTHON env var (explicit override)
#   2. dali_testing conda env bin/python (has DALI + optuna + pytorch-lightning)
#   3. Current python if optuna is importable
_DALI_PYTHON="$(conda run -n dali_testing which python 2>/dev/null || true)"

if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
  PYTHON_CMD=("${PYTHON}")
elif [[ -n "$_DALI_PYTHON" && -x "$_DALI_PYTHON" ]]; then
  PYTHON_CMD=("$_DALI_PYTHON")
elif python -c "import optuna" 2>/dev/null; then
  PYTHON_CMD=(python)
else
  echo "ERROR: optuna not found and dali_testing conda env does not exist."
  echo "  Install optuna:  pip install optuna optuna-dashboard"
  echo "  Or activate env: conda activate dali_testing"
  exit 1
fi

SWEEP_SCRIPT="$SCRIPT_DIR/optuna_sweep.py"

# ── Verify optuna is importable in the chosen interpreter ────────────────────
if ! "${PYTHON_CMD[@]}" -c "import optuna" 2>/dev/null; then
  echo "ERROR: optuna not importable via: ${PYTHON_CMD[*]}"
  echo "  Install with: pip install optuna optuna-dashboard"
  exit 1
fi

# ── Build ordered model list ─────────────────────────────────────────────────
MODELS_DEFAULT="catfish alsi dcn bahtnet sscp_mobile"
IFS=' ' read -ra ORDER <<< "${MODELS:-$MODELS_DEFAULT}"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  UATR Optuna Sweep Driver                                            ║"
echo "║  Models  : ${ORDER[*]}"
echo "║  Python  : ${PYTHON_CMD[*]}"
echo "║  Data    : $DATA_DIR"
echo "║  Trials  : $N_TRIALS per model"
echo "║  Logs    : $LOG_DIR"
echo "║  DRY_RUN : ${DRY_RUN:-0}"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# ── Launch optuna-dashboard once at the start ─────────────────────────────────
if [[ "${NO_DASHBOARD:-0}" != "1" ]]; then
  echo "[driver] Starting optuna-dashboard on port $DASHBOARD_PORT…"
  "${PYTHON_CMD[@]}" "$SWEEP_SCRIPT" --dashboard_only --port "$DASHBOARD_PORT" || true
  echo ""
fi

# ── Per-model sweeps ──────────────────────────────────────────────────────────
for m in "${ORDER[@]}"; do
  _SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "$_SEP"
  echo "SWEEP: $m   ($(date +%H:%M:%S))"
  echo "$_SEP"
  # Mirror the header into the model log so it's self-contained
  { echo "$_SEP"; echo "SWEEP: $m   ($(date +%H:%M:%S))"; echo "$_SEP"; } \
    >> "${LOG_DIR}/${m}.log"

  # Build the command as a bash array — avoids all eval/quoting hazards
  CMD_ARGS=(
    "${PYTHON_CMD[@]}" "$SWEEP_SCRIPT"
    --model "$m"
    --data_dir "$DATA_DIR"
    --n_trials "$N_TRIALS"
    --no_dashboard
    --port "$DASHBOARD_PORT"
  )

  [[ "${DRY_RUN:-0}" == "1" ]] && CMD_ARGS+=(--dry_run)

  # ── SSCP-Mobile: resolve BAHTNet teacher ──────────────────────────────────
  if [[ "$m" == "sscp_mobile" ]]; then
    CKPT_CACHE="$REPO_ROOT/optuna_bahtnet_best.txt"
    if [[ -n "${BAHTNET_BEST_CKPT:-}" ]]; then
      CMD_ARGS+=(--teacher_ckpt "$BAHTNET_BEST_CKPT")
      echo "[driver] Teacher (env var):    $BAHTNET_BEST_CKPT"
    elif [[ -f "$CKPT_CACHE" ]]; then
      TEACHER_CKPT="$(cat "$CKPT_CACHE")"
      CMD_ARGS+=(--teacher_ckpt "$TEACHER_CKPT")
      echo "[driver] Teacher (cache file): $TEACHER_CKPT"
    else
      echo "[driver] WARNING: No BAHTNet checkpoint found."
      echo "[driver] SSCP-Mobile will run WITHOUT a teacher (supervised only)."
    fi
  fi

  echo "[driver] ${CMD_ARGS[*]}"
  echo ""

  # Append both stdout and stderr directly to the log file — no pipe, no tee.
  # Using | tee was causing SIGPIPE (exit 141): DALI forks worker processes that
  # inherit the write end of the pipe; a race between workers and tee closing its
  # read end propagated SIGPIPE into Python and killed the training process before
  # a single epoch completed.  A plain >> redirect to a regular file can never
  # produce SIGPIPE, so this is the correct approach for background sweeps.
  # Follow progress with:  tail -f "${LOG_DIR}/${m}.log"
  "${CMD_ARGS[@]}" >> "${LOG_DIR}/${m}.log" 2>&1
  EXIT_CODE="$?"

  if [[ "${EXIT_CODE}" -eq 0 ]]; then
    echo "[driver] $m sweep COMPLETED"
  else
    echo "[driver] $m sweep FAILED (exit ${EXIT_CODE}) — continuing to next model"
    echo "!! $m FAILED (exit ${EXIT_CODE})" >> "${LOG_DIR}/${m}.log"
  fi

  echo ""
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "ALL SWEEPS COMPLETE   $(date)"
echo "Logs     : $LOG_DIR"
echo "Storage  : $REPO_ROOT/optuna_studies.db"
echo "Dashboard: http://localhost:$DASHBOARD_PORT"
echo ""
echo "Results summary (best val/f1 per model):"
for m in "${ORDER[@]}"; do
  LOG="${LOG_DIR}/${m}.log"
  if [[ -f "$LOG" ]]; then
    BEST=$(grep -oP 'Best val/f1:\s+\K[\d.]+' "$LOG" | sort -n | tail -1)
    printf "  %-20s val/f1 = %s\n" "$m" "${BEST:-N/A}"
  fi
done
echo "═══════════════════════════════════════════════════════════════════════"
