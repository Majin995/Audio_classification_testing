#!/usr/bin/env bash
# Grid search for HydroPrecise — optimise for macro-precision + gated precision.
#
# Axes (training-time):
#   --lmf_margin         M : 0.30, 0.50, 0.70, 0.90
#   --lmf_gamma          γ : 2.0, 3.0
#   --label_smoothing    ε : 0.00, 0.05
#   --gambler_weight     λ : 0.0, 0.1      (λ=0 disables abstention aux loss)
#
# Post-hoc: temperature T and per-class thresholds are fit automatically by
# train_precise.py.  We additionally scan three target-coverage floors
# (0.80 / 0.85 / 0.90) by invoking a side-car Python block that replays the
# threshold search on the already-saved val probabilities.
#
# Total training runs: 4 × 2 × 2 × 2 = 32
# Each run: 60 max epochs, patience 15, ≈ 15–25 min on a single RTX GPU.
#
# Usage:
#   export DATA_DIR=/abs/path/to/Split1s_3cls
#   bash scripts/grid_precise.sh                   # run the full grid
#   DRY_RUN=1 bash scripts/grid_precise.sh         # print commands only
#   MARGINS="0.5 0.7" bash scripts/grid_precise.sh # override an axis
#
# Outputs:
#   lightning_logs/grid_precise/<run_name>/...     # per-trial logs + ckpts
#   lightning_logs/grid_precise/summary.csv        # aggregate results table

set -uo pipefail

# ─── Environment / defaults ──────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/Split1s_3cls}"
PYTHON="${PYTHON:-python}"
LOG_ROOT="$REPO_ROOT/lightning_logs"
GRID_DIR="$LOG_ROOT/grid_precise"
SUMMARY_CSV="$GRID_DIR/summary.csv"
mkdir -p "$GRID_DIR"

# ─── Swept axes (override via env vars) ──────────────────────────────────
MARGINS=(${MARGINS:-0.30 0.50 0.70 0.90})
GAMMAS=(${GAMMAS:-2.0 3.0})
SMOOTHINGS=(${SMOOTHINGS:-0.00 0.05})
GAMBLER_WEIGHTS=(${GAMBLER_WEIGHTS:-0.0 0.1})
COVERAGES=(${COVERAGES:-0.80 0.85 0.90})

# ─── Fixed params (edit here or override per-env) ────────────────────────
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-60}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-8}"
PATIENCE="${PATIENCE:-15}"
LR="${LR:-3e-4}"
SEED="${SEED:-42}"
DENOISE="${DENOISE:-off}"
PRECISION_FLAG="${PRECISION_FLAG:-bf16-mixed}"

# ─── Write header if fresh ───────────────────────────────────────────────
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "run_name,margin,gamma,smoothing,gambler_w,best_epoch,val_macro_prec,gated_prec@0.80,cov@0.80,gated_prec@0.85,cov@0.85,gated_prec@0.90,cov@0.90,temperature,best_ckpt" > "$SUMMARY_CSV"
fi

# ─── Post-hoc multi-coverage scan (Python one-liner fed per run) ─────────
extract_multi_coverage() {
  local ckpt_dir="$1"
  local best_ckpt="$2"
  local coverages_csv="$3"   # e.g. "0.80,0.85,0.90"
  $PYTHON - "$best_ckpt" "$ckpt_dir" "$coverages_csv" <<'PY'
import json, os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.environ["REPO_ROOT"])
from models.hydro_precise import HydroPrecise
from data.audio_lightning_loader import DALIAudioDataModule
from training.train_precise import _collect_logits, _fit_temperature, _search_thresholds

best_ckpt, ckpt_dir, coverages_csv = sys.argv[1], sys.argv[2], sys.argv[3]
coverages = [float(c) for c in coverages_csv.split(",")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = HydroPrecise.load_from_checkpoint(best_ckpt, map_location=device, strict=False).to(device).eval()

data = DALIAudioDataModule(
    data_dir=os.environ["DATA_DIR"],
    batch_size=int(os.environ.get("BATCH_SIZE", 64)), num_threads=8,
    target_sr=5120, fixed_len=5120, oversample_train=True,
    denoise_method=os.environ.get("DENOISE", "off"),
)
data.setup()
logits, targets = _collect_logits(m, data.val_dataloader(), device)
T = _fit_temperature(logits, targets)
probs = F.softmax(logits / T, dim=-1)

out = {"temperature": T, "by_coverage": {}}
for cov in coverages:
    res = _search_thresholds(probs, targets, num_classes=m.num_classes, target_coverage=cov)
    out["by_coverage"][f"{cov:.2f}"] = res
with open(os.path.join(ckpt_dir, "multi_coverage.json"), "w") as f:
    json.dump(out, f, indent=2)
print("WROTE", os.path.join(ckpt_dir, "multi_coverage.json"))
PY
}
export REPO_ROOT BATCH_SIZE DENOISE

# ─── Run the grid ────────────────────────────────────────────────────────
TOTAL=$((${#MARGINS[@]} * ${#GAMMAS[@]} * ${#SMOOTHINGS[@]} * ${#GAMBLER_WEIGHTS[@]}))
i=0
for M in "${MARGINS[@]}"; do
  for G in "${GAMMAS[@]}"; do
    for S in "${SMOOTHINGS[@]}"; do
      for GW in "${GAMBLER_WEIGHTS[@]}"; do

        i=$((i + 1))
        RUN="grid_precise_m${M}_g${G}_s${S}_gw${GW}"
        RUN_DIR="$GRID_DIR/$RUN"

        CMD="$PYTHON training/train_precise.py \
            --run_name \"grid_precise/$RUN\" \
            --data_dir \"$DATA_DIR\" \
            --batch_size $BATCH_SIZE \
            --max_epochs $MAX_EPOCHS \
            --warmup_epochs $WARMUP_EPOCHS \
            --patience $PATIENCE \
            --denoise $DENOISE \
            --lmf_margin $M \
            --lmf_gamma $G \
            --label_smoothing $S \
            --gambler_weight $GW \
            --lr $LR \
            --seed $SEED \
            --precision $PRECISION_FLAG \
            --target_coverage 0.85"

        if [[ "${DRY_RUN:-0}" == "1" ]]; then
          echo "[DRY_RUN $i/$TOTAL] $CMD"
          continue
        fi

        echo "=== [$i/$TOTAL] $RUN ==="
        LOG_FILE="$GRID_DIR/${RUN}.log"
        eval "$CMD" > "$LOG_FILE" 2>&1 || {
          echo "!! FAILED: $RUN (see $LOG_FILE)"
          continue
        }

        # ── Parse results ──────────────────────────────────────────────
        # Use the "Best checkpoint:" line printed by train_precise.py — this
        # path is guaranteed to exist (save_top_k=3 may delete earlier ckpts).
        BEST_CKPT_PATH=$(grep -E "^Best checkpoint:" "$LOG_FILE" | tail -1 | awk '{print $3}')
        BEST_CKPT=$(basename "$BEST_CKPT_PATH" 2>/dev/null || true)
        BEST_EPOCH=$(echo "$BEST_CKPT" | grep -oE "precise-[0-9]+" | grep -oE "[0-9]+")
        VAL_PREC=$(grep -oE "Best val/macro_prec:\s+[0-9.]+" "$LOG_FILE" | grep -oE "[0-9.]+$")
        TEMP=$(grep -oE "temperature\s+=\s+[0-9.]+" "$LOG_FILE" | grep -oE "[0-9.]+$")

        if [[ -z "$BEST_CKPT_PATH" ]] || [[ ! -f "$BEST_CKPT_PATH" ]]; then
          echo "!! No best ckpt path for $RUN"
          continue
        fi
        CKPT_DIR=$(dirname "$BEST_CKPT_PATH")

        # ── Sweep multiple coverage floors using the saved ckpt ────────
        COV_CSV=$(IFS=,; echo "${COVERAGES[*]}")
        extract_multi_coverage "$CKPT_DIR" "$BEST_CKPT_PATH" "$COV_CSV" || true

        # Extract gated precision per target coverage from the JSON
        MC_JSON="$CKPT_DIR/multi_coverage.json"
        if [[ -f "$MC_JSON" ]]; then
          gp80=$(jq -r '.by_coverage["0.80"].macro_precision // empty' "$MC_JSON")
          co80=$(jq -r '.by_coverage["0.80"].coverage        // empty' "$MC_JSON")
          gp85=$(jq -r '.by_coverage["0.85"].macro_precision // empty' "$MC_JSON")
          co85=$(jq -r '.by_coverage["0.85"].coverage        // empty' "$MC_JSON")
          gp90=$(jq -r '.by_coverage["0.90"].macro_precision // empty' "$MC_JSON")
          co90=$(jq -r '.by_coverage["0.90"].coverage        // empty' "$MC_JSON")
        else
          gp80=""; co80=""; gp85=""; co85=""; gp90=""; co90=""
        fi

        echo "$RUN,$M,$G,$S,$GW,${BEST_EPOCH:-?},${VAL_PREC:-?},${gp80},${co80},${gp85},${co85},${gp90},${co90},${TEMP:-?},${BEST_CKPT_PATH}" >> "$SUMMARY_CSV"
        echo "  → val_prec=${VAL_PREC}  gated@0.85=${gp85}  (cov=${co85})"
      done
    done
  done
done

# ─── Sort summary by gated@0.85 descending ───────────────────────────────
if [[ "${DRY_RUN:-0}" != "1" ]] && [[ -f "$SUMMARY_CSV" ]]; then
  HEAD=$(head -1 "$SUMMARY_CSV")
  TAIL=$(tail -n +2 "$SUMMARY_CSV" | sort -t, -k10,10 -gr)
  echo -e "$HEAD\n$TAIL" > "$SUMMARY_CSV"
  echo ""
  echo "─── Grid complete.  Top 5 by gated@0.85 ───"
  head -6 "$SUMMARY_CSV" | column -t -s,
fi
