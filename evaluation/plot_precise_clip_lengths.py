"""Compare HydroPrecise across 4 DeepShip clip lengths (1/5/15/30s).

Reads each run's train.log, extracts the macro test metrics and confusion
matrix (the 4x4 array printed by ``on_test_epoch_end``), computes per-class
P/R/F1, and writes 3 PNG images (one per metric) to ``reports/``.

Usage:
    python -m evaluation.plot_precise_clip_lengths
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT  = Path("/var/home/damo/Documents/Git/Audio_classification_testing")
LOGS     = PROJECT / "lightning_logs"
OUT_DIR  = PROJECT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES   = ("Cargo", "Passenger", "Tanker", "Tug")
LENGTHS_S = (1, 5, 15, 30)


# ── log parsers ─────────────────────────────────────────────────────────

def _read_log(length_s: int) -> str:
    return (LOGS / f"precise_deepship_{length_s}s" / "train.log").read_text()


_TEST_LINE = re.compile(r"│\s*(test/[a-z0-9_]+)\s*│\s*([0-9.eE+-]+|nan)\s*│")


def parse_test_metrics(log: str) -> dict:
    """Pulls every ``test/<key> = <value>`` from the rich-table block."""
    out = {}
    # Grab the LAST "Test metric" block in the file (handles smoke-then-real).
    block_starts = [m.start() for m in re.finditer(r"Test metric", log)]
    if not block_starts:
        return out
    block = log[block_starts[-1]:]
    for m in _TEST_LINE.finditer(block):
        key, val = m.group(1), m.group(2)
        try:
            out[key] = float(val)
        except ValueError:
            out[key] = float("nan")
    return out


def parse_confusion_matrix(log: str) -> np.ndarray:
    """Pulls the LAST 4x4 confusion matrix from the log.

    Format printed:
        Confusion Matrix:
        [[ a  b  c  d]
         [ e  f  g  h]
         [ i  j  k  l]
         [ m  n  o  p]]
    """
    starts = [m.end() for m in re.finditer(r"Confusion Matrix:", log)]
    if not starts:
        raise ValueError("no confusion matrix found")
    tail = log[starts[-1]:]
    # Capture text up to closing ']]'
    m = re.search(r"\[\[(.*?)\]\]", tail, flags=re.DOTALL)
    if m is None:
        raise ValueError("could not parse CM block")
    body = "[[" + m.group(1) + "]]"
    body = body.replace("\n", " ")
    # Split into rows ["[r0]", "[r1]", ...]
    rows = re.findall(r"\[([^\[\]]+)\]", body)
    cm = np.array([[int(x) for x in r.split()] for r in rows[:4]])
    if cm.shape != (4, 4):
        raise ValueError(f"unexpected CM shape: {cm.shape}")
    return cm


def per_class_metrics(cm: np.ndarray) -> dict:
    """Return per-class P/R/F1 from a square confusion matrix.

    Convention: rows=true, cols=predicted (matches torchmetrics).
    """
    out = {"P": np.zeros(4), "R": np.zeros(4), "F1": np.zeros(4)}
    for i in range(4):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out["P"][i]  = p
        out["R"][i]  = r
        out["F1"][i] = f1
    return out


# ── plot helpers ────────────────────────────────────────────────────────

def _bar_plot(metric_name: str, key: str, results: dict[int, dict],
              outpath: Path) -> None:
    """Grouped bar chart: x = clip length, bar groups = per-class + macro."""
    n_groups = len(LENGTHS_S)
    bar_w   = 0.16
    x       = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=140)

    # Per-class
    palette = plt.cm.tab10(np.linspace(0, 0.6, len(CLASSES)))
    for ci, cls in enumerate(CLASSES):
        vals = [results[L]["per_class"][key][ci] for L in LENGTHS_S]
        ax.bar(x + (ci - 2) * bar_w, vals, bar_w, label=cls, color=palette[ci])

    # Macro overlay (last group)
    macros = [results[L]["macro"][key] for L in LENGTHS_S]
    ax.bar(x + 2 * bar_w, macros, bar_w, label="Macro avg",
           color="black", alpha=0.85, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{L}s" for L in LENGTHS_S], fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"HydroPrecise — test {metric_name} vs DeepShip clip length",
                 fontsize=13, pad=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", ncol=5, fontsize=9, framealpha=0.95)
    # Numeric labels on macro bars
    for xi, v in zip(x + 2 * bar_w, macros):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def main():
    results: dict[int, dict] = {}
    for L in LENGTHS_S:
        try:
            log = _read_log(L)
        except FileNotFoundError:
            print(f"[skip] no log for {L}s")
            continue
        cm = parse_confusion_matrix(log)
        macro = parse_test_metrics(log)
        per   = per_class_metrics(cm)
        # Map macro keys to short names; map ``test/recall`` (which is macro_recall in torchmetrics)
        macro_short = {
            "P":  macro.get("test/macro_precision", float("nan")),
            "R":  macro.get("test/recall",          float("nan")),
            "F1": macro.get("test/f1",              float("nan")),
        }
        results[L] = {
            "macro":     macro_short,
            "per_class": per,
            "cm":        cm,
        }
        print(f"[{L}s] macro P={macro_short['P']:.3f}  "
              f"R={macro_short['R']:.3f}  F1={macro_short['F1']:.3f}")
        for ci, cls in enumerate(CLASSES):
            print(f"   {cls:>10}  P={per['P'][ci]:.3f}  "
                  f"R={per['R'][ci]:.3f}  F1={per['F1'][ci]:.3f}")

    if len(results) < 2:
        print("[error] need at least 2 runs; have", list(results))
        return

    _bar_plot("Precision", "P",  results, OUT_DIR / "precise_deepship_precision.png")
    _bar_plot("Recall",    "R",  results, OUT_DIR / "precise_deepship_recall.png")
    _bar_plot("F1",        "F1", results, OUT_DIR / "precise_deepship_f1.png")
    print("\nDone. Plots in", OUT_DIR)


if __name__ == "__main__":
    main()
