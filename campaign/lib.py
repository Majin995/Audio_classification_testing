"""Shared utilities for the post-hoc calibration / threshold campaign.

All functions operate on softmax probabilities (np.ndarray, shape (N, C))
and integer targets. No model forward passes happen here — all inference
is replayed from the dumps written by ``dump_probs.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


# ──────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────

def load_dump(npz_path: str | Path) -> Dict[str, np.ndarray]:
    z = np.load(npz_path)
    return {k: z[k] for k in z.files}


def load_meta(probs_dir: str | Path) -> Dict:
    with open(Path(probs_dir) / "_meta.json") as f:
        return json.load(f)


def stack_probs(probs_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return (val_P_stack, val_y, test_P_stack, test_y, ckpt_stems).

    P_stack has shape (M, N, C) — M models stacked.
    """
    meta = load_meta(probs_dir)
    val_P, test_P = [], []
    val_y = test_y = None
    stems = []
    for entry in meta["ckpts"]:
        d = load_dump(entry["npz"])
        val_P.append(d["val_probs"])
        test_P.append(d["test_probs"])
        val_y = d["val_y"]
        test_y = d["test_y"]
        stems.append(entry["stem"])
    return np.stack(val_P), val_y, np.stack(test_P), test_y, stems


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def metrics(probs: np.ndarray, y: np.ndarray, num_classes: int) -> Dict:
    pred = probs.argmax(axis=1)
    out = {
        "acc":     float(accuracy_score(y, pred)),
        "f1":      float(f1_score(y, pred, average="macro", zero_division=0)),
        "recall":  float(recall_score(y, pred, average="macro", zero_division=0)),
        "micro_P": float(precision_score(y, pred, average="micro", zero_division=0)),
        "macro_P": float(precision_score(y, pred, average="macro", zero_division=0)),
        "mcc":     float(matthews_corrcoef(y, pred)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y, probs, multi_class="ovr"))
    except Exception:
        out["auroc"] = float("nan")
    for c in range(num_classes):
        m = pred == c
        n = int(m.sum())
        out[f"P_class_{c}"] = float((y[m] == c).mean()) if n > 0 else float("nan")
        m_t = y == c
        n_t = int(m_t.sum())
        out[f"R_class_{c}"] = float((pred[m_t] == c).mean()) if n_t > 0 else float("nan")
        # per-class F1
        prec = out[f"P_class_{c}"]
        rec = out[f"R_class_{c}"]
        if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
            out[f"F1_class_{c}"] = float("nan")
        else:
            out[f"F1_class_{c}"] = 2 * prec * rec / (prec + rec)
    return out


def metrics_with_thresholds(probs: np.ndarray, y: np.ndarray,
                            thresholds: np.ndarray, num_classes: int) -> Dict:
    """Apply per-class minimum-confidence thresholds, then evaluate.

    Decision rule: predict class k iff k = argmax_c probs[c] AND probs[k] >= thresholds[k].
    Otherwise abstain. Abstained samples are NOT counted in precision (precision
    is over predicted samples) but ARE counted in recall (recall is over true samples).
    """
    argmax = probs.argmax(axis=1)
    p_max = probs[np.arange(len(probs)), argmax]
    accept = p_max >= thresholds[argmax]
    pred_full = np.where(accept, argmax, -1)  # -1 = abstain
    accepted = accept.sum()
    coverage = float(accept.mean())
    out = {"coverage": coverage, "n_accepted": int(accepted)}
    if accepted == 0:
        for k in ["acc", "f1", "recall", "micro_P", "macro_P", "mcc"]:
            out[k] = 0.0
        for c in range(num_classes):
            out[f"P_class_{c}"] = float("nan")
            out[f"R_class_{c}"] = 0.0
            out[f"F1_class_{c}"] = 0.0
        return out
    # Precision: only over accepted samples
    pred_acc = argmax[accept]
    y_acc = y[accept]
    out["acc"] = float((pred_acc == y_acc).mean())
    out["macro_P"] = float(precision_score(y_acc, pred_acc, labels=list(range(num_classes)),
                                            average="macro", zero_division=0))
    out["micro_P"] = float(precision_score(y_acc, pred_acc, average="micro", zero_division=0))
    # Recall: per true class, fraction predicted correctly across all samples
    rec_per = []
    p_per = []
    f1_per = []
    for c in range(num_classes):
        mask_t = y == c
        n_t = int(mask_t.sum())
        rec = float((pred_full[mask_t] == c).mean()) if n_t > 0 else 0.0
        out[f"R_class_{c}"] = rec
        mask_p = pred_full == c
        n_p = int(mask_p.sum())
        p = float((y[mask_p] == c).mean()) if n_p > 0 else float("nan")
        out[f"P_class_{c}"] = p
        if not np.isnan(p) and (p + rec) > 0:
            f1 = 2 * p * rec / (p + rec)
        else:
            f1 = 0.0
        out[f"F1_class_{c}"] = f1
        rec_per.append(rec)
        f1_per.append(f1)
        if not np.isnan(p):
            p_per.append(p)
    out["recall"] = float(np.mean(rec_per))
    out["f1"] = float(np.mean(f1_per))
    # MCC over predicted samples
    out["mcc"] = float(matthews_corrcoef(y_acc, pred_acc))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────

def fit_temperature(probs: np.ndarray, y: np.ndarray,
                    grid: Optional[Sequence[float]] = None) -> float:
    """Grid-search T over a wide range; return T minimizing val NLL."""
    if grid is None:
        grid = np.linspace(0.5, 5.0, 91)  # step 0.05
    eps = 1e-8
    logits = np.log(probs + eps)
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        scaled = logits / T
        # softmax
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        e = np.exp(scaled)
        p = e / e.sum(axis=1, keepdims=True)
        # NLL
        nll = -np.log(p[np.arange(len(y)), y] + eps).mean()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    """Re-softmax probs at a given temperature. Probs treated as exp(logits)/Z;
    we scale the underlying log-probs by 1/T."""
    eps = 1e-8
    logits = np.log(probs + eps)
    scaled = logits / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(axis=1, keepdims=True)


# ──────────────────────────────────────────────────────────────────────────
# Threshold search — F1 with per-class precision floor
# ──────────────────────────────────────────────────────────────────────────

def search_thresholds_f1_with_p_floor(
    probs: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    p_floor: float = 0.65,
    grid: Optional[Sequence[float]] = None,
    max_iters: int = 5,
) -> Dict:
    """Coordinate-ascent: per-class threshold maximizing macro-F1 subject to
    per-class precision >= p_floor (only enforced where the class has any
    predictions; if a class has zero accepted predictions we treat its
    precision as satisfying the floor since precision is undefined).
    """
    if grid is None:
        grid = np.linspace(0.0, 0.95, 20)
    grid = list(grid)
    thr = np.zeros(num_classes)

    def score(thr_arr):
        m = metrics_with_thresholds(probs, y, thr_arr, num_classes)
        # Check per-class precision floor (NaN treated as OK)
        ok = True
        for c in range(num_classes):
            p = m[f"P_class_{c}"]
            if not np.isnan(p) and p < p_floor:
                ok = False
                break
        return m["f1"] if ok else -1.0, m

    best_f1, best_m = score(thr)
    best_thr = thr.copy()
    for _ in range(max_iters):
        improved = False
        for c in range(num_classes):
            for g in grid:
                cand = thr.copy()
                cand[c] = g
                f1, m = score(cand)
                if f1 > best_f1 + 1e-6:
                    best_f1, best_m, best_thr = f1, m, cand
                    thr = cand
                    improved = True
        if not improved:
            break
    return {
        "p_floor": p_floor,
        "thresholds": best_thr.tolist(),
        "metrics": best_m,
    }


# ──────────────────────────────────────────────────────────────────────────
# Cost-sensitive Bayes-optimal decision rule
# ──────────────────────────────────────────────────────────────────────────

def apply_cost_rule(probs: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Bayes-optimal action under a num_classes x num_classes utility matrix
    U where U[k, j] = utility of predicting k when truth is j.
    Returns argmax over actions of expected utility = probs @ U.T.
    """
    # E[U | x, action k] = sum_j P(j|x) * U[k, j] = (probs @ U.T)[k]
    expected = probs @ U.T  # (N, K)
    return expected.argmax(axis=1)


def metrics_from_pred(pred: np.ndarray, y: np.ndarray, num_classes: int) -> Dict:
    out = {
        "acc":     float(accuracy_score(y, pred)),
        "f1":      float(f1_score(y, pred, average="macro", zero_division=0)),
        "recall":  float(recall_score(y, pred, average="macro", zero_division=0)),
        "micro_P": float(precision_score(y, pred, average="micro", zero_division=0)),
        "macro_P": float(precision_score(y, pred, average="macro", zero_division=0)),
        "mcc":     float(matthews_corrcoef(y, pred)),
    }
    for c in range(num_classes):
        m_p = pred == c
        n_p = int(m_p.sum())
        out[f"P_class_{c}"] = float((y[m_p] == c).mean()) if n_p > 0 else float("nan")
        m_t = y == c
        n_t = int(m_t.sum())
        out[f"R_class_{c}"] = float((pred[m_t] == c).mean()) if n_t > 0 else 0.0
        prec, rec = out[f"P_class_{c}"], out[f"R_class_{c}"]
        if np.isnan(prec) or (prec + rec) == 0:
            out[f"F1_class_{c}"] = 0.0
        else:
            out[f"F1_class_{c}"] = 2 * prec * rec / (prec + rec)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def fmt_metrics(m: Dict, num_classes: int = 4, prefix: str = "") -> str:
    lines = [
        f"{prefix}acc={m.get('acc', float('nan')):.4f}  f1={m.get('f1', float('nan')):.4f}"
        f"  macroP={m.get('macro_P', float('nan')):.4f}  microP={m.get('micro_P', float('nan')):.4f}"
        f"  recall={m.get('recall', float('nan')):.4f}  mcc={m.get('mcc', float('nan')):.4f}"
    ]
    if "coverage" in m:
        lines[0] += f"  cov={m['coverage']:.3f}"
    pcs = [f"P{c}={m[f'P_class_{c}']:.3f}" if not np.isnan(m.get(f'P_class_{c}', np.nan))
           else f"P{c}=nan" for c in range(num_classes)]
    rcs = [f"R{c}={m[f'R_class_{c}']:.3f}" for c in range(num_classes)]
    f1s = [f"F1_{c}={m[f'F1_class_{c}']:.3f}" for c in range(num_classes)]
    lines.append(f"{prefix}  " + "  ".join(pcs))
    lines.append(f"{prefix}  " + "  ".join(rcs))
    lines.append(f"{prefix}  " + "  ".join(f1s))
    return "\n".join(lines)
