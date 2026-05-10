"""Post-hoc calibration + per-class threshold tuning for HydroPreciseV2.

Two threshold-search modes are exposed:

1. ``search_thresholds_macroP_coverage`` — the legacy objective: maximize
   macro precision subject to a coverage floor. Reproduced verbatim from
   ``training/train_precise_v2.py:_search_thresholds`` so the trainer can
   delegate here.

2. ``search_thresholds_recall_floor`` — new precision-targeted objective:
   maximize **micro precision** subject to **per-class recall ≥ floor**.
   Coordinate ascent over the same threshold grid. Returns a richer dict
   so it can be appended to the existing ``thresholds.json``.

Both functions take softmax probabilities (already temperature-scaled) and
return a JSON-serializable result.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
#  Macro-P with coverage floor (legacy reproduction)
# ──────────────────────────────────────────────────────────────────────────

def search_thresholds_macroP_coverage(
    probs:        np.ndarray,
    targets:      np.ndarray,
    num_classes:  int,
    target_coverage: float = 0.85,
    grid:         Optional[Sequence[float]] = None,
) -> Dict:
    """Maximize macro_precision subject to coverage >= target_coverage.

    Coordinate ascent: for each class c, try every grid value g > current
    threshold[c]; accept if it improves macro_precision and coverage stays
    above the floor.
    """
    grid = list(grid) if grid is not None else list(np.linspace(0.0, 0.95, 20))
    argmax = probs.argmax(axis=1)
    p_max  = probs.max(axis=1)

    def _score(thr):
        keep = p_max >= thr[argmax]
        if keep.sum() == 0:
            return 0.0, 0.0
        preds, gts = argmax[keep], targets[keep]
        precs = []
        for c in range(num_classes):
            mask = preds == c
            if mask.sum() == 0:
                continue
            precs.append((gts[mask] == c).mean())
        if not precs:
            return 0.0, float(keep.mean())
        return float(np.mean(precs)), float(keep.mean())

    thr  = np.zeros(num_classes)
    best_prec, best_cov = _score(thr)
    best_thr = thr.copy()
    improved = True
    while improved:
        improved = False
        for c in range(num_classes):
            for g in grid:
                if g <= thr[c]:
                    continue
                cand = thr.copy(); cand[c] = g
                prec, cov = _score(cand)
                if cov < target_coverage:
                    continue
                if prec > best_prec + 1e-6:
                    best_prec, best_cov, best_thr = prec, cov, cand
                    thr = cand
                    improved = True
    return {
        "objective":       "macroP_coverage",
        "thresholds":      best_thr.tolist(),
        "macro_precision": best_prec,
        "coverage":        best_cov,
        "target_coverage": target_coverage,
    }


# ──────────────────────────────────────────────────────────────────────────
#  Micro-P with per-class recall floor (NEW)
# ──────────────────────────────────────────────────────────────────────────

def search_thresholds_recall_floor(
    probs:           np.ndarray,
    targets:         np.ndarray,
    num_classes:     int,
    recall_floor:    float | Sequence[float] = 0.6,
    grid:            Optional[Sequence[float]] = None,
) -> Dict:
    """Maximize micro_precision subject to per-class recall >= recall_floor.

    The floor is computed against the **full** target distribution
    (denominator includes rejected samples), so raising thresholds blindly
    cannot game the recall constraint by shrinking the denominator.

    Args:
        recall_floor: Either a single float applied to every class, or a
            length-``num_classes`` sequence for per-class floors.
    """
    grid = list(grid) if grid is not None else list(np.linspace(0.0, 0.95, 20))
    if np.ndim(recall_floor) == 0:
        floor = np.full(num_classes, float(recall_floor), dtype=np.float64)
    else:
        floor = np.asarray(recall_floor, dtype=np.float64)
        assert floor.shape == (num_classes,)

    argmax = probs.argmax(axis=1)
    p_max  = probs.max(axis=1)
    class_pos = np.array([(targets == c).sum() for c in range(num_classes)],
                         dtype=np.float64)

    def _score(thr):
        keep = p_max >= thr[argmax]
        if keep.sum() == 0:
            return -1.0, 0.0, [0.0] * num_classes
        preds, gts = argmax[keep], targets[keep]
        # Per-class recall against full population (gates feasibility).
        per_class_recall: List[float] = []
        feasible = True
        for c in range(num_classes):
            tp = ((preds == c) & (gts == c)).sum()
            r  = tp / max(class_pos[c], 1)
            per_class_recall.append(float(r))
            if r < floor[c] - 1e-9:
                feasible = False
        if not feasible:
            return -1.0, float(keep.mean()), per_class_recall
        micro_p = float((preds == gts).mean())
        return micro_p, float(keep.mean()), per_class_recall

    thr = np.zeros(num_classes)
    best_p, best_cov, best_recall = _score(thr)
    if best_p < 0:
        return {
            "objective":       "recall_floor",
            "thresholds":      thr.tolist(),
            "micro_precision": 0.0,
            "coverage":        0.0,
            "per_class_recall": [0.0] * num_classes,
            "recall_floor":    floor.tolist(),
            "feasible":        False,
        }
    best_thr = thr.copy()
    improved = True
    while improved:
        improved = False
        for c in range(num_classes):
            for g in grid:
                if g <= thr[c]:
                    continue
                cand = thr.copy(); cand[c] = g
                mp, cov, rec = _score(cand)
                if mp < 0:                                  # infeasible
                    continue
                if mp > best_p + 1e-6:
                    best_p, best_cov, best_recall = mp, cov, rec
                    best_thr = cand
                    thr = cand
                    improved = True
    return {
        "objective":       "recall_floor",
        "thresholds":      best_thr.tolist(),
        "micro_precision": best_p,
        "coverage":        best_cov,
        "per_class_recall": best_recall,
        "recall_floor":    floor.tolist(),
        "feasible":        True,
    }
