"""Shared utilities for the MLMC (multi-label, multi-class) campaign scripts.

These wrap the existing cached per-clip softmax dumps and re-interpret them as
**independent per-class scores** so each of the top-5 setups can emit
one-hot / multi-hot predictions instead of a single argmax label.

Source grouping mirrors ``cargo_confirm_5base.py`` /
``zero_fit_ensemble.py`` (filename ``<src>_<6digits>.wav``). The honest
contract is unchanged: per-class thresholds are tuned on **val** only and the
test split is scored exactly once.

Metric / encoding helpers (``probs_to_onehot``, ``tune_thresholds_per_class``,
``multilabel_report``) are re-exported from ``data.mlmc_windowed_loader`` so the
trainer scripts and the ensemble scripts share one implementation.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

# Make repo root importable when run as `python campaign/mlmc/<script>.py`.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.mlmc_windowed_loader import (          # noqa: E402  (after sys.path)
    probs_to_onehot, tune_thresholds_per_class, multilabel_report,
)

CLIP_RE = re.compile(r"_(\d{6})\.wav$", re.IGNORECASE)


def list_files(data_dir: Path, split: str, classes):
    """Return [(filename, class_idx, source_id)] in the on-disk sort order that
    the cached prob dumps were produced in (sorted class dirs, sorted files)."""
    out = []
    for cls in sorted((data_dir / split).iterdir()):
        if not cls.is_dir():
            continue
        ci = classes.index(cls.name)
        for fn in sorted(p.name for p in cls.iterdir()
                         if p.suffix.lower() == ".wav"):
            m = CLIP_RE.search(fn)
            src = fn[:m.start()] if m else fn
            out.append((fn, ci, src))
    return out


def groupby_source(rows, n, num_classes):
    """Group consecutive clip rows by source id (rows already in dump order).

    Returns list of (row_indices: np.ndarray, multihot_label: np.ndarray).
    Each Classifier_Dataset source is single-label → exactly one positive bit.
    """
    rows = rows[:n]
    groups, cur, idx, cy = [], None, [], None
    for i, (_, ci, src) in enumerate(rows):
        if src != cur:
            if cur is not None:
                groups.append((np.array(idx), _onehot(cy, num_classes)))
            cur, idx, cy = src, [i], ci
        else:
            idx.append(i)
    if cur is not None:
        groups.append((np.array(idx), _onehot(cy, num_classes)))
    return groups


def _onehot(ci, n):
    v = np.zeros(n, dtype=np.float32)
    v[ci] = 1.0
    return v


__all__ = [
    "list_files", "groupby_source",
    "probs_to_onehot", "tune_thresholds_per_class", "multilabel_report",
    "CLIP_RE",
]
