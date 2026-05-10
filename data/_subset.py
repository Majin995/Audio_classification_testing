"""Shared helpers for stratified train-pool subsetting.

Used by:
  * training/train_active.py  — initial labeled pool for active learning rounds
  * training/train_precise_v2.py — optional --train_subset_size flag for cheap
                                   per-trial budgets (e.g. Optuna sweeps).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

AUDIO_EXTS = {".wav", ".mp3", ".flac"}


def scan_train_pool(data_dir: Path) -> Dict[str, List[str]]:
    """Return ``{class_name: [abs_paths]}`` for the ``train/`` split."""
    train_dir = Path(data_dir) / "train"
    out: Dict[str, List[str]] = {}
    for cls in sorted(train_dir.iterdir()):
        if not cls.is_dir():
            continue
        files = sorted(
            str(f) for f in cls.rglob("*") if f.suffix.lower() in AUDIO_EXTS
        )
        if files:
            out[cls.name] = files
    return out


def stratified_init(pool: Dict[str, List[str]], total: int,
                    per_class: int, rng: random.Random
                   ) -> Dict[str, List[str]]:
    """Sample a stratified subset of the labeled pool.

    If ``per_class > 0`` we take exactly that many from each class (capped at
    availability). Otherwise we split ``total`` evenly across classes
    (round-robin remainder), capped at availability.
    """
    classes = sorted(pool.keys())
    n_cls = len(classes)
    if per_class > 0:
        targets = {c: min(per_class, len(pool[c])) for c in classes}
    else:
        base, rem = divmod(total, n_cls)
        targets = {}
        for i, c in enumerate(classes):
            t = base + (1 if i < rem else 0)
            targets[c] = min(t, len(pool[c]))
    init: Dict[str, List[str]] = {}
    for c, t in targets.items():
        init[c] = rng.sample(pool[c], t) if t > 0 else []
    return init
