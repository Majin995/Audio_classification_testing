"""Build data/Split1s_eval/{val,test}/<class>/ — symlinks into Split1s/test
with 500/class held out as val (stratified random, seed=42). Remainder
becomes the test split.

Used by inference/ensemble_dirichlet.py to fit the stacker on a real-
distribution val carved from Split1s/test, while keeping a held-out test.
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_TEST = ROOT / "Split1s" / "test"
DST = ROOT / "Split1s_eval"
N_VAL_PER_CLASS = 500
SEED = 42
AUDIO_EXTS = {".wav", ".mp3", ".flac"}


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    rng = random.Random(SEED)

    summary = []
    for cls_dir in sorted(SRC_TEST.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        files = sorted(p for p in cls_dir.iterdir()
                        if p.suffix.lower() in AUDIO_EXTS)
        n_total = len(files)
        n_val = min(N_VAL_PER_CLASS, n_total)
        val_pick = set(rng.sample(files, n_val))
        for split in ("val", "test"):
            out_dir = DST / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            split = "val" if f in val_pick else "test"
            link = DST / split / cls / f.name
            rel = os.path.relpath(f, link.parent)
            link.symlink_to(rel)
        n_test = n_total - n_val
        summary.append((cls, n_val, n_test))

    print(f"Wrote {DST}")
    print(f"{'class':<12} {'val':>5} {'test':>6}")
    for cls, v, t in summary:
        print(f"{cls:<12} {v:>5} {t:>6}")


if __name__ == "__main__":
    main()
