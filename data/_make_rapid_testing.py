"""Build data/Classification_rapid_testing/ — a small stratified subset of
Split1s for rapid prototyping.

Layout produced:
    Classification_rapid_testing/
        train/{Cargo,Passenger,Tanker,Tug}/   ~64 per class (Passenger capped)
        val/  {Cargo,Passenger,Tanker,Tug}/   16  per class (carved from test)
        test/ {Cargo,Passenger,Tanker,Tug}/   32  per class

Files are symlinked (relative paths) — near-zero disk cost, but the subset
breaks if Split1s/ is moved. Re-run the script in that case.

Deterministic: seed=42. Re-running overwrites the subset.
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "Split1s"
DST = ROOT / "Classification_rapid_testing"

CLASSES = ["Cargo", "Passenger", "Tanker", "Tug"]
N_TRAIN, N_VAL, N_TEST = 64, 16, 32
SEED = 42
AUDIO_EXTS = {".wav", ".mp3", ".flac"}


def list_audio(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO_EXTS)


def link_files(files: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        link = out_dir / f.name
        rel = os.path.relpath(f, out_dir)
        link.symlink_to(rel)


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    rng = random.Random(SEED)

    summary: list[tuple[str, int, int, int]] = []
    for cls in CLASSES:
        train_pool = list_audio(SRC / "train" / cls)
        test_pool = list_audio(SRC / "test" / cls)

        n_train = min(N_TRAIN, len(train_pool))
        train_pick = rng.sample(train_pool, n_train)

        # Carve val from test first, then test subset from the remainder.
        n_val = min(N_VAL, len(test_pool))
        val_pick = rng.sample(test_pool, n_val)
        remaining_test = [p for p in test_pool if p not in set(val_pick)]
        n_test = min(N_TEST, len(remaining_test))
        test_pick = rng.sample(remaining_test, n_test)

        link_files(train_pick, DST / "train" / cls)
        link_files(val_pick,   DST / "val"   / cls)
        link_files(test_pick,  DST / "test"  / cls)
        summary.append((cls, n_train, n_val, n_test))

    print(f"Wrote {DST}")
    print(f"{'class':<12} {'train':>6} {'val':>5} {'test':>5}")
    for cls, t, v, e in summary:
        print(f"{cls:<12} {t:>6} {v:>5} {e:>5}")


if __name__ == "__main__":
    main()
