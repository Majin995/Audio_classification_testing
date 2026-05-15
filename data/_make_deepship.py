"""Build DeepShip clip-level datasets at 3 chunk lengths (5s / 15s / 30s).

Source: ``/run/media/damo/LaCie/Ubuntu BackUp/Deepship/Raw/<Class>/*.wav``
        — 32 kHz mono float WAVs, ~6 minutes each, 4 classes
        (Cargo / Passenger / Tanker / Tug).

Output (under ``/var/mnt/5A009BF8009BD8F9/Data``):
    Deepship_5s/
        Train/<Class>/<Class>_<sourceID>_<NNNN>.wav
        Test/<Class>/...
        Holdout/<Class>/...
        train -> Train, val -> Test, test -> Holdout      (compat symlinks)
        manifest.json
        splits.jsonl
    Deepship_15s/   (same structure)
    Deepship_30s/   (same structure)

Splits: per-class file-level (so no source-file leaks across splits) at
60% / 15% / 25% Train / Test / Holdout, deterministic seed 42.

Chunking: non-overlapping windows of ``length_s × sample_rate`` samples.
Tail audio shorter than the window is **discarded** (no padding).

Sample rate: resampled from 32000 → 5120 Hz to match the project's UATR
conventions (HydroPrecise fmin=20, fmax=Nyquist=2560 Hz; same as
``Classifier_Dataset``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

# Lazy import torchaudio.functional only when needed (allows --dry_run).
SRC_DIR     = Path("/run/media/damo/LaCie/Ubuntu BackUp/Deepship/Raw")
OUT_BASE    = Path("/var/mnt/5A009BF8009BD8F9/Data")
SRC_SR      = 32_000
TARGET_SR   = 5_120
CLASSES     = ("Cargo", "Passenger", "Tanker", "Tug")
SPLIT_NAMES = ("Train", "Test", "Holdout")
SPLIT_RATIOS = (0.60, 0.15, 0.25)
LENGTHS_S   = (5, 15, 30)
SEED        = 42


def _scan_files() -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for cls in CLASSES:
        d = SRC_DIR / cls
        if not d.exists():
            raise FileNotFoundError(f"missing class dir: {d}")
        files = sorted(p for p in d.iterdir() if p.suffix.lower() == ".wav")
        if not files:
            raise FileNotFoundError(f"no WAVs in {d}")
        out[cls] = files
    return out


def _split_files(files: List[Path], rng: random.Random) -> Tuple[List[Path], List[Path], List[Path]]:
    """Per-class deterministic 60/15/25 split."""
    fs = list(files)
    rng.shuffle(fs)
    n = len(fs)
    n_train = int(round(n * SPLIT_RATIOS[0]))
    n_test  = int(round(n * SPLIT_RATIOS[1]))
    # Holdout takes the remainder so totals always add up.
    train  = fs[:n_train]
    test   = fs[n_train:n_train + n_test]
    holdout = fs[n_train + n_test:]
    return train, test, holdout


def _resample(x: np.ndarray, src_sr: int, dst_sr: int):
    import torch, torchaudio.functional as TAF
    t = torch.from_numpy(x.astype(np.float32))
    if t.ndim == 1:
        t = t.unsqueeze(0)
    y = TAF.resample(t, src_sr, dst_sr).squeeze(0).cpu().numpy()
    return y


def _process_one(src_path: Path, cls: str, split: str, len_s: int,
                 out_root: Path, target_sr: int) -> int:
    """Resample → chunk → write. Returns number of chunks written."""
    out_dir = out_root / split / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    info = sf.info(str(src_path))
    if info.samplerate != SRC_SR:
        # Permit native SR drift (some files might be 24/48k); resample anyway.
        pass
    audio, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)
    win = len_s * target_sr
    n_full = audio.shape[0] // win
    if n_full <= 0:
        return 0
    base = src_path.stem                     # e.g. "000645"
    written = 0
    for i in range(n_full):
        chunk = audio[i * win:(i + 1) * win]
        out_name = f"{cls}_{base}_{i:04d}.wav"
        sf.write(str(out_dir / out_name), chunk, target_sr,
                 subtype="FLOAT", format="WAV")
        written += 1
    return written


def _ensure_compat_symlinks(out_root: Path) -> None:
    """train→Train, val→Test, test→Holdout so DALIAudioDataModule works."""
    mapping = {"train": "Train", "val": "Test", "test": "Holdout"}
    for low, up in mapping.items():
        link = out_root / low
        if link.is_symlink() or link.exists():
            try:
                link.unlink()
            except IsADirectoryError:
                # Directory pre-existing; leave it.
                continue
        link.symlink_to(up)


def main():
    ap = argparse.ArgumentParser(description="Build Deepship_5s/15s/30s datasets")
    ap.add_argument("--lengths_s", type=int, nargs="+", default=list(LENGTHS_S))
    ap.add_argument("--target_sr", type=int, default=TARGET_SR)
    ap.add_argument("--out_base", type=Path, default=OUT_BASE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files_by_cls = _scan_files()
    print(f"[scan] sources @ {SRC_DIR}")
    for c in CLASSES:
        print(f"  {c:>10}: {len(files_by_cls[c])} files")

    # Split per class once; share splits across all length variants.
    splits: Dict[str, Dict[str, List[Path]]] = {s: {c: [] for c in CLASSES} for s in SPLIT_NAMES}
    for c in CLASSES:
        tr, te, ho = _split_files(files_by_cls[c], rng)
        splits["Train"][c]   = tr
        splits["Test"][c]    = te
        splits["Holdout"][c] = ho
        print(f"[split] {c:>10} -> Train={len(tr)} Test={len(te)} Holdout={len(ho)}")

    if args.dry_run:
        return

    for len_s in args.lengths_s:
        out_root = args.out_base / f"Deepship_{len_s}s"
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {out_root} ({len_s}s clips, {args.target_sr} Hz) ===")
        manifest = {
            "src_dir": str(SRC_DIR), "src_sr": SRC_SR,
            "target_sr": args.target_sr, "length_s": len_s,
            "classes": list(CLASSES),
            "split_ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS)),
            "seed": args.seed, "samples_per_clip": len_s * args.target_sr,
            "subtype": "FLOAT",
            "split_files": {s: {c: [str(p) for p in splits[s][c]] for c in CLASSES}
                            for s in SPLIT_NAMES},
            "counts": {s: {c: 0 for c in CLASSES} for s in SPLIT_NAMES},
        }
        t0 = time.time()
        for split in SPLIT_NAMES:
            for c in CLASSES:
                for src in splits[split][c]:
                    n = _process_one(src, c, split, len_s, out_root, args.target_sr)
                    manifest["counts"][split][c] += n
                done = manifest["counts"][split][c]
                print(f"  {split:>7}/{c:<10}: {done:>6} clips  "
                      f"(elapsed {(time.time()-t0)/60:.1f} min)")
        _ensure_compat_symlinks(out_root)
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        # JSONL split file (one row per source file)
        with (out_root / "splits.jsonl").open("w") as fh:
            for s in SPLIT_NAMES:
                for c in CLASSES:
                    for p in splits[s][c]:
                        fh.write(json.dumps({"split": s, "class": c, "src": str(p)}) + "\n")
        total = sum(sum(v.values()) for v in manifest["counts"].values())
        print(f"  TOTAL: {total} clips, "
              f"manifest.json + splits.jsonl + train/val/test symlinks written")


if __name__ == "__main__":
    main()
