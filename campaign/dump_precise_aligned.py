"""Dump HydroPrecise probs aligned to clean_redo's file enumeration.

clean_redo uses `sorted(p.name for p in cls.iterdir() if p.suffix.lower()=='.wav')`
(non-recursive, alphabetical). DALI uses rglob and may sort differently.

This script:
  1. Enumerates files the same way as clean_redo (per-class iterdir, sorted).
  2. Loads each clip via soundfile / torchaudio, resamples to 5120 Hz, pads/crops
     to fixed_len=5120 (1 s) — matching the model's training contract.
  3. Runs precise-013 inference in batches.
  4. Saves val_probs (50752, 4) and test_probs (14208, 4) — aligned with v1 cache.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

# Force Mel fallback in _CQTFrontend so we match the precise-013 ckpt
# (which has Mel-style state dict keys from when nnAudio was unavailable).
import sys
import importlib
_real_spectrogram = sys.modules.get('nnAudio.Spectrogram')
class _BlockedNnAudio:
    def __getattr__(self, name):
        raise ImportError('forced nnAudio block to use Mel fallback')
sys.modules['nnAudio'] = _BlockedNnAudio()
sys.modules['nnAudio.Spectrogram'] = _BlockedNnAudio()

from models.hydro_precise import HydroPrecise


SR_TARGET = 5120
FIXED_LEN = 5120


def list_files(data_dir: Path, classes: list, split: str):
    out = []
    split_dir = data_dir / split
    for cls in sorted(split_dir.iterdir()):
        if not cls.is_dir():
            continue
        ci = classes.index(cls.name)
        for fn in sorted(p for p in cls.iterdir() if p.suffix.lower() == '.wav'):
            out.append((str(fn), ci))
    return out


def load_clip(path: str, sr_target: int, fixed_len: int, resamplers: dict) -> torch.Tensor:
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    # data shape: (n_samples, n_channels). Mono mean.
    wav = torch.from_numpy(data.mean(axis=1))  # (n_samples,)
    if sr != sr_target:
        key = (sr, sr_target)
        if key not in resamplers:
            resamplers[key] = torchaudio.transforms.Resample(sr, sr_target)
        wav = resamplers[key](wav.unsqueeze(0)).squeeze(0)
    if wav.numel() >= fixed_len:
        wav = wav[:fixed_len]
    else:
        pad = fixed_len - wav.numel()
        wav = F.pad(wav, (0, pad))
    return wav


@torch.no_grad()
def infer_split(model, files: list, device, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    P, Y = [], []
    resamplers = {}
    batch_x, batch_y = [], []
    total = len(files)
    for i, (path, y) in enumerate(files):
        x = load_clip(path, SR_TARGET, FIXED_LEN, resamplers)
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) >= batch_size or i == total - 1:
            xb = torch.stack(batch_x).to(device, non_blocking=True)
            out = model(xb)
            if out.size(-1) > 4:
                out = out[:, :4]
            P.append(F.softmax(out.float(), dim=-1).cpu().numpy())
            Y.append(np.array(batch_y, dtype=np.int64))
            batch_x, batch_y = [], []
            if (i + 1) % 1000 == 0 or i == total - 1:
                done = i + 1
                print(f'  {done}/{total} ({100*done/total:.1f}%)', flush=True)
    return np.concatenate(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_path', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()

    classes = ['Cargo', 'Passenger', 'Tanker', 'Tug']
    data_dir = Path(args.data_dir)
    # Truncate to match v1-cache enumeration (50752 val, 14208 test)
    val_files = list_files(data_dir, classes, 'val')[:50752]
    test_files = list_files(data_dir, classes, 'test')[:14208]
    print(f'val: {len(val_files)} clips;  test: {len(test_files)} clips', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'loading {args.ckpt}', flush=True)
    model = HydroPrecise.load_from_checkpoint(args.ckpt, map_location=device, strict=False)
    model = model.to(device).eval()
    print('  model loaded', flush=True)

    print('val inference...', flush=True)
    val_probs, val_y = infer_split(model, val_files, device, args.batch_size)
    print(f'val_probs: {val_probs.shape}', flush=True)

    print('test inference...', flush=True)
    test_probs, test_y = infer_split(model, test_files, device, args.batch_size)
    print(f'test_probs: {test_probs.shape}', flush=True)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_path,
             val_probs=val_probs, val_y=val_y,
             test_probs=test_probs, test_y=test_y)
    print(f'wrote {args.out_path}', flush=True)


if __name__ == '__main__':
    main()
