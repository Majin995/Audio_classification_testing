"""Dump HydroComplete probs aligned to clean_redo's file enumeration.

Same protocol as dump_precise_aligned.py but for the HydroComplete architecture
(unified Hydra⊕Precise superset with all branches: gabor, scattering, sincnet,
tdsbe, cqt, demon).
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

# Force Mel fallback in case CQT state-dict keys mismatch
import sys
class _BlockedNnAudio:
    def __getattr__(self, name):
        raise ImportError('forced nnAudio block')
sys.modules['nnAudio'] = _BlockedNnAudio()
sys.modules['nnAudio.Spectrogram'] = _BlockedNnAudio()

from models.hydro_complete import HydroComplete

SR_TARGET = 5120
FIXED_LEN = 5120


def list_files(data_dir, classes, split):
    out = []
    for cls in sorted((data_dir / split).iterdir()):
        if not cls.is_dir():
            continue
        ci = classes.index(cls.name)
        for fn in sorted(p for p in cls.iterdir() if p.suffix.lower() == '.wav'):
            out.append((str(fn), ci))
    return out


def load_clip(path, sr_target, fixed_len, resamplers):
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    wav = torch.from_numpy(data.mean(axis=1))
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
def infer_split(model, files, device, batch_size=32):
    model.eval()
    P, Y = [], []
    resamplers = {}
    bx, by = [], []
    total = len(files)
    for i, (path, y) in enumerate(files):
        x = load_clip(path, SR_TARGET, FIXED_LEN, resamplers)
        bx.append(x)
        by.append(y)
        if len(bx) >= batch_size or i == total - 1:
            xb = torch.stack(bx).to(device, non_blocking=True)
            out = model(xb)
            if isinstance(out, tuple):
                out = out[0]
            if out.size(-1) > 4:
                out = out[:, :4]
            P.append(F.softmax(out.float(), dim=-1).cpu().numpy())
            Y.append(np.array(by, dtype=np.int64))
            bx, by = [], []
            if (i + 1) % 1000 == 0 or i == total - 1:
                print(f'  {i+1}/{total} ({100*(i+1)/total:.1f}%)', flush=True)
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
    val_files = list_files(data_dir, classes, 'val')[:50752]
    test_files = list_files(data_dir, classes, 'test')[:14208]
    print(f'val: {len(val_files)} clips;  test: {len(test_files)} clips', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'loading {args.ckpt}', flush=True)
    model = HydroComplete.load_from_checkpoint(args.ckpt, map_location=device, strict=False)
    model = model.to(device).eval()
    print('model loaded', flush=True)

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
