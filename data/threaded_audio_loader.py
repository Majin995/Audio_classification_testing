"""Standalone non-DALI audio loader for stacker training.

Pure soundfile + ThreadPoolExecutor + torchaudio.functional resampling — no
NVIDIA DALI, no torch dataloader workers, no dependency on the local
`training/` package (which is shadowed by the OpenCLIP site-packages
`training` namespace). Use this as the data layer when the DALI stack is
unavailable or you want a portable backup.

I/O and preprocess are bit-identical to ``campaign/train_recurrent.py``
(FFT-domain Butterworth-style HPF + mean/std normalization to ``target_rms``)
so cached per-clip ensemble probs computed with the original loader remain
valid features for the stacker.

Public API:
  scan_split(data_dir, split, exclude_iara_glider=False)
  read_clip(path, target_sr=5120, fixed_len=5120) -> np.ndarray
  read_batch_threaded(paths, executor) -> np.ndarray
  preprocess(audio_t, target_rms=0.1, hpf_hz=20.0, sr=5120) -> torch.Tensor
  hpf_torch(x, sr, cutoff, order=4) -> torch.Tensor
  CLASSES = ['Cargo','Passenger','Tanker','Tug']
  SEG_RE
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

CLASSES = ['Cargo', 'Passenger', 'Tanker', 'Tug']
# Source id recovery — matches train_recurrent.py exactly.
SEG_RE = re.compile(r'_seg_\d+\.wav$')


def scan_split(data_dir: str, split: str, exclude_iara_glider: bool = False):
    """Mirrors ``campaign.train_recurrent.scan_split``.

    File pattern: ``<src_prefix>-<id>_seg_<n>.wav`` (e.g. ``deepship-000645_seg_3.wav``,
    ``iara-A-0004_seg_12.wav``). Source id is the filename up to ``_seg_``.
    """
    out = {c: {} for c in CLASSES}
    for c in CLASSES:
        d = Path(data_dir) / split / c
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() != '.wav':
                continue
            m = SEG_RE.search(p.name)
            if not m:
                continue
            src = p.name[:m.start()]
            if exclude_iara_glider and src.startswith('iara-'):
                parts = src.split('-')
                if len(parts) >= 2 and parts[1] in ('F', 'G'):
                    continue
            out[c].setdefault(src, []).append(str(p))
    return out


def read_clip(path: str, target_sr: int = 5120, fixed_len: int = 5120):
    a, sr = sf.read(path, dtype='float32', always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != target_sr:
        a = AF.resample(torch.from_numpy(a), sr, target_sr).numpy()
    if len(a) < fixed_len:
        a = np.pad(a, (0, fixed_len - len(a)))
    else:
        a = a[:fixed_len]
    return a


def read_batch_threaded(paths, executor: ThreadPoolExecutor,
                        target_sr: int = 5120, fixed_len: int = 5120):
    fn = lambda p: read_clip(p, target_sr, fixed_len)
    return np.stack(list(executor.map(fn, paths))).astype(np.float32)


def hpf_torch(x: torch.Tensor, sr: int, cutoff: float, order: int = 4):
    """FFT-domain Butterworth-style high-pass — matches train_recurrent.py."""
    if cutoff <= 0:
        return x
    T = x.shape[-1]
    freqs = torch.fft.rfftfreq(T, 1.0 / sr).to(x.device)
    ratio = freqs / max(cutoff, 1e-9)
    mag = ratio.pow(2 * order)
    mask = (mag / (1.0 + mag)).sqrt().to(x.dtype)
    Xf = torch.fft.rfft(x.float(), dim=-1)
    return torch.fft.irfft(Xf * mask, n=T, dim=-1).to(x.dtype)


def preprocess(audio: torch.Tensor, target_rms: float = 0.1,
               hpf_hz: float = 20.0, sr: int = 5120):
    """Mean-zero / unit-std → scale to ``target_rms`` → 4th-order HPF.

    Bit-identical to ``campaign.train_recurrent.preprocess``.
    """
    x = audio
    x = ((x - x.mean(dim=-1, keepdim=True))
         / (x.std(dim=-1, keepdim=True) + 1e-9) * target_rms)
    return hpf_torch(x, sr, hpf_hz, 4)


__all__ = ["CLASSES", "SEG_RE", "scan_split", "read_clip",
           "read_batch_threaded", "preprocess", "hpf_torch"]
