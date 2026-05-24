"""Non-DALI Lightning DataModule for long-audio classification.

Takes a folder of extended-duration audio files (each typically tens of
seconds to several minutes) and splits each one into fixed-length windows at
load time, yielding one window per item — exactly matching the shape and
preprocessing of ``DALIAudioDataModule`` so callers can swap one for the
other in a Lightning model without changing the training loop.

Per-window preprocessing is **bit-identical** to ``DALIAudioDataModule``:
  - soundfile decode → mono → resample to ``target_sr`` (if needed)
  - pad/slice to ``fixed_len = window_sec * target_sr``
  - optional normalize ``(x - mean) / (std + 1e-9) * target_rms`` (biased
    std, matching DALI's ``fn.normalize`` with epsilon=1e-9)

Window enumeration is done once at ``setup()`` via ``sf.info`` (header read
only — no audio bytes decoded), so multi-million-window manifests build in
seconds. The Dataset reads each window on demand via ``sf.read`` with
``start``/``stop`` offsets so we never load whole long files into RAM.

Drop-in usage with a Lightning model originally trained against
``DALIAudioDataModule``::

    dm = ExtendedThreadedAudioDataModule(
        data_dir="/data/LongFiles_root",  # train/<class>/*.wav etc.
        batch_size=64, window_sec=1.0, target_sr=5120,
        rms_normalize=False,
    )
    trainer.fit(model, dm)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


DEFAULT_SR = 5120
DEFAULT_WINDOW_SEC = 1.0


def _normalize_dali_style(x: torch.Tensor, target_rms: float,
                          epsilon: float = 1e-9) -> torch.Tensor:
    """Bit-identical to ``fn.normalize(audio, axes=[0], epsilon=1e-9) * target_rms``.

    DALI computes the BIASED (population) std internally. We do the same.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = var.sqrt()
    return (x - mean) / (std + epsilon) * float(target_rms)


def enumerate_windows(files_by_class, class_to_idx, window_samples, hop_samples):
    """Return list of (path, label, start_sample, stop_sample) for every window.

    Files shorter than ``window_samples`` contribute exactly one window
    spanning [0, file_len] which is later zero-padded by the dataset reader.
    """
    out = []
    for cls, paths in files_by_class.items():
        label = class_to_idx[cls]
        for p in paths:
            try:
                info = sf.info(p)
            except Exception:
                continue
            n_samples = info.frames
            if n_samples <= 0:
                continue
            if n_samples <= window_samples:
                out.append((p, label, 0, n_samples))
                continue
            start = 0
            while start + window_samples <= n_samples:
                out.append((p, label, start, start + window_samples))
                start += hop_samples
            # Optional trailing partial window so the file's tail isn't lost.
            if start < n_samples:
                out.append((p, label, n_samples - window_samples, n_samples))
    return out


class WindowedAudioDataset(Dataset):
    """Materialises one window per index. Reads bytes lazily."""

    def __init__(self, windows, target_sr: int, fixed_len: int,
                 rms_normalize: bool, target_rms: float):
        self.windows = windows
        self.target_sr = int(target_sr)
        self.fixed_len = int(fixed_len)
        self.rms_normalize = bool(rms_normalize)
        self.target_rms = float(target_rms)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        path, label, start, stop = self.windows[idx]
        a, sr = sf.read(path, dtype="float32", start=int(start), stop=int(stop),
                        always_2d=False)
        if a.ndim > 1:
            a = a.mean(axis=1)
        x = torch.from_numpy(a)
        if sr != self.target_sr:
            x = AF.resample(x, sr, self.target_sr)
        # pad → slice (matches the DALI pipeline order: pad first, then slice)
        if x.numel() < self.fixed_len:
            x = F.pad(x, (0, self.fixed_len - x.numel()))
        else:
            x = x[: self.fixed_len]
        if self.rms_normalize:
            x = _normalize_dali_style(x, self.target_rms)
        return x, int(label)


def _scan_split(data_dir: Path, split: str):
    out: dict[str, list[str]] = {}
    split_dir = data_dir / split
    if not split_dir.is_dir():
        return out
    for cls in sorted(split_dir.iterdir()):
        if not cls.is_dir():
            continue
        files = sorted(str(f) for f in cls.rglob("*")
                       if f.suffix.lower() in {".wav", ".mp3", ".flac"})
        if files:
            out[cls.name] = files
    return out


class ExtendedThreadedAudioDataModule(pl.LightningDataModule):
    """Drop-in non-DALI replacement for ``DALIAudioDataModule`` with windowing.

    Keeps the same public attributes ``class_to_idx``, ``idx_to_class``,
    ``num_classes``, ``class_weights`` so Lightning models referencing them
    don't need to change.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 8,
        target_sr: int = DEFAULT_SR,
        window_sec: float = DEFAULT_WINDOW_SEC,
        hop_sec: Optional[float] = None,
        rms_normalize: bool = False,
        target_rms: float = 0.1,
        oversample_train: bool = True,
        merge_classes: Optional[Dict[str, str]] = None,
        shuffle_train: bool = True,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.target_sr = int(target_sr)
        self.window_sec = float(window_sec)
        self.hop_sec = float(hop_sec) if hop_sec is not None else float(window_sec)
        self.fixed_len = int(round(self.window_sec * self.target_sr))
        self.hop_samples = int(round(self.hop_sec * self.target_sr))
        self.rms_normalize = bool(rms_normalize)
        self.target_rms = float(target_rms)
        self.oversample_train = bool(oversample_train)
        self.merge_classes = dict(merge_classes or {})
        self.shuffle_train = bool(shuffle_train)
        self.pin_memory = bool(pin_memory)

        self.class_to_idx: dict = {}
        self.idx_to_class: dict = {}
        self.num_classes: int = 0
        self.class_weights: list = []
        self._splits: dict = {}            # split -> files_by_class
        self._windows: dict = {}           # split -> list of windows

    def _apply_merge(self, fbc):
        if not self.merge_classes:
            return fbc
        result = {k: list(v) for k, v in fbc.items()}
        for src, tgt in self.merge_classes.items():
            if src in result and tgt in result:
                result[tgt] += result.pop(src)
            elif src in result:
                result[tgt] = result.pop(src)
        return result

    def _oversample(self, fbc):
        if not fbc:
            return fbc
        max_count = max(len(v) for v in fbc.values())
        out = {}
        for cls, fs in fbc.items():
            reps = (max_count + len(fs) - 1) // len(fs)
            out[cls] = (fs * reps)[:max_count]
        return out

    def setup(self, stage: Optional[str] = None):
        for split in ("train", "val", "test"):
            self._splits[split] = self._apply_merge(_scan_split(self.data_dir, split))

        classes = sorted(self._splits["train"].keys())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes = len(classes)

        train_fbc = (self._oversample(self._splits["train"])
                     if self.oversample_train else self._splits["train"])
        counts = [len(self._splits["train"].get(c, [])) for c in classes]
        total = sum(counts) or 1
        self.class_weights = [total / (self.num_classes * max(c, 1)) for c in counts]

        for split, fbc in (("train", train_fbc),
                           ("val", self._splits["val"]),
                           ("test", self._splits["test"])):
            self._windows[split] = enumerate_windows(
                fbc, self.class_to_idx, self.fixed_len, self.hop_samples,
            )

        n_train_files = sum(len(v) for v in self._splits["train"].values())
        print(f"Classes        : {self.class_to_idx}")
        print(f"Train files    : {n_train_files} (oversampled→{sum(len(v) for v in train_fbc.values())})")
        print(f"Train windows  : {len(self._windows['train'])}  "
              f"(window={self.window_sec}s hop={self.hop_sec}s)")
        print(f"Val windows    : {len(self._windows['val'])}")
        print(f"Test windows   : {len(self._windows['test'])}")
        print(f"Class weights  : { {c: f'{w:.2f}' for c, w in zip(classes, self.class_weights)} }")

    def _make_dataset(self, split):
        return WindowedAudioDataset(
            self._windows[split], target_sr=self.target_sr,
            fixed_len=self.fixed_len, rms_normalize=self.rms_normalize,
            target_rms=self.target_rms,
        )

    def train_dataloader(self):
        return DataLoader(self._make_dataset("train"), batch_size=self.batch_size,
                          shuffle=self.shuffle_train, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self._make_dataset("val"), batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=False, persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self._make_dataset("test"), batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=False, persistent_workers=False)


__all__ = ["ExtendedThreadedAudioDataModule", "WindowedAudioDataset",
           "enumerate_windows"]
