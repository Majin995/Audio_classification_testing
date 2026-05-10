"""
CachedFeatureDataset & OmniFeatureDataModule
============================================

Provides a caching PyTorch Dataset that pre-computes and persists
the full ``UnderwaterFeatureExtractor`` feature dict to per-clip ``.pt``
files, eliminating repeated computation of expensive transforms
(WVD, Bispectrum, HHT) across training epochs.

Cache layout
------------
  <cache_dir>/
    train/
      Cargo_000123_segment_1.pt
      ...
    val/
      ...
    test/
      ...

Each ``.pt`` file contains a ``dict[str, torch.Tensor]`` matching the
schema returned by ``UnderwaterFeatureExtractor.extract()``.

``OmniFeatureDataModule`` wraps the datasets in a Lightning DataModule
with WeightedRandomSampler for training oversampling (counteracting the
185x Cargo/Passenger imbalance) and standard sequential loaders for
validation and test splits.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.underwater_feature_extractor import (
    GRAM_KEYS,
    SCALAR_KEYS,
    INPUT_DIM_1D,
    UnderwaterFeatureExtractor,
)


# ═══════════════════════════════════════════════════════════════════════
#  Collate function
# ═══════════════════════════════════════════════════════════════════════

def omni_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of (feature_dict, label) pairs into batched tensors.

    Returns
    -------
    feat_1d : Tensor (B, 138)
        Concatenated scalar + PSD + spectral features.
    feat_2d : Tensor (B, 9, H, W)
        Stacked spectro-temporal gram channels in ``GRAM_KEYS`` order.
    labels : Tensor (B,) int64
    """
    feat_dicts, labels = zip(*batch)

    feat_1d = torch.stack([
        torch.cat([fd[k] for k in SCALAR_KEYS], dim=0)
        for fd in feat_dicts
    ])                                                          # (B, 138)

    feat_2d = torch.stack([
        torch.stack([fd[k] for k in GRAM_KEYS], dim=0)
        for fd in feat_dicts
    ])                                                          # (B, 9, H, W)

    labels_t = torch.tensor(labels, dtype=torch.long)          # (B,)

    return feat_1d, feat_2d, labels_t


# ═══════════════════════════════════════════════════════════════════════
#  Worker helpers (must be module-level for multiprocessing pickling)
# ═══════════════════════════════════════════════════════════════════════

def _worker_precompute(args: Tuple) -> Optional[str]:
    """
    Pre-compute and cache features for one audio file.

    Parameters
    ----------
    args : (wav_path, cache_path, sample_rate, extractor_kwargs, denoise_cfg)
        ``denoise_cfg`` is a dict with key ``'method'`` (str) and optional
        kwargs forwarded to the denoising function.  Pass ``None`` or
        ``{'method': 'off'}`` to skip denoising.

    Returns
    -------
    str or None
        Error message if failed; None on success.
    """
    wav_path, cache_path, sample_rate, ext_kwargs, denoise_cfg = args
    if Path(cache_path).exists():
        return None

    try:
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T).mean(dim=0)  # (samples,) mono
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        signal = waveform.numpy().astype(np.float32)

        # ── Optional denoising (EMD-Wavelet or NMF-ICA) ──────────────────
        if denoise_cfg and denoise_cfg.get("method", "off") != "off":
            try:
                from processing.denoise.emd_wavelet import emd_wavelet_denoise
                from processing.denoise.bss import nmf_ica_separate
                method  = denoise_cfg["method"]
                dn_kw   = {k: v for k, v in denoise_cfg.items() if k != "method"}
                if method == "emd_wavelet":
                    signal = emd_wavelet_denoise(signal, sample_rate, **dn_kw)
                elif method in ("nmf", "ica", "nmf_ica"):
                    signal = nmf_ica_separate(signal, sample_rate,
                                              method=method, **dn_kw)
                elif method == "emd_nmf":
                    signal = emd_wavelet_denoise(signal, sample_rate)
                    signal = nmf_ica_separate(signal, sample_rate, method="nmf_ica")
            except Exception:
                pass   # denoising failure → use original signal

        ext = UnderwaterFeatureExtractor(**ext_kwargs)
        feat_dict = ext.extract(signal)

        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(feat_dict, cache_path)
    except Exception as exc:
        return f"{wav_path}: {exc}"

    return None


# ═══════════════════════════════════════════════════════════════════════
#  CachedFeatureDataset
# ═══════════════════════════════════════════════════════════════════════

class CachedFeatureDataset(Dataset):
    """
    Pre-computed feature cache Dataset.

    On first access of any clip, features are computed via
    ``UnderwaterFeatureExtractor`` and saved as a ``.pt`` file.
    Subsequent epochs load the cache file directly, avoiding all
    repeated computation (WVD: ~0.3 s, HHT: ~2-5 s, CWT: ~0.2 s
    per clip on a modern CPU).

    Parameters
    ----------
    split_dir : str or Path
        Directory containing per-class subdirectories of audio files,
        e.g. ``data/Split1s/train/``.
    cache_dir : str or Path
        Root directory for ``.pt`` cache files.  Created if absent.
    extractor : UnderwaterFeatureExtractor
        Configured feature extractor instance.
    sample_rate : int
        Target audio sample rate for resampling on load.
    force_recompute : bool
        If True, overwrite existing cache files.
    """

    #: Supported audio file extensions.
    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

    def __init__(
        self,
        split_dir:       str | Path,
        cache_dir:       str | Path,
        extractor:       UnderwaterFeatureExtractor,
        sample_rate:     int  = 5_120,
        force_recompute: bool = False,
        denoise_cfg:     Optional[Dict] = None,
    ):
        self.split_dir       = Path(split_dir)
        self.cache_dir       = Path(cache_dir)
        self.extractor       = extractor
        self.sample_rate     = sample_rate
        self.force_recompute = force_recompute
        # ``denoise_cfg`` example: {'method': 'emd_wavelet', 'wavelet': 'sym8'}
        self.denoise_cfg     = denoise_cfg or {"method": "off"}

        self._samples: List[Tuple[Path, int, Path]] = []   # (wav, label, cache)
        self._class_to_idx: Dict[str, int] = {}

        self._scan()

    # ── Setup ────────────────────────────────────────────────────────

    def _scan(self) -> None:
        """Walk split_dir and build the (wav, label, cache) sample list."""
        class_dirs = sorted(
            d for d in self.split_dir.iterdir() if d.is_dir()
        )
        self._class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

        for cls_dir in class_dirs:
            label = self._class_to_idx[cls_dir.name]
            for wav in sorted(cls_dir.rglob("*")):
                if wav.suffix.lower() not in self.AUDIO_EXTS:
                    continue
                cache = self.cache_dir / f"{cls_dir.name}_{wav.stem}.pt"
                self._samples.append((wav, label, cache))

    # ── Dataset API ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], int]:
        wav_path, label, cache_path = self._samples[index]

        if not self.force_recompute and cache_path.exists():
            feat_dict = torch.load(cache_path, weights_only=True)
            return feat_dict, label

        # On-demand computation (fallback; precompute_all() is preferred)
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T).mean(dim=0)  # (samples,) mono
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        signal = waveform.numpy().astype(np.float32)
        feat_dict = self.extractor.extract(signal)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feat_dict, cache_path)

        return feat_dict, label

    # ── Pre-computation ──────────────────────────────────────────────

    def precompute_all(self, num_workers: int = 8) -> None:
        """
        Pre-compute and cache all features in parallel.

        Uses ``concurrent.futures.ProcessPoolExecutor`` so each worker
        process owns its own ``UnderwaterFeatureExtractor`` instance,
        avoiding GIL contention on numpy and PyEMD operations.

        Parameters
        ----------
        num_workers : int
            Number of parallel worker processes.
        """
        pending = [
            (wav, str(cache), self.sample_rate,
             self._extractor_kwargs(), self.denoise_cfg)
            for wav, _, cache in self._samples
            if self.force_recompute or not cache.exists()
        ]

        if not pending:
            print("  All cache files present — skipping pre-computation.")
            return

        n = len(pending)
        print(f"  Pre-computing features for {n} clips "
              f"using {num_workers} workers …")
        errors = []

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_worker_precompute, args): args[0]
                       for args in pending}
            with tqdm(total=n, unit="clip", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    err = future.result()
                    if err:
                        errors.append(err)
                    pbar.update(1)

        if errors:
            warnings.warn(
                f"{len(errors)} clip(s) failed feature extraction:\n"
                + "\n".join(errors[:10])
                + ("\n  ..." if len(errors) > 10 else ""),
                RuntimeWarning,
            )

    def _extractor_kwargs(self) -> dict:
        """Return constructor kwargs for pickling into worker processes."""
        e = self.extractor
        return dict(
            sample_rate       = e.sample_rate,
            n_fft             = e.n_fft,
            hop_length        = e.hop_length,
            n_mels            = e.n_mels,
            n_gammatone       = e.n_gammatone,
            n_cqt_bins        = e.n_cqt_bins,
            target_h          = e.target_h,
            target_w          = e.target_w,
            k_max             = e.k_max,
            welch_nperseg     = e.welch_nperseg,
            wvd_window        = e.wvd_window,
            wvd_n_time        = e.wvd_n_time,
            hht_max_imf       = e.hht_max_imf,
            bispectrum_n_freq = e.bispectrum_n_freq,
        )

    # ── Properties ───────────────────────────────────────────────────

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return self._class_to_idx

    @property
    def labels(self) -> List[int]:
        return [lbl for _, lbl, _ in self._samples]


# ═══════════════════════════════════════════════════════════════════════
#  OmniFeatureDataModule
# ═══════════════════════════════════════════════════════════════════════

class OmniFeatureDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for the pre-computed feature cache.

    Follows the ``DALIAudioDataModule`` interface: after ``setup()``,
    exposes ``class_to_idx``, ``idx_to_class``, ``num_classes``, and
    ``class_weights`` so the model can configure ``FocalLoss`` without
    knowing about the dataset internals.

    Parameters
    ----------
    data_dir : str
        Path to ``Split1s/`` root (contains train/, val/, test/).
    cache_dir : str
        Root for ``.pt`` cache files.  Sub-directories per split are
        created automatically.
    batch_size : int
    num_workers : int
        DataLoader worker threads.
    precompute_workers : int
        Process-pool workers for feature pre-computation.
    sample_rate : int
    oversample_train : bool
        Use WeightedRandomSampler to equalise training class frequencies.
    force_recompute : bool
        Overwrite existing cache files.
    extractor_kwargs : dict or None
        Passed to ``UnderwaterFeatureExtractor`` constructor.
    """

    def __init__(
        self,
        data_dir:            str,
        cache_dir:           str    = "data/cache",
        batch_size:          int    = 64,
        num_workers:         int    = 8,
        precompute_workers:  int    = 8,
        sample_rate:         int    = 5_120,
        oversample_train:    bool   = True,
        force_recompute:     bool   = False,
        extractor_kwargs:    Optional[dict] = None,
        denoise_cfg:         Optional[dict] = None,
    ):
        super().__init__()
        self.data_dir            = Path(data_dir)
        self.cache_dir           = Path(cache_dir)
        self.batch_size          = batch_size
        self.num_workers         = num_workers
        self.precompute_workers  = precompute_workers
        self.sample_rate         = sample_rate
        self.oversample_train    = oversample_train
        self.force_recompute     = force_recompute
        self.extractor_kwargs    = extractor_kwargs or {}
        self.denoise_cfg         = denoise_cfg or {"method": "off"}

        self.class_to_idx:  Dict[str, int] = {}
        self.idx_to_class:  Dict[int, str] = {}
        self.num_classes:   int             = 0
        self.class_weights: List[float]     = []

        self._train_ds: Optional[CachedFeatureDataset] = None
        self._val_ds:   Optional[CachedFeatureDataset] = None
        self._test_ds:  Optional[CachedFeatureDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets, compute class mappings, and trigger pre-computation."""
        extractor = UnderwaterFeatureExtractor(
            sample_rate=self.sample_rate,
            **self.extractor_kwargs,
        )

        splits = {
            "train": self.data_dir / "train",
            "val":   self.data_dir / "val",
            "test":  self.data_dir / "test",
        }

        ds: Dict[str, CachedFeatureDataset] = {}
        for split, split_dir in splits.items():
            if not split_dir.exists():
                warnings.warn(f"Split directory not found: {split_dir}")
                continue
            cache = self.cache_dir / split
            cache.mkdir(parents=True, exist_ok=True)
            ds[split] = CachedFeatureDataset(
                split_dir=split_dir,
                cache_dir=cache,
                extractor=extractor,
                sample_rate=self.sample_rate,
                force_recompute=self.force_recompute,
                denoise_cfg=self.denoise_cfg,
            )

        # Derive class mappings from training split (alphabetical)
        if "train" in ds:
            self.class_to_idx = ds["train"].class_to_idx
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            self.num_classes   = len(self.class_to_idx)

            # Inverse-frequency class weights (same formula as DALIAudioDataModule)
            labels      = ds["train"].labels
            counts      = [labels.count(i) for i in range(self.num_classes)]
            total       = sum(counts)
            self.class_weights = [
                total / (self.num_classes * c) if c > 0 else 1.0
                for c in counts
            ]

            print(f"\nClasses       : {self.class_to_idx}")
            print(f"Train counts  : { {k: counts[v] for k, v in self.class_to_idx.items()} }")
            print(f"Class weights : { {k: f'{self.class_weights[v]:.2f}' for k, v in self.class_to_idx.items()} }")

        # Trigger parallel pre-computation for any missing cache files
        for split_name, dataset in ds.items():
            print(f"\nChecking {split_name} cache …")
            dataset.precompute_all(num_workers=self.precompute_workers)

        self._train_ds = ds.get("train")
        self._val_ds   = ds.get("val")
        self._test_ds  = ds.get("test")

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.oversample_train and self._train_ds is not None:
            labels  = self._train_ds.labels
            weights = [1.0 / max(labels.count(l), 1) for l in labels]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(weights, dtype=torch.float32),
                num_samples=len(weights),
                replacement=True,
            )
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            collate_fn=omni_collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=omni_collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=omni_collate_fn,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )
