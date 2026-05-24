"""DALI Lightning DataModule for long-audio classification.

Same API and per-window output as ``ExtendedThreadedAudioDataModule`` and
``DALIAudioDataModule`` — drop-in swappable into Lightning models.

Implementation: ``fn.external_source`` feeds a Python window-extractor (uses
soundfile to read each window's byte range), then DALI applies the same
post-processing ops as ``DALIAudioDataModule`` (pad/slice to ``fixed_len``,
optional ``fn.normalize`` + scale to ``target_rms``). Result is bit-identical
to the original DALI loader for 1-window-per-file inputs at the target SR.

Why external_source rather than ``fn.readers.file`` + ``fn.decoders.audio``:
- ``fn.decoders.audio`` always decodes the WHOLE file. There is no native
  way to ask DALI to emit multiple samples per input file.
- Pre-extracting windows to disk is wasteful for multi-million windows.
- ``fn.external_source`` with parallel python workers gives equivalent
  throughput in practice while letting us slice each window's byte range.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import pytorch_lightning as pl

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False

from data.extended_audio_threaded_loader import (
    DEFAULT_SR, DEFAULT_WINDOW_SEC, enumerate_windows, _scan_split,
)


class _ExternalWindowSource:
    """Stateful callable: returns a batch of (window, label) numpy arrays.

    Maintains an internal cursor + shuffled index list per epoch. Reset by
    DALI between epochs via ``reset()`` (we call it manually in the wrapper).
    """

    def __init__(self, windows, batch_size, fixed_len, target_sr,
                 shuffle, drop_last, seed):
        self.windows = windows
        self.batch_size = int(batch_size)
        self.fixed_len = int(fixed_len)
        self.target_sr = int(target_sr)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(seed)
        self._epoch_idx = None
        self._cursor = 0
        self._reshuffle()

    def _reshuffle(self):
        order = np.arange(len(self.windows))
        if self.shuffle:
            self.rng.shuffle(order)
        self._epoch_idx = order
        self._cursor = 0

    def __len__(self):
        n = len(self.windows)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __call__(self, sample_info):
        # DALI calls us with sample_info; we use batched form below instead.
        raise RuntimeError("use the batched callable via fn.external_source")

    def batch(self):
        """Return one batch: (list[(fixed_len,) float32 arrays], list[int labels])."""
        if self._cursor >= len(self._epoch_idx):
            self._reshuffle()
            raise StopIteration
        end = min(self._cursor + self.batch_size, len(self._epoch_idx))
        chunk = self._epoch_idx[self._cursor:end]
        if self.drop_last and len(chunk) < self.batch_size:
            self._reshuffle()
            raise StopIteration
        self._cursor = end

        audios, labels = [], []
        for i in chunk:
            path, label, start, stop = self.windows[i]
            a, sr = sf.read(path, dtype="float32", start=int(start),
                            stop=int(stop), always_2d=False)
            if a.ndim > 1:
                a = a.mean(axis=1)
            if sr != self.target_sr:
                # Defer SR mismatch to DALI's fn.audio_resample. Pass the
                # raw rate via padding signal — but we don't have a band for
                # variable SR per sample. So we fail loud: dataset builder
                # must deliver target_sr, matching DALIAudioDataModule's
                # contract on the curated 1s tree.
                raise ValueError(
                    f"sr={sr} != target_sr={self.target_sr} for {path}. "
                    "Pre-resample the dataset (matches DALIAudioDataModule "
                    "behavior on Combined_*_1s)."
                )
            audios.append(a.astype(np.float32))
            labels.append(np.int64(label))
        return audios, labels


@pipeline_def
def _extended_audio_pipeline(source, fixed_len, rms_normalize, target_rms,
                              pad_last_batch=True):
    audio, label = fn.external_source(
        source=source, num_outputs=2,
        layout=["t", ""], batch=True, parallel=False,
    )
    # Pad+slice (no-op if input is exactly fixed_len)
    audio = fn.pad(audio, axes=(0,), shape=(fixed_len,), fill_value=0.0)
    audio = fn.slice(audio, 0, fixed_len, axes=[0])
    if rms_normalize:
        audio = fn.normalize(audio, axes=[0], epsilon=1e-9)
        audio = audio * float(target_rms)
    return audio.gpu(), label.gpu()


class _DALIExtendedWrapper:
    def __init__(self, dali_iter, source: _ExternalWindowSource):
        self._iter = dali_iter
        self._source = source

    def __iter__(self):
        # Do NOT reshuffle here: DALI prefetches K batches at pipeline build
        # time which advances the source cursor. Calling _reshuffle() here
        # would reset the cursor to 0 and DALI would then yield K duplicate
        # batches (cached pre-fetch) followed by another full pass from
        # cursor=0. Instead we let the source's batch() raise StopIteration
        # at the natural epoch boundary; DALI's auto_reset=True picks up the
        # reshuffle done inside batch() before each new epoch begins.
        for batch in self._iter:
            audio = batch[0]["audio"]                       # (B, T) GPU
            label = batch[0]["label"].squeeze(-1).long()    # (B,)
            yield audio, label

    def __len__(self):
        return len(self._source)


class ExtendedDALIAudioDataModule(pl.LightningDataModule):
    """DALI counterpart of ``ExtendedThreadedAudioDataModule``.

    Same constructor signature plus DALI-specific ``num_threads`` and
    ``device_id``. Produces identical batches modulo float-order differences
    when compared against the threaded variant (verified by test).
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_threads: int = 8,
        device_id: int = 0,
        target_sr: int = DEFAULT_SR,
        window_sec: float = DEFAULT_WINDOW_SEC,
        hop_sec: Optional[float] = None,
        rms_normalize: bool = False,
        target_rms: float = 0.1,
        oversample_train: bool = True,
        merge_classes: Optional[Dict[str, str]] = None,
        shuffle_train: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        if not DALI_AVAILABLE:
            raise RuntimeError("nvidia-dali is required. "
                               "Install nvidia-dali-cuda120 — or use "
                               "ExtendedThreadedAudioDataModule.")
        self.data_dir = Path(data_dir).resolve()
        self.batch_size = int(batch_size)
        self.num_threads = int(num_threads)
        self.device_id = int(device_id)
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
        self.seed = int(seed)

        self.class_to_idx: dict = {}
        self.idx_to_class: dict = {}
        self.num_classes: int = 0
        self.class_weights: list = []
        self._splits: dict = {}
        self._windows: dict = {}

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

        print(f"[DALI] Classes        : {self.class_to_idx}")
        print(f"[DALI] Train windows  : {len(self._windows['train'])}")
        print(f"[DALI] Val windows    : {len(self._windows['val'])}")
        print(f"[DALI] Test windows   : {len(self._windows['test'])}")
        print(f"[DALI] Class weights  : { {c: f'{w:.2f}' for c, w in zip(classes, self.class_weights)} }")

    def _build_loader(self, split: str, shuffle: bool, drop_last: bool):
        source = _ExternalWindowSource(
            self._windows[split], batch_size=self.batch_size,
            fixed_len=self.fixed_len, target_sr=self.target_sr,
            shuffle=shuffle, drop_last=drop_last, seed=self.seed,
        )

        # Pre-bind source.batch as the per-batch callable expected by external_source.
        def _src_cb():
            try:
                return source.batch()
            except StopIteration:
                # DALI external_source signals epoch end by raising StopIteration
                # on the underlying generator-style callable. With batch=True it
                # treats a return of empty lists as no-data — instead we re-raise.
                raise

        pipe = _extended_audio_pipeline(
            source=_src_cb, fixed_len=self.fixed_len,
            rms_normalize=self.rms_normalize, target_rms=self.target_rms,
            batch_size=self.batch_size, num_threads=self.num_threads,
            device_id=self.device_id,
        )
        pipe.build()

        dali_iter = DALIGenericIterator(
            pipe, output_map=["audio", "label"],
            size=len(source) * self.batch_size,
            last_batch_policy=(LastBatchPolicy.DROP if drop_last
                               else LastBatchPolicy.PARTIAL),
            auto_reset=True,
        )
        return _DALIExtendedWrapper(dali_iter, source)

    def train_dataloader(self):
        return self._build_loader("train", shuffle=self.shuffle_train, drop_last=True)

    def val_dataloader(self):
        return self._build_loader("val", shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._build_loader("test", shuffle=False, drop_last=False)


__all__ = ["ExtendedDALIAudioDataModule"]
