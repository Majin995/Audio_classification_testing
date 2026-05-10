"""
DALI-accelerated Audio DataModule for Underwater Acoustic Classification.

Handles:
  - GPU-accelerated decoding of float32 PCM WAV files via NVIDIA DALI
  - Severe class imbalance via oversampled training file list
  - Automatic class-to-index mapping consistent with alphabetical order
  - Epoch-safe DALI iterator wrapping for PyTorch Lightning
  - Optional CPU-side waveform denoising (EMD-Wavelet / NMF-ICA) applied
    between DALI output and model forward pass.
  - Optional unlabeled data stream for DART-MT semi-supervised training.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Literal, Optional

import torch
import pytorch_lightning as pl

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    print("WARNING: NVIDIA DALI not available. Install nvidia-dali-cuda120.")

DATA_DIR         = os.environ.get("DATA_DIR", "")
MAX_DISPLAY_FREQ = 2_560     # Hz — highest frequency of interest for vessel classification
TARGET_SR        = MAX_DISPLAY_FREQ * 2   # Nyquist theorem: SR ≥ 2 × f_max → 5120 Hz
FIXED_LEN        = TARGET_SR             # One second of audio at TARGET_SR


# ─────────────────────────── DALI Pipeline ──────────────────────────────────

@pipeline_def
def audio_pipeline(file_list: str, shuffle: bool = True,
                   target_sr: int = TARGET_SR, fixed_len: int = FIXED_LEN,
                   pad_last_batch: bool = True,
                   rms_normalize: bool = False,
                   target_rms:    float = 0.1):
    """
    DALI pipeline: read → decode (float32 PCM) → resample → pad/crop →
    optional RMS normalization.

    Args:
        file_list : Path to text file with lines: "/abs/path/to/file.wav  label"
        shuffle   : Shuffle file order each epoch.
        target_sr : Target sample rate (no-op if audio is already at this rate).
        fixed_len : Output waveform length in samples.
        pad_last_batch : If True, the last batch is padded with duplicates to
                         keep batch size constant. Set False for order-preserving
                         scoring so we get exactly len(files) outputs (paired
                         with LastBatchPolicy.PARTIAL on the iterator).
        rms_normalize : If True, zero-mean + unit-std (≈ unit-RMS for centered
                        audio), then scale to ``target_rms``. Helps reduce
                        intra-class amplitude variance from recording-rig
                        differences.
        target_rms    : Post-normalization RMS magnitude.
    """
    files, labels = fn.readers.file(
        file_list=file_list,
        random_shuffle=shuffle,
        name="Reader",
        pad_last_batch=pad_last_batch,
    )

    # libsndfile inside DALI handles WAVE_FORMAT_IEEE_FLOAT (format 0x0003)
    audio, sr = fn.decoders.audio(
        files,
        dtype=types.FLOAT,
        downmix=True,          # Mono output: shape (samples,)
    )

    # Resample only if SR doesn't match target (no-op when they match)
    audio = fn.audio_resample(audio, in_rate=sr, out_rate=target_sr)

    # Pad short clips to fixed_len, then slice to fixed_len (handles both)
    audio = fn.pad(audio, axes=(0,), shape=(fixed_len,), fill_value=0.0)
    audio = fn.slice(audio, 0, fixed_len, axes=[0])

    # Per-clip zero-mean + unit-std normalization (≈ unit-RMS), then rescale.
    if rms_normalize:
        audio = fn.normalize(audio, axes=[0], epsilon=1e-9)
        audio = audio * float(target_rms)

    return audio, labels.gpu()


# ─────────────────────────── Iterator Wrapper ───────────────────────────────

class _DALIWrapper:
    """
    Wraps DALIGenericIterator to yield (waveform, label) tuples per batch.

    Uses LastBatchPolicy.DROP so DALI always produces full batches and sets
    _end_of_epoch=True cleanly, allowing reset() to be called safely at the
    start of the next epoch without triggering DALI's reset warning.
    """
    def __init__(self, iterator: "DALIGenericIterator", n_batches: int):
        self._iter      = iterator
        self._n_batches = n_batches
        self._started   = False      # Avoid reset() on the very first iteration

    def __iter__(self):
        if self._started:
            # Safe only once the previous epoch has fully completed (DROP policy
            # guarantees _end_of_epoch=True after the last complete batch).
            self._iter.reset()
        self._started = True

        for batch in self._iter:
            audio  = batch[0]["audio"]                     # (B, T) GPU tensor
            labels = batch[0]["label"].squeeze(-1).long()  # (B,)
            yield audio, labels

    def __len__(self):
        return self._n_batches


# ─────────────────────────── Denoising Hook ─────────────────────────────────

class _DenoisingBatchHook:
    """
    Wraps a ``_DALIWrapper`` and applies CPU-side waveform denoising to each
    batch between DALI output and GPU model forward.

    Args:
        wrapper    : Underlying ``_DALIWrapper`` instance.
        method     : Denoising method string (see ``DenoiseTransform``).
        sample_rate: Audio sample rate for the denoising functions.
        **kwargs   : Forwarded to the denoising function.
    """
    def __init__(self, wrapper: _DALIWrapper, method: str,
                 sample_rate: int = TARGET_SR, **kwargs):
        self._wrapper     = wrapper
        self._sample_rate = sample_rate
        self._method      = method
        self._kwargs      = kwargs
        # Import lazily so the file loads even without processing/ on sys.path
        from processing.denoise.transform import DenoiseTransform
        self._transform = DenoiseTransform(method=method,
                                           sample_rate=sample_rate, **kwargs)

    def __iter__(self):
        for audio, labels in self._wrapper:
            audio = self._transform(audio)
            yield audio, labels

    def __len__(self):
        return len(self._wrapper)


# ─────────────────────────── HPF Hook ───────────────────────────────────────

class _HPFBatchHook:
    """Apply an n-th order Butterworth-magnitude high-pass filter to each
    GPU batch via FFT-mask multiplication.

    The mask is the magnitude response of an n-th order Butterworth HPF
    (zero-phase since the mask is real-symmetric). For SR=5120 / cutoff=20 Hz
    / order=4 this strips DC and sub-bass rumble while leaving the cavitation
    and BPF bands intact.
    """

    def __init__(self, wrapper, sample_rate: int = TARGET_SR,
                 cutoff_hz: float = 20.0, order: int = 4):
        self._wrapper      = wrapper
        self._sample_rate  = int(sample_rate)
        self._cutoff       = float(cutoff_hz)
        self._order        = int(order)
        self._mask_cache:  dict[int, "torch.Tensor"] = {}

    def _get_mask(self, T: int, device, dtype):
        key = (T, str(device), str(dtype))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        freqs = torch.fft.rfftfreq(T, d=1.0 / self._sample_rate).to(device)
        # Magnitude response of an n-th order Butterworth HPF:
        #   |H(f)|² = (f/fc)^(2n) / (1 + (f/fc)^(2n))
        ratio = freqs / max(self._cutoff, 1e-9)
        mag = ratio.pow(2 * self._order)
        mask = (mag / (1.0 + mag)).sqrt().to(dtype)
        self._mask_cache[key] = mask
        return mask

    def __iter__(self):
        for audio, labels in self._wrapper:
            T    = audio.shape[-1]
            mask = self._get_mask(T, audio.device, audio.dtype)
            Xf   = torch.fft.rfft(audio.float(), dim=-1)
            audio = torch.fft.irfft(Xf * mask, n=T, dim=-1).to(audio.dtype)
            yield audio, labels

    def __len__(self):
        return len(self._wrapper)


# ─────────────────────────── Score Loader (AL) ──────────────────────────────

class _ScoreLoader:
    """
    Order-preserving iterator over an explicit list of audio files. Yields
    (waveform: (B, T) GPU tensor, indices: (B,) long CPU tensor) per batch,
    where ``indices`` are the positions in the original ``files`` list passed
    to ``DALIAudioDataModule.make_score_loader``. The iterator is single-pass
    (no reset across calls) — call ``make_score_loader`` again for another
    pass.

    The original file paths are kept on ``self.file_paths`` so callers don't
    need to track them externally.
    """
    def __init__(self, iterator, n_files: int, file_paths: list, batch_size: int):
        self._iter      = iterator
        self.n_files    = n_files
        self.file_paths = file_paths
        self.batch_size = batch_size

    def __iter__(self):
        if self._iter is None or self.n_files == 0:
            return
        seen = 0
        for batch in self._iter:
            audio = batch[0]["audio"]                       # (B, T) GPU
            idx   = batch[0]["label"].squeeze(-1).long()    # (B,) abs index
            # Trim padding from a partial last batch (PARTIAL still sometimes
            # returns full batches with the last samples being valid duplicates
            # — we cap at n_files to be safe).
            remaining = self.n_files - seen
            if remaining <= 0:
                break
            if audio.shape[0] > remaining:
                audio = audio[:remaining]
                idx   = idx[:remaining]
            seen += audio.shape[0]
            yield audio, idx

    def __len__(self):
        if self.n_files == 0:
            return 0
        return (self.n_files + self.batch_size - 1) // self.batch_size


# ─────────────────────────── Unlabeled Stream ────────────────────────────────

class _UnlabeledDALIWrapper:
    """
    Like ``_DALIWrapper`` but yields only waveforms (no labels).
    Labels from the DALI pipeline are discarded; used for DART-MT
    unlabeled consistency training.
    """
    def __init__(self, iterator: "DALIGenericIterator", n_batches: int):
        self._iter      = iterator
        self._n_batches = n_batches
        self._started   = False

    def __iter__(self):
        if self._started:
            self._iter.reset()
        self._started = True
        for batch in self._iter:
            audio = batch[0]["audio"]                   # (B, T) GPU tensor
            yield audio                                 # no labels

    def __len__(self):
        return self._n_batches


# ─────────────────────────── DataModule ─────────────────────────────────────

class DALIAudioDataModule(pl.LightningDataModule):
    """
    Lightning DataModule backed by NVIDIA DALI for fast audio loading.

    Training loader uses an oversampled file list to counteract class imbalance
    (minority classes are repeated until each class has `max_class_count` files).
    Val/test loaders use the files as-is.

    Attributes after setup():
        class_to_idx  (dict): e.g. {"Cargo": 0, "Passenger": 1, ...}
        idx_to_class  (dict): reverse mapping
        num_classes   (int):  number of classes
        class_weights (list): inverse-frequency weights for loss functions
    """

    def __init__(
        self,
        data_dir:       str = DATA_DIR,
        batch_size:     int = 64,
        num_threads:    int = 8,
        device_id:      int = 0,
        target_sr:      int = TARGET_SR,
        fixed_len:      int = FIXED_LEN,
        oversample_train: bool = True,
        merge_classes:  Optional[dict[str, str]] = None,
        denoise_method: str = "off",
        unlabeled_dir:  Optional[str] = None,
        unlabeled_batch_size: int = 0,
        train_files_override: Optional[Dict[str, list]] = None,
        rms_normalize:  bool  = False,
        target_rms:     float = 0.1,
        hpf_hz:         float = 0.0,
        hpf_order:      int   = 4,
    ):
        """
        Args:
            denoise_method    : Waveform denoising applied after DALI output.
                                One of ``'off'``, ``'emd_wavelet'``, ``'nmf'``,
                                ``'ica'``, ``'nmf_ica'``, ``'emd_nmf'``.
            unlabeled_dir     : Path to a directory of unlabeled audio files
                                (flat or class-subdirected — labels are ignored).
                                If ``None`` or empty, unlabeled stream is disabled.
            unlabeled_batch_size : Batch size for the unlabeled DALI iterator.
                                   Defaults to ``batch_size`` if 0.
        """
        super().__init__()
        if not DALI_AVAILABLE:
            raise RuntimeError("nvidia-dali is required. Run: pip install nvidia-dali-cuda120")
        self.data_dir    = Path(data_dir).resolve()   # must be absolute for DALI workers
        self.batch_size  = batch_size
        self.num_threads = num_threads
        self.device_id   = device_id
        self.target_sr   = target_sr
        self.fixed_len   = fixed_len
        self.oversample_train = oversample_train

        # merge_classes: {source_class: target_class} — source files get target's label.
        # e.g. {"Passenger": "Cargo"} folds Passenger into Cargo.
        self.merge_classes = merge_classes or {}

        self.denoise_method = denoise_method
        self.unlabeled_dir  = Path(unlabeled_dir) if unlabeled_dir else None
        self.unlabeled_batch_size = unlabeled_batch_size or batch_size

        # Waveform preprocessing (sub-task β)
        self.rms_normalize = bool(rms_normalize)
        self.target_rms    = float(target_rms)
        self.hpf_hz        = float(hpf_hz)
        self.hpf_order     = int(hpf_order)
        if self.denoise_method != "off" and self.hpf_hz > 0:
            print(f"WARNING: --denoise={self.denoise_method} + HPF {self.hpf_hz} Hz "
                  "is mostly redundant — denoising already strips low-frequency content.")

        # Active-learning support: when set, the train split scan is replaced
        # by this {class_name: [abs_paths]} dict. Val/test splits are
        # untouched. Setter ``set_train_files_override`` lets callers update
        # the labeled subset between rounds without rebuilding the module.
        self.train_files_override: Optional[Dict[str, list]] = train_files_override

        self.class_to_idx:  dict = {}
        self.idx_to_class:  dict = {}
        self.num_classes:   int  = 0
        self.class_weights: list = []

        self._tmp_files: list = []   # tempfiles, cleaned up at teardown

    # ── helpers ─────────────────────────────────────────────────────────────

    def _apply_merge(self, files_by_class: dict[str, list[str]]) -> dict[str, list[str]]:
        """Merge source classes into their target classes, then drop source keys."""
        if not self.merge_classes:
            return files_by_class
        result = {k: list(v) for k, v in files_by_class.items()}
        for src, tgt in self.merge_classes.items():
            if src in result and tgt in result:
                result[tgt] = result[tgt] + result.pop(src)
            elif src in result:
                result[tgt] = result.pop(src)
        return result

    def _scan_split(self, split: str) -> dict[str, list[str]]:
        """Returns {class_name: [abs_file_paths]} for a split directory."""
        split_dir = self.data_dir / split
        result: dict[str, list[str]] = {}
        for cls in sorted(split_dir.iterdir()):
            if not cls.is_dir():
                continue
            files = sorted(
                str(f) for f in cls.rglob("*")
                if f.suffix.lower() in {".wav", ".mp3", ".flac"}
            )
            if files:
                result[cls.name] = files
        return result

    def _write_file_list(self, files_by_class: dict[str, list[str]],
                         oversample: bool = False) -> tuple[str, int]:
        """
        Writes a DALI-format file list to a tempfile.
        Returns (tempfile_path, total_sample_count).
        """
        lines: list[str] = []
        counts = {cls: len(fs) for cls, fs in files_by_class.items()}
        max_count = max(counts.values()) if oversample else 0

        for cls, fs in files_by_class.items():
            label = self.class_to_idx[cls]
            if oversample:
                reps   = (max_count + len(fs) - 1) // len(fs)
                fs_aug = (fs * reps)[:max_count]
            else:
                fs_aug = fs
            for path in fs_aug:
                lines.append(f"{path} {label}")

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                          delete=False, prefix="dali_flist_")
        tmp.write("\n".join(lines))
        tmp.flush()
        tmp.close()
        self._tmp_files.append(tmp.name)
        return tmp.name, len(lines)

    def _scan_unlabeled(self) -> list[str]:
        """
        Return a flat list of audio file paths under ``self.unlabeled_dir``.
        Labels are not used — all files are given sentinel label 0 so DALI
        can still build a file-list manifest.
        """
        if self.unlabeled_dir is None or not self.unlabeled_dir.exists():
            return []
        exts = {".wav", ".mp3", ".flac"}
        files = sorted(
            str(f) for f in self.unlabeled_dir.rglob("*")
            if f.suffix.lower() in exts
        )
        return files

    def _write_unlabeled_file_list(self, files: list[str]) -> tuple[str, int]:
        """Write a DALI file list for unlabeled files (all labelled 0)."""
        lines = [f"{p} 0" for p in files]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                          delete=False, prefix="dali_unlabeled_")
        tmp.write("\n".join(lines))
        tmp.flush()
        tmp.close()
        self._tmp_files.append(tmp.name)
        return tmp.name, len(lines)

    def _make_loader(self, split: str, oversample: bool) -> _DALIWrapper:
        if split == "train" and self.train_files_override is not None:
            files_by_class = self._apply_merge(self.train_files_override)
        else:
            files_by_class = self._apply_merge(self._scan_split(split))
        shuffle    = (split == "train")
        flist, n   = self._write_file_list(files_by_class, oversample=oversample)

        pipe = audio_pipeline(
            file_list=flist,
            shuffle=shuffle,
            target_sr=self.target_sr,
            fixed_len=self.fixed_len,
            rms_normalize=self.rms_normalize,
            target_rms=self.target_rms,
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
        )
        pipe.build()

        n_batches = n // self.batch_size   # Floor: DROP discards the last partial batch

        dali_iter = DALIGenericIterator(
            pipe,
            output_map=["audio", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.DROP,
        )
        wrapper = _DALIWrapper(dali_iter, n_batches=n_batches)

        # Optionally wrap with CPU denoising
        if self.denoise_method != "off":
            wrapper = _DenoisingBatchHook(wrapper, method=self.denoise_method,
                                          sample_rate=self.target_sr)

        # Optionally wrap with HPF (post-DALI / post-denoise GPU FFT-mask)
        if self.hpf_hz > 0:
            wrapper = _HPFBatchHook(wrapper, sample_rate=self.target_sr,
                                    cutoff_hz=self.hpf_hz, order=self.hpf_order)
        return wrapper

    def _make_unlabeled_loader(self) -> Optional[_UnlabeledDALIWrapper]:
        """Build the unlabeled DALI stream for DART-MT.  Returns None if disabled."""
        files = self._scan_unlabeled()
        if not files:
            return None

        flist, n = self._write_unlabeled_file_list(files)

        pipe = audio_pipeline(
            file_list=flist,
            shuffle=True,
            target_sr=self.target_sr,
            fixed_len=self.fixed_len,
            batch_size=self.unlabeled_batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
        )
        pipe.build()

        n_batches = n // self.unlabeled_batch_size

        dali_iter = DALIGenericIterator(
            pipe,
            output_map=["audio", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.DROP,
        )
        return _UnlabeledDALIWrapper(dali_iter, n_batches=n_batches)

    # ── Lightning interface ──────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None):
        # Class index must be stable across AL rounds — derive it from the
        # full disk train scan even when train_files_override is set, so
        # rounds with smaller subsets keep the same label space.
        full_train = self._apply_merge(self._scan_split("train"))
        if self.train_files_override is not None:
            train_files = self._apply_merge(self.train_files_override)
        else:
            train_files = full_train
        classes = sorted(full_train.keys())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes   = len(classes)

        counts = [len(train_files.get(c, [])) for c in classes]
        total  = sum(counts)
        # max(c, 1) guards against a class temporarily absent from the
        # active-learning subset (early rounds, small init).
        self.class_weights = [total / (self.num_classes * max(c, 1)) for c in counts]

        print(f"\nClasses        : {self.class_to_idx}")
        print(f"Train counts   : { {c: len(train_files[c]) for c in classes} }")
        print(f"Class weights  : { {c: f'{w:.2f}' for c, w in zip(classes, self.class_weights)} }")

    def train_dataloader(self):
        return self._make_loader("train", oversample=self.oversample_train)

    def val_dataloader(self):
        return self._make_loader("val", oversample=False)

    def test_dataloader(self):
        return self._make_loader("test", oversample=False)

    def unlabeled_dataloader(self) -> Optional[_UnlabeledDALIWrapper]:
        """
        Return the unlabeled DALI stream for DART-MT semi-supervised training.
        Returns ``None`` if ``unlabeled_dir`` was not set or is empty.

        Usage in HydroDARTMT.training_step:
            unlabeled_loader = self.trainer.datamodule.unlabeled_dataloader()
        """
        return self._make_unlabeled_loader()

    # ── Active-learning support ──────────────────────────────────────────────

    def set_train_files_override(self, files_by_class: Dict[str, list]) -> None:
        """Update the labeled subset between AL rounds. Recomputes class weights."""
        self.train_files_override = files_by_class
        # Refresh class weights from the new labeled count distribution.
        classes = sorted(self.class_to_idx.keys())
        counts  = [len(files_by_class.get(c, [])) for c in classes]
        total   = sum(counts)
        self.class_weights = [total / (self.num_classes * max(c, 1)) for c in counts]

    def make_score_loader(self, files: list, batch_size: Optional[int] = None) -> "_ScoreLoader":
        """
        Build an order-preserving DALI iterator over an explicit list of files.
        Yields (waveform, original_index) per batch so callers can map predictions
        back to source paths. Uses LastBatchPolicy.PARTIAL so every file is
        scored exactly once.
        """
        if not files:
            return _ScoreLoader(iterator=None, n_files=0,
                                file_paths=[], batch_size=batch_size or self.batch_size)

        bs = batch_size or self.batch_size
        # Encode the absolute index into the DALI label so we can recover order
        # even though DALI's reader output ordering is not contractually
        # documented for non-shuffled input.
        lines = [f"{p} {i}" for i, p in enumerate(files)]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                          delete=False, prefix="dali_score_")
        tmp.write("\n".join(lines))
        tmp.flush(); tmp.close()
        self._tmp_files.append(tmp.name)

        pipe = audio_pipeline(
            file_list=tmp.name,
            shuffle=False,
            target_sr=self.target_sr,
            fixed_len=self.fixed_len,
            pad_last_batch=False,
            batch_size=bs,
            num_threads=self.num_threads,
            device_id=self.device_id,
        )
        pipe.build()

        dali_iter = DALIGenericIterator(
            pipe,
            output_map=["audio", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )
        return _ScoreLoader(iterator=dali_iter, n_files=len(files),
                            file_paths=list(files), batch_size=bs)

    def teardown(self, stage: Optional[str] = None):
        for f in self._tmp_files:
            try:
                os.unlink(f)
            except OSError:
                pass
        self._tmp_files.clear()
