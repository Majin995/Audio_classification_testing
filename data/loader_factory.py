"""Loader factory — pick DALI vs threaded backend and splitting vs non-splitting.

Drop-in entry point for trainers that historically constructed
``DALIAudioDataModule`` directly. Exposes a ``build_loader(loader, ...)``
that returns a Lightning DataModule with a uniform attribute surface
(``class_to_idx``, ``idx_to_class``, ``num_classes``, ``class_weights``,
``train_files_override``, bound ``_apply_merge``, bound ``_scan_split``).

Loader IDs
----------
- ``dali``           : ``DALIAudioDataModule``. Non-splitting (files must
                      already be exactly ``fixed_len`` samples).
- ``dali_split``     : ``ExtendedDALIAudioDataModule``. Long files chunked
                      at load time using ``window_sec`` / ``hop_sec``.
- ``threaded``       : ``ExtendedThreadedAudioDataModule`` configured as
                      non-splitting (one window per file, length =
                      ``fixed_len``). Pure soundfile, no CUDA.
- ``threaded_split`` : ``ExtendedThreadedAudioDataModule`` with windowing.

Extended loaders do not natively expose ``train_files_override`` /
``_scan_split(split)`` / ``set_train_files_override``. We attach lightweight
stand-ins so trainers that touch those (LDAM cls_num_list, ocean-noise pool
build, AL round refresh) keep working without per-trainer branches.
"""
from __future__ import annotations

from typing import Optional

LOADER_CHOICES = ("dali", "dali_split", "threaded", "threaded_split")


def build_loader(
    loader: str,
    *,
    data_dir: str,
    batch_size: int = 64,
    target_sr: int = 5120,
    fixed_len: int = 5120,
    oversample_train: bool = True,
    merge_classes: Optional[dict] = None,
    # DALI-only
    num_threads: int = 8,
    device_id: int = 0,
    denoise_method: str = "off",
    # Splitting-only
    window_sec: Optional[float] = None,
    hop_sec: Optional[float] = None,
    # Threaded-only
    num_workers: int = 8,
    pin_memory: bool = True,
    # Shared waveform preprocessing
    rms_normalize: bool = False,
    target_rms: float = 0.1,
):
    """Return a Lightning DataModule for the requested backend.

    The returned object always carries the attribute surface the trainers
    in ``training/`` already expect from ``DALIAudioDataModule``.
    """
    if loader not in LOADER_CHOICES:
        raise ValueError(f"--loader must be one of {LOADER_CHOICES}, got {loader!r}")

    if loader == "dali":
        from data.audio_lightning_loader import DALIAudioDataModule
        dm = DALIAudioDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            target_sr=target_sr,
            fixed_len=fixed_len,
            oversample_train=oversample_train,
            merge_classes=merge_classes,
            denoise_method=denoise_method,
            rms_normalize=rms_normalize,
            target_rms=target_rms,
        )
        return dm

    # All Extended variants share the same window-aware constructor.
    if loader == "dali_split":
        from data.extended_audio_dali_loader import ExtendedDALIAudioDataModule as _DM
        backend_kwargs = dict(num_threads=num_threads, device_id=device_id)
    elif loader == "threaded_split":
        from data.extended_audio_threaded_loader import ExtendedThreadedAudioDataModule as _DM
        backend_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    else:  # "threaded" — non-splitting via window == fixed_len
        from data.extended_audio_threaded_loader import ExtendedThreadedAudioDataModule as _DM
        backend_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
        # Force non-splitting: one window per file of length fixed_len.
        window_sec = fixed_len / float(target_sr)
        hop_sec = window_sec

    if window_sec is None:
        window_sec = fixed_len / float(target_sr)
    if hop_sec is None:
        hop_sec = window_sec

    dm = _DM(
        data_dir=data_dir,
        batch_size=batch_size,
        target_sr=target_sr,
        window_sec=window_sec,
        hop_sec=hop_sec,
        rms_normalize=rms_normalize,
        target_rms=target_rms,
        oversample_train=oversample_train,
        merge_classes=merge_classes,
        **backend_kwargs,
    )
    _attach_dali_compat(dm)
    return dm


def _attach_dali_compat(dm) -> None:
    """Add the few DALI-only attributes/methods trainers in ``training/`` use.

    Specifically:
      - ``train_files_override``: defaults to ``None``; AL trainers may set it
        via ``set_train_files_override``.
      - ``_scan_split(split)``: trainers call this after ``setup()`` to
        recompute per-class file counts. Maps onto the loader's cached
        ``_splits`` dict (already merge-applied).
      - ``set_train_files_override(...)``: simple setter that triggers
        re-setup so the new file list takes effect.
    """
    if not hasattr(dm, "train_files_override"):
        dm.train_files_override = None

    if not hasattr(dm, "_orig_scan_split"):
        from data.extended_audio_threaded_loader import _scan_split as _module_scan

        def _scan_split(split, _self=dm):
            # Prefer the post-setup cache so callers see the merged view; fall
            # back to a fresh disk scan if setup() hasn't run yet.
            if getattr(_self, "_splits", None) and split in _self._splits:
                return _self._splits[split]
            return _module_scan(_self.data_dir, split)

        dm._scan_split = _scan_split

    if not hasattr(dm, "set_train_files_override"):
        def set_train_files_override(files_by_class, _self=dm):
            _self.train_files_override = files_by_class
            _self._splits = {}
            _self._windows = {}
            _self.setup()

        dm.set_train_files_override = set_train_files_override
