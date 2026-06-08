"""Self-splitting (windowing) **multi-label** audio DataModule — MLMC variant.

This is the multi-label, multi-class ("MLMC") sibling of
``data/extended_audio_threaded_loader.py``. Two differences from that file:

1. **Self-splitting.** Each long file is chunked into fixed ``window_sec``
   windows at load time (header-only enumeration via ``sf.info`` so manifests
   build in seconds; bytes read lazily per window via ``sf.read(start, stop)``).
   No pre-chopped dataset is required — point it at a folder of arbitrary-length
   recordings and it produces the clips itself.

2. **Multi-hot targets.** Every window yields a 0/1 vector of length
   ``num_classes`` instead of a single integer label, so a model can predict
   *several* classes at once (independent sigmoid heads + BCE), and the
   prediction is emitted in one-hot / multi-hot form. The single-label
   folder-per-class datasets in this repo are the degenerate case: exactly one
   positive bit per window (a true one-hot row).

Label sources (in priority order)
----------------------------------
- ``labels_map``: ``{abs_path: [class_name, ...]}`` for genuine multi-label
  ground truth (a file may carry >1 label, or zero).
- folder-per-class layout ``<split>/<class>/*.wav`` → single positive bit
  (classic one-hot). This is the default and matches every existing dataset.

Preprocessing (decode → mono → resample → pad/slice → optional RMS-normalize)
is bit-identical to ``ExtendedThreadedAudioDataModule`` so a checkpoint trained
against the single-label loader sees the same audio here.

Companion inference helpers (``probs_to_onehot``, ``tune_thresholds_per_class``,
``aggregate_source_logmean``, ``multilabel_report``) live at the bottom so the
campaign scripts share one honest threshold-tuning + one-hot-encoding path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

DEFAULT_SR = 5120
DEFAULT_WINDOW_SEC = 1.0
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


# ─── preprocessing (matches the DALI / extended-threaded pipeline) ──────────

def _normalize_dali_style(x: torch.Tensor, target_rms: float,
                          epsilon: float = 1e-9) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = var.sqrt()
    return (x - mean) / (std + epsilon) * float(target_rms)


# ─── split scanning + multi-hot label resolution ───────────────────────────

def scan_split_files(data_dir: Path, split: str) -> Dict[str, List[str]]:
    """folder-per-class: ``<split>/<class>/**.wav`` → {class: [path, ...]}."""
    out: Dict[str, List[str]] = {}
    split_dir = data_dir / split
    if not split_dir.is_dir():
        return out
    for cls in sorted(split_dir.iterdir()):
        if not cls.is_dir():
            continue
        files = sorted(str(f) for f in cls.rglob("*")
                       if f.suffix.lower() in AUDIO_EXTS)
        if files:
            out[cls.name] = files
    return out


def enumerate_windows_multilabel(path_labels, class_to_idx,
                                  window_samples, hop_samples):
    """Build (path, multihot, start, stop) for every window.

    ``path_labels`` : iterable of (abs_path, [class_name, ...]).
    Files shorter than one window contribute exactly one (zero-padded) window.
    Returns a list of (path, np.ndarray[float32, num_classes], start, stop).
    """
    n_cls = len(class_to_idx)
    out = []
    for p, names in path_labels:
        vec = np.zeros(n_cls, dtype=np.float32)
        for nm in names:
            if nm in class_to_idx:
                vec[class_to_idx[nm]] = 1.0
        try:
            info = sf.info(p)
        except Exception:
            continue
        n = info.frames
        if n <= 0:
            continue
        if n <= window_samples:
            out.append((p, vec, 0, n))
            continue
        start = 0
        while start + window_samples <= n:
            out.append((p, vec, start, start + window_samples))
            start += hop_samples
        if start < n:                     # trailing partial so tail isn't lost
            out.append((p, vec, n - window_samples, n))
    return out


class MLMCWindowedDataset(Dataset):
    """One window per index → (wav[fixed_len], multihot[num_classes] float)."""

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
        path, vec, start, stop = self.windows[idx]
        a, sr = sf.read(path, dtype="float32", start=int(start), stop=int(stop),
                        always_2d=False)
        if a.ndim > 1:
            a = a.mean(axis=1)
        x = torch.from_numpy(a)
        if sr != self.target_sr:
            x = AF.resample(x, sr, self.target_sr)
        if x.numel() < self.fixed_len:
            x = F.pad(x, (0, self.fixed_len - x.numel()))
        else:
            x = x[: self.fixed_len]
        if self.rms_normalize:
            x = _normalize_dali_style(x, self.target_rms)
        return x, torch.from_numpy(vec)


class MLMCWindowedDataModule(pl.LightningDataModule):
    """Self-windowing multi-label DataModule.

    Public attributes match the single-label loaders so callers can swap:
    ``class_to_idx``, ``idx_to_class``, ``num_classes``, ``class_weights``
    (the last is inverse-frequency *positive* weight per class, suitable for
    ``pos_weight`` in ``BCEWithLogitsLoss``).

    Parameters
    ----------
    data_dir : root holding ``train``/``val``/``test`` (case-sensitive; pass the
        lowercase symlink for ``Classifier_Dataset`` — see repo gotchas).
    labels_map : optional ``{abs_path: [class_name, ...]}`` for true multi-label
        ground truth. When ``None`` the folder name is the single label.
    classes : optional explicit class order. When ``None`` it is the sorted
        union of train folder names (and any names appearing in ``labels_map``).
    window_sec / hop_sec : self-splitting window + stride in seconds.
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
        labels_map: Optional[Dict[str, List[str]]] = None,
        classes: Optional[Sequence[str]] = None,
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
        self.labels_map = {str(Path(k).resolve()): list(v)
                           for k, v in (labels_map or {}).items()}
        self._forced_classes = list(classes) if classes is not None else None
        self.shuffle_train = bool(shuffle_train)
        self.pin_memory = bool(pin_memory)

        self.class_to_idx: dict = {}
        self.idx_to_class: dict = {}
        self.num_classes: int = 0
        self.class_weights: list = []
        self._windows: dict = {}

    # path → list[label]. Uses labels_map override when present, else folder.
    def _path_labels(self, files_by_class):
        out = []
        for cls, paths in files_by_class.items():
            for p in paths:
                key = str(Path(p).resolve())
                names = self.labels_map.get(key, [cls])
                out.append((p, names))
        return out

    def setup(self, stage: Optional[str] = None):
        splits = {s: scan_split_files(self.data_dir, s)
                  for s in ("train", "val", "test")}

        if self._forced_classes is not None:
            classes = list(self._forced_classes)
        else:
            names = set(splits["train"].keys())
            for v in self.labels_map.values():
                names.update(v)
            classes = sorted(names)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes = len(classes)

        for split in ("train", "val", "test"):
            self._windows[split] = enumerate_windows_multilabel(
                self._path_labels(splits[split]), self.class_to_idx,
                self.fixed_len, self.hop_samples,
            )

        # pos_weight = (#neg / #pos) per class on TRAIN windows (clamped).
        tr = self._windows["train"]
        if tr:
            Y = np.stack([w[1] for w in tr])      # (n_win, C)
            pos = Y.sum(0)
            neg = len(Y) - pos
            self.class_weights = [float(neg[i] / max(pos[i], 1.0))
                                  for i in range(self.num_classes)]
        else:
            self.class_weights = [1.0] * self.num_classes

        print(f"Classes        : {self.class_to_idx}")
        for split in ("train", "val", "test"):
            print(f"{split:>5s} windows  : {len(self._windows[split])}")
        print(f"pos_weight     : { {c: f'{w:.2f}' for c, w in zip(classes, self.class_weights)} }")
        print(f"window={self.window_sec}s hop={self.hop_sec}s fixed_len={self.fixed_len}")

    def _make_dataset(self, split):
        return MLMCWindowedDataset(
            self._windows[split], target_sr=self.target_sr,
            fixed_len=self.fixed_len, rms_normalize=self.rms_normalize,
            target_rms=self.target_rms,
        )

    def train_dataloader(self):
        return DataLoader(self._make_dataset("train"), batch_size=self.batch_size,
                          shuffle=self.shuffle_train, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=True,
                          persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self._make_dataset("val"), batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=False,
                          persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self._make_dataset("test"), batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=False,
                          persistent_workers=False)


# ─── shared inference helpers (one honest path for all MLMC scripts) ────────

def aggregate_source_logmean(probs: np.ndarray, groups, eps: float = 1e-8):
    """Per-source aggregation of per-window class probs.

    ``groups`` : iterable of (row_indices, multihot_label) — one per source.
    Returns (src_scores[S, C] in [0,1], src_labels[S, C] float multihot).
    Uses log-mean over clips then re-exponentiates (matches the repo's
    source-level aggregation), so a source score stays a per-class probability.
    """
    groups = list(groups)
    S = len(groups)
    C = probs.shape[-1]
    scores = np.zeros((S, C), dtype=np.float32)
    labels = np.zeros((S, C), dtype=np.float32)
    for gi, (idx, lab) in enumerate(groups):
        scores[gi] = np.exp(np.log(probs[np.asarray(idx)] + eps).mean(0))
        labels[gi] = lab
    return scores, labels


def tune_thresholds_per_class(scores: np.ndarray, y_multihot: np.ndarray,
                              grid: Optional[np.ndarray] = None):
    """Pick a per-class probability threshold maximizing that class's F1 on
    the given (val) set. Returns thresholds[C]. HONEST: call on val only."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    C = scores.shape[1]
    thr = np.full(C, 0.5, dtype=np.float32)
    for c in range(C):
        yc = y_multihot[:, c].astype(bool)
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = scores[:, c] >= t
            tp = int((pred & yc).sum()); fp = int((pred & ~yc).sum())
            fn = int((~pred & yc).sum())
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thr[c] = best_t
    return thr


def probs_to_onehot(scores: np.ndarray, thresholds, force_one: bool = True,
                    single_label: bool = False):
    """Threshold per-class probs → multi-hot 0/1 matrix.

    ``single_label=True`` ignores ``thresholds`` and returns a strict argmax
    one-hot (exactly one bit per row) — the faithful single-label decision rule
    for single-label datasets. ``thresholds`` may then be passed as ``None``.

    ``force_one=True`` (multi-label mode) guarantees each row has at least the
    argmax bit set, so a row is never all-zeros — it degrades gracefully to a
    valid one-hot when no class clears its threshold.
    """
    if single_label:
        onehot = np.zeros_like(scores, dtype=np.int64)
        onehot[np.arange(len(scores)), scores.argmax(1)] = 1
        return onehot
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    onehot = (scores >= thresholds).astype(np.int64)
    if force_one:
        empty = onehot.sum(1) == 0
        if empty.any():
            onehot[empty, scores[empty].argmax(1)] = 1
    return onehot


def multilabel_report(name, y_multihot: np.ndarray, pred_onehot: np.ndarray,
                      classes: Sequence[str]):
    """Print per-class P/R/F1 + macro/micro-F1, subset-accuracy, Hamming."""
    y = y_multihot.astype(bool); p = pred_onehot.astype(bool)
    C = y.shape[1]
    f1s, prs, rcs = [], [], []
    tp_tot = fp_tot = fn_tot = 0
    lines = []
    for c in range(C):
        tp = int((p[:, c] & y[:, c]).sum())
        fp = int((p[:, c] & ~y[:, c]).sum())
        fn = int((~p[:, c] & y[:, c]).sum())
        tp_tot += tp; fp_tot += fp; fn_tot += fn
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        f1s.append(f1); prs.append(pr); rcs.append(rc)
        lines.append(f"   {classes[c]:<10s} P={pr:.3f} R={rc:.3f} F1={f1:.3f}")
    macro_f1 = float(np.mean(f1s))
    micro_pr = tp_tot / (tp_tot + fp_tot) if tp_tot + fp_tot else 0.0
    micro_rc = tp_tot / (tp_tot + fn_tot) if tp_tot + fn_tot else 0.0
    micro_f1 = (2 * micro_pr * micro_rc / (micro_pr + micro_rc)
                if micro_pr + micro_rc else 0.0)
    subset_acc = float((p == y).all(1).mean())
    hamming = float((p != y).mean())
    print(f"[{name}] macroF1={macro_f1:.4f} microF1={micro_f1:.4f} "
          f"subsetAcc={subset_acc:.4f} hamming={hamming:.4f}")
    print("\n".join(lines))
    return dict(macro_f1=macro_f1, micro_f1=micro_f1,
                macro_p=float(np.mean(prs)), macro_r=float(np.mean(rcs)),
                subset_acc=subset_acc, hamming=hamming,
                per_class_f1={classes[c]: f1s[c] for c in range(C)})


__all__ = [
    "MLMCWindowedDataModule", "MLMCWindowedDataset",
    "enumerate_windows_multilabel", "scan_split_files",
    "aggregate_source_logmean", "tune_thresholds_per_class",
    "probs_to_onehot", "multilabel_report",
]
