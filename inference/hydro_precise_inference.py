"""
HydroPrecise — standalone inference + saliency for app integration.

Loads a grid_precise run (checkpoint + temperature.pt + thresholds.json) and
exposes a single class, ``HydroPreciseInferencer``, with two entry points:

    predict(waveform) -> PredictionResult
        Calibrated probabilities + per-class gated decision (may abstain).

    saliency(waveform, target=None) -> np.ndarray (T,)
        Gradient × input saliency aligned to the input samples, optionally
        smoothed.  Positive values mean "increased logit for ``target``".

A run directory is expected to look like::

    <run_dir>/
        precise-EEE-pP.PPPP.ckpt      # Lightning checkpoint (any name)
        temperature.pt                # torch.save({"temperature": float})
        thresholds.json               # {"thresholds": [...], ...}

Dependencies match the training env (torch, torchaudio, pytorch-lightning,
optionally nnAudio for the CQT branch — falls back to mel if absent).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.hydro_precise import HydroPrecise


# ═══════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    probs: np.ndarray             # (C,) calibrated softmax over classes
    pred_index: int               # argmax class index
    pred_label: Optional[str]     # label if class_names provided
    confidence: float             # probs[pred_index]
    abstained: bool               # True when gating threshold not met
    threshold: float              # threshold applied to pred_index
    temperature: float            # scalar T used for calibration


# ═══════════════════════════════════════════════════════════════════════
#  Run-directory resolution
# ═══════════════════════════════════════════════════════════════════════

def _resolve_run(run_dir: Union[str, Path]) -> tuple[Path, Path, Path]:
    """Locate ckpt / temperature.pt / thresholds.json inside a run directory.

    ``run_dir`` can point at either the top-level run (e.g. the
    ``grid_precise_m0.30_g2.0_s0.00_gw0.0`` folder containing ``version_0/``)
    or directly at the ``checkpoints/`` folder that holds the artifacts.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    candidates = [run_dir]
    candidates += sorted(run_dir.glob("version_*/checkpoints"))
    candidates += sorted(run_dir.glob("checkpoints"))
    candidates += sorted(run_dir.glob("**/checkpoints"))

    for d in candidates:
        ckpts = sorted(d.glob("*.ckpt"))
        if not ckpts:
            continue
        # Prefer highest-precision filename (precise-EEE-pP.PPPP.ckpt) → highest P
        def _score(p: Path) -> float:
            stem = p.stem
            if "-p" in stem:
                try:
                    return float(stem.rsplit("-p", 1)[1])
                except ValueError:
                    return -1.0
            return -1.0
        ckpt = max(ckpts, key=_score)
        temp = d / "temperature.pt"
        thr  = d / "thresholds.json"
        if temp.exists() and thr.exists():
            return ckpt, temp, thr

    raise FileNotFoundError(
        f"Could not find ckpt + temperature.pt + thresholds.json under {run_dir}. "
        "Point --run_dir at the run folder (containing version_*/checkpoints) "
        "or directly at the checkpoints directory."
    )


# ═══════════════════════════════════════════════════════════════════════
#  Inferencer
# ═══════════════════════════════════════════════════════════════════════

class HydroPreciseInferencer:
    """Calibrated, gated inference + gradient saliency for HydroPrecise.

    Parameters
    ----------
    run_dir
        Directory containing the trained checkpoint, ``temperature.pt`` and
        ``thresholds.json``.  See ``_resolve_run`` for accepted layouts.
    class_names
        Optional list of class label strings, length = ``num_classes``.
    device
        ``"cuda"``, ``"cpu"``, or an explicit ``torch.device``.  Defaults to
        CUDA when available.
    target_sr
        Expected input sample rate.  Inputs at other rates must be resampled
        externally.  Read from the checkpoint's hparams when present.
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        class_names: Optional[Sequence[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        target_sr: Optional[int] = None,
    ):
        ckpt_path, temp_path, thr_path = _resolve_run(run_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = HydroPrecise.load_from_checkpoint(
            str(ckpt_path), map_location=self.device, strict=False,
        ).to(self.device).eval()

        self.num_classes: int = int(self.model.num_classes)
        self.target_sr: int = int(
            target_sr if target_sr is not None
            else self.model.hparams.get("sample_rate", 5_120)
        )
        self.fixed_len: int = self.target_sr  # HydroPrecise trains on 1 s clips

        t_blob = torch.load(str(temp_path), map_location="cpu", weights_only=False)
        self.temperature: float = float(t_blob["temperature"])

        with open(thr_path) as f:
            t_cfg = json.load(f)
        self.thresholds: np.ndarray = np.asarray(t_cfg["thresholds"], dtype=np.float32)
        if self.thresholds.shape[0] != self.num_classes:
            raise ValueError(
                f"thresholds.json has {self.thresholds.shape[0]} entries but model "
                f"has {self.num_classes} classes."
            )

        if class_names is not None:
            if len(class_names) != self.num_classes:
                raise ValueError(
                    f"class_names length {len(class_names)} != num_classes {self.num_classes}"
                )
            self.class_names: Optional[list[str]] = list(class_names)
        else:
            self.class_names = None

        self.run_dir = str(run_dir)
        self.ckpt_path = str(ckpt_path)

    # ── Input helpers ───────────────────────────────────────────────────

    def _to_tensor(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(waveform, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(waveform)).float()
        elif isinstance(waveform, torch.Tensor):
            x = waveform.detach().float()
        else:
            raise TypeError(f"Unsupported waveform type: {type(waveform)}")

        if x.ndim == 1:
            x = x.unsqueeze(0)            # (1, T)
        elif x.ndim == 2:
            pass                          # (B, T)
        else:
            raise ValueError(f"waveform must be 1-D or 2-D, got shape {tuple(x.shape)}")

        L = x.shape[-1]
        if L < self.fixed_len:
            x = F.pad(x, (0, self.fixed_len - L))
        elif L > self.fixed_len:
            x = x[..., : self.fixed_len]

        return x.to(self.device)

    # ── Inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_proba(
        self, waveform: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Return temperature-calibrated per-class probabilities.

        Accepts a single clip ``(T,)`` or a batch ``(B, T)``.  Returns shape
        ``(B, num_classes)`` — always 2-D for consistent downstream handling.
        """
        x = self._to_tensor(waveform)
        logits = self.model(x)[:, : self.num_classes]
        probs = F.softmax(logits / self.temperature, dim=-1)
        return probs.cpu().numpy()

    def predict(
        self, waveform: Union[np.ndarray, torch.Tensor]
    ) -> PredictionResult:
        """Single-clip gated prediction.  Abstains if confidence < threshold."""
        probs = self.predict_proba(waveform)
        if probs.shape[0] != 1:
            raise ValueError(
                "predict() takes a single clip; pass predict_proba() for batches."
            )
        p = probs[0]
        idx = int(p.argmax())
        conf = float(p[idx])
        thr = float(self.thresholds[idx])
        abstained = conf < thr
        label = self.class_names[idx] if self.class_names is not None else None
        return PredictionResult(
            probs=p,
            pred_index=idx,
            pred_label=label,
            confidence=conf,
            abstained=abstained,
            threshold=thr,
            temperature=self.temperature,
        )

    # ── Saliency ────────────────────────────────────────────────────────

    def saliency(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        target: Optional[int] = None,
        smooth_window: int = 0,
        normalize: bool = True,
    ) -> np.ndarray:
        """Gradient × input saliency on the waveform.

        Parameters
        ----------
        waveform
            Single 1-D clip or 2-D batch ``(B, T)``.
        target
            Class index to attribute to.  Defaults to the argmax class.
        smooth_window
            If > 1, moving-average smoothing window (in samples) applied to
            ``|saliency|`` for display.  0 disables smoothing.
        normalize
            If True, scale per-example so ``max(|s|) == 1``.

        Returns
        -------
        np.ndarray
            Shape ``(T,)`` for a 1-D input, ``(B, T)`` for a batch.  Values
            are aligned to the padded/truncated input (length ``fixed_len``).
            Sign is preserved — positive samples pushed the target logit up.
        """
        x = self._to_tensor(waveform).clone().detach().requires_grad_(True)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        logits = self.model(x)[:, : self.num_classes]

        if target is None:
            tgt_idx = int(logits.argmax(dim=-1)[0].item())
        else:
            tgt_idx = int(target)
        if not 0 <= tgt_idx < self.num_classes:
            raise ValueError(f"target {tgt_idx} out of range [0, {self.num_classes})")

        score = logits[:, tgt_idx].sum()
        grads = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]

        sal = (grads * x).detach()                 # (B, T)

        if smooth_window and smooth_window > 1:
            k = int(smooth_window)
            kernel = torch.ones(1, 1, k, device=sal.device) / k
            mag = sal.abs().unsqueeze(1)           # (B, 1, T)
            mag = F.pad(mag, (k // 2, k - 1 - k // 2), mode="replicate")
            smooth = F.conv1d(mag, kernel).squeeze(1)
            sal = sal.sign() * smooth

        if normalize:
            denom = sal.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            sal = sal / denom

        # Restore params' grad flag (defensive — ``eval()`` doesn't freeze them
        # by default, and a caller may want to fine-tune later).
        for p in self.model.parameters():
            p.requires_grad_(True)

        sal_np = sal.cpu().numpy()
        return sal_np[0] if sal_np.shape[0] == 1 else sal_np


# ═══════════════════════════════════════════════════════════════════════
#  CLI smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="HydroPrecise inference smoke test")
    ap.add_argument("--run_dir", required=True,
                    help="Run directory or checkpoints dir with ckpt + temperature.pt + thresholds.json")
    ap.add_argument("--wav", default=None,
                    help="Optional .wav file to run inference on (resampled externally).")
    ap.add_argument("--class_names", nargs="*", default=None)
    args = ap.parse_args()

    inf = HydroPreciseInferencer(args.run_dir, class_names=args.class_names)
    print(f"Loaded  ckpt: {inf.ckpt_path}")
    print(f"        T   = {inf.temperature:.3f}")
    print(f"   thresholds: {inf.thresholds.tolist()}")
    print(f"   num_classes: {inf.num_classes}  sr: {inf.target_sr}")

    if args.wav:
        import torchaudio
        wav, sr = torchaudio.load(args.wav)
        if sr != inf.target_sr:
            wav = torchaudio.functional.resample(wav, sr, inf.target_sr)
        wav = wav.mean(dim=0).numpy()
        res = inf.predict(wav)
        print(f"\nprediction: {res}")
        sal = inf.saliency(wav, target=res.pred_index, smooth_window=64)
        print(f"saliency shape: {sal.shape}  min/max: {sal.min():+.3f} / {sal.max():+.3f}")
    else:
        rng = np.random.default_rng(0)
        dummy = rng.standard_normal(inf.fixed_len).astype(np.float32) * 1e-2
        res = inf.predict(dummy)
        print(f"\ndummy prediction: {res}")
