"""
Export any LightningModule under ``models/`` to ONNX.

Usage
-----
    python scripts/export_onnx.py \
        --model HydroPrecise \
        --ckpt  lightning_logs/hydro_precise/version_0/checkpoints/precise-011-p0.7084.ckpt \
        --output exports/hydro_precise.onnx

Defaults to a 1-D waveform input of shape ``(B, sample_rate)`` — override with
``--input-shape`` (e.g. ``"1,1,5120"``) for models that take a different layout.

The exported graph fixes ``training=False`` so train-only modules (waveform
augmentation, SpecAugment) are bypassed. Batch dim is dynamic.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import models as _models  # noqa: E402


def _resolve_model_class(name: str) -> type:
    """Look up a model class by name (e.g. 'HydroPrecise') or 'module:Class'."""
    if ":" in name:
        mod_name, cls_name = name.split(":", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)
    if hasattr(_models, name):
        return getattr(_models, name)
    raise AttributeError(
        f"Model class '{name}' not found in models/__init__.py. "
        "Use 'module.path:ClassName' for classes not exported there."
    )


def _parse_shape(spec: str) -> tuple[int, ...]:
    return tuple(int(s) for s in spec.split(",") if s.strip())


def _infer_default_shape(model: nn.Module, batch: int) -> tuple[int, ...]:
    """Default input shape: (B, sample_rate) — the convention for these waveform models."""
    sr = None
    if hasattr(model, "hparams") and model.hparams is not None:
        sr = model.hparams.get("sample_rate", None)
    if sr is None:
        sr = 5_120
    return (batch, int(sr))


# ────────────────────────────────────────────────────────────────────────
#  ONNX-friendly replacements for torch.fft / torch.stft
#
#  The default torch ONNX exporters (both Dynamo and TorchScript paths) lack
#  decompositions for aten::fft_rfft, aten::fft_rfftfreq, and aten::stft.
#  These helpers swap the FFT-based ops in DEMONChannel for fixed-basis
#  conv1d implementations that produce numerically identical output for the
#  trained input length.
# ────────────────────────────────────────────────────────────────────────


class _ConvSTFTMag(nn.Module):
    """STFT magnitude implemented as a fixed conv1d against a DFT basis."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int | None = None,
                 power: float = 1.0, center: bool = True, pad_mode: str = "reflect"):
        super().__init__()
        win_length = win_length or n_fft
        window = torch.hann_window(win_length, periodic=True)
        if win_length < n_fft:
            pad = (n_fft - win_length) // 2
            w = torch.zeros(n_fft)
            w[pad:pad + win_length] = window
            window = w
        n_freqs = n_fft // 2 + 1
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_freqs, dtype=torch.float64).unsqueeze(1)
        angle = 2.0 * math.pi * k * n / n_fft
        cos_basis = (torch.cos(angle) * window.double()).float()
        sin_basis = (-torch.sin(angle) * window.double()).float()
        self.register_buffer("cos_basis", cos_basis.unsqueeze(1))   # (n_freqs, 1, n_fft)
        self.register_buffer("sin_basis", sin_basis.unsqueeze(1))
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.power = power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)                             # (B, 1, T)
        if self.center:
            pad = self.n_fft // 2
            x = F.pad(x, (pad, pad), mode=self.pad_mode)
        real = F.conv1d(x, self.cos_basis, stride=self.hop_length)
        imag = F.conv1d(x, self.sin_basis, stride=self.hop_length)
        mag = torch.sqrt(real * real + imag * imag + 1e-12)
        if self.power != 1.0:
            mag = mag.pow(self.power)
        return mag                                         # (B, n_freqs, T_frames)


def _maybe_force_mel_cqt_fallback(model: nn.Module, ckpt: str | os.PathLike) -> bool:
    """Match the trained-time CQT frontend.

    HydroPrecise's _CQTFrontend uses nnAudio CQT when available, else falls
    back to a torchaudio MelSpectrogram. Older checkpoints were trained with
    the Mel fallback (nnAudio absent), but inference-time loads now pick
    nnAudio CQT — silently breaking the downstream conv stack. Detect the
    Mel signature in the ckpt's state dict and force the Mel branch back in,
    then reload the matching weights.
    """
    try:
        from models.hydro_precise import _CQTFrontend
    except ImportError:
        return False

    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", {})
    mel_keys = [k for k in sd
                if "branch_b.front.cqt.spectrogram." in k
                or "branch_b.front.cqt.mel_scale." in k]
    if not mel_keys:
        return False

    import torchaudio.transforms as TA
    swapped = 0
    for _name, mod in model.named_modules():
        if not isinstance(mod, _CQTFrontend):
            continue
        sr   = int(getattr(mod.cqt, "sr", None)
                   or getattr(mod.cqt, "sample_rate", 5_120))
        # n_bins / hop_length are stored on either the CQT or Mel module.
        n_bins = int(getattr(mod, "n_bins", 84))
        hop = int(getattr(mod.cqt, "hop_length", 64))
        fmin = float(getattr(mod.cqt, "fmin", 20.0))
        mod.cqt = TA.MelSpectrogram(
            sample_rate=sr, n_fft=512, hop_length=hop,
            n_mels=n_bins, f_min=fmin, f_max=sr / 2.0, power=1.0,
        )
        mod._use_cqt = False
        swapped += 1

    if swapped == 0:
        return False

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[export] forced Mel fallback for {swapped} _CQTFrontend module(s) "
          f"to match trained-time frontend "
          f"(reload: missing={len(missing)} unexpected={len(unexpected)})")
    return True


def _patch_demon_channels(model: nn.Module, T_len: int) -> int:
    """Swap FFT-based ops in every DEMONChannel for ONNX-friendly equivalents.

    Returns the count of patched modules.
    """
    try:
        from models.hydro_net import DEMONChannel
    except ImportError:
        return 0

    count = 0
    for _name, mod in model.named_modules():
        if not isinstance(mod, DEMONChannel):
            continue

        # 1) Bandpass: y = irfft(rfft(x) * mask, n=T) is exactly the circular
        #    convolution of x with h = irfft(mask, n=T). Implement that as
        #    conv1d over a circularly padded input.
        sr     = mod.sample_rate
        f_cav  = mod.f_cav
        freqs  = torch.fft.rfftfreq(T_len, d=1.0 / sr)
        mask_c = (freqs >= f_cav).to(torch.complex64)
        h      = torch.fft.irfft(mask_c, n=T_len).float()
        # Conv1d does correlation, so flip h to get true convolution.
        mod.register_buffer("_bp_kernel", h.flip(-1).view(1, 1, -1))
        mod._bp_T = int(T_len)

        # 2) Replace the inner torchaudio Spectrogram with the conv1d version.
        spec = mod.spec
        ta_spec = spec.spec  # torchaudio.transforms.Spectrogram
        n_fft = int(getattr(ta_spec, "n_fft", spec.n_fft))
        hop_length = int(getattr(ta_spec, "hop_length", n_fft // 4))
        win_length = int(getattr(ta_spec, "win_length", n_fft) or n_fft)
        pad_mode = str(getattr(ta_spec, "pad_mode", "reflect"))
        power = float(getattr(ta_spec, "power", 1.0) or 1.0)

        new_stft = _ConvSTFTMag(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            power=power, center=True, pad_mode=pad_mode,
        )
        mod.add_module("_export_stft", new_stft)

        def _new_forward(self, waveform: torch.Tensor) -> torch.Tensor:
            T = self._bp_T
            x = waveform.unsqueeze(1)                       # (B, 1, T)
            pad_left = x[..., -(T - 1):]
            x_padded = torch.cat([pad_left, x], dim=-1)     # (B, 1, 2T-1)
            x_bp = F.conv1d(x_padded, self._bp_kernel)      # (B, 1, T)
            x_env = x_bp.squeeze(1) ** 2
            full_spec = self._export_stft(x_env)            # (B, n_freqs, T_frames)
            dem = full_spec[:, self.spec.lo:self.spec.hi, :].clamp(min=1e-9)
            return self.pcen(dem)

        mod.forward = types.MethodType(_new_forward, mod)
        count += 1

    return count


class _InferenceWrapper(nn.Module):
    """Pin training=False so train-only branches are tracer-stable."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return self.model(input)


def export(
    model_name: str,
    ckpt: str | os.PathLike,
    output: str | os.PathLike,
    input_shape: tuple[int, ...] | None = None,
    opset: int = 18,
    dynamic_batch: bool = True,
    verify: bool = True,
    atol: float = 5e-2,
    patch_fft: bool = True,
    single_file: bool = True,
) -> Path:
    cls = _resolve_model_class(model_name)
    print(f"[export] resolved model class: {cls.__module__}.{cls.__name__}")

    print(f"[export] loading checkpoint: {ckpt}")
    model = cls.load_from_checkpoint(str(ckpt), map_location="cpu", strict=False)
    _maybe_force_mel_cqt_fallback(model, ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if input_shape is None:
        input_shape = _infer_default_shape(model, batch=1)
    print(f"[export] input shape: {input_shape}")

    if patch_fft:
        T_len = input_shape[-1]
        n_patched = _patch_demon_channels(model, T_len=T_len)
        if n_patched:
            print(f"[export] patched {n_patched} DEMONChannel module(s) "
                  f"to ONNX-friendly conv1d (T_len={T_len})")

    # When dynamic batch is requested, trace with batch >= 2 — torch.export
    # collapses size-1 dims to static even when a Dim is provided.
    trace_shape = input_shape
    if dynamic_batch and input_shape[0] == 1:
        trace_shape = (2,) + tuple(input_shape[1:])
    dummy = torch.randn(*trace_shape, dtype=torch.float32)

    wrapper = _InferenceWrapper(model)

    with torch.no_grad():
        ref_out = wrapper(dummy)
    if isinstance(ref_out, (tuple, list)):
        out_names = [f"output_{i}" for i in range(len(ref_out))]
    else:
        out_names = ["output"]
    print(f"[export] reference output: "
          f"{tuple(ref_out.shape) if isinstance(ref_out, torch.Tensor) else type(ref_out)}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dyn = None
    if dynamic_batch:
        dyn = {"input": {0: "batch"}}
        for n in out_names:
            dyn[n] = {0: "batch"}

    print(f"[export] writing ONNX → {output} (opset={opset})")
    # Try Dynamo exporter first (handles dynamic shapes + adaptive ops better),
    # fall back to the legacy TorchScript exporter on failure.
    common_kwargs = dict(
        input_names=["input"],
        output_names=out_names,
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
    )
    try:
        if dyn is not None:
            # Dynamo export uses torch.export semantics — dynamic_shapes is a
            # sequence matching positional args, with {dim_idx: Dim} per arg.
            batch_dim = torch.export.Dim("batch", min=1, max=4096)
            dynamic_shapes = ({0: batch_dim},)
        else:
            dynamic_shapes = None
        torch.onnx.export(
            wrapper, (dummy,), str(output), dynamo=True,
            dynamic_shapes=dynamic_shapes, **common_kwargs,
        )
    except (TypeError, Exception) as e:  # noqa: BLE001
        print(f"[export] dynamo export failed ({type(e).__name__}); "
              f"falling back to legacy TorchScript exporter")
        legacy_kwargs = dict(common_kwargs, dynamic_axes=dyn)
        try:
            torch.onnx.export(wrapper, (dummy,), str(output), dynamo=False, **legacy_kwargs)
        except TypeError:
            torch.onnx.export(wrapper, (dummy,), str(output), **legacy_kwargs)

    try:
        import onnx
        m = onnx.load(str(output))
        onnx.checker.check_model(m)
        print(f"[export] onnx.checker passed  ({len(m.graph.node)} nodes, "
              f"ir_version={m.ir_version})")
        if single_file:
            # The Dynamo exporter always splits weights into a sidecar
            # (<output>.data). Re-save with weights inlined for a single
            # self-contained file, then drop the sidecar.
            sidecar = output.with_suffix(output.suffix + ".data")
            onnx.save(m, str(output), save_as_external_data=False)
            if sidecar.exists():
                sidecar.unlink()
            print(f"[export] inlined weights into {output} "
                  f"({output.stat().st_size / 1e6:.2f} MB)")
    except ImportError:
        print("[export] (onnx not installed — skipping graph check)")

    if verify:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(
                str(output), providers=["CPUExecutionProvider"]
            )
            ort_out = sess.run(None, {"input": dummy.numpy()})[0]
            torch_out = (ref_out[0] if isinstance(ref_out, (tuple, list)) else ref_out).numpy()
            diff = float(np.max(np.abs(ort_out - torch_out)))
            ok = diff <= atol
            tag = "OK" if ok else "MISMATCH"
            print(f"[verify] max|onnx - torch| = {diff:.3e}  ({tag}, atol={atol})")
            if not ok:
                print("[verify] outputs differ beyond atol — inspect for nondeterministic ops.")
        except ImportError:
            print("[verify] (onnxruntime not installed — skipping numerical check)")

    print(f"[export] done: {output.resolve()}")
    return output


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Export a LightningModule to ONNX.")
    ap.add_argument("--model", required=True,
                    help="Class name from models/__init__.py (e.g. 'HydroPrecise') "
                         "or 'package.module:ClassName'.")
    ap.add_argument("--ckpt", required=True, help="Path to a Lightning .ckpt file.")
    ap.add_argument("--output", required=True, help="Output .onnx path.")
    ap.add_argument("--input-shape", default=None,
                    help="Comma-separated dims for the dummy input "
                         "(default: '1,<sample_rate>' inferred from hparams).")
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset (default 18).")
    ap.add_argument("--no-dynamic-batch", action="store_true",
                    help="Fix batch dim to the dummy size instead of marking it dynamic.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip onnxruntime numerical comparison.")
    ap.add_argument("--atol", type=float, default=5e-2,
                    help="Absolute tolerance for the verify check (default 5e-2).")
    ap.add_argument("--no-patch-fft", action="store_true",
                    help="Skip the DEMONChannel FFT→conv1d patch. Models that "
                         "use torch.fft / torch.stft will fail to export.")
    ap.add_argument("--external-data", action="store_true",
                    help="Keep weights in a sidecar <output>.data file "
                         "(default: inline weights into a single .onnx file). "
                         "Use for models that exceed protobuf's 2 GB limit.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    shape = _parse_shape(args.input_shape) if args.input_shape else None
    export(
        model_name=args.model,
        ckpt=args.ckpt,
        output=args.output,
        input_shape=shape,
        opset=args.opset,
        dynamic_batch=not args.no_dynamic_batch,
        verify=not args.no_verify,
        atol=args.atol,
        patch_fft=not args.no_patch_fft,
        single_file=not args.external_data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
