"""
DenoiseTransform — nn.Module wrapper for CPU denoising.

Operates on CPU (even if the surrounding model is on GPU).  Designed to wrap
the DALI iterator output or to be dropped into a model's forward() before the
feature frontend.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from processing.denoise.emd_wavelet import emd_wavelet_denoise
from processing.denoise.bss import nmf_ica_separate


_METHODS = {"off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"}


class DenoiseTransform(nn.Module):
    """
    Batch-level waveform denoising module.

    Moves tensors to CPU, applies NumPy-based denoising per sample,
    then returns a tensor on the original device.

    Args:
        method     : One of ``'off'``, ``'emd_wavelet'``, ``'nmf'``,
                     ``'ica'``, ``'nmf_ica'``, ``'emd_nmf'``.
                     ``'emd_nmf'`` chains EMD-wavelet → NMF-ICA.
        sample_rate: Audio sample rate (passed to denoising functions).
        **kwargs   : Forwarded to the underlying denoising function.
    """

    def __init__(
        self,
        method:      Literal["off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"] = "off",
        sample_rate: int = 5_120,
        **kwargs,
    ):
        super().__init__()
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got '{method}'")
        self.method      = method
        self.sample_rate = sample_rate
        self.kwargs      = kwargs

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) float tensor on any device.
        Returns:
            Denoised waveform, same device and dtype.
        """
        if self.method == "off":
            return waveform

        device = waveform.device
        x_np   = waveform.detach().cpu().numpy()          # (B, T) float32

        out = np.stack(
            [self._denoise_one(x_np[i]) for i in range(x_np.shape[0])],
            axis=0,
        )
        return torch.from_numpy(out).to(device=device, dtype=waveform.dtype)

    def _denoise_one(self, x: np.ndarray) -> np.ndarray:
        fs = self.sample_rate
        kw = self.kwargs

        if self.method == "emd_wavelet":
            return emd_wavelet_denoise(x, fs, **kw)

        elif self.method in ("nmf", "ica", "nmf_ica"):
            return nmf_ica_separate(x, fs, method=self.method, **kw)

        elif self.method == "emd_nmf":
            x_emd = emd_wavelet_denoise(x, fs)
            return nmf_ica_separate(x_emd, fs, method="nmf_ica")

        return x   # fallback — should never reach here

    def extra_repr(self) -> str:
        return f"method='{self.method}', sample_rate={self.sample_rate}"
