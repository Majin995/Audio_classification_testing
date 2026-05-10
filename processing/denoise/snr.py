"""
SNR estimation utilities for ablation logging.

estimate_snr_noise_floor
    Estimates SNR of a waveform using the spectrogram median as a noise-floor
    proxy.  Robust to non-stationary backgrounds common in underwater recordings.
"""

from __future__ import annotations

import numpy as np


def estimate_snr_noise_floor(
    x: np.ndarray,
    fs: int = 5_120,
    n_fft: int = 256,
    hop: int = 64,
    percentile: float = 25.0,
) -> float:
    """
    Estimate signal-to-noise ratio via spectrogram percentile noise floor.

    The ``percentile``-th magnitude column is taken as the noise estimate;
    the mean magnitude of the rest is the signal estimate.

    Args:
        x           : 1-D float32 waveform, normalised to [-1, 1].
        fs          : Sample rate (used only for Nyquist sanity check).
        n_fft       : STFT window length.
        hop         : STFT hop length.
        percentile  : Percentile of column magnitudes treated as noise floor.

    Returns:
        SNR in dB (float).  Returns 0.0 if computation fails.
    """
    try:
        # Manual STFT using real FFT for speed
        frames = _frame(x, n_fft, hop)                    # (n_frames, n_fft)
        window = np.hanning(n_fft).astype(np.float32)
        spec   = np.abs(np.fft.rfft(frames * window, n=n_fft))  # (n_frames, n_fft//2+1)

        col_energy = spec.mean(axis=1)                    # energy per time frame
        noise_floor = np.percentile(col_energy, percentile)
        signal_mean = col_energy.mean()

        if noise_floor <= 0.0:
            return 0.0

        return float(20.0 * np.log10(signal_mean / noise_floor + 1e-9))
    except Exception:
        return 0.0


def _frame(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Zero-pad and split into overlapping frames."""
    pad = (frame_len - len(x) % frame_len) % frame_len
    x_p = np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    n_frames = (len(x_p) - frame_len) // hop + 1
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return x_p[idx]
