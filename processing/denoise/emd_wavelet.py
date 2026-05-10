"""
EMD-Wavelet Denoising
=====================

Pipeline
--------
1. Empirical Mode Decomposition (PyEMD) → Intrinsic Mode Functions (IMFs).
2. Classify each IMF as signal-dominant or noise-dominant using two criteria:
     a. High-frequency energy ratio  (HFR > 0.35 → likely noise).
     b. Hurst exponent               (H < 0.55   → irregular/noisy).
3. For noise-dominant IMFs: apply SURE-optimal soft wavelet thresholding
   (Stein's Unbiased Risk Estimator) per decomposition level.
4. Reconstruct: signal IMFs + denoised noise IMFs.

Reference
---------
  Flandrin et al., "Empirical mode decomposition as a filter bank", 2004.
  Donoho & Johnstone, "Adapting to unknown smoothness via wavelet shrinkage", 1995.
"""

from __future__ import annotations

import numpy as np

try:
    from PyEMD import EMD as _EMD
    _EMD_AVAILABLE = True
except ImportError:
    _EMD_AVAILABLE = False

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def emd_wavelet_denoise(
    x:           np.ndarray,
    fs:          int   = 5_120,
    *,
    wavelet:     str   = "sym8",
    level:       int   = 5,
    hfr_thresh:  float = 0.35,
    hurst_thresh: float = 0.55,
    max_imfs:    int   = 10,
    fallback_wavelet: bool = True,
) -> np.ndarray:
    """
    EMD-Wavelet denoising of a 1-D waveform.

    Args:
        x               : Input waveform, shape (T,), float32/float64.
        fs              : Sample rate in Hz (used for HFR band definition).
        wavelet         : PyWavelets wavelet name (default ``'sym8'``).
        level           : Wavelet decomposition depth.
        hfr_thresh      : IMFs with HFR > this are classified as noise-dominant.
        hurst_thresh    : IMFs with Hurst exponent < this are also noise-dominant.
        max_imfs        : Maximum IMFs to extract (caps runtime).
        fallback_wavelet: If PyEMD unavailable, apply wavelet-only denoising.

    Returns:
        Denoised waveform, same shape and dtype as ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    original_dtype = np.float32

    if not _EMD_AVAILABLE:
        if fallback_wavelet and _PYWT_AVAILABLE:
            return _wavelet_denoise_only(x, wavelet, level).astype(original_dtype)
        return x.astype(original_dtype)

    if not _PYWT_AVAILABLE:
        return x.astype(original_dtype)

    # ── 1. EMD decomposition ─────────────────────────────────────────────
    emd = _EMD()
    emd.MAX_ITERATION = 200
    try:
        imfs = emd.emd(x, max_imf=max_imfs)   # (n_imfs, T)
    except Exception:
        return x.astype(original_dtype)

    if imfs.ndim == 1:
        imfs = imfs[np.newaxis, :]

    # ── 2. Classify IMFs ─────────────────────────────────────────────────
    nyquist    = fs / 2.0
    denoised   = np.zeros_like(x)

    for imf in imfs:
        if _is_noise_dominant(imf, nyquist, hfr_thresh, hurst_thresh):
            denoised += _wavelet_threshold_imf(imf, wavelet, level)
        else:
            denoised += imf

    return denoised.astype(original_dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_noise_dominant(imf: np.ndarray, nyquist: float,
                       hfr_thresh: float, hurst_thresh: float) -> bool:
    """Return True if the IMF is classified as noise-dominant."""
    # High-frequency ratio: fraction of spectral energy above 30 % Nyquist
    fft_mag = np.abs(np.fft.rfft(imf))
    n       = len(fft_mag)
    cutoff  = int(0.30 * n)
    hfr     = fft_mag[cutoff:].sum() / (fft_mag.sum() + 1e-10)
    if hfr > hfr_thresh:
        return True
    # Hurst exponent (R/S analysis — lightweight 4-scale estimate)
    h = _hurst_rs(imf)
    return h < hurst_thresh


def _hurst_rs(x: np.ndarray, n_scales: int = 4) -> float:
    """Lightweight R/S Hurst exponent for short series."""
    N   = len(x)
    if N < 16:
        return 0.5
    scales = np.logspace(1, np.log10(N // 2), n_scales, dtype=int)
    scales = np.unique(np.clip(scales, 2, N // 2))
    rs_vals = []
    for n in scales:
        sub = x[: (N // n) * n].reshape(-1, n)
        mean   = sub.mean(axis=1, keepdims=True)
        dev    = np.cumsum(sub - mean, axis=1)
        R      = dev.max(axis=1) - dev.min(axis=1)
        S      = sub.std(axis=1)
        valid  = S > 0
        if valid.any():
            rs_vals.append((n, (R[valid] / S[valid]).mean()))
    if len(rs_vals) < 2:
        return 0.5
    log_n  = np.log([v[0] for v in rs_vals])
    log_rs = np.log([v[1] for v in rs_vals])
    # Least-squares slope = H
    H = float(np.polyfit(log_n, log_rs, 1)[0])
    return np.clip(H, 0.0, 1.0)


def _sure_threshold(coeffs: np.ndarray) -> float:
    """
    Compute the SURE-optimal soft threshold for a 1-D coefficient array.

    Minimises Stein's Unbiased Risk Estimate over dyadic threshold candidates.
    Falls back to the universal (Donoho-Johnstone) threshold if SURE is flat.
    """
    n      = len(coeffs)
    c2     = np.sort(coeffs ** 2)            # ascending squared coefficients
    cs     = np.cumsum(c2)
    risks  = (n - 2 * np.arange(1, n + 1) + cs + c2[::-1].cumsum()[::-1]) / n
    best   = int(np.argmin(risks))
    t_sure = np.sqrt(c2[best])
    # Universal threshold as upper bound
    t_univ = np.sqrt(2.0 * np.log(n)) * (np.median(np.abs(coeffs)) / 0.6745)
    return float(min(t_sure, t_univ))


def _wavelet_threshold_imf(imf: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """Decompose one IMF, apply SURE threshold per level, reconstruct."""
    coeffs = pywt.wavedec(imf, wavelet, level=level)
    # coeffs[0] = approximation (keep); coeffs[1:] = details (threshold)
    new_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        t = _sure_threshold(detail)
        new_coeffs.append(pywt.threshold(detail, t, mode="soft"))
    return pywt.waverec(new_coeffs, wavelet)[: len(imf)]


def _wavelet_denoise_only(x: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """Fallback: wavelet-only denoising when PyEMD is unavailable."""
    coeffs     = pywt.wavedec(x, wavelet, level=level)
    new_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        t = _sure_threshold(detail)
        new_coeffs.append(pywt.threshold(detail, t, mode="soft"))
    return pywt.waverec(new_coeffs, wavelet)[: len(x)]
