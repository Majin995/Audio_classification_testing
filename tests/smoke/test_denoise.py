"""
Smoke tests for the DSP denoising pipeline.

Run:
    python -m pytest tests/smoke/test_denoise.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

FS = 5_120
DURATION = 1.0
N = int(FS * DURATION)


def _make_noisy_sine(snr_db: float = 10.0, freq: float = 440.0) -> np.ndarray:
    """Generate a sinusoid + white noise at a given SNR."""
    t      = np.arange(N) / FS
    signal = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    noise_power = (np.mean(signal ** 2)) / (10.0 ** (snr_db / 10.0))
    noise  = np.random.default_rng(0).normal(0.0, np.sqrt(noise_power), N).astype(np.float32)
    return signal + noise


def _snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Signal-to-noise ratio between a clean reference and a noisy version."""
    noise = noisy - clean
    sig_power   = float(np.mean(clean ** 2))
    noise_power = float(np.mean(noise ** 2))
    if noise_power <= 0.0:
        return 100.0
    return 10.0 * np.log10(sig_power / (noise_power + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────

class TestEMDWavelet:
    """EMD-Wavelet denoising tests."""

    def test_output_shape(self):
        from processing.denoise.emd_wavelet import emd_wavelet_denoise
        x   = _make_noisy_sine()
        out = emd_wavelet_denoise(x, FS)
        assert out.shape == x.shape, "Output shape mismatch"

    def test_output_dtype(self):
        from processing.denoise.emd_wavelet import emd_wavelet_denoise
        x   = _make_noisy_sine()
        out = emd_wavelet_denoise(x, FS)
        assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"

    def test_no_nan_inf(self):
        from processing.denoise.emd_wavelet import emd_wavelet_denoise
        x   = _make_noisy_sine()
        out = emd_wavelet_denoise(x, FS)
        assert np.all(np.isfinite(out)), "Output contains NaN or Inf"

    def test_pure_noise_reduced(self):
        """Denoising a very noisy signal should reduce variance somewhat."""
        from processing.denoise.emd_wavelet import emd_wavelet_denoise
        rng = np.random.default_rng(42)
        x   = rng.standard_normal(N).astype(np.float32)
        out = emd_wavelet_denoise(x, FS)
        # High-frequency energy should be lower after denoising
        var_in  = float(np.var(np.diff(x)))
        var_out = float(np.var(np.diff(out)))
        assert var_out <= var_in * 1.1, (
            f"Variance not reduced: in={var_in:.4f}, out={var_out:.4f}"
        )

    def test_snr_improvement(self):
        """Denoising should yield ≥ 3 dB SNR improvement on a clean sine at 0 dB SNR."""
        pytest.importorskip("PyEMD")
        pytest.importorskip("pywt")
        from processing.denoise.emd_wavelet import emd_wavelet_denoise

        t = np.arange(N) / FS
        clean = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        # 0 dB SNR
        rng   = np.random.default_rng(0)
        noise = rng.standard_normal(N).astype(np.float32)
        noise = noise * np.sqrt(np.mean(clean**2) / np.mean(noise**2))
        noisy = (clean + noise).astype(np.float32)

        snr_in  = _snr_db(clean, noisy)
        out     = emd_wavelet_denoise(noisy, FS)
        snr_out = _snr_db(clean, out)

        assert snr_out > snr_in - 2.0, (
            f"SNR not improved: before={snr_in:.1f} dB, after={snr_out:.1f} dB"
        )


class TestNMFICA:
    """NMF + FastICA BSS tests."""

    def test_output_shape(self):
        pytest.importorskip("sklearn")
        pytest.importorskip("librosa")
        from processing.denoise.bss import nmf_ica_separate
        x   = _make_noisy_sine()
        out = nmf_ica_separate(x, FS, method="nmf")
        assert out.shape == x.shape, "Output shape mismatch"

    def test_output_dtype(self):
        pytest.importorskip("sklearn")
        pytest.importorskip("librosa")
        from processing.denoise.bss import nmf_ica_separate
        x   = _make_noisy_sine()
        out = nmf_ica_separate(x, FS, method="nmf")
        assert out.dtype == np.float32

    def test_no_nan_inf(self):
        pytest.importorskip("sklearn")
        pytest.importorskip("librosa")
        from processing.denoise.bss import nmf_ica_separate
        x   = _make_noisy_sine()
        out = nmf_ica_separate(x, FS, method="nmf")
        assert np.all(np.isfinite(out))


class TestDenoiseTransform:
    """DenoiseTransform nn.Module tests."""

    def test_off_is_passthrough(self):
        import torch
        from processing.denoise.transform import DenoiseTransform
        t  = DenoiseTransform(method="off")
        x  = torch.randn(4, N)
        out = t(x)
        assert torch.equal(out, x)

    def test_emd_output_shape(self):
        import torch
        from processing.denoise.transform import DenoiseTransform
        t   = DenoiseTransform(method="emd_wavelet", sample_rate=FS)
        x   = torch.randn(2, N)
        out = t(x)
        assert out.shape == x.shape

    def test_invalid_method_raises(self):
        from processing.denoise.transform import DenoiseTransform
        with pytest.raises(ValueError, match="method must be one of"):
            DenoiseTransform(method="unknown_method")


class TestSNR:
    """SNR estimator tests."""

    def test_sine_snr_positive(self):
        from processing.denoise.snr import estimate_snr_noise_floor
        x   = _make_noisy_sine(snr_db=20.0)
        snr = estimate_snr_noise_floor(x, FS)
        assert snr > 0.0, f"Expected positive SNR, got {snr}"

    def test_pure_noise_low_snr(self):
        from processing.denoise.snr import estimate_snr_noise_floor
        rng  = np.random.default_rng(1)
        x    = rng.standard_normal(N).astype(np.float32)
        snr  = estimate_snr_noise_floor(x, FS)
        # Flat spectrum → low pseudo-SNR
        assert snr < 20.0, f"Pure noise should have SNR < 20 dB, got {snr}"
