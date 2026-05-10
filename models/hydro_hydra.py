"""
HydroHydra — Unified Spectrogram-Free UATR Classifier
=====================================================

Single model, dual 1D stream, no STFT / Mel / CQT / DEMON / spectrogram image
ops anywhere in the forward graph.

Stream A — Learnable Gabor filterbank + 1D SE-Res2 backbone
Stream B — Kymatio wavelet scattering (fixed) + 1D conv projection
Fuse    — align to common T_f → concat → 1D projection → SaShiMi (S4D) × 2
         → AttentiveStatisticsPool → LMF head + Deep-Gamblers abstention logit

Either stream can be ablated via `use_gabor` / `use_scattering` flags; at least
one must be on.  Designed for macro-precision on the DeepShip-like 4-class
(Cargo/Passenger/Tanker/Tug, 5120 Hz, 1 s).

Kymatio is an optional dependency; if missing, `use_scattering=True` raises a
clear ImportError — Gabor-only configurations remain usable without kymatio.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassRecall, MulticlassAUROC, MulticlassConfusionMatrix,
    MulticlassMatthewsCorrCoef,
)

from models.hydro_catfish   import LearnableGaborFilterbank, SpecAugment1D
from models.hydro_conformer import FocalLoss
from models.hydro_fusion    import _SERes2Block, _AttentiveStatisticsPool
from models.hydro_s4        import SaShiMiBlock
from models.heads           import build_head
from models.hydro_precise_v2 import (
    _WaveformAug as _WaveformAugV2,
    _BranchDropout,
)
from processing.losses      import LargeMarginFocalLoss, LDAMLoss


# ═══════════════════════════════════════════════════════════════════════
#  Stream C — SincNet (parametric narrowband sinc bandpass)
# ═══════════════════════════════════════════════════════════════════════

class _SincConv1d(nn.Module):
    """Ravanelli & Bengio sinc-band conv. Each filter parameterised by
    (low_hz, band_hz). Kernel is recomputed each forward — cheap at
    n_filters≤128, kernel≤257 — so updates flow through the parameters.
    """

    def __init__(self, n_filters: int, kernel_size: int, sample_rate: int,
                 min_low_hz: float = 20.0, min_band_hz: float = 20.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1                       # force odd for symmetric kernel
        self.n_filters    = int(n_filters)
        self.kernel_size  = int(kernel_size)
        self.sample_rate  = int(sample_rate)
        self.min_low_hz   = float(min_low_hz)
        self.min_band_hz  = float(min_band_hz)

        # Mel-spaced initialisation between min_low_hz and Nyquist.
        nyquist = self.sample_rate / 2.0
        edges = self._mel_band_edges(
            n_filters, self.min_low_hz, nyquist - self.min_band_hz
        )                                                       # length n_filters + 1
        low_init  = edges[:-1].clone()
        band_init = torch.tensor([
            max(self.min_band_hz, float(edges[i + 1] - edges[i]))
            for i in range(n_filters)
        ])
        self.low_hz_  = nn.Parameter(low_init.float())
        self.band_hz_ = nn.Parameter(band_init.float())

        # Pre-computed time / window buffers.
        n = (kernel_size - 1) // 2
        t = torch.arange(-n, n + 1, dtype=torch.float32) / self.sample_rate
        self.register_buffer("t", t)                              # (K,)
        # Hamming window
        idx = torch.arange(kernel_size, dtype=torch.float32)
        self.register_buffer(
            "window",
            0.54 - 0.46 * torch.cos(2 * math.pi * idx / (kernel_size - 1)),
        )

    @staticmethod
    def _mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @classmethod
    def _mel_band_edges(cls, n: int, lo: float, hi: float) -> torch.Tensor:
        """Return ``n + 1`` mel-spaced band edges in Hz spanning ``[lo, hi]``."""
        m_lo, m_hi = cls._mel(lo), cls._mel(hi)
        edges = [cls._hz(m_lo + (m_hi - m_lo) * i / n) for i in range(n + 1)]
        return torch.tensor(edges, dtype=torch.float32)

    def _build_filters(self) -> torch.Tensor:
        low  = self.min_low_hz + torch.abs(self.low_hz_)                          # (F,)
        band = self.min_band_hz + torch.abs(self.band_hz_)
        high = torch.clamp(low + band, max=self.sample_rate / 2.0)

        # 2*hi*sinc(2π hi t) - 2*lo*sinc(2π lo t) — DC-handled by sinc(0)=1.
        # Use torch.sinc which is sinc(πx)/πx so we feed 2*hi*t.
        t = self.t.to(self.low_hz_.dtype)
        kernel = (
            2.0 * high.unsqueeze(1) * torch.sinc(2.0 * high.unsqueeze(1) * t)
            - 2.0 * low.unsqueeze(1) * torch.sinc(2.0 * low.unsqueeze(1) * t)
        )                                                                          # (F, K)
        kernel = kernel * self.window
        # Per-filter L2 normalise so kernel magnitude doesn't explode.
        kernel = kernel / kernel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return kernel.unsqueeze(1)                                                 # (F, 1, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) → unsqueeze to (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        kernel = self._build_filters().to(x.dtype)
        pad = (self.kernel_size - 1) // 2
        return F.conv1d(x, kernel, padding=pad)                                    # (B, F, T)


class _SincNetStream(nn.Module):
    def __init__(self, sample_rate: int = 5_120, n_filters: int = 64,
                 kernel_size: int = 251, out_ch: int = 128, dropout: float = 0.1):
        super().__init__()
        self.filterbank = _SincConv1d(
            n_filters=n_filters, kernel_size=kernel_size, sample_rate=sample_rate,
        )
        # Apply log-magnitude compression (matches Gabor stream's implicit log envelope).
        self.stem = nn.Sequential(
            nn.Conv1d(n_filters, out_ch, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.blocks = nn.Sequential(
            _SERes2Block(out_ch, scale=8, kernel_size=3, dilation=2, dropout=dropout),
            _SERes2Block(out_ch, scale=8, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.out_ch = out_ch

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.filterbank(waveform)             # (B, n_filters, T)
        x = torch.log1p(x.abs())                  # log envelope, matches Gabor's compression
        x = self.stem(x)
        return self.blocks(x)


# ═══════════════════════════════════════════════════════════════════════
#  Stream D — Time-domain SubBand Envelope statistics (no STFT)
# ═══════════════════════════════════════════════════════════════════════

class _TDSubBandEnvelopeStream(nn.Module):
    """Fixed FIR bandpass × Hilbert envelope × per-frame statistics.

    Pipeline (waveform-only, no STFT):
      1. Apply N fixed FIR bandpass filters via grouped 1-D convolution.
      2. For each subband, compute the Hilbert envelope using the rfft
         analytic-signal trick — purely a 1-D operation.
      3. Frame the envelope (frame, hop) and compute 5 statistics:
         RMS, log-variance, kurtosis, ZCR (sign-change density on the
         band-passed signal — *not* the envelope), crest factor.
      4. Stack to (B, N×5, T') and project through Conv → BN → GELU →
         a single SE-Res2 block.

    Kurtosis is numerically delicate under bf16; the framewise stats are
    computed in fp32 and cast back to the model dtype.
    """

    def __init__(
        self,
        sample_rate: int = 5_120,
        fir_taps:    int = 129,
        bands: tuple = ((20.0, 250.0), (250.0, 1000.0),
                        (1000.0, 2000.0), (2000.0, 2540.0)),
        frame: int = 64,
        hop:   int = 32,
        out_ch: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if fir_taps % 2 == 0:
            fir_taps += 1
        self.fir_taps    = int(fir_taps)
        self.frame       = int(frame)
        self.hop         = int(hop)
        self.sample_rate = int(sample_rate)
        self.n_bands     = len(bands)
        self.n_stats     = 5

        # Build FIR bandpass kernels via firwin. Use scipy if available;
        # fall back to a tapered ideal-bandpass if scipy is absent.
        kernels = []
        for lo, hi in bands:
            kernels.append(self._design_bandpass(fir_taps, lo, hi, sample_rate))
        kernel = torch.stack(kernels, dim=0).unsqueeze(1)        # (N, 1, K)
        self.register_buffer("fir_kernel", kernel)

        in_ch = self.n_bands * self.n_stats
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.block = _SERes2Block(out_ch, scale=8, kernel_size=3,
                                  dilation=2, dropout=dropout)
        self.out_ch = int(out_ch)

    @staticmethod
    def _design_bandpass(taps: int, lo: float, hi: float, sr: int) -> torch.Tensor:
        nyq = sr / 2.0
        try:
            from scipy.signal import firwin
            h = firwin(
                taps,
                [max(1.0, lo) / nyq, min(nyq - 1.0, hi) / nyq],
                pass_zero=False, window="hamming",
            )
            return torch.tensor(h, dtype=torch.float32)
        except Exception:
            # Fallback: ideal bandpass × hamming window (truncated sinc).
            n = torch.arange(taps, dtype=torch.float32) - (taps - 1) / 2.0
            t = n / sr
            ideal = (
                2.0 * hi * torch.sinc(2.0 * hi * t)
                - 2.0 * lo * torch.sinc(2.0 * lo * t)
            )
            window = 0.54 - 0.46 * torch.cos(
                2 * math.pi * torch.arange(taps, dtype=torch.float32) / (taps - 1)
            )
            h = ideal * window
            return h / h.norm().clamp(min=1e-6)

    def _bandpass(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) → (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        kernel = self.fir_kernel.to(x.dtype)
        pad = (self.fir_taps - 1) // 2
        # Replicate the single input channel across N filter groups.
        x_rep = x.expand(-1, self.n_bands, -1).contiguous()       # (B, N, T)
        # Apply each FIR independently via grouped conv (groups = N).
        return F.conv1d(x_rep, kernel, padding=pad, groups=self.n_bands)   # (B, N, T)

    @staticmethod
    def _hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
        """rfft analytic-signal envelope. x: (..., T) real → |analytic| of same shape."""
        # Cast to fp32 for FFT precision; cast back at the end.
        in_dtype = x.dtype
        x32 = x.float()
        T = x32.size(-1)
        Xf = torch.fft.rfft(x32, dim=-1)
        # Analytic mask: 1 at DC and Nyquist (if T even), 2 elsewhere.
        n_freq = Xf.size(-1)
        mask = torch.ones(n_freq, device=x.device, dtype=Xf.dtype)
        if n_freq > 1:
            mask[1:-1] = 2.0
            if T % 2 == 1:
                mask[-1] = 2.0
        # Reshape mask for broadcasting over leading dims.
        mask = mask.view(*([1] * (Xf.dim() - 1)), -1)
        analytic = torch.fft.irfft(Xf * mask, n=T, dim=-1)
        env = (x32.pow(2) + analytic.pow(2)).clamp_min(1e-12).sqrt()
        return env.to(in_dtype)

    def _frame_stats(self, band: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """band: (B, N, T) signal; env: (B, N, T) envelope.
        Returns (B, N, S, T') where S=5 stats and T' = frames.
        Computed in fp32 to keep kurtosis stable under bf16 mixed precision.
        """
        B, N, T = band.shape
        f, h = self.frame, self.hop
        if T < f:
            band = F.pad(band, (0, f - T))
            env  = F.pad(env,  (0, f - T))
            T = f
        # unfold → (B, N, n_frames, frame)
        b = band.float().unfold(-1, f, h)
        e = env .float().unfold(-1, f, h)

        rms      = e.pow(2).mean(dim=-1).clamp_min(1e-12).sqrt()
        log_var  = torch.log1p(e.var(dim=-1, unbiased=False).clamp_min(1e-12))
        # Kurtosis on envelope. Excess kurtosis: m4 / m2^2 - 3.
        mean = e.mean(dim=-1, keepdim=True)
        d    = e - mean
        m2   = d.pow(2).mean(dim=-1).clamp_min(1e-9)
        m4   = d.pow(4).mean(dim=-1)
        kurt = m4 / m2.pow(2) - 3.0

        # ZCR on the band-passed signal (not envelope).
        signs = torch.sign(b)
        zc    = (signs[..., 1:] != signs[..., :-1]).float().mean(dim=-1)

        peak = e.abs().amax(dim=-1)
        crest = peak / rms.clamp_min(1e-12)

        stats = torch.stack([rms, log_var, kurt, zc, crest], dim=-2)   # (B, N, 5, T')
        return stats.to(band.dtype)

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        band = self._bandpass(waveform)                                # (B, N, T)
        env  = self._hilbert_envelope(band)                            # (B, N, T)
        stats = self._frame_stats(band, env)                           # (B, N, S, T')
        B, N, S, Tp = stats.shape
        x = stats.reshape(B, N * S, Tp)                                # (B, N*S, T')
        x = self.proj(x)
        return self.block(x)


# ═══════════════════════════════════════════════════════════════════════
#  Stream E — Frozen wav2vec2 conv front-end (1-D conv only, no STFT)
# ═══════════════════════════════════════════════════════════════════════

class _Wav2Vec2Stream(nn.Module):
    """Frozen wav2vec2 feature_extractor (purely a 1-D conv stack).

    The transformer is dropped; only the conv front-end is kept. Output
    is projected to ``out_ch`` and run through a single SE-Res2 block.
    Resamples the input from ``sample_rate`` to 16 kHz internally.
    """

    def __init__(
        self,
        sample_rate: int = 5_120,
        out_ch:      int = 128,
        model_name:  str = "facebook/wav2vec2-base",
        dropout:     float = 0.1,
        target_sr:   int = 16_000,
    ):
        super().__init__()
        from transformers import Wav2Vec2Model
        w2v = Wav2Vec2Model.from_pretrained(model_name)
        self.feature_extractor = w2v.feature_extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)
        self.feature_extractor.eval()
        self.in_sr  = int(sample_rate)
        self.out_sr = int(target_sr)
        in_ch = 512                                                    # wav2vec2-base default
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.block = _SERes2Block(out_ch, scale=8, kernel_size=3,
                                  dilation=2, dropout=dropout)
        self.out_ch = int(out_ch)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen extractor stays in eval regardless of parent mode.
        self.feature_extractor.eval()
        return self

    @torch.no_grad()
    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        T_in  = x.size(-1)
        T_out = int(round(T_in * self.out_sr / self.in_sr))
        return F.interpolate(
            x.unsqueeze(1).float(), size=T_out,
            mode="linear", align_corners=False,
        ).squeeze(1)

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        with torch.no_grad():
            x16  = self._resample(waveform)
            feat = self.feature_extractor(x16)                         # (B, 512, T')
        feat = self.proj(feat)
        return self.block(feat)


# ═══════════════════════════════════════════════════════════════════════
#  Stream A — Learnable Gabor + 1D SE-Res2 (no spectrogram)
# ═══════════════════════════════════════════════════════════════════════

class _GaborStream(nn.Module):
    def __init__(self, sample_rate: int = 5_120, n_filters: int = 64,
                 kernel_size: int = 257, out_ch: int = 128, dropout: float = 0.1):
        super().__init__()
        self.filterbank = LearnableGaborFilterbank(
            n_filters=n_filters, kernel_size=kernel_size, sample_rate=sample_rate,
        )
        self.spec_aug = SpecAugment1D(n_freq_masks=2, freq_mask_max=6,
                                      n_time_masks=2, time_mask_max=80)
        # strided 1D stem — two stride-4 convs → T / 16
        self.stem = nn.Sequential(
            nn.Conv1d(n_filters, out_ch, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        # out_ch must be divisible by scale=8 for _SERes2Block
        self.blocks = nn.Sequential(
            _SERes2Block(out_ch, scale=8, kernel_size=3, dilation=2, dropout=dropout),
            _SERes2Block(out_ch, scale=8, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.out_ch = out_ch

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.filterbank(waveform)             # (B, n_filters, T)
        if training:
            x = self.spec_aug(x)
        x = self.stem(x)                          # (B, out_ch, T/16)
        return self.blocks(x)                     # (B, out_ch, T')


# ═══════════════════════════════════════════════════════════════════════
#  Stream B — Kymatio Scattering1D + 1D projection (no spectrogram)
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  Stream F — Differentiable LPC residual + coefficients (no STFT)
# ═══════════════════════════════════════════════════════════════════════

class _LPCStream(nn.Module):
    """Levinson-Durbin LPC + residual as a waveform-domain branch.

    Subramani et al. Interspeech'22 (arXiv 2202.11301); LPCSE 2206.06908.

    For each frame of length ``frame``, solve the Yule-Walker equations
    via Levinson-Durbin recursion to obtain ``p`` LPC coefficients
    ``a_1..a_p``. The whitened excitation residual is computed by
    inverse-filtering the framed signal with ``[1, -a_1, ..., -a_p]``.

    Output features (B, p+1, T'):
      - ``a_1..a_p`` per frame (compact spectral-envelope summary).
      - residual energy per frame (peak / variance proxy).
    Then projected through Conv1d → BN → GELU → 1× SE-Res2 block.

    Why orthogonal to existing branches: factorises x = filter * source.
    Existing branches all see the *unfactored* signal — none of them
    expose the AR residual that highlights propeller-cavitation
    transients masked by tonal harmonics.
    """

    def __init__(self, sample_rate: int = 5_120, order: int = 12,
                 frame: int = 256, hop: int = 128, out_ch: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.order = int(order)
        self.frame = int(frame)
        self.hop   = int(hop)
        in_ch = self.order + 1                                  # coeffs + residual stat
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.block = _SERes2Block(out_ch, scale=8, kernel_size=3,
                                  dilation=2, dropout=dropout)
        self.out_ch = int(out_ch)

    @staticmethod
    def _autocorr(x: torch.Tensor, p: int) -> torch.Tensor:
        """Biased autocorrelation r[0..p], x: (..., F) frame-major."""
        # r[k] = sum_n x[n] * x[n+k] / N. Compute via direct correlation.
        # Use F.pad + conv1d in groups for batched compute.
        F_dim = x.size(-1)
        # Build lags: r[0] = sum x^2 ; r[k] = sum x[:F_dim-k] * x[k:].
        rs = []
        for k in range(p + 1):
            if k == 0:
                rs.append((x * x).mean(dim=-1, keepdim=True))
            else:
                rs.append((x[..., :-k] * x[..., k:]).mean(dim=-1, keepdim=True))
        return torch.cat(rs, dim=-1)                            # (..., p+1)

    @staticmethod
    def _levinson(r: torch.Tensor, p: int, eps: float = 1e-6) -> torch.Tensor:
        """Levinson-Durbin recursion. r: (..., p+1) → a: (..., p) coeffs.

        Returns ``a`` such that the prediction error filter is
        ``[1, -a_1, ..., -a_p]``. All math in fp32 for Toeplitz stability.
        """
        # Cast to fp32 at entry; tanh-squash reflection coefficients so |k| < 1
        # for guaranteed minimum-phase stability. Without this, coherent
        # ship-machinery harmonics produce |k| > 1 → (1 - k²) clamped to ε →
        # next k explodes → NaN gradients (I2 LPC failure mode).
        r = r.float()
        E = r[..., 0:1].clamp(min=eps)                          # (..., 1)
        a = torch.zeros(*r.shape[:-1], p, device=r.device, dtype=r.dtype)
        for i in range(1, p + 1):
            if i == 1:
                acc = r[..., 1]
            else:
                rev = torch.flip(r[..., 1:i], dims=[-1])        # (..., i-1)
                acc = r[..., i] + (a[..., :i - 1] * rev).sum(dim=-1)
            # Stable: tanh(arg) bounds the reflection coefficient in (-1, 1).
            k_raw = -acc / E.squeeze(-1).clamp(min=eps)
            k = torch.tanh(k_raw)
            if i > 1:
                a_new_head = a[..., :i - 1] + k.unsqueeze(-1) * torch.flip(
                    a[..., :i - 1], dims=[-1]
                )
                a = a.clone()
                a[..., :i - 1] = a_new_head
            a = a.clone()
            a[..., i - 1] = k
            # 1 - k² is now guaranteed in (0, 1] thanks to tanh bound.
            E = E * (1.0 - k.unsqueeze(-1).pow(2)).clamp(min=eps)
        return a                                                # (..., p)

    def _residual_stat(self, frames: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Compute per-frame log-RMS of the LPC residual.

        residual[n] = frame[n] - Σ_{k=1..p} a_k * frame[n-k].
        Implemented via grouped causal conv1d with kernel = [1, -a_1, …, -a_p].
        """
        # frames: (B, F, T'); a: (B, T', p) — but we need per-frame kernels,
        # which is awkward with grouped conv. Easiest: vectorised
        # Toeplitz construction.
        B, T_p, F_dim = frames.shape
        p = self.order
        # Build kernel per frame: shape (B*T', p+1)
        kernel = torch.cat([
            torch.ones(B, T_p, 1, device=a.device, dtype=a.dtype),
            -a,
        ], dim=-1)                                               # (B, T', p+1)
        # Apply: residual[n] = sum_{k=0..p} kernel[k] * frame[n-k]
        # For each frame independently. Use conv1d with groups = B*T'.
        x = frames.reshape(B * T_p, 1, F_dim).float()
        k = kernel.reshape(B * T_p, 1, p + 1).float()
        # We want valid convolution (causal, with left-pad p).
        x_pad = F.pad(x, (p, 0))
        # F.conv1d treats kernel as cross-correlation; flip for true conv.
        k_flip = torch.flip(k, dims=[-1])
        # Use groups = B*T' so each frame uses its own kernel
        res = F.conv1d(
            x_pad.transpose(0, 1).reshape(1, B * T_p, F_dim + p),
            k_flip,
            groups=B * T_p,
        )                                                        # (1, B*T', F_dim)
        res = res.reshape(B, T_p, F_dim)
        rms = res.pow(2).mean(dim=-1).clamp_min(1e-12).sqrt()    # (B, T')
        return torch.log1p(rms)                                  # (B, T')

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        # waveform: (B, T)
        in_dtype = waveform.dtype
        x = waveform.float()
        # Frame: (B, T') unfold → (B, T', F)
        if x.size(-1) < self.frame:
            x = F.pad(x, (0, self.frame - x.size(-1)))
        frames = x.unfold(-1, self.frame, self.hop)              # (B, T', F)
        # Autocorrelation per frame.
        r = self._autocorr(frames, self.order)                   # (B, T', p+1)
        # Levinson-Durbin per frame.
        a = self._levinson(r, self.order)                        # (B, T', p)
        # Residual log-RMS per frame.
        res_stat = self._residual_stat(frames, a)                # (B, T')
        # Stack: (B, T', p+1) → transpose → (B, p+1, T')
        feat = torch.cat([a, res_stat.unsqueeze(-1)], dim=-1)     # (B, T', p+1)
        feat = feat.transpose(1, 2).contiguous().to(in_dtype)
        feat = self.proj(feat)
        return self.block(feat)


# ═══════════════════════════════════════════════════════════════════════
#  Stream G — Recurrence Plot / phase-space branch (no STFT)
# ═══════════════════════════════════════════════════════════════════════

class _RecurrencePlotStream(nn.Module):
    """Recurrence plot of a Takens-embedded phase-space view.

    Hatami et al. (arXiv 1710.00886); Yu et al. (lake-trial UATR 94.31%).

    Pipeline:
      1. Downsample waveform to ``downsample`` (default 1024) — full RP at
         5120² = 26 M cells is infeasible at batch=64.
      2. Takens embed: y[i] = (x[i], x[i+τ], x[i+2τ], ..., x[i+(m-1)τ]).
      3. Recurrence matrix: R[i,j] = 1 iff ‖y_i - y_j‖_∞ < ε. Continuous
         differentiable variant: R[i,j] = sigmoid((ε - d) / β) so gradients
         flow back to the input via the distance kernel.
      4. 2-D conv stack collapses RP → (out_ch, T') feature map matched to
         the fusion sequence length.

    Why orthogonal: phase-space recurrence is not a frequency, envelope, or
    cepstral feature — it captures whether the trajectory of the signal
    revisits prior states, exposing quasi-periodic engine cycles
    (diagonals) and stochastic cavitation textures (granular blobs).
    """

    def __init__(self, input_len: int = 5_120, downsample: int = 1024,
                 embed_dim: int = 3, delay: int = 4,
                 eps_quantile: float = 0.10, out_ch: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.input_len = int(input_len)
        self.down = int(downsample)
        self.m = int(embed_dim)
        self.tau = int(delay)
        self.eps_q = float(eps_quantile)
        # Embedded sequence length: down - (m-1)*tau
        self.N = self.down - (self.m - 1) * self.tau
        # 2-D ResNet-ish stem on the (1, N, N) recurrence map.
        self.stem = nn.Sequential(
            nn.Conv2d(1,  16, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.AdaptiveAvgPool2d((None, 1)),                # collapse one axis
        )
        self.proj = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.out_ch = int(out_ch)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Takens embedding. x: (B, T) → (B, N, m). Differentiable."""
        # Stack m strided views via gather/index tricks. For small m this is fine.
        cols = []
        for k in range(self.m):
            start = k * self.tau
            cols.append(x[:, start:start + self.N].unsqueeze(-1))
        return torch.cat(cols, dim=-1)                          # (B, N, m)

    def _recurrence(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, N, m) → R: (B, 1, N, N) soft-recurrence in [0, 1]."""
        # Pairwise L∞ distances.
        # diff: (B, N, N, m)
        diff = (y.unsqueeze(2) - y.unsqueeze(1)).abs()
        d = diff.amax(dim=-1)                                   # (B, N, N)
        # Per-clip ε at the configured quantile of off-diagonal distances.
        eps = torch.quantile(
            d.flatten(start_dim=1),
            q=self.eps_q,
            dim=-1,
            keepdim=True,
        ).unsqueeze(-1)                                         # (B, 1, 1)
        # Soft recurrence — sigmoid for differentiability.
        beta = (eps + 1e-6) * 0.25
        R = torch.sigmoid((eps - d) / beta)
        return R.unsqueeze(1)                                   # (B, 1, N, N)

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        in_dtype = waveform.dtype
        x = waveform.float()
        # Downsample to ``self.down`` via 1-D linear interpolation.
        if x.size(-1) != self.down:
            x = F.interpolate(
                x.unsqueeze(1), size=self.down, mode="linear", align_corners=False,
            ).squeeze(1)
        y = self._embed(x)                                       # (B, N, m)
        R = self._recurrence(y)                                  # (B, 1, N, N)
        # 2-D conv stack — output (B, out_ch, H', 1)
        feat = self.stem(R)                                      # (B, out_ch, H', 1)
        feat = feat.squeeze(-1)                                  # (B, out_ch, H')
        feat = self.proj(feat).to(in_dtype)
        return feat


class _GlobalAttnBlock(nn.Module):
    """Single multi-head self-attention block over the fused (B, T, D) sequence.

    Inspired by HELIX (arXiv 2603.21316): a pure SSM backbone destabilises on
    long sequences; inserting one global-attention block closes a measured
    11.5-pt gap. Pre-norm, residual, MLP — the standard transformer recipe
    in (B, T, D) layout to match the post-S4 hidden state.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1,
                 expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        h = d_model * expansion
        self.mlp = nn.Sequential(
            nn.Linear(d_model, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class _ScatteringStream(nn.Module):
    """First-order Scattering1D + (optional) JTFS-lite modulation features.

    JTFS-lite (``use_jtfs=True``) augments the per-band log-magnitude with two
    derivative features computed in time: Δ (first-order temporal derivative
    via 1-tap difference) and ΔΔ (second-order). Captures rate-modulation
    across the scattering bands without needing the full joint-time-frequency
    scattering operator (which kymatio 0.3.0 lacks). When stacked with the
    base scattering, this surfaces blade-rate × shaft-rate cross-modulation
    that pure first-order Scattering1D cannot represent.
    """

    def __init__(self, sample_rate: int = 5_120, input_len: int = 5_120,
                 J: int = 6, Q: int = 8, out_ch: int = 128,
                 use_jtfs: bool = False):
        super().__init__()
        # Import the 1D frontend directly — kymatio.torch eagerly imports the
        # 2D/3D modules which bring in legacy scipy symbols (sph_harm) that
        # newer scipy versions renamed, causing a spurious ImportError.
        try:
            from kymatio.scattering1d.frontend.torch_frontend import (
                ScatteringTorch1D as Scattering1D,
            )
        except ImportError as e:
            raise ImportError(
                "HydroHydra with use_scattering=True requires kymatio. "
                "Install with: pip install kymatio>=0.3.0"
            ) from e
        self.scattering = Scattering1D(J=J, Q=Q, shape=(input_len,))
        # Probe output shape with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, input_len)
            probe = self.scattering(dummy)
        # Scattering1D output: (B, n_coeffs, T_scat)
        n_coeffs = probe.shape[-2]
        self.use_jtfs = bool(use_jtfs)
        in_ch = n_coeffs * (3 if self.use_jtfs else 1)
        self.norm = nn.InstanceNorm1d(in_ch, affine=True)
        self.project = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.out_ch = out_ch

    @staticmethod
    def _delta(x: torch.Tensor) -> torch.Tensor:
        """First-order temporal difference along the last axis with same length."""
        d = x[..., 1:] - x[..., :-1]
        # left-pad with zero to keep T axis length
        return F.pad(d, (1, 0))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Scattering requires float32 *and* contiguous tensors. RIR/pitch in
        # the V2 _WaveformAug can return non-contiguous views, so force layout
        # here rather than relying on the augmentation pipeline.
        in_dtype = waveform.dtype
        x = self.scattering(waveform.float().contiguous())  # (B, n_coeffs, T_scat)
        x = torch.log1p(x.abs())
        if self.use_jtfs:
            d1 = self._delta(x)
            d2 = self._delta(d1)
            x = torch.cat([x, d1, d2], dim=1)              # (B, 3*n_coeffs, T_scat)
        x = self.norm(x)
        x = self.project(x).to(in_dtype)                   # (B, out_ch, T_scat)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  HydroHydra LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroHydra(pl.LightningModule):
    """
    Unified spectrogram-free dual-stream UATR classifier.

    The head outputs ``num_classes + 1`` logits; the last is the Deep-Gamblers
    abstention logit, used only by the auxiliary loss during training.  At
    inference, predictions come from ``logits[:, :num_classes]``.
    """

    def __init__(
        self,
        num_classes:      int = 4,
        class_weights:    Optional[list] = None,
        sample_rate:      int = 5_120,
        input_len:        int = 5_120,
        # ── Stream toggles ───────────────────────────────────────────────
        use_gabor:        bool = True,
        use_scattering:   bool = True,
        use_sincnet:      bool = False,
        use_tdsbe:        bool = False,
        use_w2v:          bool = False,
        use_lpc:          bool = False,
        use_rp:           bool = False,
        # ── Stream widths / hyperparams ──────────────────────────────────
        gabor_n_filters:  int = 64,
        gabor_kernel:     int = 257,
        gabor_ch:         int = 128,
        scat_J:           int = 6,
        scat_Q:           int = 8,
        scat_ch:          int = 128,
        use_jtfs:         bool = False,
        sinc_n_filters:   int = 64,
        sinc_kernel:      int = 251,
        sinc_ch:          int = 128,
        tdsbe_ch:         int = 64,
        w2v_model:        str = "facebook/wav2vec2-base",
        w2v_ch:           int = 128,
        w2v_target_sr:    int = 16_000,
        lpc_order:        int = 12,
        lpc_frame:        int = 256,
        lpc_hop:          int = 128,
        lpc_ch:           int = 64,
        rp_downsample:    int = 1024,
        rp_dim:           int = 3,
        rp_delay:         int = 4,
        rp_eps_quantile:  float = 0.10,
        rp_ch:            int = 64,
        fusion_T:         int = 80,
        fusion_dim:       int = 192,
        s4_n_blocks:      int = 2,
        s4_d_state:       int = 64,
        use_global_attn:  bool = False,
        global_attn_heads: int = 4,
        dropout:          float = 0.15,
        drop_path:        float = 0.0,
        # ── Head ─────────────────────────────────────────────────────────
        head_type:        str   = "mlp",
        feature_norm:     str   = "none",
        arcface_margin:   float = 0.2,
        arcface_scale:    float = 30.0,
        arcface_subcenters: int = 1,
        moe_n_experts:    int   = 4,
        moe_gate_temperature: float = 1.0,
        moe_aux_weight:   float = 0.05,
        # ── Loss ─────────────────────────────────────────────────────────
        loss:             str   = "lmf",
        lmf_gamma:        float = 2.0,
        lmf_margin:       float = 0.5,
        label_smoothing:  float = 0.05,
        gambler_o:        float = 0.3,
        gambler_weight:   float = 0.1,
        cls_num_list:     Optional[list] = None,
        ldam_max_m:       float = 0.5,
        ldam_s:           float = 30.0,
        ldam_drw_epoch:   int   = 40,
        ldam_drw_beta:    float = 0.99999,
        # ── Augmentation (waveform-domain, V2 stack) ─────────────────────
        noise_prob:       float = 0.5,
        noise_snr_min:    float = 15.0,
        noise_snr_max:    float = 30.0,
        gain_prob:        float = 0.5,
        gain_range:       float = 0.3,
        ocean_noise_pool       = None,
        corpus_noise_prob:    float = 0.0,
        corpus_noise_snr_min: float = -3.0,
        corpus_noise_snr_max: float = 15.0,
        rir_prob:        float = 0.0,
        rir_max_delay_s: float = 0.030,
        pitch_prob:      float = 0.0,
        pitch_range:     float = 0.015,
        branch_dropout_p: float = 0.0,
        manifold_mixup_alpha: float = 0.0,
        manifold_mixup_prob:  float = 0.5,
        # ── Optimiser ────────────────────────────────────────────────────
        learning_rate:    float = 3e-4,
        weight_decay:     float = 1e-2,
        warmup_epochs:    int   = 10,
        max_epochs:       int   = 100,
    ):
        super().__init__()
        # ocean_noise_pool is a live object; never serialise into hparams.
        self.save_hyperparameters(ignore=["class_weights", "ocean_noise_pool", "cls_num_list"])

        if not (use_gabor or use_scattering or use_sincnet or use_tdsbe
                or use_w2v or use_lpc or use_rp):
            raise ValueError(
                "HydroHydra needs at least one of "
                "use_gabor / use_scattering / use_sincnet / use_tdsbe / "
                "use_w2v / use_lpc / use_rp."
            )

        self.num_classes    = num_classes
        self.fusion_T       = fusion_T
        self.gambler_o      = gambler_o
        self.manifold_mixup_alpha = float(manifold_mixup_alpha)
        self.manifold_mixup_prob  = float(manifold_mixup_prob)
        # ArcFace + Deep-Gamblers have an unresolved interaction (see plan).
        # Force gambler_weight=0 when a non-MLP head is used.
        if head_type != "mlp" and gambler_weight > 0.0:
            gambler_weight = 0.0
        self.gambler_weight = gambler_weight
        self.head_type      = head_type
        self._has_abstain_logit = (head_type == "mlp" and gambler_weight > 0.0)

        self.wave_aug = _WaveformAugV2(
            noise_prob=noise_prob,
            noise_snr_min=noise_snr_min, noise_snr_max=noise_snr_max,
            gain_prob=gain_prob, gain_range=gain_range,
            ocean_noise_pool=ocean_noise_pool,
            corpus_noise_prob=corpus_noise_prob,
            corpus_noise_snr_min=corpus_noise_snr_min,
            corpus_noise_snr_max=corpus_noise_snr_max,
            rir_prob=rir_prob, rir_max_delay_s=rir_max_delay_s,
            sample_rate=sample_rate,
            pitch_prob=pitch_prob, pitch_range=pitch_range,
        )
        self.branch_drop = _BranchDropout(p=branch_dropout_p)

        cat_ch = 0
        if use_gabor:
            self.stream_a = _GaborStream(
                sample_rate=sample_rate, n_filters=gabor_n_filters,
                kernel_size=gabor_kernel, out_ch=gabor_ch, dropout=dropout,
            )
            cat_ch += gabor_ch
        else:
            self.stream_a = None

        if use_scattering:
            self.stream_b = _ScatteringStream(
                sample_rate=sample_rate, input_len=input_len,
                J=scat_J, Q=scat_Q, out_ch=scat_ch,
                use_jtfs=use_jtfs,
            )
            cat_ch += scat_ch
        else:
            self.stream_b = None

        if use_sincnet:
            self.stream_c = _SincNetStream(
                sample_rate=sample_rate, n_filters=sinc_n_filters,
                kernel_size=sinc_kernel, out_ch=sinc_ch, dropout=dropout,
            )
            cat_ch += sinc_ch
        else:
            self.stream_c = None

        if use_tdsbe:
            self.stream_d = _TDSubBandEnvelopeStream(
                sample_rate=sample_rate, out_ch=tdsbe_ch, dropout=dropout,
            )
            cat_ch += tdsbe_ch
        else:
            self.stream_d = None

        if use_w2v:
            self.stream_e = _Wav2Vec2Stream(
                sample_rate=sample_rate, out_ch=w2v_ch,
                model_name=w2v_model, dropout=dropout,
                target_sr=w2v_target_sr,
            )
            cat_ch += w2v_ch
        else:
            self.stream_e = None

        if use_lpc:
            self.stream_f = _LPCStream(
                sample_rate=sample_rate, order=lpc_order,
                frame=lpc_frame, hop=lpc_hop,
                out_ch=lpc_ch, dropout=dropout,
            )
            cat_ch += lpc_ch
        else:
            self.stream_f = None

        if use_rp:
            self.stream_g = _RecurrencePlotStream(
                input_len=input_len, downsample=rp_downsample,
                embed_dim=rp_dim, delay=rp_delay,
                eps_quantile=rp_eps_quantile,
                out_ch=rp_ch, dropout=dropout,
            )
            cat_ch += rp_ch
        else:
            self.stream_g = None

        self.fuse_proj = nn.Sequential(
            nn.Conv1d(cat_ch, fusion_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(fusion_dim), nn.GELU(),
        )

        self.s4_blocks = nn.ModuleList([
            SaShiMiBlock(d_model=fusion_dim, d_state=s4_d_state,
                         expansion=4, dropout=dropout, drop_path=drop_path)
            for _ in range(s4_n_blocks)
        ])

        if use_global_attn:
            self.global_attn = _GlobalAttnBlock(
                d_model=fusion_dim, n_heads=global_attn_heads, dropout=dropout,
            )
        else:
            self.global_attn = None

        self.pool = _AttentiveStatisticsPool(fusion_dim)

        # Optional pre-head normalisation (required for ArcFace / cosine).
        self.feature_norm_kind = feature_norm
        if feature_norm == "layernorm_l2":
            self.feat_norm = nn.LayerNorm(fusion_dim * 2)
        elif feature_norm == "none":
            self.feat_norm = None
        else:
            raise ValueError(f"Unknown feature_norm: {feature_norm!r}")

        if head_type == "mlp" and self._has_abstain_logit:
            # Legacy head: emits num_classes+1 logits, last reserved for Deep-Gamblers.
            self.head = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, num_classes + 1),
            )
        else:
            # build_head: emits exactly num_classes logits.
            self.head = build_head(
                name=head_type,
                in_dim=fusion_dim * 2,
                num_classes=num_classes,
                fusion_dim=fusion_dim,
                dropout=dropout,
                arcface_margin=arcface_margin,
                arcface_scale=arcface_scale,
                arcface_subcenters=arcface_subcenters,
                moe_n_experts=moe_n_experts,
                moe_gate_temperature=moe_gate_temperature,
            )
        self.moe_aux_weight = float(moe_aux_weight)

        if loss == "lmf":
            self.criterion = LargeMarginFocalLoss(
                num_classes=num_classes, alpha=class_weights,
                gamma=lmf_gamma, margin=lmf_margin,
                label_smoothing=label_smoothing,
            )
        elif loss == "ldam":
            # cls_num_list is excluded from save_hyperparameters, so when the
            # checkpoint is reloaded for post-cal / inference there is no live
            # data module supplying it. Fall back to a uniform list — the
            # criterion's m_list is rebuilt from this, but at eval we don't
            # care about the margin (forward only uses targets), and the DRW
            # weight swap is gated on training mode.
            if cls_num_list is None:
                cls_num_list = [1.0] * num_classes
            self.criterion = LDAMLoss(
                cls_num_list=cls_num_list,
                max_m=ldam_max_m, s=ldam_s,
                weight=None,
                label_smoothing=label_smoothing,
            )
            self._cls_num_list = list(cls_num_list)
            self._ldam_drw_epoch = int(ldam_drw_epoch)
            self._ldam_drw_beta = float(ldam_drw_beta)
        else:
            self.criterion = FocalLoss(
                class_weights=class_weights,
                gamma=lmf_gamma, label_smoothing=label_smoothing,
            )

        m_macro = dict(num_classes=num_classes, average="macro")
        self.train_acc           = MulticlassAccuracy(**m_macro)
        self.val_acc             = MulticlassAccuracy(**m_macro)
        self.val_f1              = MulticlassF1Score(**m_macro)
        self.val_recall          = MulticlassRecall(**m_macro)
        self.val_precision_macro = MulticlassPrecision(**m_macro)
        self.val_precision_per   = MulticlassPrecision(num_classes=num_classes, average=None)
        self.val_mcc             = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.val_auroc           = MulticlassAUROC(num_classes=num_classes)
        self.test_acc            = MulticlassAccuracy(**m_macro)
        self.test_f1             = MulticlassF1Score(**m_macro)
        self.test_precision      = MulticlassPrecision(**m_macro)
        self.test_recall         = MulticlassRecall(**m_macro)
        self.test_mcc            = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_auroc          = MulticlassAUROC(num_classes=num_classes)
        self.test_cm             = MulticlassConfusionMatrix(num_classes=num_classes)

    # ── Forward ──────────────────────────────────────────────────────────

    def _features(self, waveform: torch.Tensor) -> torch.Tensor:
        feats = []
        if self.stream_a is not None:
            feats.append(self.stream_a(waveform, self.training))
        if self.stream_b is not None:
            feats.append(self.stream_b(waveform))
        if self.stream_c is not None:
            feats.append(self.stream_c(waveform, self.training))
        if self.stream_d is not None:
            feats.append(self.stream_d(waveform, self.training))
        if self.stream_e is not None:
            feats.append(self.stream_e(waveform, self.training))
        if self.stream_f is not None:
            feats.append(self.stream_f(waveform, self.training))
        if self.stream_g is not None:
            feats.append(self.stream_g(waveform, self.training))

        Tf = self.fusion_T
        feats = [F.adaptive_avg_pool1d(f, Tf) for f in feats]
        feats = self.branch_drop(feats)
        z = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]    # (B, cat_ch, Tf)

        z = self.fuse_proj(z)                                          # (B, D, Tf)
        z = z.transpose(1, 2)                                          # (B, Tf, D)
        for blk in self.s4_blocks:
            z = blk(z)
        if self.global_attn is not None:
            z = self.global_attn(z)
        z = z.transpose(1, 2)                                          # (B, D, Tf)
        feat = self.pool(z)                                            # (B, 2D)
        if self.feat_norm is not None:
            feat = self.feat_norm(feat)
            feat = F.normalize(feat, dim=-1)
        return feat

    def forward(self, waveform: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self._features(waveform)
        # Legacy MLP+Gamblers head is an nn.Sequential ignoring labels;
        # build_head heads accept (x, labels) and only ArcFace consumes them.
        if isinstance(self.head, nn.Sequential):
            return self.head(feat)                                     # (B, num_classes [+1])
        if self.training:
            return self.head(feat, labels)
        return self.head(feat, None)

    # ── Loss helpers ─────────────────────────────────────────────────────

    def _split(self, logits: torch.Tensor):
        """Returns (class_logits, full_softmax_or_None).

        With abstention head (legacy MLP + Gamblers) ``logits`` has
        ``num_classes + 1`` columns and the last is the abstention logit.
        Otherwise ``logits`` is already class-only.
        """
        if self._has_abstain_logit:
            class_logits = logits[:, :self.num_classes]
            full = F.softmax(logits, dim=-1)
            return class_logits, full
        return logits, None

    def _gambler_loss(self, full: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p_y       = full.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_abstain = full[:, -1]
        return -torch.log(p_y + self.gambler_o * p_abstain + 1e-8).mean()

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        class_logits, full = self._split(logits)
        primary = self.criterion(class_logits, y)
        if self._has_abstain_logit and self.gambler_weight > 0.0:
            aux = self._gambler_loss(full, y)
            return primary + self.gambler_weight * aux
        # DEMON-MoE load-balance term (only active when training and head is MoE).
        if self.head_type == "demon_moe" and self.moe_aux_weight > 0.0 and self.training:
            primary = primary + self.moe_aux_weight * self.head.aux_loss
        return primary

    def _class_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self._has_abstain_logit:
            return logits[:, :self.num_classes]
        return logits

    # ── Lightning steps ──────────────────────────────────────────────────

    def on_train_epoch_start(self):
        # Deferred Re-Weighting: at the chosen epoch, swap the LDAM weight
        # from None (uniform) to class-balanced (effective-number) weights.
        # Cao et al. NeurIPS'19 — DRW lifts tail classes by deferring the
        # imbalanced weighting until late training.
        if not isinstance(self.criterion, LDAMLoss):
            return
        # Use ``>=`` so the swap fires even if a checkpoint resumed at a
        # later epoch, and so resumed runs still apply DRW. Once weights are
        # set we won't re-apply them next epoch (idempotent — the test
        # below also short-circuits when weight is already not None).
        if self.criterion.weight is None and self.current_epoch >= self._ldam_drw_epoch:
            beta = self._ldam_drw_beta
            n = torch.tensor(self._cls_num_list, dtype=torch.float64)
            eff_n = (1.0 - beta ** n) / (1.0 - beta)
            # If β saturates (β^n → 0 for all classes), eff_n collapses to a
            # constant and weights become uniform. Fall back to inverse-freq.
            if eff_n.std() / eff_n.mean().clamp(min=1e-9) < 1e-3:
                w = (n.sum() / (len(n) * n.clamp(min=1.0)))
            else:
                w = 1.0 / eff_n
                w = w * len(self._cls_num_list) / w.sum()
            # Use Lightning's print so the message clears the progress bar
            # cleanly and lands in the log file.
            self.criterion.weight = w.float().to(self.device)
            self.print(
                f"[LDAM-DRW] epoch={self.current_epoch}: swapped to weights "
                f"{w.tolist()}  (β={beta})",
                flush=True,
            )

    def _mixup_loss(self, logits: torch.Tensor, y_a: torch.Tensor,
                    y_b: torch.Tensor, lam: float) -> torch.Tensor:
        la = self._compute_loss(logits, y_a)
        lb = self._compute_loss(logits, y_b)
        return lam * la + (1.0 - lam) * lb

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.wave_aug(x)
        # Manifold mixup: mix at the pooled-feature stage. Disabled when
        # ArcFace is in use (label-margin geometry is incompatible with
        # mixed labels).
        head_supports_mixup = (
            isinstance(self.head, nn.Sequential) or self.head_type == "mlp"
        )
        do_mixup = (
            self.manifold_mixup_alpha > 0.0
            and head_supports_mixup
            and torch.rand(()).item() < self.manifold_mixup_prob
        )
        if do_mixup:
            feat = self._features(x)
            lam = float(torch.distributions.Beta(
                self.manifold_mixup_alpha, self.manifold_mixup_alpha
            ).sample().clamp(min=0.05, max=0.95).item())
            perm = torch.randperm(feat.size(0), device=feat.device)
            feat_mix = lam * feat + (1.0 - lam) * feat[perm]
            logits = self.head(feat_mix) if isinstance(self.head, nn.Sequential) \
                     else self.head(feat_mix, None)
            y_b = y[perm]
            loss = self._mixup_loss(logits, y, y_b, lam)
            class_logits = self._class_logits(logits)
            self.train_acc(class_logits, y)        # rough — mixup blurs accuracy
        else:
            logits = self(x, y)
            loss = self._compute_loss(logits, y)
            class_logits = self._class_logits(logits)
            self.train_acc(class_logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        class_logits = self._class_logits(logits)
        probs = F.softmax(class_logits, dim=-1)

        self.val_acc(class_logits, y)
        self.val_f1(class_logits, y)
        self.val_recall(class_logits, y)
        self.val_precision_macro(class_logits, y)
        self.val_precision_per(class_logits, y)
        self.val_mcc(class_logits, y)
        self.val_auroc(probs, y)

        self.log("val/loss",            loss,                     on_epoch=True, prog_bar=True)
        self.log("val/acc",             self.val_acc,             on_epoch=True, prog_bar=True)
        self.log("val/f1",              self.val_f1,              on_epoch=True, prog_bar=True)
        self.log("val/recall",          self.val_recall,          on_epoch=True)
        self.log("val/macro_precision", self.val_precision_macro, on_epoch=True, prog_bar=True)
        self.log("val/mcc",             self.val_mcc,             on_epoch=True)
        self.log("val/auroc",           self.val_auroc,           on_epoch=True)

    def on_validation_epoch_end(self):
        per = self.val_precision_per.compute()
        for i, v in enumerate(per):
            self.log(f"val/precision_c{i}", v, prog_bar=False)
        self.val_precision_per.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        class_logits = self._class_logits(logits)
        probs = F.softmax(class_logits, dim=-1)

        self.test_acc(class_logits, y)
        self.test_f1(class_logits, y)
        self.test_precision(class_logits, y)
        self.test_recall(class_logits, y)
        self.test_mcc(class_logits, y)
        self.test_auroc(probs, y)
        self.test_cm(class_logits, y)

        self.log("test/loss",            loss,                on_epoch=True)
        self.log("test/acc",             self.test_acc,       on_epoch=True)
        self.log("test/f1",              self.test_f1,        on_epoch=True)
        self.log("test/macro_precision", self.test_precision, on_epoch=True)
        self.log("test/recall",          self.test_recall,    on_epoch=True)
        self.log("test/mcc",             self.test_mcc,       on_epoch=True)
        self.log("test/auroc",           self.test_auroc,     on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix:\n{cm.cpu().numpy()}")
        self.test_cm.reset()

    # ── Optimiser ────────────────────────────────────────────────────────

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [{"params": decay,    "weight_decay": self.hparams.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.learning_rate, betas=(0.9, 0.98), eps=1e-8,
        )

        def lr_lambda(epoch):
            wu, total = self.hparams.warmup_epochs, self.hparams.max_epochs
            if epoch < wu:
                return (epoch + 1) / max(wu, 1)
            p = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
