"""
HydroNet — ECAPA-TDNN-style Vessel Acoustic Classifier
=======================================================

Architecture
------------
  Raw waveform (32 kHz, 1 s)
      ↓
  EnhancedFrontEnd        — 4-channel acoustic front-end:
      ↓                       Ch 0: Wideband mel + PCEN
      ↓                       Ch 1: Narrowband mel + PCEN
      ↓                       Ch 2: Gammatone filterbank + PCEN
      ↓                       Ch 3: DEMON envelope spectrogram
  SpecAugment (train)     — per-channel masking
      ↓
  CepstralLifter          — removes channel/transmission effects from Ch 0-2
      ↓
  CNNInput                — Conv2d front-end (4-channel) → (B, C, T)
      ↓
  SERes2Block × 3         — dilated [2, 4, 8] Res2Net + SE attention
      ↓
  MFA                     — multi-scale feature aggregation
      ↓
  AttentiveStatisticsPool — mean + std → (B, 2C)
      ↓
  SubBandEnvelope         — 9 waveform-level features (RMS, var, kurtosis × 3 bands)
      ↓                     concatenated to (B, 2C+9)
  Classifier              — Linear(2C+9→C) → BN → ReLU → Dropout → Linear(C→classes)
      ↓
  FocalLoss (class-weighted)

Signal processing additions (vs v1)
------------------------------------
  Gammatone filterbank : ERB-spaced Gaussian filters — finer resolution at low
                         frequencies (<500 Hz) where vessel machinery noise lives.
  DEMON                : Envelope of the cavitation band (800Hz+) captures blade-rate
                         modulations — the standard passive-sonar vessel discriminator.
  Cepstral liftering   : Removes smooth spectral envelope (range/depth effects) from
                         mel/gammatone channels, preserving harmonic ridge structure.
  Sub-band envelope    : RMS, variance, kurtosis across 3 physically-motivated bands:
                         0-100 Hz (machinery), 100-800 Hz (propeller), 800-2560 Hz (cavitation).
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as TAF
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from .hydro_conformer import TrainablePCEN, SpecAugment, DropPath, FocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  1. Gammatone filterbank
# ═══════════════════════════════════════════════════════════════════════

class GammatoneSpectrogram(nn.Module):
    """
    Gammatone filterbank applied to an STFT magnitude spectrum.

    Uses ERB (Equivalent Rectangular Bandwidth) spacing for centre
    frequencies — significantly narrower than mel filters below 500 Hz,
    giving much better resolution of engine and propeller harmonics.

    The filterbank matrix is stored as a fixed buffer (not learned).
    PCEN normalisation is applied separately after this module.

    ERB formula: ERB(f) = 24.7 * (4.37 * f/1000 + 1)
    ERB scale  : ERBS(f) = 21.4 * log10(4.37 * f/1000 + 1)
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft:       int,
        hop_length:  int,
        n_bands:     int,
        f_min:       float = 20.0,
        f_max:       Optional[float] = None,
    ):
        super().__init__()
        f_max = f_max or sample_rate / 2.0
        self.spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)

        n_freqs = n_fft // 2 + 1
        freqs   = torch.linspace(0.0, sample_rate / 2.0, n_freqs)

        # ERB-spaced centre frequencies
        erbs_min = 21.4 * math.log10(max(4.37 * f_min / 1000.0 + 1.0, 1e-9))
        erbs_max = 21.4 * math.log10(4.37 * f_max / 1000.0 + 1.0)
        erbs     = torch.linspace(erbs_min, erbs_max, n_bands)
        fc       = (10.0 ** (erbs / 21.4) - 1.0) * 1000.0 / 4.37  # Hz

        # Gaussian filter shape with σ = ERB(fc) / 2
        erb_bw = 24.7 * (4.37 * fc / 1000.0 + 1.0)
        fc_c   = fc.unsqueeze(1)      # (n_bands, 1)
        bw_c   = erb_bw.unsqueeze(1)  # (n_bands, 1)
        f_r    = freqs.unsqueeze(0)   # (1, n_freqs)

        weights = torch.exp(-((f_r - fc_c) / (bw_c / 2.0)) ** 2)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        self.register_buffer("fb", weights)   # (n_bands, n_freqs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mag = self.spec(waveform)               # (B, n_freqs, T)
        mag_t = mag.permute(0, 2, 1)            # (B, T, n_freqs)
        out   = F.linear(mag_t, self.fb)        # (B, T, n_bands)
        return out.permute(0, 2, 1)             # (B, n_bands, T)


# ═══════════════════════════════════════════════════════════════════════
#  2. DEMON envelope channel
# ═══════════════════════════════════════════════════════════════════════

def _hilbert_envelope(x: torch.Tensor) -> torch.Tensor:
    """Analytic-signal magnitude (canonical Hilbert envelope).

    For real input ``x`` of length ``N`` the analytic signal is

        x_a = IFFT(FFT(x) * h),  where
        h[k] = 1 if k==0 or k==N/2 (even N) else 2 if k<N/2 else 0,

    and the envelope is ``env(t) = |x_a(t)|``. Implemented with full
    ``torch.fft.fft`` because the analytic signal is complex-valued.
    """
    N = x.shape[-1]
    Xf = torch.fft.fft(x.float(), dim=-1)              # (..., N) complex
    h = torch.zeros(N, device=x.device, dtype=Xf.dtype)
    h[0] = 1.0
    if N % 2 == 0:
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[1:(N + 1) // 2] = 2.0
    x_a = torch.fft.ifft(Xf * h, dim=-1)
    return x_a.abs().to(x.dtype)


class _DEMONGram(nn.Module):
    """Canonical DEMON-gram: envelope → DC-remove → (optional) decimate → STFT-mag.

    Replaces the simple square-then-STFT path when ``envelope_kind != 'square'``.
    The 'square' path stays in :class:`_LinearModulationSpec` for back-compat
    with pre-rewrite checkpoints.

    Shapes: input ``(B, L)`` (cavitation-bandpassed waveform),
    output ``(B, n_bins, T_spec)`` where

        n_bins = clip(floor(n_fft * mod_f_max / dec_sr) + 1, 1, n_fft//2 + 1)
                 - floor(n_fft * mod_f_min / dec_sr)
        T_spec = floor((L_dec - n_fft) / hop_length) + 1
                 with L_dec = L // decimate (when decimate > 1).
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft:       int,
        hop_length:  int,
        decimate:    int   = 1,
        mod_f_min:   float = 0.0,
        mod_f_max:   float = 250.0,
        envelope_kind: str = "hilbert",
    ):
        super().__init__()
        if envelope_kind not in ("hilbert", "fwr"):
            raise ValueError(f"_DEMONGram envelope_kind: {envelope_kind!r}")
        self.envelope_kind = envelope_kind
        self.sample_rate   = int(sample_rate)
        self.decimate      = max(1, int(decimate))
        self.dec_sr        = self.sample_rate // self.decimate
        self.n_fft         = int(n_fft)
        self.hop_length    = int(hop_length)
        self.mod_f_min     = float(mod_f_min)
        self.mod_f_max     = float(mod_f_max)

        df = self.dec_sr / self.n_fft
        self.lo = max(0, int(round(self.mod_f_min / df)))
        self.hi = min(self.n_fft // 2 + 1, int(round(self.mod_f_max / df)) + 1)
        if self.hi <= self.lo:
            raise ValueError(
                f"_DEMONGram: empty bin range [{self.lo}, {self.hi}). "
                f"Raise n_fft, raise mod_f_max, or reduce decimate "
                f"(dec_sr={self.dec_sr}, df={df:.3f} Hz)."
            )
        self.n_bins = self.hi - self.lo
        self.register_buffer("_window", torch.hann_window(self.n_fft))

    def _envelope(self, x_bp: torch.Tensor) -> torch.Tensor:
        if self.envelope_kind == "hilbert":
            return _hilbert_envelope(x_bp)
        return x_bp.abs()                                  # full-wave rect

    def forward(self, x_bp: torch.Tensor) -> torch.Tensor:
        env = self._envelope(x_bp)
        if self.decimate > 1:
            env = TAF.resample(
                env.float(),
                orig_freq=self.sample_rate, new_freq=self.dec_sr,
                lowpass_filter_width=64, rolloff=0.99,
            ).to(x_bp.dtype)
        env = env - env.mean(dim=-1, keepdim=True)         # DC removal
        spec = torch.stft(
            env.float(),
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self._window, return_complex=True, center=True,
        )
        return spec.abs().to(x_bp.dtype)[:, self.lo:self.hi, :]


class _LOFARSpec(nn.Module):
    """1D LOFAR-gram frontend: high-resolution linear-frequency log-power
    spectrogram, bandlimited to the mid-frequency tonal range.

    Adapted from :class:`models.hydro_lofar_resnet.LofarFrontend` (which produces
    a 4D image for ResNet-50). This 1D variant outputs ``(B, freq_bins, T_frames)``
    matching the contract of the other HydroPreciseV2 branch frontends so it can
    slot in as a 5th branch without reshape gymnastics.

    Pipeline: STFT power → bandlimit to ``max_freq`` → log → per-sample
    instance-norm → adaptive-pool freq axis to ``freq_bins`` (keeps the time
    axis at its natural frame count, then upstream branches pool to ``fusion_T``).
    """

    def __init__(
        self,
        sample_rate: int   = 5_120,
        n_fft:       int   = 4_096,
        hop_length:  int   = 160,
        max_freq:    float = 2_560.0,
        freq_bins:   int   = 256,
        log_floor:   float = 1e-9,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft       = int(n_fft)
        self.hop_length  = int(hop_length)
        self.max_freq    = float(max_freq)
        self.freq_bins   = int(freq_bins)
        self.log_floor   = float(log_floor)

        self.spec = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=2.0,
            center=True, normalized=False,
        )
        freqs = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
        self.register_buffer("freq_mask", (freqs <= max_freq).float())
        # Adaptive pool over the frequency axis only — time axis is left at its
        # natural frame count and downstream branches handle T pooling.
        self.pool_f = nn.AdaptiveAvgPool1d(self.freq_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spec(x.float())                            # (B, F_stft, T)
        s = s * self.freq_mask.unsqueeze(-1)                # bandlimit
        s = torch.log(s + self.log_floor)
        # Per-sample instance norm
        mu  = s.mean(dim=(-2, -1), keepdim=True)
        sig = s.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        s   = (s - mu) / sig                                # (B, F_stft, T)
        # Pool only over the frequency axis: transpose so AdaptiveAvgPool1d
        # operates on freq, then transpose back. (B, T, F) → pool → (B, T, freq_bins)
        # → (B, freq_bins, T)
        st = s.transpose(1, 2).contiguous()                 # (B, T, F_stft)
        st = self.pool_f(st)                                # (B, T, freq_bins)
        return st.transpose(1, 2).to(x.dtype)               # (B, freq_bins, T)


class _LinearModulationSpec(nn.Module):
    """
    Linear-frequency spectrogram restricted to the modulation band.

    The squared cavitation-band envelope carries amplitude-modulation rates
    at multiples of the blade-passage frequency (~1–30 Hz, harmonics out
    to ~250 Hz). A mel filterbank wastes resolution on that band because
    nearly the entire bank sits above the modulation range. This module
    takes a plain STFT magnitude and slices to the bins inside
    [mod_f_min, mod_f_max], giving uniform resolution where it matters.

    Output shape: (B, n_bins, T_spec) where n_bins is determined by
    n_fft / sample_rate / mod_f_max (exposed via `.n_bins`).
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft:       int,
        hop_length:  int,
        mod_f_min:   float = 0.0,
        mod_f_max:   float = 50.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.mod_f_min   = mod_f_min
        self.mod_f_max   = mod_f_max
        self.spec = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length,
            power=1.0, pad_mode="reflect",
        )
        df = sample_rate / n_fft
        self.lo = max(0, int(round(mod_f_min / df)))
        self.hi = min(n_fft // 2 + 1, int(round(mod_f_max / df)) + 1)
        if self.hi <= self.lo:
            raise ValueError(
                f"_LinearModulationSpec: empty bin range "
                f"[lo={self.lo}, hi={self.hi}). Increase n_fft or widen mod_f_max."
            )
        self.n_bins = self.hi - self.lo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spec(x)                       # (B, n_fft//2+1, T)
        return s[:, self.lo:self.hi, :]        # (B, n_bins, T)


class DEMONChannel(nn.Module):
    """
    DEMON (Detection of Envelope Modulation on Noise).

    Ships' propellers produce cavitation noise amplitude-modulated at the
    blade-passage frequency (BPF = shaft_rpm/60 × n_blades, typically 1–30 Hz).
    DEMON extracts this modulation pattern, which is a primary discriminator
    between vessel types.

    Implementation:
      1. Spectral bandpass to cavitation band [f_cav, Nyquist]
         (preserves only the part of the signal that carries modulation).
      2. Square the bandpassed signal (envelope detection via self-multiplication).
      3. Linear-frequency STFT of the squared signal, sliced to the modulation
         band [mod_f_min, mod_f_max]. This gives uniform resolution across the
         band where BPF and shaft-rate harmonics live (a mel filterbank would
         spend almost all of its bins above the band of interest).
      4. Apply TrainablePCEN to normalise across recording conditions. PCEN
         is bin-agnostic — its parameters are per-row scalars and do not
         depend on the mel-vs-linear choice.

    Output shape: (B, n_bins, T_spec) where n_bins is derived from
    n_fft and mod_f_max (exposed via the `.n_bins` property).
    """

    def __init__(
        self,
        sample_rate: int,
        hop_length:  int,
        n_fft:       int   = 2048,
        f_cav:       float = 800.0,
        mod_f_min:   float = 0.0,
        mod_f_max:   float = 50.0,
        envelope:    str   = "square",   # "square" | "hilbert" | "fwr"
        decimate:    int   = 1,
    ):
        super().__init__()
        if envelope not in ("square", "hilbert", "fwr"):
            raise ValueError(f"DEMONChannel envelope: {envelope!r}")
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.f_cav       = f_cav
        self.envelope    = envelope

        if envelope == "square":
            # Legacy path — keep _LinearModulationSpec for back-compat.
            self.spec = _LinearModulationSpec(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                mod_f_min=mod_f_min, mod_f_max=mod_f_max,
            )
        else:
            self.spec = _DEMONGram(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                decimate=decimate, mod_f_min=mod_f_min, mod_f_max=mod_f_max,
                envelope_kind=envelope,
            )
        self.pcen = TrainablePCEN(self.spec.n_bins)

    @property
    def n_bins(self) -> int:
        return self.spec.n_bins

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        B, T_len = waveform.shape

        # Spectral bandpass via FFT masking (force fp32 — cuFFT rejects fp16 for non-power-of-2 lengths)
        wf32  = waveform.float()
        X     = torch.fft.rfft(wf32, dim=-1)
        freqs = torch.fft.rfftfreq(T_len, d=1.0 / self.sample_rate,
                                   device=waveform.device)
        mask  = (freqs >= self.f_cav).float()
        x_bp  = torch.fft.irfft(X * mask, n=T_len, dim=-1).to(waveform.dtype)

        if self.envelope == "square":
            # Legacy: square + STFT-mag
            x_env = x_bp ** 2
            dem   = self.spec(x_env).clamp(min=1e-9)
        else:
            # Canonical: _DEMONGram does envelope + DC-remove + (optional) decimate + STFT-mag
            dem   = self.spec(x_bp).clamp(min=1e-9)
        return self.pcen(dem)


# ═══════════════════════════════════════════════════════════════════════
#  3. Cepstral liftering
# ═══════════════════════════════════════════════════════════════════════

class CepstralLifter(nn.Module):
    """
    Cepstral liftering applied along the frequency axis of a spectrogram.

    The cepstrum of a log spectrum separates the slow-varying spectral
    envelope (caused by transmission path, recording distance, hydrophone
    response) from the fast-varying harmonic structure (propeller/engine).

    This module keeps only the quefrency range [low_q, high_q], removing:
      - Low quefrencies  (< low_q): smooth envelope → recording condition effects
      - High quefrencies (> high_q): fine noise → not carrier of tonal information

    The kept range highlights harmonic ridge patterns that discriminate
    vessel types regardless of recording geometry.

    Applied to channels 0, 1, 2 (mel + gammatone).  Channel 3 (DEMON) is
    already a modulation spectrum and is left unchanged.

    Fully differentiable via real FFT/iFFT along the frequency axis.
    """

    def __init__(self, n_mels: int, low_q: int = 3, high_q: int = 25):
        super().__init__()
        n_ceps = n_mels // 2 + 1
        mask   = torch.zeros(n_ceps)
        mask[low_q : min(high_q, n_ceps)] = 1.0
        self.register_buffer("mask", mask)
        self.n_mels = n_mels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, n_mels, T)
        Returns:
            x with channels 0–2 cepstrally liftered; channel 3 unchanged.
        """
        # Apply to channels 0, 1, 2 only
        x_sp = x[:, :3]                              # (B, 3, F, T)
        x_log = torch.log(x_sp.abs() + 1e-9)
        X_cep = torch.fft.rfft(x_log, dim=2)         # (B, 3, F//2+1, T)
        X_cep = X_cep * self.mask.view(1, 1, -1, 1)
        x_lift = torch.fft.irfft(X_cep, n=self.n_mels, dim=2)  # (B, 3, F, T)
        return torch.cat([x_lift, x[:, 3:]], dim=1)  # (B, C, F, T)


# ═══════════════════════════════════════════════════════════════════════
#  4. Sub-band envelope features
# ═══════════════════════════════════════════════════════════════════════

class SubBandEnvelope(nn.Module):
    """
    Compute RMS, variance, and kurtosis for 3 physically motivated sub-bands.
    Returns a (B, 9) feature vector concatenated to the pooled embedding.

    Bands:
      0 –  100 Hz : ship machinery (engine, pump, low harmonics)
      100 – 800 Hz : propeller blade fundamentals
      800 – 2560 Hz : cavitation broadband + higher harmonics

    RMS captures overall energy; variance captures temporal dynamics;
    kurtosis captures impulsive/tonal character (high kurtosis = tonal).
    """

    BANDS = [(0.0, 100.0), (100.0, 800.0), (800.0, 2560.0)]

    def __init__(self, sample_rate: int = 32_000, n_fft: int = 2048):
        super().__init__()
        n_freqs = n_fft // 2 + 1
        freqs   = torch.linspace(0.0, sample_rate / 2.0, n_freqs)

        masks = []
        for f_lo, f_hi in self.BANDS:
            m = ((freqs >= f_lo) & (freqs < f_hi)).float()
            masks.append(m / m.sum().clamp(min=1.0))
        self.register_buffer("masks", torch.stack(masks))  # (3, n_freqs)
        self.n_fft = n_fft

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        B  = waveform.shape[0]
        # Force fp32 — torch.stft / cuFFT does not support fp16 for non-power-of-2 lengths
        wf32 = waveform.float()
        hw = torch.hann_window(self.n_fft, device=waveform.device)
        X  = torch.stft(wf32, n_fft=self.n_fft, hop_length=self.n_fft // 4,
                        window=hw, return_complex=True)   # (B, n_freqs, T_spec)
        power = X.abs().pow(2)                            # (B, n_freqs, T_spec)

        feats = []
        for i in range(3):
            # Band-averaged power over time  (B, T_spec)
            bp = (power * self.masks[i].view(1, -1, 1)).sum(dim=1)
            rms  = bp.mean(dim=-1).sqrt()                  # (B,)
            var  = bp.var(dim=-1)                          # (B,)
            mu   = bp.mean(dim=-1, keepdim=True)
            sig  = bp.std(dim=-1, keepdim=True).clamp(min=1e-9)
            kurt = ((bp - mu) / sig).pow(4).mean(dim=-1)   # (B,)
            feats += [rms, var, kurt]

        return torch.stack(feats, dim=-1)   # (B, 9)


# ═══════════════════════════════════════════════════════════════════════
#  Combined 4-channel front-end
# ═══════════════════════════════════════════════════════════════════════

class EnhancedFrontEnd(nn.Module):
    """
    Produces a 4-channel time-frequency tensor from raw waveform:
      Ch 0: Wideband mel PCEN
      Ch 1: Narrowband mel PCEN
      Ch 2: Gammatone PCEN
      Ch 3: DEMON envelope PCEN
    """

    def __init__(
        self,
        sample_rate: int   = 32_000,
        n_mels:      int   = 128,
        hop_length:  int   = 320,
        wb_n_fft:    int   = 1_024,
        nb_n_fft:    int   = 4_096,
        f_min:       float = 20.0,
    ):
        super().__init__()
        f_max = sample_rate / 2.0
        mel_kw = dict(sample_rate=sample_rate, hop_length=hop_length,
                      n_mels=n_mels, f_min=f_min, f_max=f_max, power=1.0)

        self.wb_mel  = T.MelSpectrogram(n_fft=wb_n_fft, **mel_kw)
        self.nb_mel  = T.MelSpectrogram(n_fft=nb_n_fft, **mel_kw)
        self.wb_pcen = TrainablePCEN(n_mels)
        self.nb_pcen = TrainablePCEN(n_mels)

        self.gt_spec = GammatoneSpectrogram(sample_rate, wb_n_fft, hop_length,
                                            n_bands=n_mels, f_min=f_min, f_max=f_max)
        self.gt_pcen = TrainablePCEN(n_mels)

        # DEMON now uses a linear modulation spectrogram (see DEMONChannel
        # docstring). Its native bin count is derived from n_fft + mod_f_max,
        # so we interpolate to n_mels rows downstream to preserve the
        # 4-stream stack contract used by HydroNet.
        self.demon       = DEMONChannel(sample_rate, hop_length, n_fft=wb_n_fft)
        self.demon_n_out = n_mels

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        wb  = self.wb_pcen(self.wb_mel(waveform).clamp(min=1e-9))
        nb  = self.nb_pcen(self.nb_mel(waveform).clamp(min=1e-9))
        gt  = self.gt_pcen(self.gt_spec(waveform).clamp(min=1e-9))
        dem = self.demon(waveform)
        if dem.shape[-2] != self.demon_n_out:
            dem = F.interpolate(
                dem.unsqueeze(1),
                size=(self.demon_n_out, dem.shape[-1]),
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        t = min(wb.shape[-1], nb.shape[-1], gt.shape[-1], dem.shape[-1])
        return torch.stack([wb[..., :t], nb[..., :t],
                            gt[..., :t], dem[..., :t]], dim=1)  # (B, 4, F, T)


# ═══════════════════════════════════════════════════════════════════════
#  Input projection: N-channel spectrogram → 1-D feature sequence
# ═══════════════════════════════════════════════════════════════════════

class CNNInput(nn.Module):
    """
    Compress (B, in_channels, n_mels, T) → (B, channels, T) via two strided
    Conv2d stages along the frequency axis, then AdaptiveAvgPool to collapse
    remaining freq bins.
    """

    def __init__(self, channels: int, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, channels // 2, kernel_size=(3, 3),
                      stride=(2, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=(3, 3),
                      stride=(2, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.freq_pool(x)
        return x.squeeze(2)


# ═══════════════════════════════════════════════════════════════════════
#  Res2Net multi-scale dilated convolution
# ═══════════════════════════════════════════════════════════════════════

class Res2DilatedConv(nn.Module):
    def __init__(self, channels: int, scale: int = 8,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        pad = dilation * (kernel_size // 2)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size,
                      dilation=dilation, padding=pad, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.width) for _ in range(scale - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks  = x.chunk(self.scale, dim=1)
        outputs = [chunks[0]]
        carry: Optional[torch.Tensor] = None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            inp   = chunks[i + 1] if carry is None else chunks[i + 1] + carry
            carry = F.relu(bn(conv(inp)))
            outputs.append(carry)
        return torch.cat(outputs, dim=1)


# ═══════════════════════════════════════════════════════════════════════
#  Squeeze-Excitation channel attention
# ═══════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 8))
        self.fc2 = nn.Linear(max(channels // reduction, 8), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════════════
#  SE-Res2Block
# ═══════════════════════════════════════════════════════════════════════

class SERes2Block(nn.Module):
    def __init__(self, channels: int, scale: int = 8, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.pw_in  = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.res2   = Res2DilatedConv(channels, scale, kernel_size, dilation)
        self.bn     = nn.BatchNorm1d(channels)
        self.pw_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.se      = SEBlock(channels)
        self.dropout = nn.Dropout(dropout)
        self.dp      = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw_in(x)
        out = self.res2(out)
        out = self.bn(out)
        out = self.pw_out(out)
        out = self.se(out)
        out = self.dropout(out)
        return x + self.dp(out)


# ═══════════════════════════════════════════════════════════════════════
#  Multi-scale Feature Aggregation
# ═══════════════════════════════════════════════════════════════════════

class MFALayer(nn.Module):
    def __init__(self, n_inputs: int, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(channels * n_inputs, channels, 1, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )

    def forward(self, *feature_maps: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat(feature_maps, dim=1))


# ═══════════════════════════════════════════════════════════════════════
#  Attentive Statistics Pooling
# ═══════════════════════════════════════════════════════════════════════

class AttentiveStatisticsPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 1),
            nn.Tanh(),
            nn.Conv1d(channels // 4, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w    = F.softmax(self.attn(x), dim=-1)
        mean = (x * w).sum(dim=-1)
        var  = (x ** 2 * w).sum(dim=-1) - mean ** 2
        std  = torch.sqrt(var.clamp(min=1e-9))
        return torch.cat([mean, std], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  HydroNet LightningModule
# ═══════════════════════════════════════════════════════════════════════

_N_SUBBAND = 9   # SubBandEnvelope output dimension (3 bands × 3 stats)


class HydroNet(pl.LightningModule):
    """
    ECAPA-TDNN-style underwater vessel classifier with enhanced signal processing.

    Args:
        num_classes     : Number of vessel classes.
        class_weights   : Inverse-frequency weights (from DALIAudioDataModule).
        sample_rate     : Audio sample rate in Hz.
        n_mels          : Mel/gammatone filterbank bins.
        hop_length      : STFT hop in samples.
        wb_n_fft        : Wideband FFT size.
        nb_n_fft        : Narrowband FFT size.
        channels (C)    : Feature channels throughout the encoder.
        scale           : Res2Net split factor (must divide channels).
        dilation_rates  : Dilation per SERes2Block.
        kernel_size     : Depthwise conv kernel size.
        lifter_low_q    : Lower quefrency cutoff for cepstral liftering.
        lifter_high_q   : Upper quefrency cutoff for cepstral liftering.
        dropout         : Dropout rate.
        drop_path_rate  : Max stochastic depth rate.
        learning_rate   : Peak AdamW LR.
        weight_decay    : AdamW weight decay.
        warmup_epochs   : Linear LR warmup epochs.
        max_epochs      : Total epochs for cosine schedule.
        mixup_alpha     : Waveform Mixup beta parameter (0 = off).
        noise_prob      : Probability of additive noise augmentation.
        noise_snr_db    : (min, max) SNR in dB for noise.
        gain_prob       : Probability of random gain augmentation.
        gain_range      : (min, max) multiplicative gain.
        focal_gamma     : Focal loss γ.
        label_smoothing : Label smoothing ε.
    """

    def __init__(
        self,
        num_classes:     int         = 4,
        class_weights:   Optional[list] = None,
        sample_rate:     int         = 32_000,
        n_mels:          int         = 128,
        hop_length:      int         = 320,
        wb_n_fft:        int         = 1_024,
        nb_n_fft:        int         = 4_096,
        channels:        int         = 128,
        scale:           int         = 8,
        dilation_rates:  List[int]   = (2, 4, 8),
        kernel_size:     int         = 3,
        lifter_low_q:    int         = 3,
        lifter_high_q:   int         = 25,
        dropout:         float       = 0.24,
        drop_path_rate:  float       = 0.10,
        learning_rate:   float       = 3e-4,
        weight_decay:    float       = 0.012,
        warmup_epochs:   int         = 10,
        max_epochs:      int         = 100,
        mixup_alpha:     float       = 0.20,
        noise_prob:      float       = 0.50,
        noise_snr_db:    tuple       = (20.0, 40.0),
        gain_prob:       float       = 0.70,
        gain_range:      tuple       = (0.6, 1.4),
        focal_gamma:     float       = 2.0,
        label_smoothing: float       = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        n_blocks = len(dilation_rates)

        # ── Front-end ───────────────────────────────────────────────────
        self.frontend  = EnhancedFrontEnd(
            sample_rate=sample_rate, n_mels=n_mels,
            hop_length=hop_length, wb_n_fft=wb_n_fft, nb_n_fft=nb_n_fft,
        )
        self.spec_aug  = SpecAugment(n_freq_masks=2, freq_mask_max=16,
                                     n_time_masks=2, time_mask_max=20)
        self.lifter    = CepstralLifter(n_mels, low_q=lifter_low_q,
                                        high_q=lifter_high_q)
        # n_fft scales with wb_n_fft so padding never exceeds 1-s clip at any SR
        # (at 32 kHz wb_n_fft=1024 → sub_band n_fft=2048, same as original default)
        self.sub_band  = SubBandEnvelope(sample_rate=sample_rate, n_fft=wb_n_fft * 2)

        # ── Encoder ────────────────────────────────────────────────────
        self.cnn_input = CNNInput(channels, in_channels=4)

        dp_rates = [drop_path_rate * i / max(n_blocks - 1, 1)
                    for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            SERes2Block(channels=channels, scale=scale, kernel_size=kernel_size,
                        dilation=d, dropout=dropout, drop_path=dp_rates[i])
            for i, d in enumerate(dilation_rates)
        ])

        self.mfa  = MFALayer(n_blocks + 1, channels)
        self.pool = AttentiveStatisticsPool(channels)

        # ── Classifier (includes sub-band features) ─────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2 + _N_SUBBAND, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

        # ── Loss ────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

        # ── Metrics ─────────────────────────────────────────────────────
        m_kw = dict(num_classes=num_classes, average="macro")
        self.train_acc     = MulticlassAccuracy(**m_kw)
        self.val_acc       = MulticlassAccuracy(**m_kw)
        self.val_f1        = MulticlassF1Score(**m_kw)
        self.val_precision = MulticlassPrecision(**m_kw)
        self.val_mcc       = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_acc      = MulticlassAccuracy(**m_kw)
        self.test_f1       = MulticlassF1Score(**m_kw)
        self.test_mcc      = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_auroc    = MulticlassAUROC(num_classes=num_classes)
        self.test_cm       = MulticlassConfusionMatrix(num_classes=num_classes)

    # ── Core forward ────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) float32 at sample_rate Hz
        Returns:
            logits: (B, num_classes)
        """
        # Sub-band features computed before any augmentation
        sb = self.sub_band(waveform)              # (B, 9)

        # 4-channel spectrogram
        x = self.frontend(waveform)               # (B, 4, F, T)
        x = self.spec_aug(x)                      # (B, 4, F, T)
        # Note: CepstralLifter is NOT applied here — its log-cepstral output
        # (range ≈ [-22, +10]) is incompatible with the ReLU-based encoder and
        # causes gradient collapse.  The module is kept for future experimentation.

        # Encoder
        x0 = self.cnn_input(x)                    # (B, C, T)
        feat = x0
        block_outputs = [x0]
        for block in self.blocks:
            feat = block(feat)
            block_outputs.append(feat)

        x = self.mfa(*block_outputs)              # (B, C, T)
        x = self.pool(x)                          # (B, 2C)

        # Append sub-band features
        x = torch.cat([x, sb], dim=-1)            # (B, 2C+9)
        return self.classifier(x)                  # (B, num_classes)

    # ── Augmentation ────────────────────────────────────────────────────

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1).item() < self.hparams.gain_prob:
            lo, hi = self.hparams.gain_range
            gain = lo + (hi - lo) * torch.rand(x.shape[0], 1, device=x.device)
            x = x * gain
        if torch.rand(1).item() < self.hparams.noise_prob:
            snr_min, snr_max = self.hparams.noise_snr_db
            snr_db  = snr_min + (snr_max - snr_min) * torch.rand(1, device=x.device)
            snr_lin = 10.0 ** (snr_db / 10.0)
            sig_pwr = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-9)
            noise   = torch.randn_like(x) * (sig_pwr / snr_lin).sqrt()
            x = x + noise
        return x

    # ── Mixup ───────────────────────────────────────────────────────────

    def _mixup(self, x, y):
        alpha = self.hparams.mixup_alpha
        if not self.training or alpha <= 0.0:
            return x, y, y, 1.0
        lam  = torch.distributions.Beta(alpha, alpha).sample().to(x)
        perm = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam

    def _loss(self, logits, y, y_perm=None, lam=1.0):
        if y_perm is None or lam == 1.0:
            return self.criterion(logits, y)
        return (lam * self.criterion(logits, y)
                + (1.0 - lam) * self.criterion(logits, y_perm))

    # ── Lightning steps ─────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y = batch
        x    = self._augment(x)
        x, y, y_p, lam = self._mixup(x, y)
        logits = self(x)
        loss   = self._loss(logits, y, y_p, lam)
        self.train_acc(logits, y)
        self.log("train/loss", loss,           on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        self.val_acc(logits, y);       self.val_f1(logits, y)
        self.val_precision(logits, y); self.val_mcc(logits, y)
        self.log("val/loss",      loss,               on_epoch=True, prog_bar=True)
        self.log("val/acc",       self.val_acc,       on_epoch=True, prog_bar=True)
        self.log("val/f1",        self.val_f1,        on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True, prog_bar=True)
        self.log("val/mcc",       self.val_mcc,       on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        probs  = F.softmax(logits, dim=-1)
        self.test_acc(logits, y);    self.test_f1(logits, y)
        self.test_mcc(logits, y);    self.test_auroc(probs, y)
        self.test_cm(logits, y)
        self.log("test/loss",  self.criterion(logits, y), on_epoch=True)
        self.log("test/acc",   self.test_acc,   on_epoch=True)
        self.log("test/f1",    self.test_f1,    on_epoch=True)
        self.log("test/mcc",   self.test_mcc,   on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix (rows=true, cols=pred):\n{cm.cpu().numpy()}")
        self.test_cm.reset()

    # ── Optimiser ───────────────────────────────────────────────────────

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

        def lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = HydroNet(num_classes=4).to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total:,}")

    x      = torch.randn(4, 32_000, device=device)
    logits = model(x)
    print(f"Input  : {tuple(x.shape)}")
    print(f"Logits : {tuple(logits.shape)}")
    assert logits.shape == (4, 4)
    print("Smoke test passed.")
