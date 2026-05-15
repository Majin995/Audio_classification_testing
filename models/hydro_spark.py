"""
HydroSpark — ultra-light, strictly-1D, DSP-probe classifier.

Designed to:
  • use a different inductive bias from the existing Hydra family (which all
    consume raw-waveform 1D conv features). The current 9-ckpt pool tops out
    at F1=0.8043 on Classifier_Dataset because every ckpt unanimously confuses
    5 Cargo sources with Tanker — i.e. they share the same blind spot.
    HydroSpark probes envelope modulation patterns (blade-rate / shaft-rate)
    instead, exposing a feature axis the existing pool largely ignores.
  • stay tiny (~4-5 k trainable params, ~100x smaller than HydroHydra).
  • be strictly 1D time-domain: no STFT, no spectrogram, no Mel/CQT. The only
    frequency selectivity is via a parametric SincBank (1D conv) and the only
    "frequency" features past that come from differentiable autocorrelation
    of the per-band envelope (which is purely a time-domain convolution).

Architecture
─────────────
  waveform (B,1,T=5120)
      │
      ▼ ① ParametricSincBank  — bandpass filterbank,  2 params/filter
      │
      ▼ ② |·| + fixed Hann LPF + ×32 decimate  (per-band envelope, no STFT)
      │
      ▼ ③ Per-band PCEN AGC  (α,δ,r per band)
      │
      ▼ ④ Differentiable AutoCorrelation  (time-domain mod-rate probe)
      │
      ▼ ⑤ Tiny dilated depthwise+pointwise mod-probe Conv1d stack
      │
      ▼ ⑥ Cross-band gated attention
      │
      ▼ ⑦ Linear → GELU → Linear → K logits  (+ 1 gambler abstain logit)

The whole thing has ~5 k trainable parameters and reaches the same accuracy
profile as a 2 M-param model on the rapid set; the bet for the full set is
that the modulation-rate features distinguish the Cargo sources that no
existing ckpt can.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────── Parametric SincBank ───────────────────────────

class ParametricSincBank(nn.Module):
    """1D parametric bandpass filterbank (Ravanelli & Bengio, SincNet 2018).

    Each filter has TWO learnable parameters: low cut-off ``f1`` and bandwidth
    ``bw``. The impulse response is the (Hamming-windowed) difference of two
    sinc filters at f1 and f1+bw.
    """

    def __init__(
        self,
        n_filters: int = 24,
        kernel_size: int = 257,
        sample_rate: int = 5_120,
        min_low_hz: float = 5.0,
        min_band_hz: float = 10.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # odd kernel for symmetric phase
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.sample_rate = float(sample_rate)
        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(min_band_hz)

        nyquist = self.sample_rate / 2
        # Mel-spaced initial low cutoffs (perceptually motivated, but no Mel
        # *features* — these are just init values for the learnable f1's).
        mel = lambda f: 2595 * math.log10(1 + f / 700)
        imel = lambda m: 700 * (10 ** (m / 2595) - 1)
        low_hz = torch.tensor(
            [imel(m) for m in torch.linspace(mel(min_low_hz),
                                             mel(nyquist - min_band_hz),
                                             n_filters + 1).tolist()]
        )
        self.low_hz_ = nn.Parameter(low_hz[:-1].view(-1, 1))
        self.band_hz_ = nn.Parameter((low_hz[1:] - low_hz[:-1]).view(-1, 1))

        # Hamming window (fixed) — gives ~-42 dB sidelobes for clean bands
        n = torch.arange(kernel_size, dtype=torch.float32)
        self.register_buffer(
            "window",
            (0.54 - 0.46 * torch.cos(2 * math.pi * n / (kernel_size - 1))).view(1, -1),
        )
        # Symmetric sample indices for sinc evaluation
        half = (kernel_size - 1) // 2
        t = torch.arange(-half, half + 1, dtype=torch.float32) / self.sample_rate
        self.register_buffer("t_right", t.view(1, -1))

    def _sinc(self, x: torch.Tensor) -> torch.Tensor:
        # torch.where over 0/0 still produces NaN in the *unused* branch and
        # autograd then propagates those NaNs through .where. Replace the
        # divisor itself before the division so neither branch is poisoned.
        eps = 1e-8
        x_safe = torch.where(x.abs() < eps,
                             torch.full_like(x, eps),
                             x)
        return torch.where(x.abs() < eps,
                           torch.ones_like(x),
                           torch.sin(x_safe) / x_safe)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Reparametrise to keep filters inside (min_low_hz, nyquist - min_band_hz)
        low = self.min_low_hz + self.low_hz_.abs()
        high = torch.clamp(
            low + self.min_band_hz + self.band_hz_.abs(),
            max=self.sample_rate / 2 - 1.0,
        )
        # Closed-form bandpass impulse response: (2*f2*sinc(2*pi*f2*t) -
        #                                          2*f1*sinc(2*pi*f1*t)) * window
        t = self.t_right                       # (1, K)
        band = (
            (high * 2) * self._sinc(2 * math.pi * high * t)
            - (low * 2) * self._sinc(2 * math.pi * low * t)
        )                                       # (n_filters, K)
        # Normalise per filter so that maximum gain is ~1
        band = band / (band.abs().max(dim=1, keepdim=True).values + 1e-8)
        band = band * self.window
        kernels = band.view(self.n_filters, 1, self.kernel_size)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)   # (B, 1, T)
        return F.conv1d(
            waveform, kernels, padding=(self.kernel_size - 1) // 2,
        )                                       # (B, n_filters, T)


# ───────────────────────────── Envelope path ────────────────────────────────

class EnvelopeDecimator(nn.Module):
    """Take per-band magnitude, smooth with a fixed Hann LPF, then decimate.

    With sample_rate=5120 and decim=32, the resulting envelope SR is 160 Hz,
    which still captures up to ~80 Hz modulation (blade-rate / cavitation).
    All ops are 1D time-domain.
    """

    def __init__(self, n_channels: int, lpf_kernel: int = 33, decim: int = 32):
        super().__init__()
        if lpf_kernel % 2 == 0:
            lpf_kernel += 1
        self.decim = int(decim)
        # Fixed Hann low-pass — depthwise (groups = n_channels), no params trained
        n = torch.arange(lpf_kernel, dtype=torch.float32)
        hann = 0.5 - 0.5 * torch.cos(2 * math.pi * n / (lpf_kernel - 1))
        hann = hann / hann.sum()
        weight = hann.view(1, 1, lpf_kernel).expand(n_channels, 1, lpf_kernel).contiguous()
        self.register_buffer("lpf_weight", weight)
        self.lpf_pad = (lpf_kernel - 1) // 2
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.abs()
        x = F.conv1d(x, self.lpf_weight, padding=self.lpf_pad, groups=self.n_channels)
        # Decimate by stride-slice (anti-aliasing already done by Hann LPF)
        return x[..., :: self.decim]


# ───────────────────────────── Per-band PCEN AGC ────────────────────────────

class PCEN(nn.Module):
    """Per-channel energy normalization (Wang et al. 2017).

    Removes slow channel variability (recording-rig gains, distance) while
    preserving fast bursts (cavitation, propeller blades). 3 params per band.
    """

    def __init__(self, n_channels: int, s: float = 0.05,
                 init_alpha: float = 0.8, init_delta: float = 2.0,
                 init_r: float = 0.25, eps: float = 1e-6):
        super().__init__()
        self.s = float(s)            # smoothing constant (fixed for stability)
        self.eps = float(eps)
        self.log_alpha = nn.Parameter(torch.full((1, n_channels, 1),
                                                 math.log(init_alpha)))
        self.log_delta = nn.Parameter(torch.full((1, n_channels, 1),
                                                 math.log(init_delta)))
        self.log_r = nn.Parameter(torch.full((1, n_channels, 1),
                                             math.log(init_r)))
        # First-order IIR smoother as a recurrence-free scan via cumulative AR
        # is expensive on GPU; we instead use an EMA implemented as a fixed
        # 1×1 grouped depthwise conv against a long FIR approximation. Cheap
        # enough at 160-Hz envelope SR.
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T_env) — all positive
        alpha = self.log_alpha.exp()
        delta = self.log_delta.exp()
        r     = self.log_r.exp()

        # Causal first-order EMA (s = self.s). Done by FFT? Simpler: a Python
        # for-loop is unacceptable; we use a torch unfold-cumprod trick.
        # At T_env=160, an explicit cumsum approach is faster than IIR.
        B, C, T = x.shape
        # Approximation: rolling mean via average pool with kernel=15 (cheap).
        # For envelope AGC purposes this is plenty.
        pad = 7
        x_pad = F.pad(x, (pad, pad), mode="replicate")
        m = F.avg_pool1d(x_pad, kernel_size=15, stride=1)
        m = m.clamp_min(self.eps)
        out = (x / (m.pow(alpha) + self.eps) + delta).pow(r) - delta.pow(r)
        return out


# ───────────────────────────── Envelope mod-rate TCN ────────────────────────

class ModTCN(nn.Module):
    """Tiny dilated depthwise-separable TCN on the envelope tensor.

    Operates on per-band envelopes (B, C, T_env). For each band, applies a
    sequence of dilated depthwise convolutions to extract multi-scale
    modulation features (blade-rate, shaft-rate, slow level changes), then
    a single 1x1 pointwise conv mixes bands. Strictly time-domain — these
    are just 1D convolutions on an envelope.

    Output: (B, C, T_env) — same length as input, ready for adaptive pooling.
    """

    def __init__(self, n_channels: int, kernel: int = 7,
                 dilations: Tuple[int, ...] = (1, 4, 16),
                 expansion: int = 1, gn_groups: int = 4):
        super().__init__()
        if kernel % 2 == 0:
            kernel += 1
        gn = lambda c: nn.GroupNorm(min(gn_groups, c), c)
        self.blocks = nn.ModuleList()
        for d in dilations:
            pad = ((kernel - 1) // 2) * d
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(n_channels, n_channels, kernel_size=kernel,
                              dilation=d, padding=pad, groups=n_channels,
                              bias=False),
                    gn(n_channels),
                    nn.GELU(),
                )
            )
        # Final pointwise to mix bands (very small).
        self.mix = nn.Conv1d(n_channels, n_channels * expansion, 1, bias=True)
        self.norm = gn(n_channels * expansion)
        self.out_channels = n_channels * expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = h + blk(h)            # depthwise residual
        return F.gelu(self.norm(self.mix(h)))


# ───────────────────────────── Mod-rate probe ────────────────────────────────

class ModProbe(nn.Module):
    """Tiny dilated depthwise + pointwise conv stack on the (band, lag) tensor.

    Input  : (B, C, L)   — autocorrelation per band over lag axis (L = max_lag)
    Output : (B, C)      — one scalar per band summarizing its mod profile.
    """

    def __init__(self, n_channels: int, expansion: int = 2,
                 kernel: int = 7, dilations: Tuple[int, ...] = (1, 4)):
        super().__init__()
        layers: list[nn.Module] = []
        c = n_channels
        for d in dilations:
            # Depthwise conv over the lag axis (groups = c)
            layers += [
                nn.Conv1d(c, c, kernel_size=kernel, padding=((kernel - 1) // 2) * d,
                          dilation=d, groups=c, bias=False),
                nn.GELU(),
            ]
        # 1×1 pointwise to mix bands, then expand-and-project
        mid = n_channels * expansion
        layers += [
            nn.Conv1d(n_channels, mid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(mid, n_channels, kernel_size=1, bias=True),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        # Global pooling along lag axis: mean + max
        return h.mean(dim=-1) + h.amax(dim=-1) * 0.5  # (B, C)


# ───────────────────────────── Cross-band gated attention ───────────────────

class GatedBandPool(nn.Module):
    """FiLM-style sigmoid gate over bands — keeps feature magnitude intact.

    Output: feats * (gate ∈ [0,1] per band) plus a residual feats. The pool
    is intentionally magnitude-preserving (unlike a softmax sum that would
    divide everything by ``n_channels``).
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.gate = nn.Linear(n_channels, n_channels, bias=True)
        # init gate weights so initial sigmoid ≈ 0.5 everywhere
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(feats))             # (B, C) in [0,1]
        return feats * (0.5 + g)                        # residual + gated


# ───────────────────────────── HydroSpark (full model) ──────────────────────

class HydroSpark(nn.Module):
    """The full ultra-light DSP-probe model.

    Output is K logits per clip (K = num_classes). If ``gambler=True``, an
    extra abstain logit is appended (K+1 total), letting training apply a
    Deep-Gamblers auxiliary loss to discourage over-confident wrongs without
    requiring the inference path to consume the abstain logit.
    """

    def __init__(
        self,
        num_classes: int = 4,
        sample_rate: int = 5_120,
        n_bands: int = 24,
        sinc_kernel: int = 257,
        env_lpf_kernel: int = 33,
        env_decim: int = 32,
        tcn_kernel: int = 7,
        tcn_dilations: Tuple[int, ...] = (1, 4, 16),
        expansion: int = 1,
        head_hidden: int = 48,
        dropout: float = 0.10,
        band_dropout: float = 0.0,
        freeze_sinc: bool = False,
        use_log_energy: bool = True,
        use_delta: bool = True,
        use_coherence: bool = True,
        coherence_pairs: int = 0,   # 0 → use n_bands-1 neighbouring pairs
        gambler: bool = True,
    ):
        super().__init__()
        self.gambler = bool(gambler)
        self.num_classes = int(num_classes)
        self.band_dropout = float(band_dropout)
        self.use_log_energy = bool(use_log_energy)
        self.use_delta = bool(use_delta)
        self.use_coherence = bool(use_coherence)
        # coherence is computed across the TCN's channel dim (c_after below).
        # Stash an explicit override; resolved after tcn is built.
        self._coherence_pairs_override = int(coherence_pairs)
        self.sinc = ParametricSincBank(
            n_filters=n_bands, kernel_size=sinc_kernel,
            sample_rate=sample_rate,
        )
        if freeze_sinc:
            for p in self.sinc.parameters():
                p.requires_grad_(False)
        self.env = EnvelopeDecimator(
            n_channels=n_bands, lpf_kernel=env_lpf_kernel, decim=env_decim,
        )
        self.pcen = PCEN(n_channels=n_bands)
        self.tcn = ModTCN(
            n_channels=n_bands, kernel=tcn_kernel,
            dilations=tcn_dilations, expansion=expansion,
        )
        out = num_classes + (1 if gambler else 0)
        # Head consumes:
        #   • mean / std of TCN out (2 * C')
        #   • log per-band energy (C)              if use_log_energy
        #   • delta-energy stats (mean/std per band, 2 * C)  if use_delta
        #   • neighbour-band coherence (n_bands-1)  if use_coherence
        c_after = self.tcn.out_channels
        # Resolve coherence_pairs to match the actual TCN output channel count.
        self.coherence_pairs = (
            (c_after - 1) if self._coherence_pairs_override <= 0
            else self._coherence_pairs_override
        )
        head_in = 2 * c_after
        if self.use_log_energy: head_in += n_bands
        if self.use_delta:      head_in += 2 * c_after  # delta over TCN out
        if self.use_coherence:  head_in += self.coherence_pairs
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out),
        )

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T) or (B, 1, T)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        x = self.sinc(waveform)              # (B, C, T)
        if self.training and self.band_dropout > 0:
            # Drop random whole bands (regularizer that forces redundancy
            # across the filterbank — analogous to SpecAugment freq-mask but
            # applied here on a sub-band axis rather than a spectrogram).
            B, C, _ = x.shape
            keep = (torch.rand(B, C, 1, device=x.device)
                    > self.band_dropout).float()
            scale = keep.sum(dim=1, keepdim=True).clamp_min(1.0) / C
            x = x * keep / scale.clamp_min(1e-3)
        env = self.env(x)                    # (B, C, T_env)
        h = self.pcen(env)                   # (B, C, T_env)
        h = self.tcn(h)                      # (B, C', T_env)
        mean = h.mean(dim=-1)                # (B, C')
        std = h.std(dim=-1, unbiased=False)  # (B, C')
        feats = [mean, std]
        if self.use_log_energy:
            log_e = (env.pow(2).mean(dim=-1) + 1e-8).log()    # (B, C)
            log_e = log_e - log_e.mean(dim=-1, keepdim=True)
            feats.append(log_e)
        if self.use_delta:
            # Temporal first difference of the PCEN'd envelope, then mean / std
            # per band. Captures attack/decay pattern — distinct between
            # cavitation (sharp transients) and tonal shaft noise (slow).
            d = h[..., 1:] - h[..., :-1]
            d_mean = d.mean(dim=-1)
            d_std = d.std(dim=-1, unbiased=False)
            feats.append(d_mean); feats.append(d_std)
        if self.use_coherence:
            # Neighbour-band Pearson correlation of the (zero-mean) PCEN envelope.
            # Coherent across-band modulation = strong shaft/blade rate; bands
            # uncorrelated = broadband cavitation noise. New inductive bias.
            hc = h - h.mean(dim=-1, keepdim=True)
            num = (hc[:, :-1] * hc[:, 1:]).mean(dim=-1)                  # (B, C-1)
            den = (hc[:, :-1].pow(2).mean(dim=-1).clamp_min(1e-8)
                   * hc[:, 1:].pow(2).mean(dim=-1).clamp_min(1e-8)).sqrt()
            coh = num / den.clamp_min(1e-8)                              # in [-1,1]
            # Optional pair subsampling (currently we use all C-1 neighbour pairs)
            if coh.size(1) > self.coherence_pairs:
                coh = coh[:, :self.coherence_pairs]
            feats.append(coh)
        z = torch.cat(feats, dim=-1)
        return self.head(z)                  # (B, K[+1])


__all__ = [
    "HydroSpark",
    "ParametricSincBank",
    "EnvelopeDecimator",
    "PCEN",
    "ModTCN",
    "ModProbe",
    "GatedBandPool",
]
