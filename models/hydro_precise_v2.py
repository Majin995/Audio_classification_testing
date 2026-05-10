"""
HydroPrecise v2 — Multi-stream UATR classifier with config-flag ablation surface.

Builds on hydro_precise.py (Gabor + CQT + DEMON cross-attention fusion) and adds:
  • SE-Res2 stacks on every branch (configurable depth)
  • Trainable PCEN on the CQT branch
  • Optional Gammatone branch (ERB-spaced, complementary low-frequency resolution)
  • Optional multi-band DEMON
  • Boundary-aware self-attention (onset-biased cross-attention)
  • Optional S4D blocks for narrow-band tonal tracking
  • Optional DART parallel-residual block (local depthwise conv ‖ MHA)
  • Waveform Mixup
  • SpecAugment on every branch (not just Gabor)
  • Auxiliary supervised contrastive head
  • Drops the v1 abstention logit + Deep-Gamblers auxiliary
  • Logit adjustment at evaluation
  • Loss family flag: focal | lmf | ldam | cb_focal

Mean-Teacher / EMA / SWA live in the trainer (training/train_precise_v2.py),
not here, since they are training-recipe additions, not architectural ones.
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TA
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassRecall, MulticlassAUROC, MulticlassConfusionMatrix,
    MulticlassMatthewsCorrCoef,
)

from models.hydro_catfish    import LearnableGaborFilterbank, SpecAugment1D
from models.hydro_conformer  import TrainablePCEN, FocalLoss
from models.hydro_net        import GammatoneSpectrogram, _LinearModulationSpec, _DEMONGram
from models.hydro_fusion     import _SERes2Block, _AttentiveStatisticsPool
from models.hydro_s4         import SaShiMiBlock
from models.hydro_dart_mt    import DARTBlock
from models.hydro_bahtnet    import SpectralFluxOnset, BoundaryAwareAttention
from models.heads             import build_head, PrototypeHead
from processing.losses       import (
    LargeMarginFocalLoss, LDAMLoss, ClassBalancedFocalLoss,
)


# ═══════════════════════════════════════════════════════════════════════
#  Augmentation
# ═══════════════════════════════════════════════════════════════════════

class _WaveformAug(nn.Module):
    """Waveform-level augmentation. Train-only by default; pass
    ``force_train=True`` from TTA to fire while the model is in eval mode.

    Pipeline (each step independent, gated by its own probability):
      1. Corpus ocean-noise injection (mixed at random SNR in dB).
      2. Synthetic Gaussian-noise injection (legacy). When corpus noise fires,
         synthetic is auto-suppressed to avoid SNR stacking.
      3. RIR / multipath convolution (random 3-5 tap delay line).
      4. Pitch ±``pitch_range`` via interpolate-resample (cheap; co-shifts
         duration ~ pitch_range %).
      5. Random gain (legacy).
    """

    def __init__(
        self,
        noise_prob:    float = 0.3,
        noise_snr_min: float = 15.0,
        noise_snr_max: float = 30.0,
        gain_prob:     float = 0.6,
        gain_range:    float = 0.3,
        # ── Corpus noise (NEW) ─────────────────────────────────────────
        ocean_noise_pool=None,                              # OceanNoisePool | None
        corpus_noise_prob:    float = 0.0,
        corpus_noise_snr_min: float = -3.0,
        corpus_noise_snr_max: float = 15.0,
        # ── RIR / multipath (NEW) ──────────────────────────────────────
        rir_prob:        float = 0.0,
        rir_max_delay_s: float = 0.030,
        rir_min_taps:    int   = 3,
        rir_max_taps:    int   = 5,
        sample_rate:     int   = 5_120,
        # ── Pitch shift (NEW) ──────────────────────────────────────────
        pitch_prob:  float = 0.0,
        pitch_range: float = 0.015,                         # ±1.5 %
    ):
        super().__init__()
        self.noise_prob    = noise_prob
        self.noise_snr_min = noise_snr_min
        self.noise_snr_max = noise_snr_max
        self.gain_prob     = gain_prob
        self.gain_range    = gain_range
        # Corpus noise
        self.ocean_noise_pool     = ocean_noise_pool
        self.corpus_noise_prob    = corpus_noise_prob
        self.corpus_noise_snr_min = corpus_noise_snr_min
        self.corpus_noise_snr_max = corpus_noise_snr_max
        # RIR
        self.rir_prob        = rir_prob
        self.rir_max_delay_s = rir_max_delay_s
        self.rir_min_taps    = max(1, int(rir_min_taps))
        self.rir_max_taps    = max(self.rir_min_taps, int(rir_max_taps))
        self.sample_rate     = int(sample_rate)
        # Pitch
        self.pitch_prob  = pitch_prob
        self.pitch_range = float(pitch_range)

    # ── primitives ─────────────────────────────────────────────────────

    def _add_corpus_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.ocean_noise_pool is None:
            return x
        n = self.ocean_noise_pool.sample(x.size(0), x.device).to(x.dtype)
        if n.shape[-1] != x.shape[-1]:
            n = n[..., : x.shape[-1]] if n.shape[-1] > x.shape[-1] \
                else F.pad(n, (0, x.shape[-1] - n.shape[-1]))
        snr = self.corpus_noise_snr_min + (
            self.corpus_noise_snr_max - self.corpus_noise_snr_min
        ) * torch.rand(1).item()
        sig_p = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12)
        n_p   = n.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = (sig_p / (10.0 ** (snr / 10.0)) / n_p).sqrt()
        return x + n * scale

    def _add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        snr = self.noise_snr_min + (
            self.noise_snr_max - self.noise_snr_min
        ) * torch.rand(1).item()
        sig_pow   = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12)
        noise_pow = sig_pow / (10.0 ** (snr / 10.0))
        return x + torch.randn_like(x) * noise_pow.sqrt()

    def _apply_rir(self, x: torch.Tensor) -> torch.Tensor:
        max_delay_n = max(1, int(self.rir_max_delay_s * self.sample_rate))
        n_taps = int(torch.randint(self.rir_min_taps, self.rir_max_taps + 1, (1,)).item())
        delays = torch.randint(0, max_delay_n, (n_taps,))
        delays[0] = 0                                       # direct path
        gains = torch.tensor([
            math.exp(-0.1 * i) * (0.5 + 0.5 * float(torch.rand(1).item()))
            for i in range(n_taps)
        ])
        L = int(delays.max().item()) + 1
        h = torch.zeros(L, device=x.device, dtype=x.dtype)
        for d, g in zip(delays.tolist(), gains.tolist()):
            h[d] = h[d] + g
        denom = h.abs().sum().clamp(min=1e-6)
        h = h / denom                                       # roughly preserve RMS
        T = x.size(-1)
        Hf = torch.fft.rfft(F.pad(h, (0, T - 1)).float())
        Xf = torch.fft.rfft(F.pad(x, (0, L - 1)).float())
        y  = torch.fft.irfft(Xf * Hf, n=T + L - 1)[..., :T]
        return y.to(x.dtype)

    def _apply_pitch(self, x: torch.Tensor) -> torch.Tensor:
        ratio = 1.0 + (2.0 * float(torch.rand(1).item()) - 1.0) * self.pitch_range
        T     = x.size(-1)
        new_T = max(1, int(round(T * ratio)))
        if new_T == T:
            return x
        y = F.interpolate(
            x.unsqueeze(1).float(), size=new_T,
            mode="linear", align_corners=False,
        ).squeeze(1).to(x.dtype)
        if new_T >= T:
            return y[..., :T]
        return F.pad(y, (0, T - new_T))

    # ── forward ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, force_train: bool = False) -> torch.Tensor:
        if not (self.training or force_train):
            return x

        # Corpus noise (priority over synthetic Gaussian to avoid stacking).
        corpus_fired = False
        if (
            self.ocean_noise_pool is not None
            and self.corpus_noise_prob > 0.0
            and torch.rand(1).item() < self.corpus_noise_prob
        ):
            x = self._add_corpus_noise(x)
            corpus_fired = True

        if not corpus_fired and torch.rand(1).item() < self.noise_prob:
            x = self._add_gaussian_noise(x)

        if self.rir_prob > 0.0 and torch.rand(1).item() < self.rir_prob:
            x = self._apply_rir(x)

        if self.pitch_prob > 0.0 and torch.rand(1).item() < self.pitch_prob:
            x = self._apply_pitch(x)

        if torch.rand(1).item() < self.gain_prob:
            g = 1.0 + (2.0 * float(torch.rand(1).item()) - 1.0) * self.gain_range
            x = x * g
        return x


class _BranchDropout(nn.Module):
    """Stochastic per-batch zeroing of one branch's pooled feature tensor.

    Forces the fusion layer to learn a representation robust to losing any
    single branch. With probability ``p`` (and only when training) one of the
    feature tensors in the input list is replaced with zeros (NOT removed) —
    this preserves the static channel count expected by ``fuse_proj``.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, feats):
        if not self.training or self.p <= 0.0 or len(feats) <= 1:
            return feats
        if float(torch.rand(1).item()) >= self.p:
            return feats
        idx = int(torch.randint(0, len(feats), (1,)).item())
        out = list(feats)
        out[idx] = torch.zeros_like(out[idx])
        return out


# ═══════════════════════════════════════════════════════════════════════
#  CQT front-end (with optional PCEN)
# ═══════════════════════════════════════════════════════════════════════

class _CQTFrontend(nn.Module):
    def __init__(
        self,
        sample_rate:     int   = 5_120,
        n_bins:          int   = 96,
        bins_per_octave: int   = 12,
        hop_length:      int   = 64,
        fmin:            float = 20.0,
        use_pcen:        bool  = True,
    ):
        super().__init__()
        self.n_bins = n_bins
        try:
            from nnAudio.Spectrogram import CQT1992v2
            self.cqt = CQT1992v2(
                sr=sample_rate, hop_length=hop_length, fmin=fmin,
                n_bins=n_bins, bins_per_octave=bins_per_octave,
                output_format="Magnitude", verbose=False,
            )
            self._use_cqt = True
        except Exception:
            self.cqt = TA.MelSpectrogram(
                sample_rate=sample_rate, n_fft=512, hop_length=hop_length,
                n_mels=n_bins, f_min=fmin, f_max=sample_rate / 2.0, power=1.0,
            )
            self._use_cqt = False
        self.use_pcen = use_pcen
        if use_pcen:
            self.norm = TrainablePCEN(n_bins)
        else:
            self.norm = nn.InstanceNorm1d(n_bins, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.cqt(x).abs().clamp(min=1e-9)
        if self.use_pcen:
            return self.norm(spec)
        return self.norm(torch.log1p(spec))


class _CQT2DBackbone(nn.Module):
    """Two strided 2D conv stages → collapse freq into channels → (B, C_out, T)."""

    def __init__(self, in_freq: int, base_ch: int = 32, out_ch: int = 128):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1,       base_ch,     3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.GELU(),
            nn.Conv2d(base_ch, base_ch,     3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_ch,     base_ch * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2), nn.GELU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2), nn.GELU(),
        )
        freq_after = math.ceil(in_freq / 4)
        self.project = nn.Conv1d(base_ch * 2 * freq_after, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                               # (B, 1, F, T)
        x = self.stage2(self.stage1(x))
        B, C, Fm, T = x.shape
        x = x.reshape(B, C * Fm, T)
        return self.project(x)                           # (B, out_ch, T)


# ═══════════════════════════════════════════════════════════════════════
#  Multi-band DEMON (per-band FFT bandpass → square → linear-mod-spec + PCEN)
# ═══════════════════════════════════════════════════════════════════════

class _MultiBandDEMON(nn.Module):
    """
    Per-subband DEMON with linear modulation spectrogram.

    For each (lo, hi) subband: bandpass via FFT mask → square → linear-frequency
    STFT magnitude truncated to [mod_f_min, mod_f_max] → trainable PCEN. Output
    is the per-subband features concatenated along the channel axis.

    The linear scale replaces the previous mel filterbank: a mel bank devotes
    almost all of its bins to the [50 Hz–Nyquist] range, while DEMON's
    discriminative content lives at 1–~30 Hz (BPF + harmonics out to ~250 Hz).
    PCEN is bin-agnostic — its parameters are per-row scalars and work
    identically on linear and mel bins.
    """

    def __init__(
        self,
        sample_rate: int,
        hop_length:  int,
        subbands:    Sequence[Tuple[float, float]],
        n_fft:       int   = 2048,
        mod_f_min:   float = 0.0,
        mod_f_max:   float = 50.0,
        envelope:    str   = "square",   # "square" | "hilbert" | "fwr"
        decimate:    int   = 1,
    ):
        super().__init__()
        if envelope not in ("square", "hilbert", "fwr"):
            raise ValueError(f"_MultiBandDEMON envelope: {envelope!r}")
        self.sample_rate = sample_rate
        self.subbands    = list(subbands)
        self.envelope    = envelope
        if envelope == "square":
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
        self.pcens = nn.ModuleList(
            [TrainablePCEN(self.spec.n_bins) for _ in self.subbands]
        )

    @property
    def n_bins(self) -> int:
        return self.spec.n_bins

    @property
    def out_channels(self) -> int:
        return self.spec.n_bins * len(self.subbands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wf32  = x.float()
        T_len = x.shape[-1]
        X     = torch.fft.rfft(wf32, dim=-1)
        freqs = torch.fft.rfftfreq(T_len, d=1.0 / self.sample_rate, device=x.device)
        outs = []
        nyq = self.sample_rate / 2.0
        for (lo, hi), pcen in zip(self.subbands, self.pcens):
            mask = ((freqs >= lo) & (freqs <= min(hi, nyq))).float()
            x_bp = torch.fft.irfft(X * mask, n=T_len, dim=-1).to(x.dtype)
            if self.envelope == "square":
                ms = self.spec(x_bp ** 2).clamp(min=1e-9)
            else:
                ms = self.spec(x_bp).clamp(min=1e-9)
            outs.append(pcen(ms))
        return torch.cat(outs, dim=1)                    # (B, n_bins * n_bands, T)


# ═══════════════════════════════════════════════════════════════════════
#  Branch wrappers
# ═══════════════════════════════════════════════════════════════════════

def _seres2_stack(channels: int, n_blocks: int, dropout: float, drop_path: float) -> nn.Sequential:
    """SE-Res2 stack with linearly increasing dilation, scale=8 (channels must be /8)."""
    blocks = []
    for i in range(n_blocks):
        blocks.append(_SERes2Block(
            channels=channels, scale=8, kernel_size=3,
            dilation=2 * (i + 1), dropout=dropout, drop_path=drop_path,
        ))
    return nn.Sequential(*blocks)


class _GaborBranch(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_filters:   int   = 96,
        out_ch:      int   = 192,
        kernel_size: int   = 257,
        n_blocks:    int   = 2,
        spec_aug:    bool  = True,
        dropout:     float = 0.15,
        drop_path:   float = 0.0,
    ):
        super().__init__()
        self.filterbank = LearnableGaborFilterbank(
            n_filters=n_filters, kernel_size=kernel_size, sample_rate=sample_rate,
        )
        self.spec_aug = SpecAugment1D(
            n_freq_masks=2, freq_mask_max=6, n_time_masks=2, time_mask_max=80,
        ) if spec_aug else nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv1d(n_filters, out_ch, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 5, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.blocks = _seres2_stack(out_ch, n_blocks, dropout, drop_path)

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.filterbank(waveform)
        if training:
            x = self.spec_aug(x)
        x = self.stem(x)
        return self.blocks(x)


class _CQTBranch(nn.Module):
    def __init__(
        self,
        sample_rate:     int,
        n_bins:          int  = 96,
        bins_per_octave: int  = 12,
        hop_length:      int  = 64,
        out_ch:          int  = 192,
        n_blocks:        int  = 2,
        spec_aug:        bool = True,
        use_pcen:        bool = True,
        dropout:         float = 0.15,
        drop_path:       float = 0.0,
    ):
        super().__init__()
        self.front = _CQTFrontend(
            sample_rate=sample_rate, n_bins=n_bins,
            bins_per_octave=bins_per_octave, hop_length=hop_length,
            use_pcen=use_pcen,
        )
        self.spec_aug = SpecAugment1D(
            n_freq_masks=2, freq_mask_max=8, n_time_masks=2, time_mask_max=15,
        ) if spec_aug else nn.Identity()
        self.back = _CQT2DBackbone(in_freq=n_bins, base_ch=32, out_ch=out_ch)
        self.blocks = _seres2_stack(out_ch, n_blocks, dropout, drop_path)

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        spec = self.front(waveform)                      # (B, F, T)
        if training:
            spec = self.spec_aug(spec)
        x = self.back(spec)                              # (B, out_ch, T)
        return self.blocks(x)


class _DEMONBranch(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        hop_length:  int   = 64,
        out_ch:      int   = 128,
        n_blocks:    int   = 1,
        subbands:    Sequence[Tuple[float, float]] = ((800.0, 2560.0),),
        n_fft:       int   = 2048,
        mod_f_min:   float = 0.0,
        mod_f_max:   float = 50.0,
        envelope:    str   = "square",
        decimate:    int   = 1,
        spec_aug:    bool  = True,
        dropout:     float = 0.15,
        drop_path:   float = 0.0,
    ):
        super().__init__()
        self.demon = _MultiBandDEMON(
            sample_rate=sample_rate, hop_length=hop_length, subbands=subbands,
            n_fft=n_fft, mod_f_min=mod_f_min, mod_f_max=mod_f_max,
            envelope=envelope, decimate=decimate,
        )
        in_dim = self.demon.out_channels
        self.spec_aug = SpecAugment1D(
            n_freq_masks=1, freq_mask_max=4, n_time_masks=2, time_mask_max=15,
        ) if spec_aug else nn.Identity()
        self.proj = nn.Sequential(
            nn.Conv1d(in_dim, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.blocks = _seres2_stack(out_ch, n_blocks, dropout, drop_path)
        gru_h = out_ch // 2
        self.gru = nn.GRU(out_ch, gru_h, num_layers=1, batch_first=True, bidirectional=True)
        self.out_ch = gru_h * 2

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.demon(waveform)                         # (B, n_mels*n_bands, T)
        if training:
            x = self.spec_aug(x)
        x = self.proj(x)
        x = self.blocks(x)
        x = x.transpose(1, 2)                            # (B, T, C)
        x, _ = self.gru(x)
        return x.transpose(1, 2)                         # (B, 2*gru_h, T)


class _GammatoneBranch(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_bands:     int   = 64,
        n_fft:       int   = 512,
        hop_length:  int   = 64,
        out_ch:      int   = 128,
        n_blocks:    int   = 1,
        spec_aug:    bool  = True,
        dropout:     float = 0.15,
        drop_path:   float = 0.0,
    ):
        super().__init__()
        self.gammatone = GammatoneSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_bands=n_bands, f_min=20.0,
        )
        self.pcen = TrainablePCEN(n_bands)
        self.spec_aug = SpecAugment1D(
            n_freq_masks=2, freq_mask_max=6, n_time_masks=2, time_mask_max=15,
        ) if spec_aug else nn.Identity()
        self.proj = nn.Sequential(
            nn.Conv1d(n_bands, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.blocks = _seres2_stack(out_ch, n_blocks, dropout, drop_path)

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.gammatone(waveform).clamp(min=1e-9)
        x = self.pcen(x)
        if training:
            x = self.spec_aug(x)
        x = self.proj(x)
        return self.blocks(x)


class _PretrainedBranch(nn.Module):
    """Frozen pretrained acoustic encoder as a 6th feature stream.

    Uses the *conv front-end only* of a pretrained wav2vec2 model — the
    transformer is dropped to keep parameter count low (~4M frozen vs.
    ~95M for the full model). Output is projected to ``out_ch`` and run
    through one SE-Res2 block, mirroring the other branches.

    Resamples 5120 → 16 kHz internally (wav2vec2's expected SR). Aliasing
    irrelevant downstream of the conv stack's stride-10 first layer.

    Trainable params: ``proj + blocks`` (~0.3M). The pretrained extractor
    is frozen and put in ``eval()``.
    """

    def __init__(
        self,
        sample_rate: int   = 5_120,
        out_ch:      int   = 128,
        model_name:  str   = "facebook/wav2vec2-base",
        spec_aug:    bool  = True,
        dropout:     float = 0.15,
        drop_path:   float = 0.0,
        target_sr:   int   = 16_000,
    ):
        super().__init__()
        from transformers import Wav2Vec2Model
        w2v = Wav2Vec2Model.from_pretrained(model_name)
        # Keep only the conv front-end (CNN); discard pos-conv-embed + transformer.
        self.feature_extractor = w2v.feature_extractor
        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)
        self.feature_extractor.eval()
        self.in_sr     = int(sample_rate)
        self.out_sr    = int(target_sr)
        # The conv stack outputs 512 channels in wav2vec2-base.
        in_ch = 512
        self.spec_aug = SpecAugment1D(
            n_freq_masks=2, freq_mask_max=16, n_time_masks=1, time_mask_max=4,
        ) if spec_aug else nn.Identity()
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.blocks = _seres2_stack(out_ch, 1, dropout, drop_path)
        self.out_ch = int(out_ch)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen pretrained extractor in eval mode regardless of
        # parent's training flag — its BN running stats must not update.
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
            feat = self.feature_extractor(x16)              # (B, 512, T')
        if training:
            feat = self.spec_aug(feat)
        feat = self.proj(feat)
        return self.blocks(feat)


class _LOFARBranch(nn.Module):
    """High-resolution linear-narrowband branch — slots in next to Gammatone
    as the 5th HydroPreciseV2 stream. Complements CQT (log-spaced) by giving
    uniform linear resolution at low frequencies where machinery tonals live."""

    def __init__(
        self,
        sample_rate: int,
        n_bins:      int   = 256,
        n_fft:       int   = 4_096,
        hop_length:  int   = 160,
        max_freq:    float = 2_560.0,
        out_ch:      int   = 128,
        n_blocks:    int   = 1,
        spec_aug:    bool  = True,
        dropout:     float = 0.15,
        drop_path:   float = 0.0,
    ):
        super().__init__()
        from models.hydro_net import _LOFARSpec
        self.lofar = _LOFARSpec(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            max_freq=max_freq, freq_bins=n_bins,
        )
        self.spec_aug = SpecAugment1D(
            n_freq_masks=2, freq_mask_max=8, n_time_masks=2, time_mask_max=4,
        ) if spec_aug else nn.Identity()
        self.proj = nn.Sequential(
            nn.Conv1d(n_bins, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.blocks = _seres2_stack(out_ch, n_blocks, dropout, drop_path)

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.lofar(waveform)                           # (B, n_bins, T)
        if training:
            x = self.spec_aug(x)
        x = self.proj(x)
        return self.blocks(x)


# ═══════════════════════════════════════════════════════════════════════
#  Fused attention block (boundary-aware optional)
# ═══════════════════════════════════════════════════════════════════════

class _FusedAttnBlock(nn.Module):
    """Pre-norm self-attention + FFN; optional onset-biased attention."""

    def __init__(
        self,
        d_model:      int,
        n_heads:      int   = 4,
        dropout:      float = 0.15,
        use_boundary: bool  = True,
    ):
        super().__init__()
        self.use_boundary = use_boundary
        self.ln1 = nn.LayerNorm(d_model)
        if use_boundary:
            self.attn = BoundaryAwareAttention(d_model, n_heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
        self.attn_drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor, onset_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        if self.use_boundary:
            h = self.attn(h, onset_mask)
        else:
            h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_drop(h)
        x = x + self.ff(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Supervised contrastive loss
# ═══════════════════════════════════════════════════════════════════════

def _supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """SupCon (Khosla et al. NeurIPS 2020). features: (B, D), L2-normalized."""
    B = features.size(0)
    if B < 2:
        return features.sum() * 0.0
    sim = features @ features.t() / temperature                  # (B, B)
    self_mask = torch.eye(B, dtype=torch.bool, device=features.device)
    sim = sim.masked_fill(self_mask, -1e9)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & ~self_mask               # (B, B)
    log_prob = F.log_softmax(sim, dim=1)
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_count
    return -mean_log_prob_pos[pos_count > 0].mean() if (pos_count > 0).any() else features.sum() * 0.0


# ═══════════════════════════════════════════════════════════════════════
#  HydroPreciseV2
# ═══════════════════════════════════════════════════════════════════════

class HydroPreciseV2(pl.LightningModule):
    """v2 high-precision UATR classifier — ablation surface for hydro_precise."""

    def __init__(
        self,
        # ── Data ────────────────────────────────────────────────────────
        num_classes:       int   = 4,
        class_weights:     Optional[List[float]] = None,
        cls_num_list:      Optional[List[float]] = None,
        sample_rate:       int   = 5_120,
        # ── Branches ────────────────────────────────────────────────────
        gabor_n_filters:   int   = 96,
        gabor_kernel:      int   = 257,
        gabor_ch:          int   = 192,
        cqt_n_bins:        int   = 96,
        cqt_bpo:           int   = 12,
        cqt_hop:           int   = 64,
        cqt_ch:            int   = 192,
        pcen_on_cqt:       bool  = True,
        demon_hop:         int   = 64,
        demon_ch:          int   = 128,
        demon_subbands:    Optional[List[Tuple[float, float]]] = None,
        demon_n_fft:       int   = 2048,
        demon_mod_f_min:   float = 0.0,
        demon_mod_f_max:   float = 50.0,
        demon_envelope:    str   = "square",   # "square" | "hilbert" | "fwr"
        demon_decimate:    int   = 1,
        use_gammatone_branch: bool = False,
        gammatone_n_bands: int   = 64,
        gammatone_ch:      int   = 128,
        use_lofar_branch:  bool  = False,
        lofar_n_bins:      int   = 256,
        lofar_n_fft:       int   = 4_096,
        lofar_hop:         int   = 160,
        lofar_max_freq:    float = 2_560.0,
        lofar_ch:          int   = 128,
        lofar_n_blocks:    int   = 1,
        use_pretrained_branch: bool = False,
        pretrained_model:      str = "facebook/wav2vec2-base",
        pretrained_ch:         int = 128,
        pretrained_target_sr:  int = 16_000,
        seres2_blocks_per_branch: Tuple[int, int, int, int] = (2, 2, 1, 1),
        spec_aug_all_branches: bool = True,
        # ── Fusion ──────────────────────────────────────────────────────
        fusion_T:          int   = 64,
        fusion_dim:        int   = 256,
        n_heads:           int   = 4,
        n_attn_blocks:     int   = 1,
        use_boundary_attn: bool  = True,
        use_dart_block:    bool  = False,
        n_s4d_blocks:      int   = 1,
        s4d_d_state:       int   = 64,
        dropout:           float = 0.25,
        drop_path:         float = 0.10,
        # ── Loss ────────────────────────────────────────────────────────
        loss:              str   = "focal",      # "focal" | "lmf" | "ldam" | "cb_focal"
        focal_gamma:       float = 2.0,
        lmf_margin:        float = 0.5,
        ldam_max_m:        float = 0.5,
        ldam_s:            float = 30.0,
        cb_beta:           float = 0.999,
        label_smoothing:   float = 0.05,
        aux_supcon_weight: float = 0.1,
        supcon_temperature: float = 0.07,
        supcon_proj_dim:   int   = 128,
        logit_adjust_tau:  float = 0.0,
        # ── Mixup (waveform) ────────────────────────────────────────────
        mixup_alpha:       float = 0.2,
        # ── Mean-Teacher (EMA self-distillation) ────────────────────────
        mean_teacher_weight:    float = 0.5,
        mean_teacher_ema_decay: float = 0.999,
        mean_teacher_rampup_epochs: int = 10,
        # ── Waveform aug ────────────────────────────────────────────────
        noise_prob:        float = 0.3,
        noise_snr_min:     float = 15.0,
        noise_snr_max:     float = 30.0,
        gain_prob:         float = 0.6,
        gain_range:        float = 0.3,
        # ── Domain-aware aug (NEW) ──────────────────────────────────────
        ocean_noise_pool=None,                              # OceanNoisePool | None
        corpus_noise_prob:    float = 0.0,
        corpus_noise_snr_min: float = -3.0,
        corpus_noise_snr_max: float = 15.0,
        rir_prob:           float = 0.0,
        rir_max_delay_s:    float = 0.030,
        pitch_prob:         float = 0.0,
        pitch_range:        float = 0.015,
        branch_dropout_p:   float = 0.0,
        # ── Optimiser ───────────────────────────────────────────────────
        learning_rate:     float = 3e-4,
        weight_decay:      float = 1e-3,
        warmup_epochs:     int   = 5,
        max_epochs:        int   = 100,
        ssm_lr_mult:       float = 0.1,
        # ── Head + embedding-norm surface ───────────────────────────────
        head_type:         str   = "mlp",      # "mlp" | "cosine" | "prototype" | "arcface" | "mlp_wide"
        feature_norm:      str   = "none",     # "none" | "layernorm_l2"
        arcface_margin:    float = 0.2,
        arcface_scale:     float = 30.0,
        cosine_scale_init: float = 10.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "cls_num_list", "ocean_noise_pool"])

        if demon_subbands is None:
            demon_subbands = [(800.0, sample_rate / 2.0)]

        self.num_classes      = num_classes
        self.fusion_T         = fusion_T
        self.use_boundary     = use_boundary_attn
        self.use_dart         = use_dart_block
        self.n_s4d            = n_s4d_blocks
        self.aux_supcon_weight = aux_supcon_weight
        self.supcon_temperature = supcon_temperature
        self.mixup_alpha       = mixup_alpha
        self.mean_teacher_weight = mean_teacher_weight
        self.mean_teacher_ema_decay = mean_teacher_ema_decay
        self.mean_teacher_rampup_epochs = mean_teacher_rampup_epochs

        # ── Logit adjustment prior (set later via set_class_prior) ──────
        if class_weights is not None and logit_adjust_tau > 0:
            # class_weights are inverse-frequency; recover ~prior
            cw = torch.tensor(class_weights, dtype=torch.float32)
            prior = 1.0 / cw
            prior = prior / prior.sum()
            self.register_buffer("log_prior", torch.log(prior + 1e-12))
        else:
            self.register_buffer("log_prior", torch.zeros(num_classes))
        self.logit_adjust_tau = logit_adjust_tau

        # ── Augmentation ────────────────────────────────────────────────
        self.wave_aug = _WaveformAug(
            noise_prob=noise_prob, noise_snr_min=noise_snr_min,
            noise_snr_max=noise_snr_max,
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

        # ── Branches ────────────────────────────────────────────────────
        n_gab, n_cqt, n_dem, n_gam = seres2_blocks_per_branch
        self.branch_a = _GaborBranch(
            sample_rate=sample_rate, n_filters=gabor_n_filters,
            out_ch=gabor_ch, kernel_size=gabor_kernel, n_blocks=n_gab,
            spec_aug=spec_aug_all_branches, dropout=dropout, drop_path=drop_path,
        )
        self.branch_b = _CQTBranch(
            sample_rate=sample_rate, n_bins=cqt_n_bins,
            bins_per_octave=cqt_bpo, hop_length=cqt_hop, out_ch=cqt_ch,
            n_blocks=n_cqt, spec_aug=spec_aug_all_branches,
            use_pcen=pcen_on_cqt, dropout=dropout, drop_path=drop_path,
        )
        self.branch_c = _DEMONBranch(
            sample_rate=sample_rate,
            hop_length=demon_hop, out_ch=demon_ch, n_blocks=n_dem,
            subbands=demon_subbands,
            n_fft=demon_n_fft, mod_f_min=demon_mod_f_min, mod_f_max=demon_mod_f_max,
            envelope=demon_envelope, decimate=demon_decimate,
            spec_aug=spec_aug_all_branches,
            dropout=dropout, drop_path=drop_path,
        )
        if use_lofar_branch:
            self.branch_e = _LOFARBranch(
                sample_rate=sample_rate, n_bins=lofar_n_bins,
                n_fft=lofar_n_fft, hop_length=lofar_hop,
                max_freq=lofar_max_freq,
                out_ch=lofar_ch, n_blocks=lofar_n_blocks,
                spec_aug=spec_aug_all_branches,
                dropout=dropout, drop_path=drop_path,
            )
            e_branch_ch = lofar_ch
        else:
            self.branch_e = None
            e_branch_ch = 0

        if use_gammatone_branch:
            self.branch_d = _GammatoneBranch(
                sample_rate=sample_rate, n_bands=gammatone_n_bands,
                out_ch=gammatone_ch, n_blocks=n_gam,
                spec_aug=spec_aug_all_branches,
                dropout=dropout, drop_path=drop_path,
            )
            d_branch_ch = gammatone_ch
        else:
            self.branch_d = None
            d_branch_ch = 0

        if use_pretrained_branch:
            self.branch_f = _PretrainedBranch(
                sample_rate=sample_rate, out_ch=pretrained_ch,
                model_name=pretrained_model,
                spec_aug=spec_aug_all_branches,
                dropout=dropout, drop_path=drop_path,
                target_sr=pretrained_target_sr,
            )
            f_branch_ch = pretrained_ch
        else:
            self.branch_f = None
            f_branch_ch = 0

        # ── Onset detector (computed on Gabor envelope) ─────────────────
        self.onset = SpectralFluxOnset(patch_size=1, std_mult=1.5) if use_boundary_attn else None

        # ── Fusion projection ───────────────────────────────────────────
        cat_ch = gabor_ch + cqt_ch + self.branch_c.out_ch + d_branch_ch + e_branch_ch + f_branch_ch
        self.fuse_proj = nn.Sequential(
            nn.Conv1d(cat_ch, fusion_dim, 1, bias=False),
            nn.BatchNorm1d(fusion_dim), nn.GELU(),
        )

        # ── Sequence-level blocks (fused attention, S4D, optional DART) ─
        self.attn_blocks = nn.ModuleList([
            _FusedAttnBlock(
                d_model=fusion_dim, n_heads=n_heads,
                dropout=dropout, use_boundary=use_boundary_attn,
            )
            for _ in range(n_attn_blocks)
        ])
        self.s4d_blocks = nn.ModuleList([
            SaShiMiBlock(
                d_model=fusion_dim, d_state=s4d_d_state,
                dropout=dropout, drop_path=drop_path * 0.5,
            )
            for _ in range(n_s4d_blocks)
        ])
        if use_dart_block:
            self.dart_block = DARTBlock(
                dim=fusion_dim, n_heads=n_heads, kernel_size=15,
                ff_expansion=2, dropout=dropout, attn_drop=dropout, drop_path=drop_path,
            )
        else:
            self.dart_block = None
        self.final_norm = nn.LayerNorm(fusion_dim)

        # ── Pool + (optional) embedding norm + classification head ──────
        self.pool = _AttentiveStatisticsPool(fusion_dim)

        self.feature_norm_kind = feature_norm
        if feature_norm == "layernorm_l2":
            self.feat_norm = nn.LayerNorm(fusion_dim * 2)
        elif feature_norm == "none":
            self.feat_norm = None
        else:
            raise ValueError(f"Unknown feature_norm: {feature_norm!r}")

        self.head = build_head(
            name=head_type,
            in_dim=fusion_dim * 2,
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            dropout=dropout,
            arcface_margin=arcface_margin,
            arcface_scale=arcface_scale,
            cosine_scale_init=cosine_scale_init,
        )
        self.head_type = head_type

        # ── SupCon projection head ──────────────────────────────────────
        if aux_supcon_weight > 0.0:
            self.supcon_head = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.GELU(),
                nn.Linear(fusion_dim, supcon_proj_dim),
            )
        else:
            self.supcon_head = None

        # ── Loss ────────────────────────────────────────────────────────
        self.criterion = self._build_loss(
            loss=loss, num_classes=num_classes, class_weights=class_weights,
            cls_num_list=cls_num_list, focal_gamma=focal_gamma,
            lmf_margin=lmf_margin, ldam_max_m=ldam_max_m, ldam_s=ldam_s,
            cb_beta=cb_beta, label_smoothing=label_smoothing,
        )

        # ── Metrics ─────────────────────────────────────────────────────
        m_macro = dict(num_classes=num_classes, average="macro")
        m_micro = dict(num_classes=num_classes, average="micro")
        self.train_acc           = MulticlassAccuracy(**m_macro)
        self.val_acc             = MulticlassAccuracy(**m_macro)
        self.val_f1              = MulticlassF1Score(**m_macro)
        self.val_recall          = MulticlassRecall(**m_macro)
        self.val_precision_macro = MulticlassPrecision(**m_macro)
        self.val_precision_micro = MulticlassPrecision(**m_micro)
        self.val_precision_per   = MulticlassPrecision(num_classes=num_classes, average=None)
        self.val_mcc             = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.val_auroc           = MulticlassAUROC(num_classes=num_classes)
        self.test_acc            = MulticlassAccuracy(**m_macro)
        self.test_f1             = MulticlassF1Score(**m_macro)
        self.test_precision      = MulticlassPrecision(**m_macro)
        self.test_precision_micro = MulticlassPrecision(**m_micro)
        self.test_recall         = MulticlassRecall(**m_macro)
        self.test_mcc            = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_auroc          = MulticlassAUROC(num_classes=num_classes)
        self.test_cm             = MulticlassConfusionMatrix(num_classes=num_classes)

        # ── Mean-Teacher EMA copy (eager, so SWA + checkpoint see it) ───
        if mean_teacher_weight > 0.0:
            teacher = copy.deepcopy(self)
            teacher.mean_teacher_weight = 0.0      # prevent recursive teachers / cons-loss in teacher path
            teacher._teacher = None                # belt-and-braces
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher.eval()
            self._teacher = teacher
        else:
            self._teacher = None

    # ── Backward-compatibility for pre-heads-refactor checkpoints ───────

    def on_load_checkpoint(self, checkpoint):
        """Remap legacy head keys (``head.0.*`` / ``head.3.*``) to the new
        :class:`MLPHead` layout (``head.net.0.*`` / ``head.net.3.*``) so old
        checkpoints continue to load. Only applied when the current head is the
        default MLP — other head types intentionally start from random."""
        if self.head_type != "mlp":
            return
        sd = checkpoint.get("state_dict", checkpoint)
        rename = {}
        for k in list(sd.keys()):
            for prefix in ("head.", "_teacher.head."):
                if k.startswith(prefix) and not k.startswith(prefix + "net."):
                    suffix = k[len(prefix):]                           # e.g. "0.weight"
                    rename[k] = prefix + "net." + suffix
                    break
        for old, new in rename.items():
            sd[new] = sd.pop(old)

    # ── Loss factory ────────────────────────────────────────────────────

    @staticmethod
    def _build_loss(
        loss: str, num_classes: int,
        class_weights: Optional[List[float]],
        cls_num_list: Optional[List[float]],
        focal_gamma: float, lmf_margin: float,
        ldam_max_m: float, ldam_s: float, cb_beta: float,
        label_smoothing: float,
    ) -> nn.Module:
        if loss == "lmf":
            return LargeMarginFocalLoss(
                num_classes=num_classes, alpha=class_weights,
                gamma=focal_gamma, margin=lmf_margin,
                label_smoothing=label_smoothing,
            )
        if loss == "ldam":
            if cls_num_list is None:
                raise ValueError("loss='ldam' requires cls_num_list (per-class sample counts)")
            return LDAMLoss(
                cls_num_list=cls_num_list, max_m=ldam_max_m, s=ldam_s,
                weight=class_weights, label_smoothing=label_smoothing,
            )
        if loss == "cb_focal":
            if cls_num_list is None:
                raise ValueError("loss='cb_focal' requires cls_num_list")
            return ClassBalancedFocalLoss(
                cls_num_list=cls_num_list, beta=cb_beta,
                gamma=focal_gamma, label_smoothing=label_smoothing,
            )
        return FocalLoss(
            class_weights=class_weights, gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

    # ── Forward ─────────────────────────────────────────────────────────

    def _features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return the (B, 2*fusion_dim) post-pool embedding."""
        a = self.branch_a(waveform, self.training)
        b = self.branch_b(waveform, self.training)
        c = self.branch_c(waveform, self.training)

        Tf = self.fusion_T
        a_p = F.adaptive_avg_pool1d(a, Tf)
        b_p = F.adaptive_avg_pool1d(b, Tf)
        c_p = F.adaptive_avg_pool1d(c, Tf)
        feats = [a_p, b_p, c_p]
        if self.branch_d is not None:
            d = self.branch_d(waveform, self.training)
            feats.append(F.adaptive_avg_pool1d(d, Tf))
        if self.branch_e is not None:
            e = self.branch_e(waveform, self.training)
            feats.append(F.adaptive_avg_pool1d(e, Tf))
        if self.branch_f is not None:
            f = self.branch_f(waveform, self.training)
            feats.append(F.adaptive_avg_pool1d(f, Tf))

        # Stochastic branch dropout (zero one branch's features per batch)
        feats = self.branch_drop(feats)

        # Onset mask from pooled Gabor features (B, gabor_ch, Tf)
        onset_mask = None
        if self.onset is not None:
            with torch.no_grad():
                onset_mask = self.onset(a_p)             # (B, ~Tf)
                # Pad/truncate to exactly Tf
                if onset_mask.shape[-1] < Tf:
                    onset_mask = F.pad(onset_mask, (0, Tf - onset_mask.shape[-1]))
                else:
                    onset_mask = onset_mask[:, :Tf]

        z = torch.cat(feats, dim=1)                      # (B, sum_C, Tf)
        z = self.fuse_proj(z)                            # (B, D, Tf)
        z = z.transpose(1, 2)                            # (B, Tf, D)

        for blk in self.attn_blocks:
            z = blk(z, onset_mask)
        for blk in self.s4d_blocks:
            z = blk(z)
        if self.dart_block is not None:
            z = self.dart_block(z)
        z = self.final_norm(z)

        z = z.transpose(1, 2)                            # (B, D, Tf)
        feat = self.pool(z)                              # (B, 2D)
        if self.feat_norm is not None:
            feat = self.feat_norm(feat)
            feat = F.normalize(feat, dim=-1)
        return feat

    def forward(self, waveform: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self._features(waveform)
        # Only ArcFace consumes labels; others ignore them.
        logits = self.head(feat, labels) if self.training else self.head(feat, None)
        if not self.training and self.logit_adjust_tau > 0:
            logits = logits - self.logit_adjust_tau * self.log_prior
        return logits

    # ── Mixup ───────────────────────────────────────────────────────────

    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
        if not self.training or self.mixup_alpha <= 0.0:
            return x, y, y, 1.0
        lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample())
        perm = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam

    def _mixup_loss(self, logits, y, y_p, lam):
        if lam == 1.0:
            return self.criterion(logits, y)
        return lam * self.criterion(logits, y) + (1.0 - lam) * self.criterion(logits, y_p)

    # ── Mean-Teacher ────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_teacher(self):
        if self._teacher is None:
            return
        m = self.mean_teacher_ema_decay
        for ps, pt in zip(self.parameters(), self._teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)

    def _consistency_lambda(self) -> float:
        if self._teacher is None or self.mean_teacher_weight <= 0.0:
            return 0.0
        rampup = float(self.mean_teacher_rampup_epochs)
        if rampup <= 0:
            return self.mean_teacher_weight
        x = max(0.0, min(1.0, float(self.current_epoch) / rampup))
        sig = float(torch.sigmoid(torch.tensor(5.0 * (x - 0.5))))
        return self.mean_teacher_weight * sig

    # ── Lightning steps ─────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Two stochastic waveform views (only differ when mean-teacher is on)
        x_student = self.wave_aug(x)
        x_m, y, y_p, lam = self._mixup(x_student, y)

        feat = self._features(x_m)
        # ArcFace needs labels at train time. With mixup (lam < 1) the head
        # sees mixed inputs but unmixed primary labels — same convention as the
        # mixup loss combines the two label streams downstream.
        logits = self.head(feat, y)
        loss = self._mixup_loss(logits, y, y_p, lam)

        if self.supcon_head is not None and self.aux_supcon_weight > 0.0:
            proj = F.normalize(self.supcon_head(feat), dim=-1)
            sup = _supcon_loss(proj, y, self.supcon_temperature)
            loss = loss + self.aux_supcon_weight * sup
            self.log("train/supcon", sup, on_epoch=True, prog_bar=False)

        lam_cons = self._consistency_lambda()
        if lam_cons > 0.0:
            x_teacher = self.wave_aug(x)                 # second stochastic view
            with torch.no_grad():
                t_logits = self._teacher(x_teacher)
            cons = F.mse_loss(F.softmax(logits, dim=-1),
                              F.softmax(t_logits.detach(), dim=-1))
            loss = loss + lam_cons * cons
            self.log("train/cons", cons,         on_epoch=True, prog_bar=False)
            self.log("train/cons_lam", lam_cons, on_epoch=True, prog_bar=False)
        self._update_teacher()

        self.train_acc(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = F.softmax(logits, dim=-1)

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_recall(logits, y)
        self.val_precision_macro(logits, y)
        self.val_precision_micro(logits, y)
        self.val_precision_per(logits, y)
        self.val_mcc(logits, y)
        self.val_auroc(probs, y)

        self.log("val/loss",            loss,                     on_epoch=True, prog_bar=True)
        self.log("val/acc",             self.val_acc,             on_epoch=True, prog_bar=True)
        self.log("val/f1",              self.val_f1,              on_epoch=True, prog_bar=True)
        self.log("val/recall",          self.val_recall,          on_epoch=True)
        self.log("val/macro_precision", self.val_precision_macro, on_epoch=True, prog_bar=True)
        self.log("val/micro_precision", self.val_precision_micro, on_epoch=True, prog_bar=True)
        self.log("val/mcc",             self.val_mcc,             on_epoch=True)
        self.log("val/auroc",           self.val_auroc,           on_epoch=True)

    def on_validation_epoch_end(self):
        per = self.val_precision_per.compute()
        for i, v in enumerate(per):
            self.log(f"val/precision_c{i}", v, prog_bar=False)
        self.val_precision_per.reset()

        # Composite F1/Precision score for tuning. macro_f1 already balances P
        # and R, so this lifts precision specifically without abandoning recall.
        f1   = self.val_f1.compute()
        pmac = self.val_precision_macro.compute()
        self.log("val/f1_p_score", (f1 + pmac) / 2.0, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = F.softmax(logits, dim=-1)

        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.test_precision(logits, y)
        self.test_precision_micro(logits, y)
        self.test_recall(logits, y)
        self.test_mcc(logits, y)
        self.test_auroc(probs, y)
        self.test_cm(logits, y)

        self.log("test/loss",            loss,                       on_epoch=True)
        self.log("test/acc",             self.test_acc,              on_epoch=True)
        self.log("test/f1",              self.test_f1,               on_epoch=True)
        self.log("test/macro_precision", self.test_precision,        on_epoch=True)
        self.log("test/micro_precision", self.test_precision_micro,  on_epoch=True)
        self.log("test/recall",          self.test_recall,           on_epoch=True)
        self.log("test/mcc",             self.test_mcc,              on_epoch=True)
        self.log("test/auroc",           self.test_auroc,            on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix:\n{cm.cpu().numpy()}")
        self.test_cm.reset()

    # ── Optimiser (SSM param LR split) ──────────────────────────────────

    def configure_optimizers(self):
        ssm_params, decay, no_decay = [], [], []
        ssm_keys = {"log_a_real", "log_a_imag", "log_dt", "dt_proj"}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ssm_keys):
                ssm_params.append(p)
            elif p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        lr = self.hparams.learning_rate
        groups = [
            {"params": decay,    "weight_decay": self.hparams.weight_decay, "lr": lr},
            {"params": no_decay, "weight_decay": 0.0,                       "lr": lr},
        ]
        if ssm_params:
            groups.append({"params": ssm_params, "weight_decay": 0.0,
                           "lr": lr * self.hparams.ssm_lr_mult})
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)

        def lr_lambda(epoch):
            wu, total = self.hparams.warmup_epochs, self.hparams.max_epochs
            if epoch < wu:
                return (epoch + 1) / max(wu, 1)
            p = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
