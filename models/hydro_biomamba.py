"""
HydroBioMamba — BioMamba-inspired Vessel Acoustic Classifier
=============================================================

Optimised for 5 120 Hz input (Nyquist-correct for vessel acoustics whose
energy ceiling is 2 560 Hz) and 0.5 s clips, significantly reducing compute
and sequence length vs. the 32 kHz / 1 s baseline used by HydroSSAMBA.

Architecture summary
--------------------
  Raw waveform (5 120 Hz, 0.5 s = 2 560 samples)
      ↓
  MultiScalePCEN    — dual-resolution trainable PCEN (wideband + narrowband)
                      → (B, 2, n_mels, T_frames) magnitude mel, noise-floor adapted
      ↓
  SpecAugment (train)
      ↓
  ConvTokeniser     — 1-D temporal CNN (NOT 2-D patches)
                      flatten freq channels → depthwise-separable Conv1d ×2
                      → (B, T_frames, d_model)  + learnable positional embedding
      ↓
  Stage 1: Zigzag Mamba  × n_stage1  [d_state_1, fine-scale]
      Odd blocks  scan left→right (forward)
      Even blocks scan right→left (backward, flip–scan–flip)
      ↓
  TemporalDownsample  — depthwise Conv1d stride=2
                        T_frames → T_frames // 2
      ↓
  ChannelMixer FFN    — LayerNorm → Linear(d→4d) → GELU → Linear(4d→d) + residual
      ↓
  Stage 2: Zigzag Mamba  × n_stage2  [d_state_2 > d_state_1, coarse-scale]
      Larger state captures slower vessel periodicities (shaft/engine rhythm)
      ↓
  LayerNorm
      ↓
  AttentiveStatisticsPool  → (B, 2·d_model)
      ↓
  Classifier: Linear(2D→D) → BN → ReLU → Dropout → Linear(D→C)
      ↓
  FocalLoss (class-weighted)


How HydroBioMamba differs from HydroSSAMBA
-------------------------------------------
  ┌─────────────────────┬─────────────────────────┬────────────────────────────┐
  │ Aspect              │ HydroSSAMBA             │ HydroBioMamba (this file)  │
  ├─────────────────────┼─────────────────────────┼────────────────────────────┤
  │ Frontend            │ log-Mel + AmplitudeToDB │ MultiScalePCEN (trainable) │
  │ Tokenisation        │ 2-D PatchEmbedding      │ 1-D ConvTokeniser (CNN)    │
  │                     │ (freq×time patches,     │ (depthwise-separable,      │
  │                     │  linear projection)     │  local temporal context)   │
  │ Scan pattern        │ BidirMambaBlock: fwd+bwd│ ZigzagMambaBlock: ONE dir  │
  │                     │ run IN PARALLEL within  │ per block, direction       │
  │                     │ each block, outputs SUM │ ALTERNATES across layers   │
  │ Architecture depth  │ Flat — 6 blocks at same │ Two-stage hierarchy:       │
  │                     │ resolution              │ n_stage1 fine + downsample │
  │                     │                         │ + channel mixer + n_stage2 │
  │                     │                         │ coarse (larger d_state)    │
  │ Δ (dt) rank         │ Rank 1 — single scalar  │ Rank dt_rank = d_model//16 │
  │                     │ expanded to d_inner     │ richer input-dependence    │
  │ Native SR           │ 32 000 Hz (≫ max vessel │ 5 120 Hz (Nyquist-optimal) │
  │                     │  content at 2 560 Hz)   │                            │
  │ Default clip length │ 1.0 s (32 000 samples)  │ 0.5 s (2 560 samples)      │
  └─────────────────────┴─────────────────────────┴────────────────────────────┘

BioMamba inspiration
--------------------
BioMamba applies hierarchical multi-scale Mamba processing to biological signals
where different temporal scales capture different periodicities.  For underwater
vessel acoustics, the analogous periodicities are:

  Fast  (10–50 ms)  : propeller cavitation, blade-rate harmonics
  Medium(100–500 ms): engine cylinder firing, shaft harmonics
  Slow  (~1 s)      : vessel structural resonance, Doppler envelope

Stage 1 (full resolution, small d_state) addresses fast/medium scales.
Stage 2 (half resolution, larger d_state) addresses the slow/structural scale.

The zigzag scan pattern (from VMamba / S6-based 2D SSM literature) improves
on naïve bidirectional processing by ensuring each token is contextualised by
ALL preceding and ALL following tokens via the layer sequence, without running
two entire passes per block.

References
----------
  BioMamba: Kong et al., "BioMamba: Mamba meets Biosignal", 2024
  VMamba:   Liu et al., "VMamba: Visual State Space Model", 2024 (zigzag scan)
  Mamba:    Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from .hydro_conformer import MultiScalePCEN, SpecAugment, DropPath, FocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  Bio-Selective SSM core  (multi-rank Δ — distinct from SSAMBA's rank-1)
# ═══════════════════════════════════════════════════════════════════════

class _BioMambaCore(nn.Module):
    """
    Input-selective SSM with multi-rank time-step projection.

    Difference from SSAMBA's MambaBlock
    ------------------------------------
    SSAMBA uses a rank-1 dt_proj: ``Linear(1, d_inner)`` — a single scalar
    Δ is broadcast across all d_inner channels.

    Here, dt_rank = d_model // 16 projections are computed from the input and
    then expanded to d_inner, allowing *different* time steps for different
    channel groups.  For vessel acoustics this lets the SSM simultaneously
    track fast propeller transients (small Δ) and slow structural resonance
    (large Δ) in the same layer.

    Architecture (single direction)
    --------------------------------
        x (B, L, d_model)
          → in_proj → [x_branch, z]   (B, L, d_inner) each
          → causal DepthwiseConv1d(kernel=d_conv) + SiLU
          → x_proj  → [Δ_low (dt_rank), B_ssm (N), C_ssm (N)]
          → dt_proj → Δ expanded from dt_rank → d_inner
          → selective_scan → y  (B, L, d_inner)
          → y * SiLU(z)   (output gate)
          → out_proj → (B, L, d_model)
    """

    def __init__(
        self,
        d_model:  int   = 128,
        d_state:  int   = 16,
        d_conv:   int   = 4,
        expand:   int   = 2,
        dt_rank:  int   = 8,    # multi-rank Δ — set to d_model // 16 by caller
        dropout:  float = 0.0,
    ):
        super().__init__()
        self.d_state  = d_state
        self.dt_rank  = dt_rank
        d_inner       = int(expand * d_model)
        self.d_inner  = d_inner

        self.in_proj  = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise conv for local temporal context
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        # Multi-rank Δ + B + C from input
        self.x_proj   = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        # Expand dt_rank → d_inner
        self.dt_proj  = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)

        # Fixed diagonal A (negative real) — same stability guarantee as S4D
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.log_A = nn.Parameter(
            torch.log(A_init).unsqueeze(0).expand(d_inner, -1).clone()
        )

        self.D        = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop     = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _selective_scan(
        self,
        x:     torch.Tensor,   # (B, L, d_inner)
        delta: torch.Tensor,   # (B, L, d_inner)
        B_ssm: torch.Tensor,   # (B, L, N)
        C_ssm: torch.Tensor,   # (B, L, N)
    ) -> torch.Tensor:
        B_sz, L, D = x.shape
        N  = self.d_state
        A  = -torch.exp(self.log_A)                             # (D, N)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,L,D,N)
        dBx = delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1)  # (B,L,D,N)

        h  = torch.zeros(B_sz, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h  = dA[:, t] * h + dBx[:, t]
            ys.append((h * C_ssm[:, t].unsqueeze(1)).sum(-1))  # (B, D)
        return torch.stack(ys, dim=1) + self.D * x             # (B, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_sz, L, _ = x.shape
        xz          = self.in_proj(x)
        x_br, z     = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_br.transpose(1, 2))[..., :L]
        x_conv = F.silu(x_conv.transpose(1, 2))

        xbc    = self.x_proj(x_conv)                            # (B, L, dt_rank+2N)
        delta_r = xbc[..., :self.dt_rank]                       # (B, L, dt_rank)
        B_ssm   = xbc[..., self.dt_rank : self.dt_rank + self.d_state]
        C_ssm   = xbc[..., self.dt_rank + self.d_state:]
        delta   = F.softplus(self.dt_proj(delta_r))             # (B, L, d_inner)

        y = self._selective_scan(x_conv, delta, B_ssm, C_ssm)
        y = y * F.silu(z)
        return self.drop(self.out_proj(y))


# ═══════════════════════════════════════════════════════════════════════
#  Zigzag Mamba Block  (single direction per block, alternated across layers)
# ═══════════════════════════════════════════════════════════════════════

class ZigzagMambaBlock(nn.Module):
    """
    Single-direction Mamba block with pre-norm residual.

    Difference from SSAMBA's BidirMambaBlock
    -----------------------------------------
    BidirMambaBlock (SSAMBA): contains TWO MambaBlocks running in PARALLEL
    within a single block — one forward, one backward — and their outputs are
    SUMMED inside the block.

    ZigzagMambaBlock (BioMamba): contains ONE _BioMambaCore running in a
    SINGLE direction determined by ``self.reverse``.  The caller stacks
    multiple ZigzagMambaBlocks and sets ``reverse`` to alternate directions
    across layers, forming a zigzag scan path through the sequence.

    This gives every token full left AND right context through the layer
    sequence without requiring a parallel dual-pass per block, reducing
    parameter count while maintaining bidirectional coverage.

    Args
    ----
    reverse    : If True, scans right→left (sequence is flipped before and
                 after the SSM).  Set True for even-indexed layers.
    """

    def __init__(
        self,
        d_model:   int   = 128,
        d_state:   int   = 16,
        d_conv:    int   = 4,
        expand:    int   = 2,
        dt_rank:   int   = 8,
        dropout:   float = 0.0,
        drop_path: float = 0.0,
        reverse:   bool  = False,
    ):
        super().__init__()
        self.reverse = reverse
        self.norm    = nn.LayerNorm(d_model)
        self.mamba   = _BioMambaCore(d_model, d_state, d_conv, expand, dt_rank, dropout)
        self.dp      = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        z = x.flip(1) if self.reverse else x
        z = self.mamba(self.norm(z))
        if self.reverse:
            z = z.flip(1)
        return x + self.dp(z)


# ═══════════════════════════════════════════════════════════════════════
#  1-D Convolutional Tokeniser
# ═══════════════════════════════════════════════════════════════════════

class ConvTokeniser(nn.Module):
    """
    Projects a stacked multi-scale spectrogram into a token sequence using
    a two-layer depthwise-separable 1-D CNN.

    Difference from SSAMBA's PatchEmbedding
    ----------------------------------------
    PatchEmbedding (SSAMBA): chops the 2-D spectrogram into rectangular
    freq×time patches and applies a single linear projection.  This discards
    local temporal continuity within each patch.

    ConvTokeniser (BioMamba): treats frequency bins as channels and applies
    standard + depthwise Conv1d along the TIME axis, preserving local temporal
    structure.  The two-layer design gives a receptive field of 7+7-1 = 13
    frames (~330 ms at 5 120 Hz with hop=25), appropriate for capturing
    propeller blade-rate harmonics.

    Input : (B, in_channels, T_frames)  — in_channels = 2 * n_mels
    Output: (B, T_frames, d_model)
    """

    def __init__(self, in_channels: int, d_model: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=pad,
                      groups=d_model, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, T) → (B, T, d_model)"""
        x = self.pointwise(x)   # (B, d_model, T)
        x = self.depthwise(x)   # (B, d_model, T)
        return x.transpose(1, 2)  # (B, T, d_model)


# ═══════════════════════════════════════════════════════════════════════
#  Temporal Downsampling  (stride-2 between hierarchy stages)
# ═══════════════════════════════════════════════════════════════════════

class TemporalDownsample(nn.Module):
    """Halve the sequence length via depthwise stride-2 Conv1d + LayerNorm."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, stride=2, padding=1,
            groups=d_model, bias=False,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T//2, d_model)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════════
#  Channel Mixer FFN  (cross-channel mixing between hierarchy stages)
# ═══════════════════════════════════════════════════════════════════════

class ChannelMixer(nn.Module):
    """
    Two-layer FFN applied independently at each time step.

    Sits between Stage 1 and Stage 2 to mix information across the d_model
    feature channels BEFORE the coarser Mamba stage processes longer-range
    dependencies.  Acts as a learned non-linear projection that bridges the
    two temporal scales.
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, d_model * expansion)
        self.fc2  = nn.Linear(d_model * expansion, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.norm(x)))))


# ═══════════════════════════════════════════════════════════════════════
#  Attentive Statistics Pool  (sequence-first)
# ═══════════════════════════════════════════════════════════════════════

class _AttentiveStatsPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w    = F.softmax(self.attn(x), dim=1)
        mean = (x * w).sum(dim=1)
        var  = ((x ** 2) * w).sum(dim=1) - mean ** 2
        return torch.cat([mean, var.clamp(min=1e-9).sqrt()], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  HydroBioMamba  LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroBioMamba(pl.LightningModule):
    """
    BioMamba-inspired hierarchical vessel acoustic classifier.

    Optimised for 5 120 Hz / 0.5 s input (2 560 samples).  Uses zigzag
    scanning and a two-stage temporal hierarchy instead of the flat
    bidirectional architecture in HydroSSAMBA.

    Parameters
    ----------
    num_classes     : Output classes (default 3: Cargo/Passenger merged, Tanker, Tug).
    class_weights   : Inverse-frequency weights for FocalLoss.
    sample_rate     : Native sample rate. Default 5 120 Hz.
    fixed_len       : Input length in samples. Default 2 560 (= 0.5 s at 5 120 Hz).
    n_mels          : Mel filterbank bins. Default 64.
    hop_length      : STFT hop (samples). Default 25 (~200 frames/s at 5 120 Hz).
    d_model         : Token channel width throughout. Default 128.
    d_state_1       : SSM state dim in Stage 1 (fine scale). Default 16.
    d_state_2       : SSM state dim in Stage 2 (coarse scale). Default 32.
    n_stage1        : Zigzag Mamba blocks in Stage 1. Default 3.
    n_stage2        : Zigzag Mamba blocks in Stage 2. Default 3.
    expand          : Mamba inner-dim expansion. Default 2.
    d_conv          : Causal conv kernel inside Mamba. Default 4.
    dropout         : Dropout in Mamba cores and classifier. Default 0.24.
    drop_path_rate  : Max stochastic-depth probability. Default 0.10.
    learning_rate   : AdamW peak LR. Default 3e-4.
    weight_decay    : AdamW weight decay. Default 0.012.
    warmup_epochs   : Linear LR warmup epochs. Default 10.
    max_epochs      : Total training epochs (for cosine schedule). Default 100.
    mixup_alpha     : Waveform Mixup β (0 = off). Default 0.20.
    noise_prob      : Gaussian noise augmentation probability. Default 0.50.
    noise_snr_min   : Min SNR (dB) for noise aug. Default 15.0.
    noise_snr_max   : Max SNR (dB) for noise aug. Default 35.0.
    gain_prob       : Random gain augmentation probability. Default 0.70.
    focal_gamma     : Focal loss γ. Default 2.0.
    label_smoothing : Label smoothing ε. Default 0.001.
    """

    def __init__(
        self,
        num_classes:     int            = 3,
        class_weights:   Optional[list] = None,
        sample_rate:     int            = 5_120,
        fixed_len:       int            = 2_560,
        n_mels:          int            = 64,
        hop_length:      int            = 25,
        d_model:         int            = 128,
        d_state_1:       int            = 16,
        d_state_2:       int            = 32,
        n_stage1:        int            = 3,
        n_stage2:        int            = 3,
        expand:          int            = 2,
        d_conv:          int            = 4,
        dropout:         float          = 0.24,
        drop_path_rate:  float          = 0.10,
        learning_rate:   float          = 3e-4,
        weight_decay:    float          = 0.012,
        warmup_epochs:   int            = 10,
        max_epochs:      int            = 100,
        mixup_alpha:     float          = 0.20,
        noise_prob:      float          = 0.50,
        noise_snr_min:   float          = 15.0,
        noise_snr_max:   float          = 35.0,
        gain_prob:       float          = 0.70,
        focal_gamma:     float          = 2.0,
        label_smoothing: float          = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        n_total = n_stage1 + n_stage2
        # dt_rank: multi-rank Δ — d_model//16 (vs rank-1 in SSAMBA)
        dt_rank = max(1, d_model // 16)

        # ── Dual-PCEN front-end (trainable normalisation) ────────────────
        # MultiScalePCEN internally computes wideband + narrowband mel.
        # At 5 120 Hz: n_fft=128 (wideband, ~25 ms) and n_fft=512 (narrowband, ~100 ms)
        self.pcen    = MultiScalePCEN(
            sample_rate = sample_rate,
            n_mels      = n_mels,
            hop_length  = hop_length,
            wb_n_fft    = 128,       # ~25 ms at 5 120 Hz
            nb_n_fft    = 512,       # ~100 ms at 5 120 Hz
            f_min       = 20.0,
            f_max       = float(sample_rate // 2),
        )
        self.spec_aug = SpecAugment(
            n_freq_masks=2, freq_mask_max=10,
            n_time_masks=2, time_mask_max=15,
        )

        # ── 1-D Convolutional Tokeniser ──────────────────────────────────
        # Input after PCEN: (B, 2, n_mels, T) → flatten → (B, 2*n_mels, T)
        self.tokeniser  = ConvTokeniser(in_channels=2 * n_mels, d_model=d_model)

        # Learnable positional embedding sized for max expected T
        # At 5120 Hz with hop=25: 1s → ~204 frames, 0.5s → ~102 frames
        _max_T = math.ceil(sample_rate / hop_length) + 16   # +buffer
        self.pos_embed = nn.Parameter(torch.zeros(1, _max_T, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Stage 1: Fine-scale Zigzag Mamba ────────────────────────────
        # Odd blocks = forward (index 0, 2, …); Even blocks = backward (1, 3, …)
        dp1 = [drop_path_rate * i / max(n_total - 1, 1) for i in range(n_stage1)]
        self.stage1 = nn.ModuleList([
            ZigzagMambaBlock(
                d_model   = d_model,
                d_state   = d_state_1,
                d_conv    = d_conv,
                expand    = expand,
                dt_rank   = dt_rank,
                dropout   = dropout,
                drop_path = dp1[i],
                reverse   = (i % 2 == 1),   # alternate direction
            )
            for i in range(n_stage1)
        ])

        # ── Temporal Downsampling between stages ─────────────────────────
        self.downsample   = TemporalDownsample(d_model)

        # ── Channel Mixer (cross-feature mixing before coarse stage) ─────
        self.channel_mixer = ChannelMixer(d_model, expansion=4, dropout=dropout)

        # ── Stage 2: Coarse-scale Zigzag Mamba (larger d_state) ─────────
        dp2 = [drop_path_rate * (n_stage1 + i) / max(n_total - 1, 1)
               for i in range(n_stage2)]
        self.stage2 = nn.ModuleList([
            ZigzagMambaBlock(
                d_model   = d_model,
                d_state   = d_state_2,  # larger state for vessel periodicities
                d_conv    = d_conv,
                expand    = expand,
                dt_rank   = dt_rank,
                dropout   = dropout,
                drop_path = dp2[i],
                reverse   = (i % 2 == 1),
            )
            for i in range(n_stage2)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── Attentive Statistics Pool + Classifier ───────────────────────
        self.pool       = _AttentiveStatsPool(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # ── Loss + Metrics ───────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights   = class_weights,
            gamma           = focal_gamma,
            label_smoothing = label_smoothing,
        )
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

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T_samples) float32 at sample_rate Hz
        Returns:
            logits: (B, num_classes)
        """
        # ── Dual PCEN front-end ──────────────────────────────────────────
        spec = self.pcen(waveform)                  # (B, 2, n_mels, T_frames)
        if self.training:
            spec = self.spec_aug(spec)

        B, C, F, T = spec.shape
        spec_flat   = spec.view(B, C * F, T)        # (B, 2*n_mels, T_frames)

        # ── 1-D Convolutional Tokeniser ──────────────────────────────────
        x = self.tokeniser(spec_flat)               # (B, T_frames, d_model)

        # Learnable positional embedding
        L = x.size(1)
        x = x + self.pos_embed[:, :L, :]

        # ── Stage 1: Fine-scale Zigzag Mamba ────────────────────────────
        for blk in self.stage1:
            x = blk(x)

        # ── Temporal downsampling ─────────────────────────────────────────
        x = self.downsample(x)                      # (B, T//2, d_model)

        # ── Channel mixing ────────────────────────────────────────────────
        x = self.channel_mixer(x)

        # ── Stage 2: Coarse-scale Zigzag Mamba ──────────────────────────
        for blk in self.stage2:
            x = blk(x)

        x = self.final_norm(x)

        # ── Pool + Classify ───────────────────────────────────────────────
        x = self.pool(x)                            # (B, 2*d_model)
        return self.classifier(x)                   # (B, num_classes)

    # ── Augmentation ─────────────────────────────────────────────────────

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1).item() < self.hparams.gain_prob:
            gain = 0.6 + 0.8 * torch.rand(x.shape[0], 1, device=x.device)
            x    = x * gain
        if torch.rand(1).item() < self.hparams.noise_prob:
            snr_db  = (self.hparams.noise_snr_min
                       + (self.hparams.noise_snr_max - self.hparams.noise_snr_min)
                       * torch.rand(1, device=x.device))
            snr_lin = 10.0 ** (snr_db / 10.0)
            sig_pwr = x.pow(2).mean(-1, keepdim=True).clamp(min=1e-9)
            x       = x + torch.randn_like(x) * (sig_pwr / snr_lin).sqrt()
        return x

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
        return lam * self.criterion(logits, y) + (1.0 - lam) * self.criterion(logits, y_perm)

    # ── Lightning steps ──────────────────────────────────────────────────

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
        self.log("val/precision", self.val_precision, on_epoch=True)
        self.log("val/mcc",       self.val_mcc,       on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        probs  = F.softmax(logits, dim=-1)
        self.test_acc(logits, y);    self.test_f1(logits, y)
        self.test_mcc(logits, y);    self.test_auroc(probs, y)
        self.test_cm(logits, y)
        self.log("test/acc",   self.test_acc,   on_epoch=True)
        self.log("test/f1",    self.test_f1,    on_epoch=True, prog_bar=True)
        self.log("test/mcc",   self.test_mcc,   on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix (rows=true, cols=pred):\n{cm.cpu().numpy()}")
        self.test_cm.reset()

    # ── Optimiser ────────────────────────────────────────────────────────

    def configure_optimizers(self):
        # SSM poles (log_A, dt_proj) get a lower LR to stabilise the scan
        ssm_p, decay, no_decay = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "log_A" in name or "dt_proj" in name:
                ssm_p.append(p)
            elif p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        lr     = self.hparams.learning_rate
        ssm_lr = lr * 0.1
        optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.hparams.weight_decay, "lr": lr},
                {"params": no_decay, "weight_decay": 0.0,                       "lr": lr},
                {"params": ssm_p,    "weight_decay": 0.0,                       "lr": ssm_lr},
            ],
            lr=lr, betas=(0.9, 0.98), eps=1e-8,
        )

        def lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                             "interval": "epoch"},
        }


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test + architecture comparison
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model  = HydroBioMamba(num_classes=3).to(device).eval()
    total  = sum(p.numel() for p in model.parameters())

    stage1_p = sum(p.numel() for n, p in model.named_parameters() if "stage1" in n)
    stage2_p = sum(p.numel() for n, p in model.named_parameters() if "stage2" in n)
    front_p  = sum(p.numel() for n, p in model.named_parameters()
                   if "pcen" in n or "tokeniser" in n or "pos_embed" in n)

    hp = model.hparams
    dt_rank = max(1, hp.d_model // 16)

    x_dummy = torch.randn(2, hp.fixed_len, device=device)
    with torch.no_grad():
        logits = model(x_dummy)

    print(f"\nHydroBioMamba  |  {total:,} total params")
    print(f"  Frontend      : {front_p:,}  (MultiScalePCEN + ConvTokeniser + pos_embed)")
    print(f"  Stage 1       : {stage1_p:,}  ({hp.n_stage1} × ZigzagMambaBlock, d_state={hp.d_state_1}, dt_rank={dt_rank})")
    print(f"  Stage 2       : {stage2_p:,}  ({hp.n_stage2} × ZigzagMambaBlock, d_state={hp.d_state_2}, dt_rank={dt_rank})")
    print(f"  Other         : {total - front_p - stage1_p - stage2_p:,}  (downsample + mixer + pool + classifier)")
    print(f"\n  sample_rate   : {hp.sample_rate} Hz  (vs 32 000 Hz in HydroSSAMBA)")
    print(f"  fixed_len     : {hp.fixed_len} samples = {hp.fixed_len / hp.sample_rate:.2f} s  (vs 1.0 s)")
    print(f"  d_state_1/2   : {hp.d_state_1}/{hp.d_state_2}  (vs {hp.d_state_1} flat in HydroSSAMBA)")
    print(f"  dt_rank       : {dt_rank}  (vs 1 in HydroSSAMBA)")
    print(f"  scan pattern  : zigzag (alternating dirs)  (vs parallel bidir-sum)")
    print(f"  hierarchy     : 2-stage with downsample    (vs flat)")
    print(f"\nOutput shape   : {list(logits.shape)}")
