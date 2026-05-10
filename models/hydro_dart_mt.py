"""
Hydro-DART-MT — Dual Attention Parallel Residual Network Transformer
=====================================================================

Architecture
------------
  Raw waveform (5120 Hz, 1 s)
      ↓
  LogMelPCEN              — single-scale mel spectrogram + trainable PCEN
      ↓                     (B, n_mels, T_spec) with adaptive noise normalisation
  SpecAugment (train)     — frequency & time masking
      ↓
  PatchStem               — 2-stage strided conv: (B, 1, F, T) → (B, L, D)
      ↓
  SinusoidalPosEnc + Dropout
      ↓
  DART-MT Block × N:
    ┌─ Pre-norm LN(x) ───────────────────────────────────┐
    │   ├── LocalAttnBranch   (gated depthwise conv)      │
    │   └── GlobalAttnBranch  (full MHA)                  │
    │   → DualAttnFusion: sigmoid gate α·local+(1-α)·global  │
    │   + FFN (parallel residual — same pre-norm input)    │
    └─ x = x + DropPath(dual_attn) + DropPath(ffn) ──────┘
      ↓
  AttentionPool → (B, D)
      ↓
  Classifier: LN → Linear → GELU → Dropout → Linear
      ↓
  FocalLoss (class-weighted)

Key design decisions
--------------------
  Dual Attention
    The local branch (gated depthwise conv, kernel=31) captures rhythmic, short-
    range periodicities in vessel signatures (propeller blade rate, cavitation).
    The global branch (full MHA) captures long-range harmonic structure and
    broadband noise envelopes. A learned per-token sigmoid gate α ∈ (0,1) lets
    the network dynamically weight local vs global context per token.

  Parallel Residual
    Unlike the standard sequential Transformer (Attn → FFN, each on post-attn
    input), DART-MT computes both the dual-attention branch and the FFN branch
    on the *same* pre-norm input and adds both directly to the residual:
        x = x + DropPath(dual_attn(LN(x))) + DropPath(ffn(LN_ffn(x)))
    This halves effective depth while maintaining expressivity, and empirically
    stabilises training on short sequences.

  PCEN frontend
    Trainable Per-Channel Energy Normalisation adapts to the varying noise floor
    of underwater recordings, suppressing stationary background noise while
    preserving transient vessel signatures.

Training features
-----------------
  - Waveform-level Mixup (augmentation before spectrogram)
  - SpecAugment (frequency + time masking)
  - Focal loss with inverse-frequency class weights (handles 185× imbalance)
  - AdamW + cosine annealing with linear warmup
  - Stochastic depth linearly scheduled from 0 → drop_path_rate across blocks
  - AMP-friendly (fp16/bf16 safe)
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC,
    MulticlassConfusionMatrix,
)


# ═══════════════════════════════════════════════════════════════════════
#  Shared utilities
# ═══════════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    """Stochastic depth: drop entire residual paths during training."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep  = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


def sinusoidal_pe(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """Returns (1, seq_len, dim) sinusoidal positional encoding."""
    pe  = torch.zeros(seq_len, dim, device=device)
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10_000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)   # (1, L, D)


# ═══════════════════════════════════════════════════════════════════════
#  Frontend: Log-mel spectrogram + trainable PCEN
# ═══════════════════════════════════════════════════════════════════════

class TrainablePCEN(nn.Module):
    """
    Per-Channel Energy Normalisation with learnable parameters.

    Suppresses the stationary underwater noise floor while enhancing
    transient vessel signatures.  Parameters (α, δ, r, s) are learnable
    per mel-band scalars, initialised at typical PCEN defaults.

    Reference: Wang et al., ICASSP 2017.
    """

    def __init__(self, n_mels: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.alpha = nn.Parameter(torch.full((n_mels, 1), 0.98))
        self.delta = nn.Parameter(torch.full((n_mels, 1), 2.0))
        self.r     = nn.Parameter(torch.full((n_mels, 1), 0.5))
        self.s     = nn.Parameter(torch.full((n_mels, 1), 0.025))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T)
        alpha = torch.sigmoid(self.alpha)
        delta = F.softplus(self.delta)
        r     = torch.sigmoid(self.r)
        s     = torch.sigmoid(self.s)
        M     = self._ema(x, s)
        normed = x / (self.eps + M).pow(alpha)
        return (normed + delta).pow(r) - delta.pow(r)

    def _ema(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        frames = x.unbind(dim=-1)
        M = frames[0].unsqueeze(-1)
        out = [M]
        for f in frames[1:]:
            M = (1.0 - s) * M + s * f.unsqueeze(-1)
            out.append(M)
        return torch.cat(out, dim=-1)


class LogMelPCEN(nn.Module):
    """
    Single-scale mel spectrogram (magnitude) followed by trainable PCEN.

    Chosen parameters for the 5120 Hz / 1-second underwater dataset:
      n_fft=512   → 100 ms analysis window (captures slow modulations)
      hop_length  → ~10 ms hop (100 frames per second)
      n_mels=64   → sufficient resolution across [20, 2560] Hz
    """

    def __init__(
        self,
        sample_rate: int   = 5_120,
        n_mels:      int   = 64,
        n_fft:       int   = 512,
        hop_length:  int   = 51,
        f_min:       float = 20.0,
    ):
        super().__init__()
        f_max = sample_rate / 2.0
        self.mel  = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0,          # magnitude; PCEN prefers magnitude over power
        )
        self.pcen = TrainablePCEN(n_mels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T) → (B, n_mels, T_spec)
        x = self.mel(waveform).clamp(min=1e-9)
        return self.pcen(x)


class SpecAugment(nn.Module):
    """Frequency and time masking (applied independently to each sample)."""

    def __init__(
        self,
        n_freq_masks:  int = 2,
        freq_mask_max: int = 10,
        n_time_masks:  int = 2,
        time_mask_max: int = 15,
    ):
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [T.FrequencyMasking(freq_mask_max) for _ in range(n_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [T.TimeMasking(time_mask_max) for _ in range(n_time_masks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        if not self.training:
            return x
        for m in self.freq_masks:
            x = m(x)
        for m in self.time_masks:
            x = m(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Patch embedding
# ═══════════════════════════════════════════════════════════════════════

class PatchStem(nn.Module):
    """
    Two-stage strided CNN that embeds a single-channel spectrogram
    (B, 1, F, T) into a sequence of D-dimensional tokens (B, L, D).

    Stage 1: full resolution  → (B, 32, F,   T  )
    Stage 2: ½ resolution     → (B, 64, F/2, T/2)
    Stage 3: ¼ resolution     → (B, D,  F/4, T/4)

    At n_mels=64, T_spec≈100:
      L = (64/4) × (100/4) = 16 × 25 = 400 tokens
    """

    def __init__(self, model_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, model_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(model_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        x = x.unsqueeze(1)          # (B, 1, F, T)
        x = self.net(x)             # (B, D, F', T')
        B, D, F, Tt = x.shape
        x = x.view(B, D, F * Tt)   # (B, D, L)
        return x.permute(0, 2, 1)  # (B, L, D)


# ═══════════════════════════════════════════════════════════════════════
#  DART-MT building blocks
# ═══════════════════════════════════════════════════════════════════════

class LocalAttentionBranch(nn.Module):
    """
    Gated depthwise separable convolution as a local attention proxy.

    Captures short-range temporal patterns (propeller blade rate,
    cavitation periodicity) with O(L·k) complexity vs O(L²) for MHA.

    Structure: pw_expand → sigmoid gate × depthwise_conv(value) → pw_out
    """

    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        # Expand to gate + value (2× channels, split at forward time)
        self.pw_expand = nn.Linear(dim, dim * 2)
        self.dw_conv   = nn.Conv1d(
            dim, dim, kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.bn        = nn.BatchNorm1d(dim)
        self.act       = nn.SiLU()
        self.pw_out    = nn.Linear(dim, dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        gate, value = self.pw_expand(x).chunk(2, dim=-1)   # (B, L, D) each
        gate  = torch.sigmoid(gate)
        value = self.dw_conv(value.transpose(1, 2))         # (B, D, L)
        value = self.act(self.bn(value)).transpose(1, 2)    # (B, L, D)
        return self.dropout(self.pw_out(gate * value))


class GlobalAttentionBranch(nn.Module):
    """
    Full multi-head self-attention — captures long-range harmonic structure
    and broadband envelope correlations across the entire sequence.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) → (B, L, D)
        out, _ = self.attn(x, x, x)
        return self.dropout(out)


class DualAttnFusion(nn.Module):
    """
    Learned per-token sigmoid gate that mixes local and global outputs.

    The gate α = σ(W · [local ‖ global]) ∈ (0,1)^D controls the
    per-feature contribution of each branch independently, letting the
    model allocate local vs global capacity on a per-token basis.

    Output: α ⊙ local + (1 − α) ⊙ global
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim * 2, dim, bias=True)

    def forward(self, local: torch.Tensor, global_: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(torch.cat([local, global_], dim=-1)))
        return gate * local + (1.0 - gate) * global_


class DARTBlock(nn.Module):
    """
    Dual Attention Parallel Residual Transformer block.

    Unlike the sequential Conformer (Attn then FFN applied to post-Attn input),
    DART-MT applies both the dual-attention branch and the FFN branch to the
    same pre-normalised input and sums their contributions into the residual:

        x_n   = LN_attn(x)
        local = LocalAttn(x_n)
        glob  = GlobalAttn(x_n)
        dual  = DualAttnFusion(local, glob)

        ffn   = FFN(LN_ffn(x))          ← same residual x, separate norm

        x     = x + DropPath(dual) + DropPath(ffn)

    Benefits:
      • Each branch optimises independently (no gradient coupling through Attn).
      • Halves effective sequential depth → faster convergence on small datasets.
      • Empirically stable under mixed precision.
    """

    def __init__(
        self,
        dim:          int,
        n_heads:      int,
        kernel_size:  int   = 31,
        ff_expansion: int   = 4,
        dropout:      float = 0.1,
        attn_drop:    float = 0.1,
        drop_path:    float = 0.0,
    ):
        super().__init__()
        # Separate pre-norms for the two parallel branches
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm  = nn.LayerNorm(dim)

        # Dual attention
        self.local_attn  = LocalAttentionBranch(dim, kernel_size, attn_drop)
        self.global_attn = GlobalAttentionBranch(dim, n_heads, attn_drop)
        self.fusion      = DualAttnFusion(dim)

        # FFN branch (parallel residual)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_expansion, dim),
            nn.Dropout(dropout),
        )

        self.dp_attn = DropPath(drop_path)
        self.dp_ffn  = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dual attention branch (pre-norm shared between local + global)
        x_a = self.attn_norm(x)
        local_out  = self.local_attn(x_a)
        global_out = self.global_attn(x_a)
        dual_out   = self.fusion(local_out, global_out)

        # FFN branch (parallel — sees the same residual x, not post-attn)
        ffn_out = self.ffn(self.ffn_norm(x))

        # Parallel residual update
        return x + self.dp_attn(dual_out) + self.dp_ffn(ffn_out)


# ═══════════════════════════════════════════════════════════════════════
#  Pooling
# ═══════════════════════════════════════════════════════════════════════

class AttentionPool(nn.Module):
    """Soft attention-weighted sum over the sequence → single vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        w = F.softmax(self.score(x), dim=1)   # (B, L, 1)
        return (x * w).sum(dim=1)             # (B, D)


# ═══════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss with per-class inverse-frequency weighting.

    Addresses the severe class imbalance in the vessel dataset
    (Cargo ~185× more common than Passenger) by down-weighting
    easy, high-confidence predictions.

    Reference: Lin et al., ICCV 2017.
    """

    def __init__(
        self,
        class_weights:   Optional[list] = None,
        gamma:           float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "alpha",
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ═══════════════════════════════════════════════════════════════════════
#  HydroDARTMT — Lightning Module
# ═══════════════════════════════════════════════════════════════════════

class HydroDARTMT(pl.LightningModule):
    """
    Dual Attention Parallel Residual Network Transformer for underwater
    vessel acoustic classification.

    Args:
        num_classes     : Number of output classes (default 4).
        class_weights   : Inverse-frequency weights for FocalLoss.
        sample_rate     : Input audio sample rate in Hz (default 5120).
        n_mels          : Mel filterbank bins.
        n_fft           : STFT window size in samples.
        hop_length      : STFT hop in samples (~10 ms at 5120 Hz = 51).
        model_dim       : Hidden dimension D for all layers.
        n_blocks        : Number of DART-MT blocks.
        n_heads         : MHA heads in the global attention branch.
        local_kernel    : Depthwise conv kernel in local attention branch.
        ff_expansion    : FFN expansion ratio.
        dropout         : General dropout probability.
        attn_drop       : Dropout inside attention branches.
        drop_path_rate  : Maximum stochastic depth rate (linearly scheduled).
        learning_rate   : Peak learning rate for AdamW.
        weight_decay    : L2 regularisation coefficient.
        warmup_epochs   : Linear warmup duration.
        max_epochs      : Total training epochs (for cosine schedule).
        mixup_alpha     : Beta parameter for waveform-level Mixup (0 = off).
        focal_gamma     : Focal loss γ exponent (0 = standard cross-entropy).
        label_smoothing : Cross-entropy label smoothing ε.
    """

    def __init__(
        self,
        num_classes:     int   = 4,
        class_weights:   Optional[list] = None,
        sample_rate:     int   = 5_120,
        n_mels:          int   = 64,
        n_fft:           int   = 512,
        hop_length:      int   = 51,
        model_dim:       int   = 256,
        n_blocks:        int   = 6,
        n_heads:         int   = 8,
        local_kernel:    int   = 31,
        ff_expansion:    int   = 4,
        dropout:         float = 0.1,
        attn_drop:       float = 0.1,
        drop_path_rate:  float = 0.15,
        learning_rate:   float = 3e-4,
        weight_decay:    float = 1e-2,
        warmup_epochs:   int   = 10,
        max_epochs:      int   = 100,
        mixup_alpha:     float = 0.3,
        focal_gamma:     float = 2.0,
        label_smoothing: float = 0.05,
        # ── Mean-Teacher (DART-MT semi-supervised) ───────────────────────
        use_mean_teacher:  bool  = False,
        ema_momentum:      float = 0.999,
        consistency_max:   float = 1.0,
        rampup_epochs:     int   = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Frontend ────────────────────────────────────────────────────
        self.frontend  = LogMelPCEN(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.spec_aug  = SpecAugment(
            n_freq_masks=2, freq_mask_max=10,
            n_time_masks=2, time_mask_max=15,
        )

        # ── Patch embedding ─────────────────────────────────────────────
        self.patch_stem  = PatchStem(model_dim=model_dim)
        self.pos_dropout = nn.Dropout(dropout)

        # ── DART-MT blocks ──────────────────────────────────────────────
        # Stochastic depth rate increases linearly from 0 to drop_path_rate
        dp_rates = [
            drop_path_rate * i / max(n_blocks - 1, 1)
            for i in range(n_blocks)
        ]
        self.blocks = nn.ModuleList([
            DARTBlock(
                dim=model_dim,
                n_heads=n_heads,
                kernel_size=local_kernel,
                ff_expansion=ff_expansion,
                dropout=dropout,
                attn_drop=attn_drop,
                drop_path=dp_rates[i],
            )
            for i in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(model_dim)

        # ── Pooling & classification head ───────────────────────────────
        self.pool       = AttentionPool(model_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

        # ── Loss ────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

        # ── Mean-Teacher EMA (optional) ──────────────────────────────────
        # Ported from HydroFusion._update_teacher (hydro_fusion.py:500)
        if use_mean_teacher:
            self._teacher = copy.deepcopy(self)
            for p in self._teacher.parameters():
                p.requires_grad_(False)
            # Prevent the teacher copy from itself creating another teacher
            self._teacher.hparams.use_mean_teacher = False
        else:
            self._teacher = None

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

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (B, T)  float32 waveform at self.hparams.sample_rate
        Returns:
            logits   : (B, num_classes)
        """
        x = self.frontend(waveform)     # (B, n_mels, T_spec)
        x = self.spec_aug(x)            # (B, n_mels, T_spec)

        x = self.patch_stem(x)          # (B, L, D)

        pe = sinusoidal_pe(x.shape[1], x.shape[2], x.device)
        x  = self.pos_dropout(x + pe)

        for block in self.blocks:
            x = block(x)                # (B, L, D)

        x = self.final_norm(x)
        x = self.pool(x)                # (B, D)
        return self.classifier(x)       # (B, num_classes)

    # ── Mean-Teacher EMA ────────────────────────────────────────────────

    @torch.no_grad()
    def _update_teacher(self) -> None:
        """EMA update of teacher parameters. Mirrors HydroFusion._update_teacher."""
        if self._teacher is None:
            return
        m = self.hparams.ema_momentum
        for ps, pt in zip(self.parameters(), self._teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)

    def _consistency_lambda(self) -> float:
        """Sigmoid ramp-up of the consistency weight λ(t)."""
        if self._teacher is None:
            return 0.0
        epoch     = float(self.current_epoch)
        rampup    = float(self.hparams.rampup_epochs)
        if rampup <= 0:
            return self.hparams.consistency_max
        # Sigmoid ramp-up
        x = max(0.0, epoch / rampup)
        return self.hparams.consistency_max * float(
            torch.sigmoid(torch.tensor(5.0 * (x - 0.5)))
        )

    # ── Mixup ───────────────────────────────────────────────────────────

    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
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
        # batch may be a dict {'labeled': (x_l, y_l), 'unlabeled': x_u}
        # when DART-MT semi-supervised mode is active, or a plain (x, y) tuple
        if isinstance(batch, dict):
            x_l, y    = batch["labeled"]
            x_u       = batch.get("unlabeled", None)
        else:
            x_l, y    = batch
            x_u       = None

        x_l, y, y_p, lam = self._mixup(x_l, y)
        logits = self(x_l)
        loss   = self._loss(logits, y, y_p, lam)

        # ── Mean-Teacher consistency loss on unlabeled samples ───────────
        if self._teacher is not None and x_u is not None:
            lam_cons = self._consistency_lambda()
            if lam_cons > 0.0:
                # Weak augmentation → teacher logits (no grad)
                with torch.no_grad():
                    t_logits = self._teacher(x_u)
                # Strong augmentation → student logits
                # Strong aug: apply SpecAugment in-place on the spectrogram
                s_logits = self(x_u)
                consistency_loss = F.mse_loss(s_logits, t_logits.detach())
                loss = loss + lam_cons * consistency_loss
                self.log("train/cons_loss", consistency_loss, on_epoch=True)
                self.log("train/cons_lambda", lam_cons,        on_epoch=True)

        # ── EMA teacher update ───────────────────────────────────────────
        self._update_teacher()

        self.train_acc(logits, y)
        self.log("train/loss", loss,           on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_precision(logits, y)
        self.val_mcc(logits, y)
        self.log("val/loss",      loss,               on_epoch=True, prog_bar=True)
        self.log("val/acc",       self.val_acc,       on_epoch=True, prog_bar=True)
        self.log("val/f1",        self.val_f1,        on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True)
        self.log("val/mcc",       self.val_mcc,       on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        probs  = F.softmax(logits, dim=-1)

        self.test_acc(logits,  y)
        self.test_f1(logits,   y)
        self.test_mcc(logits,  y)
        self.test_auroc(probs, y)
        self.test_cm(logits,   y)
        self.log("test/loss",  self.criterion(logits, y), on_epoch=True)
        self.log("test/acc",   self.test_acc,             on_epoch=True)
        self.log("test/f1",    self.test_f1,              on_epoch=True)
        self.log("test/mcc",   self.test_mcc,             on_epoch=True)
        self.log("test/auroc", self.test_auroc,           on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix (rows=true, cols=pred):\n{cm.cpu().numpy()}")
        self.test_cm.reset()

    # ── Optimiser ───────────────────────────────────────────────────────

    def configure_optimizers(self):
        # Weight-decay param groups: skip biases and 1-D params (norms)
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        def lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
