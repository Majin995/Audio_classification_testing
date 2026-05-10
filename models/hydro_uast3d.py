"""
Hydro-UAST3D — Uncompressed Audio Spectrogram Transformer with 3D Feature Fusion
==================================================================================

Architecture
------------
  Raw waveform (5120 Hz, 1 s)
      ↓
  SpectralTriFrontend     — three parallel spectrograms forming a 3D feature volume
      ├── Wideband  mel-PCEN  (n_fft=256 → high time resolution)
      ├── Narrowband mel-PCEN (n_fft=1024 → high frequency resolution)
      └── Standard  log-mel   (n_fft=512  → balanced reference)
      Stack → (B, 3, F=64, T≈101)   [depth × freq × time]
      ↓
  SpecAugment (train)     — freq + time masking applied per-view
      ↓
  3DFusionBlock           — explicit cross-view learning via two-stage Conv3d
      Stage 1 (intra-view):  Conv3d(1→C, kernel=(1,3,3)) — local spatial context
                              within each spectral view independently
      Stage 2 (cross-view):  Conv3d(C→C, kernel=(3,1,1)) — correlates all three
                              views at each (freq, time) location
      Compress:              Conv3d(C→1, kernel=(1,1,1)) + residual
      Output: (B, 1, 3, F, T)
      ↓
  Uncompressed Patch Embed — a single Conv3d: no prior CNN stem, no spatial
      Conv3d(1→D, kernel=(3, pF, pT), stride=(1, pF, pT))
      Fuses all 3 views while partitioning into non-overlapping (pF×pT) patches
      → (B, D, 1, nF, nT)  →  flatten  →  (B, N=nF·nT, D)
      ↓
  Prepend [CLS] token      → (B, 1+N, D)
  Add learned positional embedding (factorised 2D: freq + time)
      ↓
  ViT Encoder × n_blocks:
      Pre-norm MHSA  + DropPath residual
      Pre-norm FFN   + DropPath residual
      ↓
  CLS token → (B, D)
      ↓
  Classifier: LN → Linear(D→D/2) → GELU → Dropout → Linear(D/2→num_classes)
      ↓
  FocalLoss (class-weighted)

Design rationale
----------------
  Uncompressed
    The only spatial downsampling is the Conv3d patch embedding itself.  There is
    no preceding CNN stem that discards high-frequency details before the tokens
    are formed.  Each patch carries the full spectral and temporal fidelity of
    the (pF × pT) region in all three views simultaneously.

  Three views
    Underwater vessel acoustics span multiple timescales.  A 256-pt FFT (50 ms)
    resolves propeller-blade-rate modulations; a 1024-pt FFT (200 ms) resolves
    shaft harmonics separated by < 5 Hz; the 512-pt view balances both.
    A CNN frontend must commit to one resolution; the tri-frontend keeps all three
    intact for the 3D fusion stage to decide which matters per token.

  3D Feature Fusion
    The two-stage Conv3d design explicitly separates two distinct operations:
      • Stage 1 (intra-view, kernel depth=1): extracts spatial patterns within
        each view without cross-view contamination, acting like per-view Conv2d.
      • Stage 2 (cross-view, kernel height=width=1): fuses information across
        all three views at each (freq, time) position, learning which view
        combinations are discriminative for each spectral locus.
    The residual connection ensures the fused output retains the original views
    as a baseline, guarding against cross-view noise.

Training features
-----------------
  - Waveform-level Mixup
  - Per-view SpecAugment (independent masks per spectral view)
  - Focal loss with inverse-frequency class weights
  - AdamW + cosine warmup schedule
  - Stochastic depth linearly scheduled across blocks
  - AMP-friendly (fp16/bf16 safe)
"""

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
#  Utilities
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


# ═══════════════════════════════════════════════════════════════════════
#  Spectral frontends
# ═══════════════════════════════════════════════════════════════════════

class TrainablePCEN(nn.Module):
    """
    Trainable Per-Channel Energy Normalisation (PCEN).
    Adapts per mel-band to suppress the stationary underwater noise floor.
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


class MelPCENView(nn.Module):
    """Single mel-PCEN frontend (one spectral view)."""

    def __init__(
        self,
        sample_rate: int,
        n_mels:      int,
        n_fft:       int,
        hop_length:  int,
        f_min:       float = 20.0,
    ):
        super().__init__()
        f_max = sample_rate / 2.0
        self.mel  = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max, power=1.0,
        )
        self.pcen = TrainablePCEN(n_mels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T) → (B, n_mels, T_spec)
        return self.pcen(self.mel(waveform).clamp(min=1e-9))


class LogMelView(nn.Module):
    """Single log-mel frontend without PCEN (baseline reference view)."""

    def __init__(
        self,
        sample_rate: int,
        n_mels:      int,
        n_fft:       int,
        hop_length:  int,
        f_min:       float = 20.0,
    ):
        super().__init__()
        f_max = sample_rate / 2.0
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max, power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.amplitude_to_db(self.mel(waveform).clamp(min=1e-9))


class SpectralTriFrontend(nn.Module):
    """
    Three parallel spectral frontends stacked into a 3D feature volume.

    Views (all share the same n_mels and hop_length for consistent F and T):
      0 — Wideband  mel-PCEN  (n_fft=256  ≈  50 ms): high time resolution,
          resolves propeller blade-rate modulations and transient events.
      1 — Narrowband mel-PCEN (n_fft=1024 ≈ 200 ms): high frequency resolution,
          resolves closely-spaced shaft harmonics (< 5 Hz separation).
      2 — Standard  log-mel   (n_fft=512  ≈ 100 ms): balanced reference view,
          provides a PCEN-free baseline the transformer can contrast against.

    All three produce (B, n_mels, T_spec) with identical T_spec because
    torchaudio's center-padding makes T_spec = ⌊T/hop⌋ + 1 independent of n_fft.
    """

    def __init__(
        self,
        sample_rate: int   = 5_120,
        n_mels:      int   = 64,
        hop_length:  int   = 51,
        f_min:       float = 20.0,
    ):
        super().__init__()
        self.wb  = MelPCENView(sample_rate, n_mels, n_fft=256,  hop_length=hop_length, f_min=f_min)
        self.nb  = MelPCENView(sample_rate, n_mels, n_fft=1024, hop_length=hop_length, f_min=f_min)
        self.ref = LogMelView( sample_rate, n_mels, n_fft=512,  hop_length=hop_length, f_min=f_min)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T)
        wb  = self.wb(waveform)    # (B, F, T_spec)
        nb  = self.nb(waveform)    # (B, F, T_spec)
        ref = self.ref(waveform)   # (B, F, T_spec)

        # Align time axes (should be identical, but guard against off-by-one)
        t = min(wb.shape[-1], nb.shape[-1], ref.shape[-1])
        return torch.stack([wb[..., :t], nb[..., :t], ref[..., :t]], dim=1)
        # → (B, 3, F, T_spec)


class SpecAugment3View(nn.Module):
    """
    SpecAugment applied independently to each of the three spectral views.
    Masking each view separately forces the transformer to be robust to
    partial view dropout, preventing over-reliance on any single frontend.
    """

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
        # x: (B, 3, F, T)
        if not self.training:
            return x
        B, C, F, Tt = x.shape
        x = x.view(B * C, F, Tt)       # treat each view as a separate sample
        for m in self.freq_masks:
            x = m(x)
        for m in self.time_masks:
            x = m(x)
        return x.view(B, C, F, Tt)


# ═══════════════════════════════════════════════════════════════════════
#  3D Feature Fusion
# ═══════════════════════════════════════════════════════════════════════

class ThreeDFusionBlock(nn.Module):
    """
    Two-stage 3D convolutional fusion of the tri-frontend feature volume.

    Input/output shape: (B, 1, 3, F, T) — channel=1, depth=3 views.

    Stage 1 — Intra-view spatial:  Conv3d kernel (1, kH, kW)
      Each 3D filter covers only ONE depth slice (one spectral view), acting
      exactly like a Conv2d over each view independently.  Extracts local
      freq-time patterns (harmonic ridges, tonal stripes) within each view
      without mixing information across views yet.

    Stage 2 — Cross-view fusion:   Conv3d kernel (3, 1, 1)
      Each filter spans ALL three depth slices at a single (freq, time) point,
      learning which combinations of view features are discriminative.  e.g.
      "Cargo appears as a strong WB signal AND a weak NB signal at 200-400 Hz."

    Compress:  Conv3d(C→1, 1×1×1) projects back to 1-channel depth space.
    Residual:  adds the original (B, 1, 3, F, T) input, so fusion is additive
               — the raw views are always available to downstream layers.
    """

    def __init__(self, fusion_dim: int = 32, spatial_kernel: int = 3):
        super().__init__()
        pad = spatial_kernel // 2

        self.intra_view = nn.Sequential(
            nn.Conv3d(1, fusion_dim, (1, spatial_kernel, spatial_kernel),
                      padding=(0, pad, pad), bias=False),
            nn.BatchNorm3d(fusion_dim),
            nn.GELU(),
        )
        self.cross_view = nn.Sequential(
            nn.Conv3d(fusion_dim, fusion_dim, (3, 1, 1),
                      padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(fusion_dim),
            nn.GELU(),
        )
        self.compress = nn.Conv3d(fusion_dim, 1, (1, 1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 3, F, T)
        out = self.intra_view(x)    # (B, fusion_dim, 3, F, T)
        out = self.cross_view(out)  # (B, fusion_dim, 3, F, T)
        out = self.compress(out)    # (B, 1, 3, F, T)
        return out + x              # residual


# ═══════════════════════════════════════════════════════════════════════
#  Uncompressed patch embedding
# ═══════════════════════════════════════════════════════════════════════

class UncompressedPatchEmbed(nn.Module):
    """
    Single Conv3d that simultaneously:
      (a) fuses all 3 spectral views (kernel depth = 3, covers full depth),
      (b) partitions the freq-time plane into non-overlapping patches of
          size (patch_f × patch_t) without prior spatial downsampling.

    "Uncompressed" because no CNN stem discards spatial information before
    patching — the patch projection is the only downsampling step.

    Input:  (B, 1, 3, F, T)
    Output: (B, N_patches, model_dim)  where  N = (F//pF) × (T//pT)
    """

    def __init__(
        self,
        model_dim: int,
        patch_f:   int,
        patch_t:   int,
    ):
        super().__init__()
        # kernel_size depth=3 with stride depth=1 → fuses all 3 views into 1 output depth
        self.proj = nn.Conv3d(
            in_channels=1,
            out_channels=model_dim,
            kernel_size=(3, patch_f, patch_t),
            stride=(1, patch_f, patch_t),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 3, F, T)
        x = self.proj(x)                   # (B, D, 1, nF, nT)
        x = x.squeeze(2)                   # (B, D, nF, nT)
        B, D, nF, nT = x.shape
        x = x.view(B, D, nF * nT)         # (B, D, N)
        return x.transpose(1, 2)           # (B, N, D)


# ═══════════════════════════════════════════════════════════════════════
#  Factorised 2D positional encoding
# ═══════════════════════════════════════════════════════════════════════

class Factorised2DPosEmbed(nn.Module):
    """
    Factorised learned 2D positional embedding for the patch grid.

    For patch (f, t):  pos(f, t) = freq_emb[f] + time_emb[t]

    This outer-sum factorisation is more parameter-efficient than a flat 1D
    embedding (nF·nT vs nF·D + nT·D) and encodes explicit spatial structure
    — the transformer can more easily learn "harmonic patterns repeat along
    the frequency axis" when frequency position is encoded independently.
    """

    def __init__(self, n_freq: int, n_time: int, dim: int):
        super().__init__()
        self.n_freq = n_freq
        self.n_time = n_time
        self.freq_emb = nn.Parameter(torch.zeros(1, n_freq, 1, dim))
        self.time_emb = nn.Parameter(torch.zeros(1, 1, n_time, dim))
        nn.init.trunc_normal_(self.freq_emb, std=0.02)
        nn.init.trunc_normal_(self.time_emb, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D)  where  N = n_freq × n_time
        B, N, D = tokens.shape
        x = tokens.view(B, self.n_freq, self.n_time, D)
        x = x + self.freq_emb + self.time_emb     # broadcast outer-sum
        return x.view(B, N, D)


# ═══════════════════════════════════════════════════════════════════════
#  ViT-style Transformer block
# ═══════════════════════════════════════════════════════════════════════

class ViTBlock(nn.Module):
    """
    Standard pre-norm Vision Transformer block.

    Pre-LN → MHSA  → DropPath residual
    Pre-LN → FFN   → DropPath residual

    Following the "Pre-LN" formulation (Xiong et al. 2020) for stable
    training under mixed precision without warmup-only stabilisation.
    """

    def __init__(
        self,
        dim:          int,
        n_heads:      int,
        ff_expansion: int   = 4,
        dropout:      float = 0.1,
        attn_drop:    float = 0.1,
        drop_path:    float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            dim, n_heads, dropout=attn_drop, batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_expansion, dim),
            nn.Dropout(dropout),
        )

        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_n = self.norm1(x)
        a, _ = self.attn(x_n, x_n, x_n)
        x    = x + self.dp1(self.attn_drop(a))
        # Feed-forward
        x    = x + self.dp2(self.ffn(self.norm2(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss with per-class inverse-frequency weighting.
    Addresses the ~185× imbalance between Cargo and Passenger.
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
#  HydroUAST3D — Lightning Module
# ═══════════════════════════════════════════════════════════════════════

class HydroUAST3D(pl.LightningModule):
    """
    Uncompressed Audio Spectrogram Transformer with 3D Feature Fusion for
    underwater vessel acoustic classification.

    Args:
        num_classes     : Number of output classes (default 4).
        class_weights   : Inverse-frequency weights for FocalLoss.
        sample_rate     : Input audio sample rate in Hz (default 5120).
        n_mels          : Mel filterbank bins (freq axis F).
        hop_length      : STFT hop in samples (~10 ms at 5120 Hz = 51).
        patch_f         : Patch height in frequency bins (must divide n_mels).
        patch_t         : Patch width in time frames.
        fusion_dim      : Inner channels of the 3D fusion Conv layers.
        model_dim       : Transformer hidden dimension D.
        n_blocks        : Number of ViT blocks.
        n_heads         : Multi-head attention heads (model_dim % n_heads == 0).
        ff_expansion    : FFN expansion ratio.
        dropout         : General dropout probability.
        attn_drop       : Dropout inside attention.
        drop_path_rate  : Maximum stochastic depth rate (linearly scheduled).
        learning_rate   : Peak learning rate for AdamW.
        weight_decay    : L2 regularisation.
        warmup_epochs   : Linear warmup duration.
        max_epochs      : Total training epochs.
        mixup_alpha     : Beta parameter for Mixup (0 = off).
        focal_gamma     : Focal loss γ (0 = standard CE).
        label_smoothing : Cross-entropy label smoothing ε.
    """

    def __init__(
        self,
        num_classes:     int   = 4,
        class_weights:   Optional[list] = None,
        sample_rate:     int   = 5_120,
        n_mels:          int   = 64,
        hop_length:      int   = 51,
        patch_f:         int   = 8,
        patch_t:         int   = 10,
        fusion_dim:      int   = 32,
        model_dim:       int   = 256,
        n_blocks:        int   = 6,
        n_heads:         int   = 8,
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
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Spectral frontend ───────────────────────────────────────────
        self.frontend  = SpectralTriFrontend(
            sample_rate=sample_rate, n_mels=n_mels,
            hop_length=hop_length,
        )
        self.spec_aug  = SpecAugment3View(
            n_freq_masks=2, freq_mask_max=10,
            n_time_masks=2, time_mask_max=15,
        )

        # ── 3D feature fusion ───────────────────────────────────────────
        self.fusion3d = ThreeDFusionBlock(fusion_dim=fusion_dim, spatial_kernel=3)

        # ── Uncompressed patch embedding ────────────────────────────────
        self.patch_embed = UncompressedPatchEmbed(
            model_dim=model_dim, patch_f=patch_f, patch_t=patch_t,
        )

        # Compute patch grid dimensions for positional encoding
        # T_spec = floor(sample_rate / hop_length) + 1  (center-padding rule)
        _t_spec = sample_rate // hop_length + 1   # ≈ 101
        self.n_freq_patches = n_mels    // patch_f  # e.g. 64//8 = 8
        self.n_time_patches = _t_spec   // patch_t  # e.g. 101//10 = 10
        self.n_patches      = self.n_freq_patches * self.n_time_patches  # 80

        # ── CLS token + positional embeddings ───────────────────────────
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.cls_pos     = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.pos_embed_2d = Factorised2DPosEmbed(
            n_freq=self.n_freq_patches,
            n_time=self.n_time_patches,
            dim=model_dim,
        )
        self.pos_dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos,   std=0.02)

        # ── Transformer encoder ─────────────────────────────────────────
        dp_rates = [
            drop_path_rate * i / max(n_blocks - 1, 1)
            for i in range(n_blocks)
        ]
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=model_dim, n_heads=n_heads,
                ff_expansion=ff_expansion,
                dropout=dropout, attn_drop=attn_drop,
                drop_path=dp_rates[i],
            )
            for i in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(model_dim)

        # ── Classification head ─────────────────────────────────────────
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
        # 1. Tri-frontend → (B, 3, F, T_spec)
        x = self.frontend(waveform)
        x = self.spec_aug(x)

        # 2. 3D feature fusion — (B, 3, F, T) → (B, 1, 3, F, T)
        x = x.unsqueeze(1)       # add channel dim for Conv3d
        x = self.fusion3d(x)     # (B, 1, 3, F, T)

        # 3. Uncompressed patch embedding → (B, N, D)
        x = self.patch_embed(x)  # (B, N_patches, D)

        # 4. 2D positional encoding
        x = self.pos_embed_2d(x)    # (B, N, D)

        # 5. Prepend CLS token with its own positional embedding
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1) + self.cls_pos  # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                       # (B, 1+N, D)
        x   = self.pos_dropout(x)

        # 6. Transformer encoder
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # 7. CLS token → classifier
        cls_out = x[:, 0]        # (B, D)
        return self.classifier(cls_out)

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
        x, y = batch
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
