"""
HydroBAHTNet — Boundary-Aware Hybrid Transformer Network
=========================================================

Architecture
------------
  Raw waveform (5120 Hz, 1 s)
      ↓
  LogMelPCEN             — log-mel spectrogram + trainable PCEN
  SpecAugment (train)    — frequency & time masking
      ↓
  Patchify               — flatten mel bins into non-overlapping patches
  Patch projection       → (B, N_patches, D)
      ↓
  Boundary-Aware Transformer Encoder × 6 layers:
    Standard MHSA + learnable boundary-bias term (added to attention logits
    at patch boundaries flagged by spectral-flux onset detector).
    FFN: SwiGLU, dim_ff = 4×D.
  LayerNorm
      ↓
  AttentionPool → (B, D)
      ↓
  Classifier (LN → Linear → GELU → Dropout → Linear)
      ↓
  LargeMarginFocalLoss   — distinguishing feature of BAHTNet

Boundary-Aware Attention
------------------------
  For each sample in the batch, a spectral-flux onset detector computes a
  binary onset mask O ∈ {0,1}^{N_patches}.  A learnable scalar gate g scales
  a fixed bias matrix B_ij = (O_i OR O_j) to produce the boundary bias:
    attention_logits ← attention_logits + g · B

  This encourages the model to attend across onset boundaries, effectively
  aligning token representations with acoustic events (transient signatures,
  propeller cavitation bursts).
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
    MulticlassMatthewsCorrCoef, MulticlassAUROC, MulticlassConfusionMatrix,
)

from models.hydro_dart_mt import LogMelPCEN
from models.hydro_conformer import SpecAugment, AttentionPool
from processing.losses import LargeMarginFocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  Onset detector (spectral flux, runs per-batch on GPU)
# ═══════════════════════════════════════════════════════════════════════

class SpectralFluxOnset(nn.Module):
    """
    Lightweight half-wave-rectified spectral flux onset detector.

    Operates on a mel spectrogram (B, F, T) and returns a binary onset
    mask (B, N_patches) where N_patches = T // patch_size.

    threshold is applied as: flux > mean + std_mult * std
    """

    def __init__(self, patch_size: int = 4, std_mult: float = 1.5):
        super().__init__()
        self.patch_size = patch_size
        self.std_mult   = std_mult

    @torch.no_grad()
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: (B, F, T) mel spectrogram.
        Returns:
            onset_mask: (B, N_patches) bool tensor.
        """
        # Half-wave rectified spectral flux: sum of positive differences
        diff  = F.relu(spec[:, :, 1:] - spec[:, :, :-1])  # (B, F, T-1)
        flux  = diff.sum(dim=1)                             # (B, T-1)

        # Pool flux to patch granularity
        T_flux = flux.shape[-1]
        n_full = T_flux // self.patch_size
        flux_p = flux[:, :n_full * self.patch_size].reshape(
            flux.shape[0], n_full, self.patch_size
        ).mean(dim=-1)                                      # (B, n_patches_approx)

        # Adaptive threshold: mean + K * std
        mu  = flux_p.mean(dim=-1, keepdim=True)
        sig = flux_p.std(dim=-1, keepdim=True)
        onset = flux_p > (mu + self.std_mult * sig)        # (B, n_patches_approx)
        return onset


# ═══════════════════════════════════════════════════════════════════════
#  Boundary-Aware Attention block
# ═══════════════════════════════════════════════════════════════════════

class BoundaryAwareAttention(nn.Module):
    """
    Multi-head self-attention with a learnable onset-boundary positional bias.

    Args:
        embed_dim : Model dimension.
        num_heads : Attention heads.
        dropout   : Attention dropout.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.qkv  = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        # Learnable scalar gate for boundary bias
        self.boundary_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor,
                onset_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x          : (B, N, D) token sequence.
            onset_mask : (B, N) bool onset flags (optional).
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                     # each (B, N, H, Hd)
        q = q.permute(0, 2, 1, 3)                   # (B, H, N, Hd)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale   = self.head_dim ** -0.5
        dots    = (q @ k.transpose(-2, -1)) * scale # (B, H, N, N)

        # ── Boundary bias ────────────────────────────────────────────────
        if onset_mask is not None:
            # onset_mask: (B, N') where N' may differ from N by 1 after patchify
            N_mask = min(N, onset_mask.shape[-1])
            om = onset_mask[:, :N_mask].float()     # (B, N_mask)
            # B_ij = 1 if position i OR j is an onset
            b_row = om.unsqueeze(-1)                # (B, N_mask, 1)
            b_col = om.unsqueeze(-2)                # (B, 1, N_mask)
            bias  = (b_row + b_col).clamp(max=1.0)  # (B, N_mask, N_mask)
            # Pad/crop to match dots shape (N x N)
            if N_mask < N:
                pad_size = N - N_mask
                bias = F.pad(bias, (0, pad_size, 0, pad_size))
            bias = bias.unsqueeze(1) * self.boundary_gate   # (B, 1, N, N)
            dots = dots + bias

        attn = F.softmax(dots, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, D)
        return self.proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: x → (W1·x) * σ(W2·x) → W3."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.w1  = nn.Linear(dim, hidden, bias=False)
        self.w2  = nn.Linear(dim, hidden, bias=False)
        self.w3  = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class BAHTBlock(nn.Module):
    """One BAHTNet transformer block: BoundaryAwareAttention + SwiGLU FFN."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1,
                 drop_path: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = BoundaryAwareAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = SwiGLUFFN(dim, expansion=4, dropout=dropout)
        self.dp1   = _DropPath(drop_path)
        self.dp2   = _DropPath(drop_path)

    def forward(self, x, onset_mask=None):
        x = x + self.dp1(self.attn(self.norm1(x), onset_mask))
        x = x + self.dp2(self.ffn(x))
        return x


class _DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.bernoulli(
            torch.full((x.shape[0],) + (1,) * (x.ndim - 1), keep, device=x.device)
        ) / keep
        return x * mask


# ═══════════════════════════════════════════════════════════════════════
#  HydroBAHTNet
# ═══════════════════════════════════════════════════════════════════════

class HydroBAHTNet(pl.LightningModule):
    """
    Boundary-Aware Hybrid Transformer for underwater acoustic target recognition.

    Args:
        num_classes    : Output classes.
        class_weights  : Inverse-frequency weights for LMF loss.
        sample_rate    : Input sample rate.
        n_mels         : Mel filterbank bins.
        n_fft          : STFT window.
        hop_length     : STFT hop.
        patch_size     : Number of mel-time frames per patch token.
        model_dim      : Transformer hidden dimension.
        n_heads        : Attention heads.
        n_layers       : Transformer blocks.
        dropout        : Dropout rate.
        drop_path      : Stochastic depth max rate.
        lmf_gamma      : LMF focal exponent γ.
        lmf_margin     : LMF margin m.
        label_smoothing: Loss smoothing ε.
        learning_rate  : AdamW initial LR.
        weight_decay   : AdamW weight decay.
        warmup_epochs  : Linear warmup epochs.
        max_epochs     : Total training epochs.
        mixup_alpha    : Mixup Beta(α,α) strength.
    """

    def __init__(
        self,
        num_classes:    int   = 4,
        class_weights:  Optional[list] = None,
        sample_rate:    int   = 5_120,
        n_mels:         int   = 64,
        n_fft:          int   = 512,
        hop_length:     int   = 51,
        patch_size:     int   = 4,
        model_dim:      int   = 384,
        n_heads:        int   = 6,
        n_layers:       int   = 6,
        dropout:        float = 0.1,
        drop_path:      float = 0.15,
        lmf_gamma:      float = 2.0,
        lmf_margin:     float = 0.35,
        label_smoothing: float = 0.05,
        learning_rate:  float = 3e-4,
        weight_decay:   float = 1e-2,
        warmup_epochs:  int   = 10,
        max_epochs:     int   = 100,
        mixup_alpha:    float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Frontend ─────────────────────────────────────────────────────
        self.frontend = LogMelPCEN(
            sample_rate=sample_rate, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length,
        )
        self.spec_aug = SpecAugment(
            n_freq_masks=2, freq_mask_max=10,
            n_time_masks=2, time_mask_max=15,
        )

        # ── Onset detector ───────────────────────────────────────────────
        self.onset_detector = SpectralFluxOnset(patch_size=patch_size)

        # ── Patchify & project ───────────────────────────────────────────
        # Each patch: n_mels × patch_size mel frames → flatten → project to D
        patch_in_dim = n_mels * patch_size
        self.patch_proj = nn.Linear(patch_in_dim, model_dim, bias=False)

        # ── Transformer ──────────────────────────────────────────────────
        dp_rates = [drop_path * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.blocks = nn.ModuleList([
            BAHTBlock(model_dim, n_heads, dropout, dp_rates[i])
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(model_dim)

        # ── Pooling & head ───────────────────────────────────────────────
        self.pool       = AttentionPool(model_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

        # ── Loss ─────────────────────────────────────────────────────────
        self.criterion = LargeMarginFocalLoss(
            num_classes=num_classes,
            alpha=class_weights,
            gamma=lmf_gamma,
            margin=lmf_margin,
            label_smoothing=label_smoothing,
        )

        # ── Metrics ──────────────────────────────────────────────────────
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

    def _patchify(self, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split (B, F, T) spectrogram into non-overlapping patches.
        Returns patches (B, N, F*patch_size) and onset_mask (B, N).
        """
        B, F, T   = spec.shape
        ps        = self.hparams.patch_size
        n_patches = T // ps

        # Trim to multiple of patch_size
        spec_trim = spec[:, :, :n_patches * ps]     # (B, F, N*ps)
        patches   = spec_trim.reshape(B, F, n_patches, ps)  # (B, F, N, ps)
        patches   = patches.permute(0, 2, 1, 3).reshape(B, n_patches, F * ps)

        # Onset mask from the same spectrogram
        onset_mask = self.onset_detector(spec_trim)  # (B, N_approx)
        # Align to n_patches
        N_mask = min(n_patches, onset_mask.shape[-1])
        onset_aligned = torch.zeros(B, n_patches, dtype=torch.bool,
                                    device=spec.device)
        onset_aligned[:, :N_mask] = onset_mask[:, :N_mask]

        return patches, onset_aligned

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.frontend(waveform)          # (B, n_mels, T_spec)
        if self.training:
            spec = self.spec_aug(spec.unsqueeze(1)).squeeze(1)

        patches, onset_mask = self._patchify(spec)  # (B, N, F*ps), (B, N)
        x = self.patch_proj(patches)                # (B, N, D)

        # Sinusoidal positional encoding
        N, D = x.shape[1], x.shape[2]
        pe = self._sinusoidal_pe(N, D, x.device)
        x  = x + pe

        for block in self.blocks:
            x = block(x, onset_mask)

        x = self.final_norm(x)
        x = self.pool(x)                            # (B, D)
        return self.classifier(x)

    @staticmethod
    def _sinusoidal_pe(seq_len, dim, device):
        pe  = torch.zeros(seq_len, dim, device=device)
        pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2, device=device).float()
                        * (-math.log(10_000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    # ── Mixup + loss ──────────────────────────────────────────────────────

    def _mixup(self, x, y):
        alpha = self.hparams.mixup_alpha
        if not self.training or alpha <= 0.0:
            return x, y, y, 1.0
        lam  = float(torch.distributions.Beta(alpha, alpha).sample())
        perm = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam

    def _loss(self, logits, y, y_perm=None, lam=1.0):
        if y_perm is None or lam == 1.0:
            return self.criterion(logits, y)
        return lam * self.criterion(logits, y) + (1.0 - lam) * self.criterion(logits, y_perm)

    # ── Lightning steps ──────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y, y_p, lam = self._mixup(x, y)
        logits = self(x)
        loss   = self._loss(logits, y, y_p, lam)
        self.train_acc(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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
        self.test_acc(logits, y);  self.test_f1(logits, y)
        self.test_mcc(logits, y);  self.test_auroc(probs, y)
        self.test_cm(logits, y)
        self.log("test/loss",  self.criterion(logits, y), on_epoch=True)
        self.log("test/acc",   self.test_acc,              on_epoch=True)
        self.log("test/f1",    self.test_f1,               on_epoch=True)
        self.log("test/mcc",   self.test_mcc,              on_epoch=True)
        self.log("test/auroc", self.test_auroc,            on_epoch=True)

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
                return epoch / max(wu, 1)
            p = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
