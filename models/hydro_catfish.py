"""
HydroCATFISH — Convolutional Acoustic Transformer with Filterbank Initialised
from Sinusoidal Harmonics (CATFISH)
============================================================================

Architecture
------------
  Raw waveform (5120 Hz, 1 s)
      ↓
  LearnableGaborFilterbank     — per-filter learnable (f0, bandwidth) as
                                 grouped Conv1d; replaces fixed STFT front-end.
  Magnitude + log1p            → (B, n_filters, T_conv)
  SpecAugment-1D (train)
      ↓
  TCN backbone                 — 6 × dilated depthwise-separable Conv1d blocks
                                 with exponentially growing dilation [1,2,4,8,16,32].
      ↓
  AttentionPool1D              → (B, D)
      ↓
  Classifier (LN → Linear → GELU → Dropout → Linear)
      ↓
  FocalLoss (class-weighted)

CATFISH reference
-----------------
  Original CATFISH uses Gabor-initialised filters learned jointly with the
  classifier.  Per-filter parameters: centre frequency f0 and half-bandwidth σ.
  The filter impulse response:  g(t) = cos(2π·f0·t) · exp(−t²/(2σ²))
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

from models.hydro_conformer import FocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  Learnable Gabor Filterbank
# ═══════════════════════════════════════════════════════════════════════

class LearnableGaborFilterbank(nn.Module):
    """
    Learnable Gabor filterbank applied as grouped Conv1d.

    Each filter is parameterised by learnable (f0, log_sigma) per filter,
    initialised on a mel-spaced grid.  The real Gabor kernel:
        g_k(t) = cos(2π·f0_k·t) · exp(−t²/(2·σ_k²))
    is computed dynamically and used as a Conv1d weight.

    Args:
        n_filters   : Number of frequency filters (channels out).
        kernel_size : Filter length in samples (should be odd).
        sample_rate : Audio sample rate.
        f_min       : Minimum centre frequency.
        f_max       : Maximum centre frequency (default Nyquist).
    """

    def __init__(
        self,
        n_filters:   int   = 64,
        kernel_size: int   = 257,
        sample_rate: int   = 5_120,
        f_min:       float = 50.0,
        f_max:       Optional[float] = None,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.n_filters   = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        f_max = f_max or sample_rate / 2.0

        # Initialise on mel-spaced grid
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        f0_init = torch.tensor(
            [self._mel_to_hz(mel_min + i * (mel_max - mel_min) / (n_filters - 1))
             for i in range(n_filters)], dtype=torch.float32
        )

        # Initial bandwidth ~ one mel-filter width
        bandwidth_init = torch.full((n_filters,), fill_value=f0_init.mean().item() * 0.5)

        self.f0        = nn.Parameter(f0_init)            # (n_filters,)
        self.log_sigma = nn.Parameter(torch.log(bandwidth_init.clamp(min=1.0)))

        # Time axis [-(L//2), ..., L//2] in seconds
        t = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        self.register_buffer("t", t / sample_rate)        # (kernel_size,)

    @staticmethod
    def _hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    @staticmethod
    def _mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def _build_kernels(self) -> torch.Tensor:
        """Dynamically compute Gabor kernels from learnable parameters."""
        # f0: (n_filters,) → (n_filters, 1)
        f0    = self.f0.abs().unsqueeze(1)         # enforce positive freq
        sigma = torch.exp(self.log_sigma).unsqueeze(1).clamp(min=1e-4)
        t     = self.t.unsqueeze(0)                # (1, kernel_size)

        # Gabor: cos(2π·f0·t) · Gaussian(t, 0, σ)
        gauss   = torch.exp(-0.5 * (t / sigma) ** 2)
        carrier = torch.cos(2.0 * math.pi * f0 * t)
        kernels = gauss * carrier                  # (n_filters, kernel_size)

        # Normalise each filter to unit energy
        energy = kernels.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        kernels = kernels / energy

        return kernels.unsqueeze(1)                # (n_filters, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T) mono waveform.
        Returns:
            envelope : (B, n_filters, T_out) log1p magnitude.
        """
        kernels = self._build_kernels()            # (n_filters, 1, kernel_size)
        x = x.unsqueeze(1)                         # (B, 1, T)
        pad = self.kernel_size // 2
        out = F.conv1d(x, kernels, padding=pad)    # (B, n_filters, T)
        return torch.log1p(out.abs())


class SpecAugment1D(nn.Module):
    """Frequency-band and time masking for 1-D filterbank output."""

    def __init__(self, n_freq_masks: int = 2, freq_mask_max: int = 8,
                 n_time_masks: int = 2, time_mask_max: int = 20):
        super().__init__()
        self.n_freq_masks  = n_freq_masks
        self.freq_mask_max = freq_mask_max
        self.n_time_masks  = n_time_masks
        self.time_mask_max = time_mask_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if not self.training:
            return x
        B, C, T = x.shape
        for _ in range(self.n_freq_masks):
            f = torch.randint(1, max(2, self.freq_mask_max), (1,)).item()
            f0 = torch.randint(0, max(1, C - f), (1,)).item()
            x[:, f0: f0 + f, :] = 0.0
        for _ in range(self.n_time_masks):
            t = torch.randint(1, max(2, self.time_mask_max), (1,)).item()
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            x[:, :, t0: t0 + t] = 0.0
        return x


# ═══════════════════════════════════════════════════════════════════════
#  TCN Backbone
# ═══════════════════════════════════════════════════════════════════════

class TCNBlock(nn.Module):
    """Dilated depthwise-separable Conv1d residual block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.dw   = nn.Conv1d(in_ch, in_ch, kernel_size, padding=pad,
                               dilation=dilation, groups=in_ch, bias=False)
        self.pw   = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.bn(self.pw(self.dw(x)))))
        return out + self.res(x)


class AttentionPool1D(nn.Module):
    """Soft attention pooling over time dimension: (B, D, T) → (B, D)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Conv1d(dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        w = F.softmax(self.score(x), dim=-1)   # (B, 1, T)
        return (x * w).sum(dim=-1)             # (B, D)


# ═══════════════════════════════════════════════════════════════════════
#  HydroCATFISH
# ═══════════════════════════════════════════════════════════════════════

class HydroCATFISH(pl.LightningModule):
    """
    CATFISH: raw-waveform classifier with a learnable Gabor filterbank front-end
    and a dilated TCN backbone.

    Args:
        num_classes    : Number of target classes.
        class_weights  : Inverse-frequency weights for FocalLoss.
        sample_rate    : Audio sample rate (Hz).
        gabor_n_filters: Number of learnable Gabor filters.
        gabor_kernel   : Filter kernel length (samples, must be odd).
        tcn_channels   : Hidden channels in TCN blocks.
        n_tcn_blocks   : Number of TCN blocks (dilation doubles each block).
        dropout        : Dropout rate.
        learning_rate  : Initial LR for AdamW.
        weight_decay   : Weight decay for AdamW.
        warmup_epochs  : Linear LR warmup duration.
        max_epochs     : Total epochs (for cosine schedule).
        mixup_alpha    : Beta(α,α) mixup strength (0 = off).
        focal_gamma    : Focal loss γ.
        label_smoothing: Cross-entropy smoothing ε.
    """

    def __init__(
        self,
        num_classes:     int   = 4,
        class_weights:   Optional[list] = None,
        sample_rate:     int   = 5_120,
        gabor_n_filters: int   = 64,
        gabor_kernel:    int   = 257,
        tcn_channels:    int   = 128,
        n_tcn_blocks:    int   = 6,
        dropout:         float = 0.1,
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

        # ── Frontend ─────────────────────────────────────────────────────
        self.filterbank = LearnableGaborFilterbank(
            n_filters=gabor_n_filters,
            kernel_size=gabor_kernel,
            sample_rate=sample_rate,
        )
        self.spec_aug = SpecAugment1D(
            n_freq_masks=2, freq_mask_max=8,
            n_time_masks=2, time_mask_max=20,
        )

        # ── TCN backbone ─────────────────────────────────────────────────
        blocks = []
        in_ch  = gabor_n_filters
        for i in range(n_tcn_blocks):
            dil = 2 ** i
            blocks.append(TCNBlock(in_ch, tcn_channels, dilation=dil, dropout=dropout))
            in_ch = tcn_channels
        self.tcn = nn.Sequential(*blocks)

        # ── Pooling & classification ──────────────────────────────────────
        self.pool       = AttentionPool1D(tcn_channels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(tcn_channels),
            nn.Linear(tcn_channels, tcn_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_channels // 2, num_classes),
        )

        # ── Loss ─────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
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

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.filterbank(waveform)   # (B, n_filters, T)
        if self.training:
            x = self.spec_aug(x)
        x = self.tcn(x)                 # (B, tcn_channels, T)
        x = self.pool(x)                # (B, tcn_channels)
        return self.classifier(x)       # (B, num_classes)

    # ── Mixup ────────────────────────────────────────────────────────────

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
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
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
