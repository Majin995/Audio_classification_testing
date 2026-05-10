"""
HydroResNet — 1D Convolutional ResNet with Multi-Head Self-Attention.

Architecture
------------
  Raw waveform (32 kHz, 1 s)
      ↓
  MultiScalePCEN       — 2-channel dual-resolution mel + trainable PCEN
      ↓                  (B, 2, n_mels, T_frames)
  SpecAugment (train)
      ↓
  Flatten freq→channels  (B, 2*n_mels, T_frames)
      ↓
  Stem Conv1d            (B, C, T)
      ↓
  ResStage × n_stages    each stage: n_blocks × pre-activation ResBlock1d
      ↓                  with cycling dilations [1, 2, 4] for multi-scale context
  Multi-Head Self-Attn   MHA over time axis with residual + LayerNorm
      ↓
  Attentive Stats Pool   learned-weight mean + std → (B, 2C)
      ↓
  Classifier             Linear(2C→C) → BN → ReLU → Dropout → Linear(C→classes)
      ↓
  FocalLoss

Differences from existing models
----------------------------------
  vs HydroNet     : standard 2-layer ResBlocks (no Res2Net / SE), explicit MHA
                    instead of attentive query pooling, simpler 2-ch frontend
  vs HydroConformer: no transformer positional encoding, no Conformer FFN/depthwise
                    conv blocks — pure residual tower + single attention layer
"""

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
#  Building blocks
# ═══════════════════════════════════════════════════════════════════════

class ResBlock1d(nn.Module):
    """
    Pre-activation 1D residual block.

    BN → ReLU → Conv1d → BN → ReLU → Dropout → Conv1d  +  skip
    Dilation expands receptive field without increasing parameter count.
    """

    def __init__(
        self,
        channels:   int,
        kernel:     int   = 3,
        dilation:   int   = 1,
        dropout:    float = 0.1,
        drop_path:  float = 0.0,
    ):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel,
                      dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel,
                      dilation=dilation, padding=pad, bias=False),
        )
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dp(self.block(x))


class SelfAttention1d(nn.Module):
    """
    Multi-head self-attention over the time axis.

    Operates on (B, C, T), transposes internally for nn.MultiheadAttention,
    then adds a residual connection.  LayerNorm applied pre-attention.
    """

    def __init__(self, channels: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        xt = x.transpose(1, 2)               # (B, T, C)
        xt_n = self.norm(xt)
        attn_out, _ = self.attn(xt_n, xt_n, xt_n)
        return x + self.drop(attn_out.transpose(1, 2))


class AttentiveStatisticsPool(nn.Module):
    """Learned-weight mean + std pooling → (B, 2*channels)."""

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
#  HydroResNet LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroResNet(pl.LightningModule):
    """
    1D Conv ResNet + Multi-Head Self-Attention for vessel acoustic classification.

    Args:
        num_classes     : Number of vessel classes.
        class_weights   : Inverse-frequency weights for focal loss.
        sample_rate     : Audio sample rate (Hz).
        n_mels          : Mel filterbank bins.
        hop_length      : STFT hop in samples.
        channels (C)    : Channel width throughout the encoder.
        n_stages        : Number of residual stages.
        n_blocks        : ResBlock1d per stage (dilations cycle [1, 2, 4]).
        kernel_size     : Conv kernel in ResBlock1d.
        n_heads         : Attention heads for MHA (channels must be divisible).
        dropout         : Dropout rate in ResBlocks and classifier.
        drop_path_rate  : Max stochastic-depth rate (linearly scheduled).
        learning_rate   : Peak AdamW LR.
        weight_decay    : AdamW weight decay.
        warmup_epochs   : Linear LR warmup epochs.
        max_epochs      : Total epochs for cosine schedule.
        mixup_alpha     : Waveform Mixup β (0 = off).
        noise_prob      : Additive noise augmentation probability.
        gain_prob       : Random gain augmentation probability.
        focal_gamma     : Focal loss γ.
        label_smoothing : Label smoothing ε.
    """

    def __init__(
        self,
        num_classes:     int            = 3,
        class_weights:   Optional[list] = None,
        sample_rate:     int            = 32_000,
        n_mels:          int            = 128,
        hop_length:      int            = 320,
        channels:        int            = 128,
        n_stages:        int            = 3,
        n_blocks:        int            = 3,
        kernel_size:     int            = 3,
        n_heads:         int            = 8,
        dropout:         float          = 0.24,
        drop_path_rate:  float          = 0.10,
        learning_rate:   float          = 3e-4,
        weight_decay:    float          = 0.012,
        warmup_epochs:   int            = 10,
        max_epochs:      int            = 100,
        mixup_alpha:     float          = 0.20,
        noise_prob:      float          = 0.50,
        gain_prob:       float          = 0.70,
        focal_gamma:     float          = 2.0,
        label_smoothing: float          = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Frontend ────────────────────────────────────────────────────
        self.features = MultiScalePCEN(
            sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length,
        )
        self.spec_aug = SpecAugment(
            n_freq_masks=2, freq_mask_max=16,
            n_time_masks=2, time_mask_max=20,
        )

        # ── Stem: flatten (B, 2, n_mels, T) → (B, 2*n_mels, T) → (B, C, T)
        stem_in = 2 * n_mels
        self.stem = nn.Sequential(
            nn.Conv1d(stem_in, channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        # ── Residual stages (dilation cycles 1→2→4 per stage)
        _DILATIONS = [1, 2, 4]
        total_blocks = n_stages * n_blocks
        dp_rates = [drop_path_rate * i / max(total_blocks - 1, 1)
                    for i in range(total_blocks)]

        self.stages = nn.ModuleList()
        block_idx = 0
        for _ in range(n_stages):
            stage_blocks = nn.ModuleList([
                ResBlock1d(
                    channels   = channels,
                    kernel     = kernel_size,
                    dilation   = _DILATIONS[i % len(_DILATIONS)],
                    dropout    = dropout,
                    drop_path  = dp_rates[block_idx + i],
                )
                for i in range(n_blocks)
            ])
            self.stages.append(stage_blocks)
            block_idx += n_blocks

        # ── Multi-head self-attention
        self.attention = SelfAttention1d(channels, n_heads=n_heads, dropout=dropout)

        # ── Pooling + classifier
        self.pool = AttentiveStatisticsPool(channels)
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

        # ── Loss
        self.criterion = FocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

        # ── Metrics
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
            waveform: (B, T) float32 at sample_rate Hz
        Returns:
            logits: (B, num_classes)
        """
        x = self.features(waveform)             # (B, 2, n_mels, T_frames)
        x = self.spec_aug(x)                    # (B, 2, n_mels, T_frames)

        # Flatten freq channels: (B, 2, F, T) → (B, 2*F, T)
        B, C_in, F, T = x.shape
        x = x.reshape(B, C_in * F, T)

        x = self.stem(x)                        # (B, C, T)

        for stage in self.stages:
            for block in stage:
                x = block(x)                    # (B, C, T)

        x = self.attention(x)                   # (B, C, T)
        x = self.pool(x)                        # (B, 2C)
        return self.classifier(x)               # (B, num_classes)

    # ── Augmentation ────────────────────────────────────────────────────

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1).item() < self.hparams.gain_prob:
            gain = 0.6 + 0.8 * torch.rand(x.shape[0], 1, device=x.device)
            x = x * gain
        if torch.rand(1).item() < self.hparams.noise_prob:
            snr_db  = 20.0 + 20.0 * torch.rand(1, device=x.device)
            snr_lin = 10.0 ** (snr_db / 10.0)
            sig_pwr = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-9)
            x = x + torch.randn_like(x) * (sig_pwr / snr_lin).sqrt()
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
        self.log("test/acc",   self.test_acc,   on_epoch=True)
        self.log("test/f1",    self.test_f1,    on_epoch=True, prog_bar=True)
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
    model  = HydroResNet(num_classes=3).to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"HydroResNet  |  {total:,} parameters")
    print(f"  channels={model.hparams.channels}, "
          f"stages={model.hparams.n_stages}, "
          f"blocks/stage={model.hparams.n_blocks}, "
          f"heads={model.hparams.n_heads}")

    x = torch.randn(2, 32_000, device=device)
    with torch.no_grad():
        logits = model(x)
    print(f"Output shape : {logits.shape}")
