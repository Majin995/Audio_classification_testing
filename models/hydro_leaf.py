"""
HydroLEAF — Learnable Audio Frontend (LEAF) + 1D ResNet Backbone.

Architecture
------------
  Raw waveform (32 kHz, 1 s)
      ↓
  GaborFilterbank       — n_filters learnable complex Gabor filters
      ↓                   center_freq: mel-initialized, ERB σ init
      ↓                   output: (B, n_filters, T)  amplitude spectrum
  GaussianLowpass       — per-filter learnable Gaussian smoothing
      ↓                   depthwise Conv1d, stride-based downsampling
      ↓                   output: (B, n_filters, T_frames)
  TrainablePCEN         — per-channel energy normalization (reused)
      ↓
  SpecAugment (train)   — freq & time masking
      ↓
  Stem Conv1d            (B, n_filters, T) → (B, C, T)
      ↓
  ResStage × n_stages    pre-activation ResBlock1d, cycling dilations [1,2,4]
      ↓
  Multi-Head Self-Attn   MHA over time axis with residual + LayerNorm
      ↓
  Attentive Stats Pool   → (B, 2C)
      ↓
  Classifier             Linear(2C→C) → BN → ReLU → Dropout → Linear(C→classes)
      ↓
  FocalLoss

LEAF reference
--------------
  Zeghidour et al. "LEAF: A Learnable Frontend for Audio Classification"
  ICLR 2021.  https://arxiv.org/abs/2101.08596

Key differences from original LEAF
------------------------------------
  - Backbone: 1D ResNet + MHA (same as HydroResNet) instead of EfficientNet
  - PCEN: reuses TrainablePCEN from hydro_conformer.py
  - SpecAugment applied to (B, 1, n_filters, T) single-channel spectrogram
  - No learnable compression exponent separate from PCEN (PCEN handles it)
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

from .hydro_conformer import TrainablePCEN, SpecAugment, DropPath, FocalLoss
from .hydro_resnet import ResBlock1d, SelfAttention1d, AttentiveStatisticsPool


# ═══════════════════════════════════════════════════════════════════════
#  LEAF Frontend Components
# ═══════════════════════════════════════════════════════════════════════

class GaborFilterbank(nn.Module):
    """
    Bank of n_filters learnable complex Gabor filters applied to raw waveform.

    Each filter is parameterised by:
      - center_freq  : carrier frequency (normalized, fraction of sample_rate),
                       initialized at mel-spaced values from min_freq to max_freq.
      - log_sigma    : log of Gaussian envelope width (samples),
                       initialized from ERB scale at each center frequency.

    Forward output is the amplitude spectrum |Gabor(x)| = sqrt(real² + imag²),
    matching the non-negative "magnitude" assumption of TrainablePCEN.
    """

    def __init__(
        self,
        n_filters:   int   = 40,
        window_size: int   = 401,    # 12.5 ms at 32 kHz — must be odd
        sample_rate: int   = 32_000,
        min_freq:    float = 60.0,
        max_freq:    float = 16_000.0,
    ):
        super().__init__()
        self.n_filters   = n_filters
        self.window_size = window_size
        self.sample_rate = sample_rate

        # ── Mel-spaced center frequencies (normalized to [0, 0.5]) ──────
        mel_min  = 2595.0 * math.log10(1.0 + min_freq / 700.0)
        mel_max  = 2595.0 * math.log10(1.0 + max_freq / 700.0)
        mel_pts  = torch.linspace(mel_min, mel_max, n_filters)
        hz_pts   = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
        norm_pts = hz_pts / sample_rate          # ∈ [0, 0.5]

        # ── ERB-based sigma initialization ──────────────────────────────
        # ERB(f) = 24.7 + 0.107 * f  (Moore & Glasberg 1983)
        # σ_hz = ERB / (2π)   then normalize to fraction of sample_rate
        erb_hz   = 24.7 + 0.107 * hz_pts
        sigma_hz = erb_hz / (2.0 * math.pi)
        # Convert to samples and take log
        sigma_samples = sigma_hz / sample_rate * window_size
        sigma_samples = sigma_samples.clamp(min=1.0)

        self.center_freq = nn.Parameter(norm_pts)              # (n_filters,)
        self.log_sigma   = nn.Parameter(torch.log(sigma_samples))  # (n_filters,)

        # Fixed time-index buffer — shared across forward calls
        t = torch.arange(
            -(window_size // 2), window_size // 2 + 1,
            dtype=torch.float32,
        )                                                        # (window_size,)
        self.register_buffer("t", t)

    def _build_filters(self) -> torch.Tensor:
        """Returns (2*n_filters, 1, window_size) real-and-imaginary filter bank."""
        mu    = self.center_freq.unsqueeze(1)                   # (F, 1)
        sigma = torch.exp(self.log_sigma).unsqueeze(1)          # (F, 1)
        t     = self.t.unsqueeze(0)                             # (1, W)

        gauss = torch.exp(-0.5 * (t / sigma) ** 2)             # (F, W)
        phase = 2.0 * math.pi * mu * t                          # (F, W)
        real  = gauss * torch.cos(phase)                        # (F, W)
        imag  = gauss * torch.sin(phase)                        # (F, W)

        # Stack real and imaginary as separate output channels
        filters = torch.cat([real, imag], dim=0)                # (2F, W)
        return filters.unsqueeze(1)                             # (2F, 1, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) raw waveform
        Returns:
            (B, n_filters, T) amplitude spectrum (non-negative)
        """
        x       = x.unsqueeze(1)                                # (B, 1, T)
        filters = self._build_filters()                         # (2F, 1, W)
        pad     = self.window_size // 2
        out     = F.conv1d(x, filters, padding=pad)             # (B, 2F, T)

        real = out[:, :self.n_filters]                          # (B, F, T)
        imag = out[:, self.n_filters:]                          # (B, F, T)
        return (real ** 2 + imag ** 2 + 1e-9).sqrt()            # (B, F, T)


class GaussianLowpass(nn.Module):
    """
    Per-filter learnable Gaussian lowpass smoother with stride-based downsampling.

    Implemented as a depthwise Conv1d whose kernel weights are re-derived from
    a learnable log_sigma parameter each forward call, so that sigma remains
    differentiable end-to-end.  The stride downsamples the power envelope to
    the desired time resolution (e.g. stride=320 → 10 ms frames at 32 kHz).
    """

    def __init__(
        self,
        n_filters:   int = 40,
        kernel_size: int = 401,     # 12.5 ms — same as Gabor window
        stride:      int = 320,     # 10 ms hop
    ):
        super().__init__()
        self.n_filters   = n_filters
        self.kernel_size = kernel_size
        self.stride      = stride

        # Initialize sigma to give a smooth envelope (≈ kernel_size/6 samples)
        init_sigma = kernel_size / 6.0
        self.log_sigma = nn.Parameter(
            torch.full((n_filters,), math.log(init_sigma))
        )

    def _build_kernels(self) -> torch.Tensor:
        """Returns normalized Gaussian kernels (n_filters, 1, kernel_size)."""
        sigma = torch.exp(self.log_sigma).unsqueeze(1)          # (F, 1)
        t     = torch.arange(
            -(self.kernel_size // 2), self.kernel_size // 2 + 1,
            dtype=torch.float32, device=sigma.device,
        ).unsqueeze(0)                                           # (1, K)

        gauss = torch.exp(-0.5 * (t / sigma) ** 2)              # (F, K)
        gauss = gauss / gauss.sum(dim=-1, keepdim=True)         # normalize
        return gauss.unsqueeze(1)                               # (F, 1, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_filters, T) amplitude spectrum
        Returns:
            (B, n_filters, T') smoothed + downsampled
        """
        kernels = self._build_kernels()                          # (F, 1, K)
        pad     = self.kernel_size // 2
        return F.conv1d(
            x, kernels,
            padding=pad, stride=self.stride,
            groups=self.n_filters,
        )                                                        # (B, F, T')


class LEAFFrontend(nn.Module):
    """
    Full LEAF frontend:  GaborFilterbank → GaussianLowpass → TrainablePCEN.

    Outputs (B, 1, n_filters, T_frames) — a single-channel 'spectrogram'
    compatible with SpecAugment which expects (B, C, F, T).
    """

    def __init__(
        self,
        n_filters:   int   = 40,
        window_size: int   = 401,
        kernel_size: int   = 401,
        stride:      int   = 320,
        sample_rate: int   = 32_000,
        min_freq:    float = 60.0,
        max_freq:    float = 16_000.0,
    ):
        super().__init__()
        self.gabor   = GaborFilterbank(n_filters, window_size, sample_rate, min_freq, max_freq)
        self.lowpass = GaussianLowpass(n_filters, kernel_size, stride)
        self.pcen    = TrainablePCEN(n_filters)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) at sample_rate Hz
        Returns:
            (B, 1, n_filters, T_frames)
        """
        x = self.gabor(waveform)          # (B, n_filters, T)
        x = self.lowpass(x)               # (B, n_filters, T_frames)
        x = x.clamp(min=1e-9)
        x = self.pcen(x)                  # (B, n_filters, T_frames)
        return x.unsqueeze(1)             # (B, 1, n_filters, T_frames)


# ═══════════════════════════════════════════════════════════════════════
#  HydroLEAF LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroLEAF(pl.LightningModule):
    """
    LEAF frontend + 1D ResNet backbone for vessel acoustic classification.

    The frontend (GaborFilterbank + GaussianLowpass + TrainablePCEN) replaces
    the fixed mel filterbank used in HydroResNet / HydroConformer, allowing
    the model to learn optimal frequency resolution and temporal smoothing
    directly from data.

    Args:
        num_classes     : Number of vessel classes.
        class_weights   : Inverse-frequency weights for focal loss.
        sample_rate     : Audio sample rate (Hz).
        n_filters       : Number of LEAF filters (analogous to n_mels).
        window_size     : Gabor filter window length (samples, odd).
        lowpass_size    : Gaussian lowpass kernel length (samples, odd).
        hop_length      : Lowpass stride — controls time resolution (samples).
        min_freq        : Lowest filter center frequency (Hz).
        max_freq        : Highest filter center frequency (Hz).
        channels (C)    : Residual channel width.
        n_stages        : Number of residual stages.
        n_blocks        : ResBlock1d per stage.
        kernel_size     : Conv kernel in ResBlock1d.
        n_heads         : Attention heads.
        dropout         : Dropout rate.
        drop_path_rate  : Max stochastic-depth rate.
        learning_rate   : Peak AdamW LR.
        weight_decay    : AdamW weight decay.
        warmup_epochs   : Linear LR warmup.
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
        n_filters:       int            = 40,
        window_size:     int            = 401,
        lowpass_size:    int            = 401,
        hop_length:      int            = 320,
        min_freq:        float          = 60.0,
        max_freq:        float          = 16_000.0,
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

        # ── LEAF Frontend ────────────────────────────────────────────────
        self.frontend = LEAFFrontend(
            n_filters   = n_filters,
            window_size = window_size,
            kernel_size = lowpass_size,
            stride      = hop_length,
            sample_rate = sample_rate,
            min_freq    = min_freq,
            max_freq    = max_freq,
        )
        self.spec_aug = SpecAugment(
            n_freq_masks=2, freq_mask_max=8,
            n_time_masks=2, time_mask_max=20,
        )

        # ── Stem: (B, 1, n_filters, T) → flatten → (B, n_filters, T) → (B, C, T)
        self.stem = nn.Sequential(
            nn.Conv1d(n_filters, channels, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        # ── Residual stages (dilation cycles 1→2→4) ─────────────────────
        _DILATIONS   = [1, 2, 4]
        total_blocks = n_stages * n_blocks
        dp_rates     = [
            drop_path_rate * i / max(total_blocks - 1, 1)
            for i in range(total_blocks)
        ]

        self.stages  = nn.ModuleList()
        block_idx    = 0
        for _ in range(n_stages):
            stage_blocks = nn.ModuleList([
                ResBlock1d(
                    channels  = channels,
                    kernel    = kernel_size,
                    dilation  = _DILATIONS[i % len(_DILATIONS)],
                    dropout   = dropout,
                    drop_path = dp_rates[block_idx + i],
                )
                for i in range(n_blocks)
            ])
            self.stages.append(stage_blocks)
            block_idx += n_blocks

        # ── Attention + pooling + classifier ────────────────────────────
        self.attention  = SelfAttention1d(channels, n_heads=n_heads, dropout=dropout)
        self.pool       = AttentiveStatisticsPool(channels)
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

        # ── Loss ─────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights   = class_weights,
            gamma           = focal_gamma,
            label_smoothing = label_smoothing,
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
        """
        Args:
            waveform: (B, T) float32 at sample_rate Hz
        Returns:
            logits: (B, num_classes)
        """
        x = self.frontend(waveform)           # (B, 1, n_filters, T_frames)
        x = self.spec_aug(x)                  # (B, 1, n_filters, T_frames)

        # Squeeze the single-channel dim to get (B, n_filters, T_frames)
        x = x.squeeze(1)

        x = self.stem(x)                      # (B, C, T)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        x = self.attention(x)                 # (B, C, T)
        x = self.pool(x)                      # (B, 2C)
        return self.classifier(x)             # (B, num_classes)

    # ── Augmentation ─────────────────────────────────────────────────────

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1).item() < self.hparams.gain_prob:
            gain = 0.6 + 0.8 * torch.rand(x.shape[0], 1, device=x.device)
            x    = x * gain
        if torch.rand(1).item() < self.hparams.noise_prob:
            snr_db  = 20.0 + 20.0 * torch.rand(1, device=x.device)
            snr_lin = 10.0 ** (snr_db / 10.0)
            sig_pwr = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-9)
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
        return (lam       * self.criterion(logits, y)
                + (1.0 - lam) * self.criterion(logits, y_perm))

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

        def lr_lambda(epoch: int) -> float:
            wu       = self.hparams.warmup_epochs
            total    = self.hparams.max_epochs
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
    model  = HydroLEAF(num_classes=3).to(device).eval()

    total      = sum(p.numel() for p in model.parameters())
    frontend_p = sum(p.numel() for p in model.frontend.parameters())
    print(f"HydroLEAF  |  {total:,} params  (frontend: {frontend_p:,}  backbone: {total-frontend_p:,})")

    x = torch.randn(2, 32_000, device=device)
    with torch.no_grad():
        logits = model(x)
    print(f"Output shape : {logits.shape}  logits={logits.tolist()}")
