"""
HydroDCN — Deep Complex Network for Underwater Acoustic Recognition
====================================================================

Architecture
------------
  Raw waveform (5120 Hz, 1 s)
      ↓
  STFT → complex (B, 2, F, T)    [real + imaginary stacked on channel dim]
      ↓
  DeepComplexMatchedFilter (DCMF) — learnable complex templates
      ↓
  ComplexConv2d × 4 stages        — Trabelsi 2018 complex convolution
      ↓
  Magnitude extraction → (B, D, F', T')
      ↓
  GlobalAveragePool2D → (B, D)
      ↓
  Classifier  (LN → Linear → GELU → Dropout → Linear)
      ↓
  FocalLoss (class-weighted)

Complex arithmetic
------------------
  ComplexConv2d: W * z = (W_re * z_re − W_im * z_im) + j(W_re * z_im + W_im * z_re)
  ComplexBN: normalises real and imaginary parts independently.
  ModReLU:  z → relu(|z| + b) · z / (|z| + ε)   (Arjovsky 2016)

Reference: Trabelsi et al., "Deep Complex Networks", ICLR 2018.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC, MulticlassConfusionMatrix,
)

from models.hydro_conformer import FocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  Complex-valued building blocks
# ═══════════════════════════════════════════════════════════════════════

class ComplexConv2d(nn.Module):
    """
    Complex-valued 2-D convolution.

    Input/output convention: channel dim = (real, imag) interleaved as
    (B, 2*C, H, W) where channels [0::2] = real, [1::2] = imag.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = False):
        super().__init__()
        self.re_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, bias=bias)
        self.im_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2*C_in, H, W)  — channels interleaved (re, im, re, im …)
        Returns:
            (B, 2*C_out, H', W')
        """
        re = x[:, 0::2]   # (B, C_in, H, W)
        im = x[:, 1::2]

        re_out = self.re_conv(re) - self.im_conv(im)
        im_out = self.re_conv(im) + self.im_conv(re)

        # Interleave: (B, C_out, H', W') → (B, 2*C_out, H', W')
        B, C, H, W = re_out.shape
        out = torch.empty(B, 2 * C, H, W, device=x.device, dtype=x.dtype)
        out[:, 0::2] = re_out
        out[:, 1::2] = im_out
        return out


class ComplexBatchNorm2d(nn.Module):
    """Independent BN on real and imaginary parts."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(n_channels)
        self.bn_im = nn.BatchNorm2d(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        re = self.bn_re(x[:, 0::2])
        im = self.bn_im(x[:, 1::2])
        B, C, H, W = re.shape
        out = torch.empty(B, 2 * C, H, W, device=x.device, dtype=x.dtype)
        out[:, 0::2] = re
        out[:, 1::2] = im
        return out


class ModReLU(nn.Module):
    """
    ModReLU: magnitude-based activation for complex tensors.
        z → relu(|z| + b) · z / (|z| + ε)
    One learnable bias per channel.
    """

    def __init__(self, n_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps  = eps
        self.bias = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        re = x[:, 0::2]   # (B, C, H, W)
        im = x[:, 1::2]
        magnitude = torch.sqrt(re ** 2 + im ** 2 + self.eps)   # (B, C, H, W)
        b     = self.bias.view(1, -1, 1, 1)
        scale = F.relu(magnitude + b) / (magnitude + self.eps)  # (B, C, H, W)
        B, C, H, W = re.shape
        out = torch.empty(B, 2 * C, H, W, device=x.device, dtype=x.dtype)
        out[:, 0::2] = re * scale
        out[:, 1::2] = im * scale
        return out


class ComplexStage(nn.Module):
    """One stage: ComplexConv2d → ComplexBN → ModReLU (optional stride)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = ComplexConv2d(in_ch, out_ch, kernel_size=3,
                                   stride=stride, padding=1)
        self.bn   = ComplexBatchNorm2d(out_ch)
        self.act  = ModReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ═══════════════════════════════════════════════════════════════════════
#  Deep Complex Matched Filter (DCMF)
# ═══════════════════════════════════════════════════════════════════════

class DeepComplexMatchedFilter(nn.Module):
    """
    Learnable complex templates convolved across the time axis of the STFT.

    Implements L learnable complex spectral templates (size: F × τ) as a
    complex Conv2d with kernel height=F (full frequency), width=τ (short time).
    Magnitude of the output = match score between each input frame and template.

    Args:
        n_freqs     : STFT frequency bins (n_fft // 2 + 1).
        n_templates : Number of matched filter templates (= output channels).
        template_len: Time span (frames) of each template.
    """

    def __init__(self, n_freqs: int, n_templates: int = 16, template_len: int = 5):
        super().__init__()
        pad_t = template_len // 2
        self.conv = ComplexConv2d(
            in_channels=1,
            out_channels=n_templates,
            kernel_size=(n_freqs, template_len),
            stride=1,
            padding=(0, pad_t),
        )
        self.bn  = ComplexBatchNorm2d(n_templates)
        self.act = ModReLU(n_templates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, F, T)  — complex STFT (re, im channels).
        Returns:
            (B, 2*n_templates, 1, T)  — magnitude-based match scores.
        """
        return self.act(self.bn(self.conv(x)))


# ═══════════════════════════════════════════════════════════════════════
#  HydroDCN
# ═══════════════════════════════════════════════════════════════════════

class HydroDCN(pl.LightningModule):
    """
    Deep Complex Network for underwater acoustic target recognition.

    Args:
        num_classes    : Number of output classes.
        class_weights  : Inverse-frequency weights for FocalLoss.
        sample_rate    : Audio sample rate.
        n_fft          : STFT window size.
        hop_length     : STFT hop length.
        dcmf_templates : Number of DCMF filter templates.
        complex_depth  : Number of complex conv stages after DCMF (3 or 4).
        base_channels  : Base number of complex channels per stage.
        dropout        : Dropout rate.
        learning_rate  : AdamW initial LR.
        weight_decay   : AdamW weight decay.
        warmup_epochs  : Linear warmup epochs.
        max_epochs     : Total training epochs.
        mixup_alpha    : Mixup Beta(α,α) strength.
        focal_gamma    : Focal loss γ.
        label_smoothing: Label smoothing ε.
    """

    def __init__(
        self,
        num_classes:    int   = 4,
        class_weights:  Optional[list] = None,
        sample_rate:    int   = 5_120,
        n_fft:          int   = 512,
        hop_length:     int   = 51,
        dcmf_templates: int   = 16,
        complex_depth:  int   = 3,
        base_channels:  int   = 16,
        dropout:        float = 0.1,
        learning_rate:  float = 3e-4,
        weight_decay:   float = 1e-2,
        warmup_epochs:  int   = 10,
        max_epochs:     int   = 100,
        mixup_alpha:    float = 0.3,
        focal_gamma:    float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        n_freqs = n_fft // 2 + 1

        # ── STFT (non-learnable, returns complex) ────────────────────────
        self.register_buffer("_stft_window",
                             torch.hann_window(n_fft), persistent=False)
        self.n_fft      = n_fft
        self.hop_length = hop_length

        # ── DCMF front-end ───────────────────────────────────────────────
        self.dcmf = DeepComplexMatchedFilter(
            n_freqs=n_freqs,
            n_templates=dcmf_templates,
            template_len=5,
        )
        # After DCMF: (B, 2*dcmf_templates, 1, T)

        # ── Complex Conv backbone ────────────────────────────────────────
        stages = []
        in_ch  = dcmf_templates
        ch     = base_channels
        for i in range(complex_depth):
            stride = 2 if i > 0 else 1
            stages.append(ComplexStage(in_ch, ch, stride=stride))
            in_ch = ch
            ch    = min(ch * 2, 128)
        self.stages = nn.Sequential(*stages)

        # Magnitude: flatten real+imag channels, then GAP
        self._feature_dim = in_ch

        # ── Classification head ──────────────────────────────────────────
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_ch),
            nn.Linear(in_ch, in_ch // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_ch // 2, num_classes),
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

    # ── STFT helper ──────────────────────────────────────────────────────

    def _compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT and return complex (B, 2, F, T) tensor.
        Channel 0 = real, channel 1 = imaginary.
        """
        B, L = waveform.shape
        stft = torch.stft(
            waveform.reshape(-1, L),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._stft_window,
            return_complex=True,
        )               # (B, F, T) complex
        # Stack real/imag → (B, 2, F, T)
        return torch.stack([stft.real, stft.imag], dim=1)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self._compute_stft(waveform)       # (B, 2, F, T)
        x = self.dcmf(x)                       # (B, 2*dcmf_templates, 1, T)

        # Reshape: treat 2*C interleaved channels as C complex channels
        # Squeeze frequency dim (=1 after DCMF)
        x = x.squeeze(2)                       # (B, 2*C, T)
        # Add fake H=1 dim for ComplexStage (expects 2-D spatial)
        x = x.unsqueeze(2)                     # (B, 2*C, 1, T)

        x = self.stages(x)                     # (B, 2*C_final, 1, T')
        x = x.squeeze(2)                       # (B, 2*C_final, T')

        # Extract magnitude
        re = x[:, 0::2]   # (B, C_final, T')
        im = x[:, 1::2]
        mag = torch.sqrt(re ** 2 + im ** 2 + 1e-8)   # (B, C_final, T')

        x = mag.mean(dim=-1)                   # GAP over time → (B, C_final)
        return self.classifier(x)

    # ── Mixup + Loss helpers (same pattern as HydroCATFISH) ──────────────

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
