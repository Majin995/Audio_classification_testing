"""
HydroLofarResNet — ResNet-50 on High-Resolution LOFAR Spectrograms
===================================================================

LOFAR (LOw Frequency Analysis and Recording) is the standard passive sonar
display: a narrow-band spectrogram restricted to < 2 560 Hz (the Nyquist
limit of the 5 120 Hz system sample rate), computed with a deliberately large
FFT to achieve the fine frequency resolution required for detecting tonal
propeller-blade-rate and ship-machinery lines.

Architecture
------------
  Raw waveform (5 120 Hz, 1 s)
      ↓
  LofarFrontend
      n_fft       = 4 096  →  10 Hz / bin  (fine enough for LOFAR tonals)
      hop_length  = 160 samples  (~31 ms per frame)
      log-power + per-sample instance normalisation
      AdaptiveAvgPool2d  →  (B, 1, 32, 256)   [time-rows × freq-columns]
      ↓
  SpecAugment  (train only — 2 freq masks + 2 time masks)
      ↓
  ResNet-50 backbone
      First conv adapted:  in_channels 3 → 1
      Pretrained ImageNet weights optionally transferred via channel averaging
      Global AdaptiveAvgPool2d → 2 048-d feature vector
      ↓
  Classifier   LayerNorm → Linear(2048, num_classes)
      ↓
  FocalLoss  (class-weighted, γ=2, label-smoothing=0.05)

Why ResNet-50 for LOFAR?
------------------------
The 2-D LOFAR image has clear spatial structure: narrow horizontal tonal
lines (freq × time ridges) that convolutional filters detect well.  ResNet-50
offers a well-understood inductive bias and strong ImageNet priors that
transfer even to single-channel acoustic images.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.models as tv_models
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from .hydro_conformer import FocalLoss, DropPath


# ═══════════════════════════════════════════════════════════════════════
#  LOFAR Spectrogram Frontend
# ═══════════════════════════════════════════════════════════════════════

class LofarFrontend(nn.Module):
    """
    Produce a normalised log-power LOFAR spectrogram image.

    Steps
    -----
    1. Power spectrogram via torch STFT (Hann window, large n_fft).
    2. Bandlimit to max_freq by zeroing bins above the cutoff.
    3. Convert to log-power: log(power + floor).
    4. Per-sample instance normalisation (zero mean, unit std) to equalise
       varying ocean noise floors across recording sites and seasons.
    5. AdaptiveAvgPool2d to a fixed (time_bins, freq_bins) output shape,
       making the frontend resolution-independent.

    Output
    ------
    (B, 1, time_bins, freq_bins)  float32  — ready as single-channel image.
    The time axis is rows (vertical) and the frequency axis is columns
    (horizontal), matching standard LOFAR display convention.
    """

    def __init__(
        self,
        sample_rate: int   = 5_120,
        n_fft:       int   = 4_096,
        hop_length:  int   = 160,
        max_freq:    float = 2_560.0,
        time_bins:   int   = 32,
        freq_bins:   int   = 256,
        log_floor:   float = 1e-9,
    ):
        super().__init__()
        self.log_floor = log_floor

        self.spec = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
            center=True,
            normalized=False,
        )

        # Frequency mask: keeps bins up to max_freq (registered as buffer
        # so it moves to the correct device automatically)
        freqs = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
        self.register_buffer("freq_mask", (freqs <= max_freq).float())

        # Fixed output size — decouples model from exact STFT frame count
        self.pool = nn.AdaptiveAvgPool2d((time_bins, freq_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) float32 waveform
        Returns:
            (B, 1, time_bins, freq_bins)
        """
        s = self.spec(x)                                 # (B, F_stft, T_frames)
        s = s * self.freq_mask.unsqueeze(-1)             # bandlimit to max_freq
        s = torch.log(s + self.log_floor)                # log-power

        # Per-sample normalisation — subtract mean, scale by std
        mu  = s.mean(dim=(-2, -1), keepdim=True)
        sig = s.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        s   = (s - mu) / sig                             # (B, F_stft, T_frames)

        # Transpose to (B, T_frames, F_stft) → add channel dim → pool
        s = s.transpose(1, 2).unsqueeze(1)               # (B, 1, T_frames, F_stft)
        s = self.pool(s)                                 # (B, 1, time_bins, freq_bins)
        return s


# ═══════════════════════════════════════════════════════════════════════
#  Spec Augment  (re-implemented for 4-D tensors)
# ═══════════════════════════════════════════════════════════════════════

class LofarSpecAugment(nn.Module):
    """
    Frequency and time masking for (B, 1, H, W) spectrograms.

    Masks are drawn uniformly in [0, max_mask] independently per sample.
    Applied only during training; no-op at eval time.
    """

    def __init__(
        self,
        n_time_masks:  int = 2,
        time_mask_max: int = 8,
        n_freq_masks:  int = 2,
        freq_mask_max: int = 32,
    ):
        super().__init__()
        self.n_time_masks  = n_time_masks
        self.time_mask_max = time_mask_max
        self.n_freq_masks  = n_freq_masks
        self.freq_mask_max = freq_mask_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        # x: (B, 1, H, W) — H = time_bins, W = freq_bins
        B, _, H, W = x.shape
        out = x.clone()

        for _ in range(self.n_time_masks):
            widths = torch.randint(1, max(2, self.time_mask_max + 1), (B,))
            starts = torch.randint(0, max(1, H - self.time_mask_max), (B,))
            for b in range(B):
                out[b, :, starts[b]:starts[b] + widths[b], :] = 0.0

        for _ in range(self.n_freq_masks):
            widths = torch.randint(1, max(2, self.freq_mask_max + 1), (B,))
            starts = torch.randint(0, max(1, W - self.freq_mask_max), (B,))
            for b in range(B):
                out[b, :, :, starts[b]:starts[b] + widths[b]] = 0.0

        return out


# ═══════════════════════════════════════════════════════════════════════
#  ResNet-50 Backbone  (1-channel input)
# ═══════════════════════════════════════════════════════════════════════

class LofarResNet50Backbone(nn.Module):
    """
    ResNet-50 with the input layer adapted for single-channel spectrograms.

    If `pretrained=True`, ImageNet weights are loaded and the 3-channel
    first conv is collapsed to 1 channel by averaging across the RGB axis —
    a common practice that preserves useful low-level edge detectors.
    """

    OUT_DIM = 2048

    def __init__(self, pretrained: bool = False):
        super().__init__()
        if pretrained:
            weights = tv_models.ResNet50_Weights.DEFAULT
        else:
            weights = None

        base = tv_models.resnet50(weights=weights)

        # Swap 3-channel conv for 1-channel conv
        orig_conv = base.conv1
        base.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                # Average RGB channels → single luminance channel
                base.conv1.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))

        # Strip the original FC classification head
        self.body = nn.Sequential(*list(base.children())[:-1])  # → (B, 2048, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) → (B, 2048)"""
        return self.body(x).flatten(1)


# ═══════════════════════════════════════════════════════════════════════
#  HydroLofarResNet  LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroLofarResNet(pl.LightningModule):
    """
    ResNet-50 LOFAR vessel acoustic classifier.

    Args:
        num_classes     : Number of vessel classes (auto-detected from data).
        class_weights   : Inverse-frequency weights for focal loss.
        sample_rate     : Audio sample rate in Hz (default 5 120).
        n_fft           : STFT window size for LOFAR spectrogram.
        hop_length      : STFT hop in samples.
        time_bins       : Fixed time dimension of the spectrogram image.
        freq_bins       : Fixed frequency dimension of the spectrogram image.
        pretrained      : Initialise ResNet-50 from ImageNet weights.
        dropout         : Dropout before the final classifier.
        learning_rate   : Peak AdamW LR.
        weight_decay    : AdamW weight decay.
        warmup_epochs   : Linear LR warmup length.
        max_epochs      : Total training epochs (for cosine schedule).
        mixup_alpha     : Waveform-level Mixup β distribution α (0 = off).
        focal_gamma     : Focal loss focusing exponent γ.
        label_smoothing : Label smoothing ε.
    """

    def __init__(
        self,
        num_classes:     int            = 3,
        class_weights:   Optional[list] = None,
        sample_rate:     int            = 5_120,
        n_fft:           int            = 4_096,
        hop_length:      int            = 160,
        time_bins:       int            = 32,
        freq_bins:       int            = 256,
        pretrained:      bool           = False,
        dropout:         float          = 0.2,
        learning_rate:   float          = 3e-4,
        weight_decay:    float          = 1e-2,
        warmup_epochs:   int            = 10,
        max_epochs:      int            = 80,
        mixup_alpha:     float          = 0.3,
        focal_gamma:     float          = 2.0,
        label_smoothing: float          = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Frontend ────────────────────────────────────────────────────
        self.frontend  = LofarFrontend(
            sample_rate=sample_rate, n_fft=n_fft,
            hop_length=hop_length, time_bins=time_bins, freq_bins=freq_bins,
        )
        self.spec_aug  = LofarSpecAugment(
            n_time_masks=2, time_mask_max=8,
            n_freq_masks=2, freq_mask_max=32,
        )

        # ── Backbone ────────────────────────────────────────────────────
        self.backbone  = LofarResNet50Backbone(pretrained=pretrained)

        # ── Classifier ──────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(LofarResNet50Backbone.OUT_DIM),
            nn.Dropout(dropout),
            nn.Linear(LofarResNet50Backbone.OUT_DIM, num_classes),
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
            waveform: (B, T) float32 at sample_rate Hz
        Returns:
            logits: (B, num_classes)
        """
        x = self.frontend(waveform)   # (B, 1, time_bins, freq_bins)
        x = self.spec_aug(x)          # (B, 1, time_bins, freq_bins)
        x = self.backbone(x)          # (B, 2048)
        return self.classifier(x)     # (B, num_classes)

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
        x, y           = batch
        x, y, y_p, lam = self._mixup(x, y)
        logits         = self(x)
        loss           = self._loss(logits, y, y_p, lam)
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
    model  = HydroLofarResNet(num_classes=3).to(device).eval()
    total  = sum(p.numel() for p in model.parameters())
    print(f"HydroLofarResNet  |  {total:,} parameters")

    x = torch.randn(2, 5_120, device=device)
    with torch.no_grad():
        spec   = model.frontend(x)
        logits = model(x)
    print(f"LOFAR spec shape : {spec.shape}")   # (2, 1, 32, 256)
    print(f"Output shape     : {logits.shape}") # (2, 3)
