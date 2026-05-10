"""
HydroXLSR — Fine-tuned XLSR-53 (wav2vec 2.0 Large) for Vessel Classification.

Architecture
------------
  XLSR-53 (Cross-Lingual Speech Representations) extends Facebook's
  wav2vec 2.0 framework to 53 languages.  For vessel acoustic classification
  we treat it as a raw-waveform feature extractor:

    Raw waveform (32 kHz, 1 s)
        ↓
    Resample 32 kHz → 16 kHz  (XLSR-53 is trained at 16 kHz)
        ↓
    Normalize to zero-mean unit-variance
        ↓
    CNN Feature Extractor  (7 Conv1d layers, learnable, frozen)
        ↓
    Positional Encoding  (relative, Conv1d-based)
        ↓
    Transformer Encoder  (24 layers × 1024 hidden × 16 heads)
        Layers 0–17 : frozen
        Layers 18–23: fine-tuned (last 6 layers)
        ↓
    Mean Pooling over time  → (B, 1024)
        ↓
    Classifier  Linear(1024→256) → BN → ReLU → Dropout → Linear(256→C)
        ↓
    FocalLoss + waveform-level augmentation (gain, noise)

Why XLSR-53?
------------
XLSR-53 was pretrained on 56,000 hours of unlabelled speech in 53 languages
using a contrastive masked prediction objective.  Despite its speech origin,
the low-level convolutional feature extractor captures fine-grained acoustic
patterns directly from the waveform — analogous to what a hydrophone sensor
analysis pipeline would do.  Transfer to vessel sound is effective because
the CNN layers learn fundamental time-frequency structure regardless of the
source domain.

Key differences from spectrogram models:
  - No hand-crafted frequency analysis; the CNN learns its own filterbank
  - Relative positional encoding generalises across clip lengths
  - The transformer's self-attention operates over 20 ms acoustic frames,
    matching the timescale of vessel signature changes

Fine-Tuning Strategy (Partial)
-------------------------------
  - CNN feature extractor     : frozen (learned acoustic primitives)
  - Transformer layers 0–17   : frozen (general low-to-mid-level features)
  - Transformer layers 18–23  : fine-tuned (task-specific abstraction)
  - Classification head       : fully trainable
  - Pretrained layers LR      : 1e-5  (prevent catastrophic forgetting)
  - Classification head LR    : 3e-4  (same as other models)

Model Size: ~316 M parameters (XLSR-53 Large)
  - CNN feature extractor: ~1.5 M
  - Transformer 24 layers: ~309 M
  - Classification head: ~265 K
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchaudio
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from .hydro_conformer import FocalLoss

# HuggingFace transformers
from transformers import Wav2Vec2Model


# ═══════════════════════════════════════════════════════════════════════
#  HydroXLSR LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroXLSR(pl.LightningModule):
    """
    Fine-tuned XLSR-53 (wav2vec 2.0 Large) vessel classifier.

    Args
    ----
    num_classes         : Number of vessel classes.
    class_weights       : Inverse-frequency weights for FocalLoss (None = uniform).
    sample_rate         : Input audio sample rate (Hz).  Will be resampled to 16 kHz.
    model_name          : HuggingFace model ID (default: wav2vec2-large-xlsr-53).
    n_frozen_layers     : Number of transformer encoder layers to freeze
                          (counting from layer 0).  Default 18 of 24 layers.
    head_hidden         : Hidden dim of 2-layer classification head.
    dropout             : Dropout in classification head.
    pretrained_lr       : LR for fine-tuned transformer layers.
    head_lr             : LR for classification head.
    weight_decay        : AdamW weight decay.
    warmup_epochs       : Linear LR warmup epochs.
    max_epochs          : Total epochs (cosine schedule).
    noise_prob          : Additive Gaussian noise augmentation probability.
    gain_prob           : Random gain augmentation probability.
    focal_gamma         : Focal loss γ.
    label_smoothing     : Label smoothing ε.
    mixup_alpha         : Waveform Mixup β (0 = off).
    """

    def __init__(
        self,
        num_classes:     int            = 3,
        class_weights:   Optional[list] = None,
        sample_rate:     int            = 32_000,
        model_name:      str            = "facebook/wav2vec2-large-xlsr-53",
        n_frozen_layers: int            = 18,
        head_hidden:     int            = 256,
        dropout:         float          = 0.24,
        pretrained_lr:   float          = 1e-5,
        head_lr:         float          = 3e-4,
        weight_decay:    float          = 0.012,
        warmup_epochs:   int            = 10,
        max_epochs:      int            = 100,
        noise_prob:      float          = 0.50,
        gain_prob:       float          = 0.70,
        focal_gamma:     float          = 2.0,
        label_smoothing: float          = 0.001,
        mixup_alpha:     float          = 0.20,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Load XLSR-53 ─────────────────────────────────────────────────
        print(f"[HydroXLSR] Loading {model_name} ...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze CNN feature extractor (learns raw waveform primitives)
        self.wav2vec2.feature_extractor._freeze_parameters()

        # Freeze early transformer layers
        for layer in self.wav2vec2.encoder.layers[:n_frozen_layers]:
            for p in layer.parameters():
                p.requires_grad_(False)

        # Freeze feature projection (sits between CNN extractor and transformer)
        for p in self.wav2vec2.feature_projection.parameters():
            p.requires_grad_(False)

        # Hidden size: 1024 for Large, 768 for Base
        hidden_size = self.wav2vec2.config.hidden_size

        # ── Classification head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

        # ── Loss + metrics ───────────────────────────────────────────────
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

    def _preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, T) at sample_rate Hz
        Returns:  (B, T') normalised, resampled to 16 kHz
        """
        sr_in  = self.hparams.sample_rate
        sr_out = 16_000
        if sr_in != sr_out:
            waveform = torchaudio.functional.resample(waveform, sr_in, sr_out)
        # Per-utterance normalisation (wav2vec 2.0 convention)
        mean = waveform.mean(dim=-1, keepdim=True)
        std  = waveform.std(dim=-1,  keepdim=True).clamp(min=1e-9)
        return (waveform - mean) / std

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform : (B, T_samples) float32 at sample_rate Hz
        returns  : (B, num_classes) logits
        """
        x = self._preprocess(waveform)                 # (B, T')

        # wav2vec 2.0 forward — outputs (B, T_frames, hidden_size)
        # attention_mask = None means full attention (all frames valid)
        out    = self.wav2vec2(x, attention_mask=None)
        hidden = out.last_hidden_state                  # (B, T_frames, H)

        # Mean pooling over time
        emb    = hidden.mean(dim=1)                     # (B, H)
        return self.classifier(emb)                     # (B, num_classes)

    # ── Augmentation (waveform level — no SpecAugment on raw waveform) ───

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1).item() < self.hparams.gain_prob:
            gain = 0.6 + 0.8 * torch.rand(x.shape[0], 1, device=x.device)
            x    = x * gain
        if torch.rand(1).item() < self.hparams.noise_prob:
            snr_db  = 20.0 + 20.0 * torch.rand(1, device=x.device)
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

    # ── Optimiser: different LR groups ───────────────────────────────────

    def configure_optimizers(self):
        head_params, pretrained_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("classifier"):
                head_params.append(p)
            else:
                pretrained_params.append(p)

        hp = self.hparams
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params,       "lr": hp.head_lr,       "weight_decay": hp.weight_decay},
                {"params": pretrained_params, "lr": hp.pretrained_lr, "weight_decay": hp.weight_decay},
            ],
            betas=(0.9, 0.98), eps=1e-8,
        )

        def lr_lambda(epoch: int) -> float:
            wu       = hp.warmup_epochs
            total    = hp.max_epochs
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

    model  = HydroXLSR(num_classes=3).to(device).eval()
    total  = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    tuned  = total - frozen

    x_dummy = torch.randn(2, 32_000, device=device)
    with torch.no_grad():
        logits = model(x_dummy)

    hp = model.hparams
    print(f"\nHydroXLSR ({hp.model_name})")
    print(f"  Total params  : {total:,}")
    print(f"  Frozen params : {frozen:,}  (CNN extractor + first {hp.n_frozen_layers} transformer layers)")
    print(f"  Fine-tuned    : {tuned:,}   (last {24 - hp.n_frozen_layers} transformer layers + head)")
    print(f"\nOutput shape : {list(logits.shape)}")
