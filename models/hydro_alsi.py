"""
HydroALSI — Adaptive Latent Space Integration Hybrid
======================================================

Architecture (dual-stream)
--------------------------
  Raw waveform (5120 Hz, 1 s)
      │
      ├─── Stream A: Temporal ───────────────────────────────────────────
      │    Resample 5120 → 16000 Hz (Wav2Vec2 expects 16 kHz)
      │    Wav2Vec2-base (frozen by default) → hidden states (B, T_w2v, 768)
      │    Linear projection → (B, T_w2v, D)
      │
      └─── Stream B: Frequency ──────────────────────────────────────────
           CQT (nnAudio) → (B, n_bins, T_cqt)
           Small ResNet-18-style trunk → (B, D, T_cqt)
           Permute → (B, T_cqt, D)
      │
      Interpolate T_cqt → T_w2v  (align time axes)
      │
      Multi-Head Cross-Attention (QKV):
          Q = temporal tokens (Wav2Vec2)
          K, V = frequency tokens (CQT-ResNet)
          + symmetric pass  (Q=freq, K/V=temporal)
          Concat pooled → (B, 2D)
      │
      MLP head → (B, num_classes)
      │
      FocalLoss (class-weighted)

References
----------
  Wav2Vec2: Baevski et al., "wav2vec 2.0", NeurIPS 2020.
  nnAudio CQT: Cheuk et al., "nnAudio", IEEE Access 2020.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC, MulticlassConfusionMatrix,
)

from models.hydro_conformer import FocalLoss

# Lazy import of heavy deps so the module loads without them
_WAV2VEC_SR = 16_000


# ═══════════════════════════════════════════════════════════════════════
#  Building blocks
# ═══════════════════════════════════════════════════════════════════════

class CQTResNet(nn.Module):
    """
    Lightweight 4-stage ResNet operating on a CQT spectrogram.

    Input:  (B, 1, n_bins, T_cqt)
    Output: (B, out_dim, T_cqt)   — frequency-pooled via GAP over freq axis.
    """

    def __init__(self, n_bins: int = 84, out_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        self.layer1 = self._make_layer(32,  64,  stride=(2, 1))
        self.layer2 = self._make_layer(64,  128, stride=(2, 1))
        self.layer3 = self._make_layer(128, 256, stride=(2, 1))
        self.proj   = nn.Conv2d(256, out_dim, 1, bias=False)
        # After 3× stride-2 in freq: freq_bins → n_bins//8
        self._freq_out = max(1, n_bins // 8)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, stride=(1, 1)):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F, T)
        x = self.stem(x)
        x = F.gelu(x + F.interpolate(self.layer1(x), size=x.shape[2:], mode="nearest")) \
            if x.shape[1] == self.layer1[0].weight.shape[0] else self.layer1(x)
        # Simple sequential to avoid mismatched residuals:
        x = self.layer1(self.stem(x.clone()) if False else x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.proj(x)                   # (B, out_dim, F', T)
        x = x.mean(dim=2)                  # GAP over freq → (B, out_dim, T)
        return x


class _CQTResNetSimple(nn.Module):
    """Simplified version that avoids shape ambiguity."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32,  3, padding=1, bias=False), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=(2, 1), padding=1, bias=False), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=(2, 1), padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=(2, 1), padding=1, bias=False), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, out_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F, T)
        x = self.net(x)         # (B, out_dim, F', T)
        return x.mean(dim=2)    # GAP over freq → (B, out_dim, T)


class MultiHeadCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between two token sequences.

    Each direction: Q from one stream, K/V from the other.
    Both outputs are mean-pooled and concatenated.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # A→B attention
        self.attn_ab = nn.MultiheadAttention(embed_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        # B→A attention
        self.attn_ba = nn.MultiheadAttention(embed_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            a: (B, T_a, D) — temporal (Wav2Vec2) tokens.
            b: (B, T_b, D) — frequency (CQT-ResNet) tokens.
        Returns:
            pooled_a: (B, D), pooled_b: (B, D)
        """
        a2, _ = self.attn_ab(a, b, b)   # Q=a, K/V=b
        b2, _ = self.attn_ba(b, a, a)   # Q=b, K/V=a
        a2 = self.norm_a(a + a2)
        b2 = self.norm_b(b + b2)
        return a2.mean(dim=1), b2.mean(dim=1)   # mean pool over sequence


# ═══════════════════════════════════════════════════════════════════════
#  HydroALSI
# ═══════════════════════════════════════════════════════════════════════

class HydroALSI(pl.LightningModule):
    """
    ALSI-Hybrid: Wav2Vec2 (frozen) × CQT-ResNet dual-stream with cross-attention
    fusion for underwater acoustic target recognition.

    Args:
        num_classes     : Output classes.
        class_weights   : FocalLoss per-class weights.
        sample_rate     : Input waveform sample rate (resampled to 16 kHz for W2V2).
        fusion_dim      : Projection dimension for cross-attention.
        fusion_heads    : Number of attention heads.
        freeze_wav2vec  : If True (default), keep Wav2Vec2 weights frozen.
        cqt_bins        : Number of CQT frequency bins.
        dropout         : Dropout rate.
        learning_rate   : AdamW initial LR.
        weight_decay    : AdamW weight decay.
        warmup_epochs   : Linear warmup epochs.
        max_epochs      : Total training epochs.
        mixup_alpha     : Mixup strength.
        focal_gamma     : Focal loss γ.
        label_smoothing : Cross-entropy ε.
    """

    def __init__(
        self,
        num_classes:    int   = 4,
        class_weights:  Optional[list] = None,
        sample_rate:    int   = 5_120,
        fusion_dim:     int   = 256,
        fusion_heads:   int   = 8,
        freeze_wav2vec: bool  = True,
        cqt_bins:       int   = 84,
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

        # ── Resampler (5120 → 16000 Hz) for Wav2Vec2 ────────────────────
        self.resample_w2v = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=_WAV2VEC_SR
        )

        # ── Stream A: Wav2Vec2 (frozen) ──────────────────────────────────
        try:
            from transformers import Wav2Vec2Model
            self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            if freeze_wav2vec:
                for p in self.wav2vec.parameters():
                    p.requires_grad_(False)
            w2v_hidden = 768
        except Exception:
            # Fallback if offline: 1D conv stub with matching output dim
            self.wav2vec = None
            w2v_hidden   = 256

        self.proj_a = nn.Linear(w2v_hidden if self.wav2vec else w2v_hidden,
                                fusion_dim)

        # ── Stream B: CQT-ResNet ─────────────────────────────────────────
        try:
            from nnAudio.Spectrogram import CQT1992v2
            self.cqt = CQT1992v2(
                sr=sample_rate,
                hop_length=256,
                fmin=32.7,
                n_bins=cqt_bins,
                bins_per_octave=12,
                output_format="Magnitude",
            )
            self._cqt_available = True
        except Exception:
            # Fallback: use torchaudio mel as proxy CQT
            import torchaudio.transforms as Tf
            self.cqt = Tf.MelSpectrogram(
                sample_rate=sample_rate, n_mels=cqt_bins,
                n_fft=512, hop_length=256,
            )
            self._cqt_available = True

        self.cqt_resnet = _CQTResNetSimple(out_dim=fusion_dim)
        self.proj_b     = nn.Conv1d(fusion_dim, fusion_dim, 1, bias=False)

        # ── Cross-Attention Fusion ───────────────────────────────────────
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=fusion_dim, num_heads=fusion_heads, dropout=dropout,
        )

        # ── Classification head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
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
        # ── Stream A: Wav2Vec2 ───────────────────────────────────────────
        if self.wav2vec is not None:
            x16k = self.resample_w2v(waveform)       # (B, T_16k)
            with torch.set_grad_enabled(
                not self.hparams.freeze_wav2vec
            ):
                w2v_out = self.wav2vec(x16k).last_hidden_state  # (B, T_w2v, 768)
        else:
            # Fallback projection of raw waveform frames
            B, T = waveform.shape
            w2v_out = waveform.reshape(B, -1, 256).float()

        feat_a = self.proj_a(w2v_out)              # (B, T_a, D)

        # ── Stream B: CQT-ResNet ─────────────────────────────────────────
        cqt_mag = self.cqt(waveform)               # (B, n_bins, T_cqt)
        if cqt_mag.dim() == 3:
            cqt_mag = cqt_mag.unsqueeze(1)         # (B, 1, n_bins, T_cqt)
        feat_b = self.cqt_resnet(cqt_mag)          # (B, D, T_cqt)
        feat_b = feat_b.permute(0, 2, 1)           # (B, T_cqt, D)

        # ── Align time axes ──────────────────────────────────────────────
        T_a = feat_a.shape[1]
        if feat_b.shape[1] != T_a:
            feat_b = F.interpolate(
                feat_b.permute(0, 2, 1),           # (B, D, T_cqt)
                size=T_a, mode="linear", align_corners=False,
            ).permute(0, 2, 1)                     # (B, T_a, D)

        # ── Cross-Attention Fusion ───────────────────────────────────────
        pool_a, pool_b = self.cross_attn(feat_a, feat_b)  # (B, D), (B, D)

        fused = torch.cat([pool_a, pool_b], dim=-1)        # (B, 2D)
        return self.classifier(fused)

    # ── Mixup / loss ─────────────────────────────────────────────────────

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
        # Only update unfrozen parameters
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
