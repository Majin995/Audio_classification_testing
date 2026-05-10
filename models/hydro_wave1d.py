"""
HydroWave1D — Spectrogram-Free High-Precision UATR Classifier
=============================================================

Single-branch, strictly 1D time-domain classifier. No torch.stft, no Mel,
no CQT, no 2D/image backbones anywhere in the forward graph.

Architecture
------------
    raw waveform (B, T=5120) at 5120 Hz
      → _WaveformAug (train-only: Gaussian noise + gain)
      → LearnableGaborFilterbank (64 filters) → (B, 64, T)   [1D multichannel]
      → log1p |·|
      → SpecAugment1D (train-only: channel + time masking)
      → Conv1d stride-4 × 2  (1D stem)                 → (B, 128, T/16)
      → _SERes2Block × 3  (dilations 2 / 4 / 8)        → (B, 128, T')
      → SaShiMiBlock × 2  (S4D long-range memory)      → (B, T', 128)
      → _AttentiveStatisticsPool (1D time-axis)        → (B, 256)
      → + Gabor-band DSP statistics (9 scalars from
         Gabor output channel groups, NO STFT)         → (B, 265)
      → Head: Linear → GELU → Dropout → Linear(num_classes + 1)
      → LargeMarginFocalLoss + 0.1·Deep-Gamblers gambler loss

The last output logit is an abstention head used only during training
(Deep-Gamblers auxiliary).  Inference uses logits[:, :num_classes].

Reused components
-----------------
- LearnableGaborFilterbank, SpecAugment1D  (models/hydro_catfish.py)
- _SERes2Block, _AttentiveStatisticsPool   (models/hydro_fusion.py)
- SaShiMiBlock                             (models/hydro_s4.py)
- LargeMarginFocalLoss                     (processing/losses/lmf.py)
- FocalLoss                                (models/hydro_conformer.py)
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
    MulticlassRecall, MulticlassAUROC, MulticlassConfusionMatrix,
    MulticlassMatthewsCorrCoef,
)

from models.hydro_catfish   import LearnableGaborFilterbank, SpecAugment1D
from models.hydro_conformer import FocalLoss
from models.hydro_fusion    import _SERes2Block, _AttentiveStatisticsPool
from models.hydro_s4        import SaShiMiBlock
from processing.losses      import LargeMarginFocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  Waveform-level train augmentation (pure 1D: no reshape, no STFT)
# ═══════════════════════════════════════════════════════════════════════

class _WaveformAug(nn.Module):
    """Additive Gaussian noise at a random SNR + random ±gain.  Train-only."""

    def __init__(self, noise_prob: float = 0.5,
                 noise_snr_min: float = 15.0, noise_snr_max: float = 30.0,
                 gain_prob: float = 0.5, gain_range: float = 0.3):
        super().__init__()
        self.noise_prob, self.noise_snr_min, self.noise_snr_max = noise_prob, noise_snr_min, noise_snr_max
        self.gain_prob,  self.gain_range = gain_prob, gain_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1).item() < self.noise_prob:
            snr = self.noise_snr_min + (self.noise_snr_max - self.noise_snr_min) * torch.rand(1).item()
            sig_pow   = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12)
            noise_pow = sig_pow / (10.0 ** (snr / 10.0))
            x = x + torch.randn_like(x) * noise_pow.sqrt()
        if torch.rand(1).item() < self.gain_prob:
            g = 1.0 + (2.0 * torch.rand(1, device=x.device).item() - 1.0) * self.gain_range
            x = x * g
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Gabor-band DSP statistics (no STFT — operates on Gabor output only)
# ═══════════════════════════════════════════════════════════════════════

class _GaborBandStats(nn.Module):
    """
    Compute RMS, variance, kurtosis over 3 filter-index groups of the
    LearnableGaborFilterbank output.  Returns (B, 9) scalars.

    Pure 1D — no torch.stft, no spectrogram.  Uses the already-sorted-
    by-frequency Gabor channels (mel-spaced initialisation) and groups
    them into low/mid/high bands by index.
    """

    def __init__(self, n_filters: int = 64):
        super().__init__()
        self.n_filters = n_filters
        lo = n_filters // 3
        hi = 2 * n_filters // 3
        self.register_buffer("ranges", torch.tensor([[0, lo], [lo, hi], [hi, n_filters]]))

    def forward(self, gabor_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gabor_out: (B, n_filters, T) — envelope output of LearnableGaborFilterbank.
        Returns:
            (B, 9) scalar features.
        """
        feats = []
        # Collapse the time axis first: per-channel energy = mean(|x|^2)
        energy = gabor_out.pow(2).mean(dim=-1)           # (B, n_filters)
        for i in range(3):
            a = int(self.ranges[i, 0]); b = int(self.ranges[i, 1])
            band = energy[:, a:b]                        # (B, b-a)
            rms  = band.mean(dim=-1).clamp(min=1e-12).sqrt()
            var  = band.var(dim=-1, unbiased=False)
            mu   = band.mean(dim=-1, keepdim=True)
            sig  = band.std(dim=-1, unbiased=False, keepdim=True).clamp(min=1e-9)
            kurt = ((band - mu) / sig).pow(4).mean(dim=-1)
            feats += [rms, var, kurt]
        return torch.stack(feats, dim=-1)                # (B, 9)


# ═══════════════════════════════════════════════════════════════════════
#  HydroWave1D LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroWave1D(pl.LightningModule):
    """
    Strictly 1D time-domain UATR classifier with Deep-Gamblers abstention.

    The output layer produces ``num_classes + 1`` logits; the last one is
    the abstention logit used only by the auxiliary loss.  Inference uses
    ``logits[:, :num_classes]``.
    """

    def __init__(
        self,
        num_classes:     int   = 4,
        class_weights:   Optional[list] = None,
        sample_rate:     int   = 5_120,
        # ── Frontend ─────────────────────────────────────────────────────
        gabor_n_filters: int   = 64,
        gabor_kernel:    int   = 257,
        # ── Backbone ─────────────────────────────────────────────────────
        d_model:         int   = 128,
        stem_stride:     int   = 4,   # applied twice → total decimation 16
        se_res2_blocks:  int   = 3,
        s4_blocks:       int   = 2,
        s4_d_state:      int   = 64,
        dropout:         float = 0.15,
        drop_path:       float = 0.10,
        # ── Loss ─────────────────────────────────────────────────────────
        loss:            str   = "lmf",             # "lmf" | "focal"
        lmf_gamma:       float = 2.0,
        lmf_margin:      float = 0.5,
        label_smoothing: float = 0.05,
        gambler_o:       float = 0.3,
        gambler_weight:  float = 0.1,
        # ── Augmentation ─────────────────────────────────────────────────
        noise_prob:      float = 0.5,
        noise_snr_min:   float = 15.0,
        noise_snr_max:   float = 30.0,
        gain_prob:       float = 0.5,
        gain_range:      float = 0.3,
        # ── Optimiser ────────────────────────────────────────────────────
        learning_rate:   float = 3e-4,
        weight_decay:    float = 1e-2,
        warmup_epochs:   int   = 10,
        max_epochs:      int   = 100,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.num_classes    = num_classes
        self.gambler_o      = gambler_o
        self.gambler_weight = gambler_weight

        # ── Waveform aug (pure 1D, no STFT) ──────────────────────────────
        self.wave_aug = _WaveformAug(
            noise_prob=noise_prob, noise_snr_min=noise_snr_min,
            noise_snr_max=noise_snr_max,
            gain_prob=gain_prob, gain_range=gain_range,
        )

        # ── Frontend: LearnableGaborFilterbank + SpecAugment1D ───────────
        self.filterbank = LearnableGaborFilterbank(
            n_filters=gabor_n_filters, kernel_size=gabor_kernel,
            sample_rate=sample_rate,
        )
        self.spec_aug = SpecAugment1D(
            n_freq_masks=2, freq_mask_max=6,
            n_time_masks=2, time_mask_max=80,
        )

        # ── 1D stem: two stride-`stem_stride` convs (no 2D reshape) ──────
        self.stem = nn.Sequential(
            nn.Conv1d(gabor_n_filters, d_model, kernel_size=7,
                      stride=stem_stride, padding=3, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5,
                      stride=stem_stride, padding=2, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )

        # ── SE-Res2 stack (channel-first 1D) ─────────────────────────────
        dilations = [2, 4, 8][:se_res2_blocks] if se_res2_blocks <= 3 else \
                    [2 ** (i + 1) for i in range(se_res2_blocks)]
        self.res2 = nn.Sequential(*[
            _SERes2Block(d_model, scale=8, kernel_size=3,
                         dilation=dilations[i], dropout=dropout,
                         drop_path=drop_path)
            for i in range(se_res2_blocks)
        ])

        # ── SaShiMi (S4D) tail for long-range tonal tracking ─────────────
        self.s4 = nn.ModuleList([
            SaShiMiBlock(d_model=d_model, d_state=s4_d_state,
                         expansion=4, dropout=dropout, drop_path=drop_path)
            for _ in range(s4_blocks)
        ])

        # ── Pooling + DSP-stat concat ────────────────────────────────────
        self.pool      = _AttentiveStatisticsPool(d_model)   # → (B, 2*d_model)
        self.band_stats = _GaborBandStats(n_filters=gabor_n_filters)

        # ── Head: includes +1 abstention logit ───────────────────────────
        head_in = 2 * d_model + 9
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes + 1),
        )

        # ── Loss ─────────────────────────────────────────────────────────
        if loss == "lmf":
            self.criterion = LargeMarginFocalLoss(
                num_classes=num_classes, alpha=class_weights,
                gamma=lmf_gamma, margin=lmf_margin,
                label_smoothing=label_smoothing,
            )
        else:
            self.criterion = FocalLoss(
                class_weights=class_weights,
                gamma=lmf_gamma, label_smoothing=label_smoothing,
            )

        # ── Metrics (identical logging keys to HydroPrecise) ─────────────
        m_macro = dict(num_classes=num_classes, average="macro")
        self.train_acc           = MulticlassAccuracy(**m_macro)
        self.val_acc             = MulticlassAccuracy(**m_macro)
        self.val_f1              = MulticlassF1Score(**m_macro)
        self.val_recall          = MulticlassRecall(**m_macro)
        self.val_precision_macro = MulticlassPrecision(**m_macro)
        self.val_precision_per   = MulticlassPrecision(num_classes=num_classes, average=None)
        self.val_mcc             = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.val_auroc           = MulticlassAUROC(num_classes=num_classes)
        self.test_acc            = MulticlassAccuracy(**m_macro)
        self.test_f1             = MulticlassF1Score(**m_macro)
        self.test_precision      = MulticlassPrecision(**m_macro)
        self.test_recall         = MulticlassRecall(**m_macro)
        self.test_mcc            = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_auroc          = MulticlassAUROC(num_classes=num_classes)
        self.test_cm             = MulticlassConfusionMatrix(num_classes=num_classes)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) mono float32 at sample_rate.
        Returns:
            (B, num_classes + 1) logits — last column is abstention.
        """
        g = self.filterbank(waveform)                # (B, nF, T)
        if self.training:
            g = self.spec_aug(g)

        x = self.stem(g)                             # (B, d, T')
        x = self.res2(x)                             # (B, d, T')

        # S4D tail expects (B, T, d)
        y = x.transpose(1, 2)
        for blk in self.s4:
            y = blk(y)
        x = y.transpose(1, 2)                        # (B, d, T')

        pooled = self.pool(x)                        # (B, 2*d)
        stats  = self.band_stats(g)                  # (B, 9)
        h      = torch.cat([pooled, stats], dim=-1)  # (B, 2*d + 9)
        return self.head(h)                          # (B, num_classes + 1)

    # ── Losses (abstention-aware) ────────────────────────────────────────

    def _split(self, logits: torch.Tensor):
        class_logits = logits[:, :self.num_classes]
        full = F.softmax(logits, dim=-1)
        return class_logits, full

    def _gambler_loss(self, full: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p_y       = full.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_abstain = full[:, -1]
        return -torch.log(p_y + self.gambler_o * p_abstain + 1e-8).mean()

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        class_logits, full = self._split(logits)
        primary = self.criterion(class_logits, y)
        if self.gambler_weight > 0.0:
            return primary + self.gambler_weight * self._gambler_loss(full, y)
        return primary

    # ── Lightning steps ──────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.wave_aug(x)
        logits = self(x)
        loss = self._compute_loss(logits, y)
        class_logits = logits[:, :self.num_classes]
        self.train_acc(class_logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        class_logits = logits[:, :self.num_classes]
        probs = F.softmax(class_logits, dim=-1)

        self.val_acc(class_logits, y)
        self.val_f1(class_logits, y)
        self.val_recall(class_logits, y)
        self.val_precision_macro(class_logits, y)
        self.val_precision_per(class_logits, y)
        self.val_mcc(class_logits, y)
        self.val_auroc(probs, y)

        self.log("val/loss",            loss,                     on_epoch=True, prog_bar=True)
        self.log("val/acc",             self.val_acc,             on_epoch=True, prog_bar=True)
        self.log("val/f1",              self.val_f1,              on_epoch=True, prog_bar=True)
        self.log("val/recall",          self.val_recall,          on_epoch=True)
        self.log("val/macro_precision", self.val_precision_macro, on_epoch=True, prog_bar=True)
        self.log("val/mcc",             self.val_mcc,             on_epoch=True)
        self.log("val/auroc",           self.val_auroc,           on_epoch=True)

    def on_validation_epoch_end(self):
        per = self.val_precision_per.compute()
        for i, v in enumerate(per):
            self.log(f"val/precision_c{i}", v, prog_bar=False)
        self.val_precision_per.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        class_logits = logits[:, :self.num_classes]
        probs = F.softmax(class_logits, dim=-1)

        self.test_acc(class_logits, y)
        self.test_f1(class_logits, y)
        self.test_precision(class_logits, y)
        self.test_recall(class_logits, y)
        self.test_mcc(class_logits, y)
        self.test_auroc(probs, y)
        self.test_cm(class_logits, y)

        self.log("test/loss",            loss,                on_epoch=True)
        self.log("test/acc",             self.test_acc,       on_epoch=True)
        self.log("test/f1",              self.test_f1,        on_epoch=True)
        self.log("test/macro_precision", self.test_precision, on_epoch=True)
        self.log("test/recall",          self.test_recall,    on_epoch=True)
        self.log("test/mcc",             self.test_mcc,       on_epoch=True)
        self.log("test/auroc",           self.test_auroc,     on_epoch=True)

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
                return (epoch + 1) / max(wu, 1)
            p = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
