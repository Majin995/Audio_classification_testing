"""
HydroWaveScattering — Kymatio Wavelet Scattering + SE-Res2 + S4D Classifier
===========================================================================

Strictly spectrogram-free.  Replaces the LearnableGaborFilterbank of
HydroWave1D with a fixed multi-scale wavelet scattering transform from
Kymatio.  Output of Scattering1D is a (B, channels, T_scat) tensor of
wavelet envelope coefficients — mathematically derived, not a
time-frequency image.

Dependency
----------
- kymatio >= 0.3.0   (pip install kymatio)

Architecture
------------
    raw (B, 5120) → Scattering1D(J=6, Q=12, shape=(5120,))
                  → log1p → InstanceNorm
                  → Conv1d proj → SE-Res2 × 3 → SaShiMi × 2
                  → AttentiveStatisticsPool → Head(num_classes + 1)
                  → LargeMarginFocalLoss + Deep-Gamblers

Reused: _SERes2Block, SaShiMiBlock, _AttentiveStatisticsPool, LMF loss,
        _WaveformAug.
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

from models.hydro_conformer import FocalLoss
from models.hydro_fusion    import _SERes2Block, _AttentiveStatisticsPool
from models.hydro_s4        import SaShiMiBlock
from models.hydro_wave1d    import _WaveformAug
from processing.losses      import LargeMarginFocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  Scattering frontend (kymatio wrapper)
# ═══════════════════════════════════════════════════════════════════════

class _ScatteringFrontend(nn.Module):
    """
    Fixed-parameter Scattering1D from kymatio.  Output shape depends on
    (input_size, J, Q); determined dynamically in __init__ via a dummy pass.

    All-fp32 — kymatio does not support AMP on the scattering op.
    """

    def __init__(self, input_size: int = 5_120, J: int = 6, Q: int = 12):
        super().__init__()
        try:
            from kymatio.torch import Scattering1D
        except ImportError as e:
            raise ImportError(
                "HydroWaveScattering requires the 'kymatio' package. "
                "Install with:  pip install kymatio>=0.3.0"
            ) from e

        self.scat = Scattering1D(J=J, shape=(input_size,), Q=Q)
        with torch.no_grad():
            probe = torch.zeros(1, input_size)
            y = self.scat(probe)              # (1, C_scat, T_scat)
        self.out_channels = int(y.shape[1])
        self.out_time     = int(y.shape[2])
        self.input_size   = input_size
        self.norm = nn.InstanceNorm1d(self.out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T)  mono waveform.
        Returns:
            (B, out_channels, out_time) normalised scattering coefficients.
        """
        in_dtype = x.dtype
        y = self.scat(x.float())              # (B, C, T_scat)
        y = torch.log1p(y.abs().clamp(min=0.0))
        return self.norm(y).to(in_dtype)


# ═══════════════════════════════════════════════════════════════════════
#  HydroWaveScattering LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroWaveScattering(pl.LightningModule):
    """
    Wavelet scattering + 1D SE-Res2 + S4D classifier with Deep-Gamblers
    abstention.  Same training contract as HydroWave1D / HydroPrecise.
    """

    def __init__(
        self,
        num_classes:     int   = 4,
        class_weights:   Optional[list] = None,
        sample_rate:     int   = 5_120,
        # ── Scattering frontend ──────────────────────────────────────────
        input_size:      int   = 5_120,
        J:               int   = 6,
        Q:               int   = 12,
        # ── Backbone ─────────────────────────────────────────────────────
        d_model:         int   = 128,
        se_res2_blocks:  int   = 3,
        s4_blocks:       int   = 2,
        s4_d_state:      int   = 64,
        dropout:         float = 0.15,
        drop_path:       float = 0.10,
        # ── Loss ─────────────────────────────────────────────────────────
        loss:            str   = "lmf",
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

        self.wave_aug = _WaveformAug(
            noise_prob=noise_prob, noise_snr_min=noise_snr_min,
            noise_snr_max=noise_snr_max,
            gain_prob=gain_prob, gain_range=gain_range,
        )

        # ── Scattering frontend ──────────────────────────────────────────
        self.front = _ScatteringFrontend(input_size=input_size, J=J, Q=Q)

        # ── Projection to d_model channels ───────────────────────────────
        self.proj = nn.Sequential(
            nn.Conv1d(self.front.out_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )

        # ── SE-Res2 stack ────────────────────────────────────────────────
        dilations = [2, 4, 8][:se_res2_blocks] if se_res2_blocks <= 3 else \
                    [2 ** (i + 1) for i in range(se_res2_blocks)]
        self.res2 = nn.Sequential(*[
            _SERes2Block(d_model, scale=8, kernel_size=3,
                         dilation=dilations[i], dropout=dropout,
                         drop_path=drop_path)
            for i in range(se_res2_blocks)
        ])

        # ── SaShiMi (S4D) tail ───────────────────────────────────────────
        self.s4 = nn.ModuleList([
            SaShiMiBlock(d_model=d_model, d_state=s4_d_state,
                         expansion=4, dropout=dropout, drop_path=drop_path)
            for _ in range(s4_blocks)
        ])

        # ── Pooling + head ───────────────────────────────────────────────
        self.pool = _AttentiveStatisticsPool(d_model)    # → (B, 2*d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
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

        # ── Metrics (identical keys to HydroPrecise / HydroWave1D) ───────
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
        s = self.front(waveform)                        # (B, C_scat, T_scat)
        x = self.proj(s)                                # (B, d, T_scat)
        x = self.res2(x)                                # (B, d, T_scat)

        y = x.transpose(1, 2)                           # (B, T_scat, d)
        for blk in self.s4:
            y = blk(y)
        x = y.transpose(1, 2)                           # (B, d, T_scat)

        pooled = self.pool(x)                           # (B, 2*d)
        return self.head(pooled)                        # (B, num_classes + 1)

    # ── Losses / Lightning steps (identical pattern to HydroWave1D) ──────

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
