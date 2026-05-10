"""
HydroVQCFeatures — Variational Quantum Classifier on cached statistical features
================================================================================

Architecture
------------
  Cached scalar+PSD+spectral feature vector (B, 138)
      ↓
  LayerNorm                 — normalise heterogeneous feature scales
      ↓
  Linear(138 → 16) + tanh   — learned PCA-style compression
      ↓
  QuantumHead               — Linear → AngleEncoder → VariationalAnsatz
                              → MeasureAll(PauliZ) → Linear → logits
      ↓
  Logits (B, num_classes)

Why this model
--------------
This is the smallest possible quantum-classifier footprint in the project: no
deep classical backbone, just a learned linear compression to a low-dim
representation that the variational quantum circuit can absorb. Useful as a
clean comparison against the classical SVM / XGBoost baselines that train on
the same 138-D feature vector.

Data contract
-------------
Consumes the 3-tuple ``(feat_1d, feat_2d, labels)`` produced by
``data.cached_feature_dataset.omni_collate_fn``. Only ``feat_1d`` is used.
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

from models.quantum import QuantumHead
from processing.losses import LargeMarginFocalLoss


FEATURE_1D_DIM = 138   # ZCR/RMS/crest/kurtosis/skew/HFD (6) + Welch PSD (129) + spec (3)


class HydroVQCFeatures(pl.LightningModule):
    """
    Variational Quantum Classifier on cached 138-D acoustic feature vectors.

    Args:
        num_classes      : Output classes (default 4).
        class_weights    : Inverse-frequency α weights for LMF loss.
        compress_dim     : Width of the learned linear compression (default 16).
        n_qubits         : Qubit register width (default 8).
        n_layers         : Variational-ansatz depth (default 6 — deeper than the
                           other models since there is no classical backbone).
        n_reuploads      : Data re-uploading repetitions (default 1).
        lmf_gamma        : LMF focal exponent (default 2.0).
        lmf_margin       : LMF margin (default 0.35).
        label_smoothing  : Cross-entropy label smoothing (default 0.05).
        learning_rate    : AdamW initial LR (default 3e-4).
        weight_decay     : AdamW weight decay (default 1e-2).
        warmup_epochs    : Linear warmup epochs (default 5).
        max_epochs       : Total training epochs (default 100).
    """

    def __init__(
        self,
        num_classes:     int   = 4,
        class_weights:   Optional[list] = None,
        compress_dim:    int   = 16,
        n_qubits:        int   = 8,
        n_layers:        int   = 6,
        n_reuploads:     int   = 1,
        lmf_gamma:       float = 2.0,
        lmf_margin:      float = 0.35,
        label_smoothing: float = 0.05,
        learning_rate:   float = 3e-4,
        weight_decay:    float = 1e-2,
        warmup_epochs:   int   = 5,
        max_epochs:      int   = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.norm     = nn.LayerNorm(FEATURE_1D_DIM)
        self.compress = nn.Sequential(
            nn.Linear(FEATURE_1D_DIM, compress_dim),
            nn.Tanh(),
        )
        self.qhead = QuantumHead(
            in_dim      = compress_dim,
            n_qubits    = n_qubits,
            n_layers    = n_layers,
            n_reuploads = n_reuploads,
            num_classes = num_classes,
        )

        self.criterion = LargeMarginFocalLoss(
            num_classes     = num_classes,
            alpha           = class_weights,
            gamma           = lmf_gamma,
            margin          = lmf_margin,
            label_smoothing = label_smoothing,
        )

        m_kw = dict(num_classes=num_classes, average="macro")
        self.train_acc  = MulticlassAccuracy(**m_kw)
        self.val_acc    = MulticlassAccuracy(**m_kw)
        self.val_f1     = MulticlassF1Score(**m_kw)
        self.val_mcc    = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_acc   = MulticlassAccuracy(**m_kw)
        self.test_f1    = MulticlassF1Score(**m_kw)
        self.test_mcc   = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_auroc = MulticlassAUROC(num_classes=num_classes)
        self.test_cm    = MulticlassConfusionMatrix(num_classes=num_classes)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, feat_1d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_1d : (B, 138) cached scalar+PSD+spectral feature vector.
        Returns:
            logits  : (B, num_classes)
        """
        z = self.norm(feat_1d)
        z = self.compress(z)
        return self.qhead(z)

    # ── Lightning steps ──────────────────────────────────────────────────

    @staticmethod
    def _unpack(batch):
        # OmniFeatureDataModule yields (feat_1d, feat_2d, labels); we use only feat_1d.
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            feat_1d, _feat_2d, y = batch
        else:
            feat_1d, y = batch
        return feat_1d, y

    def training_step(self, batch, batch_idx):
        x, y = self._unpack(batch)
        logits = self(x)
        loss   = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = self._unpack(batch)
        logits = self(x)
        loss   = self.criterion(logits, y)
        self.val_acc(logits, y);  self.val_f1(logits, y);  self.val_mcc(logits, y)
        self.log("val/loss", loss,         on_epoch=True, prog_bar=True)
        self.log("val/acc",  self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val/f1",   self.val_f1,  on_epoch=True, prog_bar=True)
        self.log("val/mcc",  self.val_mcc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y   = self._unpack(batch)
        logits = self(x)
        probs  = F.softmax(logits, dim=-1)
        self.test_acc(logits, y);   self.test_f1(logits, y)
        self.test_mcc(logits, y);   self.test_auroc(probs, y)
        self.test_cm(logits, y)
        self.log("test/loss",  self.criterion(logits, y), on_epoch=True)
        self.log("test/acc",   self.test_acc,   on_epoch=True)
        self.log("test/f1",    self.test_f1,    on_epoch=True)
        self.log("test/mcc",   self.test_mcc,   on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_epoch=True)

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
