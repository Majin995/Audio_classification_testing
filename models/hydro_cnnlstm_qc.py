"""
HydroCNNLSTMQC — CNN → Bi-LSTM → Variational Quantum Classifier
================================================================

Architecture
------------
  Raw waveform (B, 5120) at 5120 Hz, 1 s
      ↓
  1-D CNN (4 strided blocks, Conv1d(1→16→32→64→128) + BN + GELU)
      → (B, 128, ~320)
      ↓
  permute → (B, ~320, 128)
      ↓
  Bi-LSTM (input=128, hidden=64, 2 layers) → take last hidden state
      → (B, 128)
      ↓
  QuantumHead (in_dim=128, n_qubits, n_layers, num_classes)
      ↓
  Logits (B, num_classes)

Why this model
--------------
The user's stated starting point: a hybrid CNN-LSTM-QC architecture for
underwater-acoustic classification. Reuses the existing raw-waveform DALI
contract (5120 Hz, 1 s) and slots a parameterised quantum circuit in for the
final classification head.
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


# ─────────────────────────────────────────────────────────────────────────
#  CNN frontend
# ─────────────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 7, s: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k,
                              stride=s, padding=k // 2, bias=False)
        self.bn   = nn.BatchNorm1d(c_out)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CNNStem1D(nn.Module):
    """Four strided Conv1d blocks: ``(B, 1, T) → (B, c_out, T // 16)``."""

    def __init__(self, channels=(16, 32, 64, 128)):
        super().__init__()
        c_in   = 1
        blocks = []
        for c_out in channels:
            blocks.append(_ConvBlock(c_in, c_out, k=7, s=2))
            c_in = c_out
        self.net = nn.Sequential(*blocks)
        self.out_channels = channels[-1]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T)  →  (B, 1, T)  →  (B, C, T_out)
        return self.net(waveform.unsqueeze(1))


# ─────────────────────────────────────────────────────────────────────────
#  Lightning model
# ─────────────────────────────────────────────────────────────────────────

class HydroCNNLSTMQC(pl.LightningModule):
    """
    Hybrid CNN → Bi-LSTM → Quantum-Classifier.

    Args:
        num_classes      : Output classes (default 4).
        class_weights    : Inverse-frequency α weights for LMF loss.
        cnn_channels     : Per-block channel widths (default (16, 32, 64, 128)).
        lstm_hidden      : Hidden dim per LSTM direction (default 64).
        lstm_layers      : Number of LSTM layers (default 2).
        n_qubits         : Qubit register width (default 8).
        n_layers         : Variational-ansatz depth (default 4).
        n_reuploads      : Encoder data re-uploading repetitions (default 1).
        lmf_gamma        : LMF focal exponent.
        lmf_margin       : LMF margin.
        label_smoothing  : Cross-entropy label smoothing.
        learning_rate    : AdamW initial LR.
        weight_decay     : AdamW weight decay.
        warmup_epochs    : Linear warmup epochs.
        max_epochs       : Total training epochs.
        mixup_alpha      : Mixup Beta(α,α) strength (default 0.3).
    """

    def __init__(
        self,
        num_classes:     int   = 4,
        class_weights:   Optional[list] = None,
        cnn_channels:    tuple = (16, 32, 64, 128),
        lstm_hidden:     int   = 64,
        lstm_layers:     int   = 2,
        n_qubits:        int   = 8,
        n_layers:        int   = 4,
        n_reuploads:     int   = 1,
        lmf_gamma:       float = 2.0,
        lmf_margin:      float = 0.35,
        label_smoothing: float = 0.05,
        learning_rate:   float = 3e-4,
        weight_decay:    float = 1e-2,
        warmup_epochs:   int   = 10,
        max_epochs:      int   = 100,
        mixup_alpha:     float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cnn = CNNStem1D(channels=tuple(cnn_channels))
        self.lstm = nn.LSTM(
            input_size    = self.cnn.out_channels,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = 0.1 if lstm_layers > 1 else 0.0,
        )
        # Bi-LSTM final hidden state: 2 directions × hidden
        lstm_out_dim = 2 * lstm_hidden
        self.qhead = QuantumHead(
            in_dim      = lstm_out_dim,
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

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (B, T) raw audio at 5120 Hz, 1 s.
        Returns:
            logits   : (B, num_classes)
        """
        feats = self.cnn(waveform)                # (B, C, T_out)
        feats = feats.permute(0, 2, 1)            # (B, T_out, C)
        _out, (h_n, _c_n) = self.lstm(feats)
        # h_n: (num_layers * 2, B, lstm_hidden) — take the last layer's two directions
        last_fwd = h_n[-2]                        # (B, hidden)
        last_bwd = h_n[-1]                        # (B, hidden)
        z = torch.cat([last_fwd, last_bwd], dim=-1)   # (B, 2*hidden)
        return self.qhead(z)

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
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        self.val_acc(logits, y);  self.val_f1(logits, y);  self.val_mcc(logits, y)
        self.log("val/loss", loss,         on_epoch=True, prog_bar=True)
        self.log("val/acc",  self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val/f1",   self.val_f1,  on_epoch=True, prog_bar=True)
        self.log("val/mcc",  self.val_mcc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y   = batch
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
