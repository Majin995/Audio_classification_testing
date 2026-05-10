"""
HydroSSCPMobile — Sub-128kB Compact CNN with Knowledge Distillation
=====================================================================

Architecture
------------
  Raw waveform (5120 Hz, 1 s)
      ↓
  LogMelPCEN (shared, lightweight: n_mels=32, n_fft=256)
      ↓
  SSCP Backbone:
    Block 1: DW-Sep Conv2d (1→16, 3×3, stride 2)
    Block 2: DW-Sep Conv2d (16→32, 3×3, stride 2)
    Block 3: Pointwise-grouped Conv2d (32→48, 1×1)
    GlobalAveragePool2D → (B, 48)
      ↓
  Classifier: Linear(48 → num_classes)
      ↓
  Loss: α·CE(student) + (1-α)·T²·KLDiv(soft_student / T ‖ soft_teacher / T)
        where teacher = BAHTNet (--teacher_ckpt required at training time)

Constraint
----------
  A hard assert at step 0 verifies: param_bytes < 128 kB.
  If the model is extended beyond the default, training will fail loudly.

Knowledge Distillation
----------------------
  At training time the teacher BAHTNet is loaded from --teacher_ckpt, frozen,
  and kept on the same GPU device.  The distillation loss is only computed when
  a teacher is present; if no checkpoint is provided the model trains from
  scratch with standard CrossEntropy + FocalLoss.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC, MulticlassConfusionMatrix,
)

from models.hydro_dart_mt import LogMelPCEN
from models.hydro_conformer import FocalLoss

_PARAM_BUDGET_BYTES = 128_000   # 128 kB


# ═══════════════════════════════════════════════════════════════════════
#  Depthwise-Separable block
# ═══════════════════════════════════════════════════════════════════════

class DWSepBlock(nn.Module):
    """Depthwise + pointwise Conv2d → BN → ReLU6."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw   = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                               padding=1, groups=in_ch, bias=False)
        self.pw   = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1  = nn.BatchNorm2d(in_ch)
        self.bn2  = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu6(self.bn1(self.dw(x)))
        x = F.relu6(self.bn2(self.pw(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  HydroSSCPMobile
# ═══════════════════════════════════════════════════════════════════════

class HydroSSCPMobile(pl.LightningModule):
    """
    Sub-128kB mobile CNN for edge-deployable UATR with Knowledge Distillation.

    Args:
        num_classes    : Output classes.
        class_weights  : Inverse-frequency weights for FocalLoss.
        sample_rate    : Input audio sample rate.
        n_mels         : Mel bins (kept small to stay within budget).
        n_fft          : STFT window size.
        hop_length     : STFT hop length.
        teacher_ckpt   : Path to a trained HydroBAHTNet checkpoint for KD.
                         If None or file missing, supervised-only training.
        kd_alpha       : Weight of supervised CE loss (1-alpha = KD weight).
        kd_temp        : Softmax temperature T for soft label distillation.
        learning_rate  : AdamW initial LR.
        weight_decay   : AdamW weight decay.
        warmup_epochs  : Linear warmup epochs.
        max_epochs     : Total training epochs.
        mixup_alpha    : Mixup Beta(α,α) strength.
        focal_gamma    : Focal loss γ.
        label_smoothing: Cross-entropy smoothing ε.
    """

    def __init__(
        self,
        num_classes:    int   = 4,
        class_weights:  Optional[list] = None,
        sample_rate:    int   = 5_120,
        n_mels:         int   = 32,
        n_fft:          int   = 256,
        hop_length:     int   = 51,
        teacher_ckpt:   Optional[str] = None,
        kd_alpha:       float = 0.3,
        kd_temp:        float = 4.0,
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

        # ── Lightweight frontend ─────────────────────────────────────────
        self.frontend = LogMelPCEN(
            sample_rate=sample_rate, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length,
        )

        # ── SSCP backbone: DW-Sep × 2 + grouped PW ──────────────────────
        self.block1 = DWSepBlock(1,  16, stride=2)   # (B, 16, F/2, T/2)
        self.block2 = DWSepBlock(16, 32, stride=2)   # (B, 32, F/4, T/4)
        # Pointwise-grouped conv (groups=8 reduces params)
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 48, 1, groups=8, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU6(),
        )
        # GlobalAveragePool2D
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(48, num_classes)

        # ── Student loss ─────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

        # ── Teacher (BAHTNet, frozen) ────────────────────────────────────
        self._teacher: Optional[nn.Module] = None
        if teacher_ckpt:
            self._teacher = self._load_teacher(teacher_ckpt)

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

    # ── Teacher loader ───────────────────────────────────────────────────

    @staticmethod
    def _load_teacher(ckpt_path: str) -> Optional[nn.Module]:
        """Load BAHTNet teacher from checkpoint, freeze all params."""
        try:
            from models.hydro_bahtnet import HydroBAHTNet
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                print(f"[SSCP-Mobile] WARNING: teacher_ckpt not found: {ckpt_path}")
                return None
            teacher = HydroBAHTNet.load_from_checkpoint(str(ckpt_path))
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            print(f"[SSCP-Mobile] Loaded BAHTNet teacher from {ckpt_path}")
            return teacher
        except Exception as exc:
            print(f"[SSCP-Mobile] WARNING: Could not load teacher: {exc}")
            return None

    # ── Param budget check ───────────────────────────────────────────────

    def on_train_start(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_bytes = n_params * 4   # float32
        if param_bytes >= _PARAM_BUDGET_BYTES:
            raise RuntimeError(
                f"HydroSSCPMobile exceeds 128 kB budget: "
                f"{n_params} params = {param_bytes / 1024:.1f} kB"
            )
        print(f"[SSCP-Mobile] {n_params} params ({param_bytes / 1024:.1f} kB) ✓")
        # Move teacher to same device if loaded
        if self._teacher is not None:
            self._teacher = self._teacher.to(self.device)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.frontend(waveform)     # (B, n_mels, T)
        x = x.unsqueeze(1)              # (B, 1, n_mels, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)      # (B, 48)
        return self.classifier(x)

    # ── KD loss ──────────────────────────────────────────────────────────

    def _compute_loss(self, waveform, logits, y):
        ce_loss = self.criterion(logits, y)
        if self._teacher is None:
            return ce_loss, ce_loss, torch.tensor(0.0, device=ce_loss.device)

        T   = self.hparams.kd_temp
        alpha = self.hparams.kd_alpha
        with torch.no_grad():
            teacher_logits = self._teacher(waveform)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        soft_student = F.log_softmax(logits / T, dim=-1)
        kd_loss  = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
        total    = alpha * ce_loss + (1.0 - alpha) * kd_loss
        return total, ce_loss, kd_loss

    # ── Mixup ────────────────────────────────────────────────────────────

    def _mixup(self, x, y):
        alpha = self.hparams.mixup_alpha
        if not self.training or alpha <= 0.0:
            return x, y, y, 1.0
        lam  = float(torch.distributions.Beta(alpha, alpha).sample())
        perm = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam

    # ── Lightning steps ──────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y, y_p, lam = self._mixup(x, y)
        logits = self(x)

        if lam < 1.0:
            # Mixup: no KD for mixed targets (teacher can't mix cleanly)
            loss = lam * self.criterion(logits, y) + (1.0 - lam) * self.criterion(logits, y_p)
        else:
            loss, ce, kd = self._compute_loss(x, logits, y)
            self.log("train/kd_loss", kd, on_epoch=True, prog_bar=False)

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
