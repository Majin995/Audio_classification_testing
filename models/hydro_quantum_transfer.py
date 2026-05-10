"""
HydroQuantumTransfer — frozen UATR backbone + dressed Quantum-Classifier head
==============================================================================

Architecture
------------
  Raw waveform (B, 5120) at 5120 Hz, 1 s
      ↓
  Frozen teacher backbone (e.g. BAHTNet, DART-MT) — ``classifier`` stripped
      → (B, feat_dim) penultimate features (no gradient)
      ↓
  Dressed block:
      Linear(feat_dim → 32) → tanh
      → QuantumHead(32 → num_classes)
      → Linear(num_classes → num_classes)
      ↓
  Logits (B, num_classes)

Why this model
--------------
Quantum Transfer Learning (Mari et al. 2020) is the cheapest way to evaluate
whether a parameterised quantum classifier offers any value-add over an
existing strong UATR backbone. The backbone is frozen — only the dressed
quantum block is trained — so convergence is fast, and the comparison
isolates the contribution of the quantum head.

Loading the teacher
-------------------
Provide ``teacher_arch`` (registry key, e.g. ``"bahtnet"``) and
``teacher_ckpt`` (a Lightning checkpoint path). The backbone's classifier
attribute (default ``"classifier"``) is replaced with ``nn.Identity()`` so
``teacher.forward(audio)`` yields penultimate features instead of logits.
"""

from __future__ import annotations

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

from models.quantum import QuantumHead
from processing.losses import LargeMarginFocalLoss


def _infer_classifier_in_dim(classifier_module: nn.Module) -> int:
    """Return the input dim of the first nn.Linear inside a classifier module."""
    if isinstance(classifier_module, nn.Linear):
        return classifier_module.in_features
    for m in classifier_module.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    raise ValueError(
        "Could not infer feature dim — no nn.Linear found in the classifier "
        "module. Pass --teacher_feat_dim explicitly."
    )


class HydroQuantumTransfer(pl.LightningModule):
    """
    Frozen-teacher hybrid quantum-classical model.

    Args:
        num_classes        : Output classes (default 4).
        class_weights      : Inverse-frequency α weights for LMF loss.
        teacher_arch       : Registry key for the backbone (default ``"bahtnet"``).
        teacher_ckpt       : Path to a Lightning ``.ckpt`` for the teacher
                             (``None`` → randomly-initialised teacher; useful
                             only for smoke testing the wiring).
        teacher_kwargs     : Optional dict forwarded to ``build_model``.
        teacher_feat_dim   : Override for the auto-detected feature dim.
        teacher_classifier_attr : Attribute name on the teacher to replace with
                             ``Identity`` (default ``"classifier"``).
        dressed_dim        : Width of the linear bridge before the quantum head
                             (default 32).
        n_qubits           : Qubit register width (default 8).
        n_layers           : Variational-ansatz depth (default 4).
        n_reuploads        : Encoder data re-uploading repetitions.
        lmf_gamma, lmf_margin, label_smoothing : LMF hyperparameters.
        learning_rate, weight_decay, warmup_epochs, max_epochs : Optimiser.
    """

    def __init__(
        self,
        num_classes:     int   = 4,
        class_weights:   Optional[list] = None,
        teacher_arch:    str   = "bahtnet",
        teacher_ckpt:    Optional[str] = None,
        teacher_kwargs:  Optional[dict] = None,
        teacher_feat_dim: Optional[int] = None,
        teacher_classifier_attr: str = "classifier",
        dressed_dim:     int   = 32,
        n_qubits:        int   = 8,
        n_layers:        int   = 4,
        n_reuploads:     int   = 1,
        lmf_gamma:       float = 2.0,
        lmf_margin:      float = 0.35,
        label_smoothing: float = 0.05,
        learning_rate:   float = 1e-3,    # head-only training tolerates higher LR
        weight_decay:    float = 1e-2,
        warmup_epochs:   int   = 3,
        max_epochs:      int   = 30,
    ):
        super().__init__()
        # Avoid pickling potentially large teacher_kwargs into checkpoints.
        self.save_hyperparameters(ignore=["teacher_kwargs", "class_weights"])

        # ── Build & load teacher backbone ───────────────────────────────
        from processing.registry import build_model
        tk = dict(teacher_kwargs or {})
        tk.setdefault("num_classes", num_classes)

        # If a checkpoint is provided, auto-pull its hyper_parameters so the
        # teacher is reconstructed with the exact architecture it was trained
        # with (fusion_dim, model_dim, n_layers, …). Explicit teacher_kwargs
        # passed in still win over the ckpt-derived ones.
        ckpt = None
        if teacher_ckpt is not None:
            ckpt_path = Path(teacher_ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Teacher checkpoint not found: {ckpt_path}\n"
                    f"  Train a {teacher_arch} backbone first (see "
                    f"training/train_{teacher_arch}.py) and pass its best "
                    f"checkpoint path via --teacher_ckpt."
                )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_hp = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
            for k, v in ckpt_hp.items():
                # ``class_weights`` are dataset-derived and may be passed in
                # explicitly via the constructor; never override.
                if k == "class_weights":
                    continue
                tk.setdefault(k, v)

        teacher = build_model(teacher_arch, **tk)

        if ckpt is not None:
            sd   = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = teacher.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"[HydroQuantumTransfer] teacher load — "
                      f"missing={len(missing)}, unexpected={len(unexpected)} keys")

        # ── Strip classifier; auto-detect penultimate dim ────────────────
        if not hasattr(teacher, teacher_classifier_attr):
            raise AttributeError(
                f"Teacher '{teacher_arch}' has no attribute "
                f"'{teacher_classifier_attr}'. Pass "
                f"--teacher_classifier_attr <name> to point at its head."
            )
        classifier_mod = getattr(teacher, teacher_classifier_attr)
        feat_dim = teacher_feat_dim or _infer_classifier_in_dim(classifier_mod)
        setattr(teacher, teacher_classifier_attr, nn.Identity())

        # Freeze every teacher parameter (fast head-only training).
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        self.teacher = teacher
        self.teacher_feat_dim = feat_dim

        # ── Dressed quantum block ────────────────────────────────────────
        self.dressed_pre = nn.Sequential(
            nn.Linear(feat_dim, dressed_dim),
            nn.Tanh(),
        )
        self.qhead = QuantumHead(
            in_dim      = dressed_dim,
            n_qubits    = n_qubits,
            n_layers    = n_layers,
            n_reuploads = n_reuploads,
            num_classes = num_classes,
        )
        self.dressed_post = nn.Linear(num_classes, num_classes)

        # ── Loss & metrics ───────────────────────────────────────────────
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
        with torch.no_grad():
            feats = self.teacher(waveform)        # (B, feat_dim)
        z = self.dressed_pre(feats)               # (B, dressed_dim)
        q = self.qhead(z)                         # (B, num_classes)
        return self.dressed_post(q)               # (B, num_classes)

    # Keep the teacher in eval() even when Lightning toggles train mode.
    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "teacher"):
            self.teacher.eval()
        return self

    # ── Lightning steps ──────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
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

    # ── Optimiser (only trainable params — i.e. the dressed quantum block) ─

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=self.hparams.weight_decay,
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
