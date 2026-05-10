"""
HydroBinaryEnsemble — One-vs-Rest Binary Ensemble Classifier.

Identical sub-model architecture to HydroEnsemble (HydroNet + HydroConformer)
but trained for binary (one-vs-rest) classification of a single vessel class.

Training three of these — one per class — produces three specialized experts
that can then be used as teachers for a multiclass student via distillation.

Binary teacher combination in distillation
------------------------------------------
The three binary logits b_0, b_1, b_2 are stacked and passed through softmax
to produce a valid multiclass soft-label distribution:

    p_teacher = softmax([b_cargo, b_tanker, b_tug])

This is the "logit voting" approach: a high-confidence binary prediction for
class i lifts p_i relative to the others in a calibrated way.
"""

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score, BinaryPrecision,
    BinaryRecall, BinaryAUROC,
)

import torch.nn as nn

from .hydro_net import HydroNet, _N_SUBBAND
from .hydro_conformer import HydroConformer


# ═══════════════════════════════════════════════════════════════════════
#  Binary focal loss
# ═══════════════════════════════════════════════════════════════════════

class BinaryFocalLoss(nn.Module):
    """
    Binary cross-entropy with focal down-weighting of easy examples.

    Args:
        pos_weight : Scalar weight on the positive class (use n_neg/n_pos
                     to counteract imbalance in binary setup).
        gamma      : Focal exponent (0 = standard BCE).
    """

    def __init__(self, pos_weight: float = 2.0, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets : (B,)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight, reduction="none"
        )
        p_t = torch.sigmoid(logits)
        pt  = torch.where(targets > 0.5, p_t, 1.0 - p_t)
        return ((1.0 - pt) ** self.gamma * bce).mean()


# ═══════════════════════════════════════════════════════════════════════
#  HydroBinaryEnsemble
# ═══════════════════════════════════════════════════════════════════════

class HydroBinaryEnsemble(pl.LightningModule):
    """
    Joint HydroNet + HydroConformer trained as a one-vs-rest binary classifier.

    target_class_idx  : Integer label of the positive class (e.g. 0 for Cargo).
    target_class_name : Human-readable name used for logging (e.g. "Cargo").

    Both sub-models output a single logit (num_classes=1).  Their logits are
    averaged before the binary focal loss.  Metrics are computed on the averaged
    logit after sigmoid.

    All other args mirror HydroEnsemble.
    """

    def __init__(
        self,
        target_class_idx:  int,
        target_class_name: str,
        # Binary loss
        pos_weight:        float          = 2.0,
        focal_gamma:       float          = 2.0,
        # Architecture (same defaults as HydroEnsemble)
        sample_rate:       int            = 32_000,
        n_mels:            int            = 128,
        hop_length:        int            = 320,
        net_channels:      int            = 128,
        net_scale:         int            = 8,
        net_dilations:     List[int]      = (2, 4, 8),
        net_dropout:       float          = 0.24,
        net_drop_path:     float          = 0.10,
        con_model_dim:     int            = 128,
        con_n_blocks:      int            = 4,
        con_n_heads:       int            = 16,
        con_conv_kernel:   int            = 31,
        con_dropout:       float          = 0.10,
        con_drop_path:     float          = 0.10,
        # Training
        learning_rate:     float          = 3e-4,
        weight_decay:      float          = 0.012,
        warmup_epochs:     int            = 10,
        max_epochs:        int            = 60,
        mixup_alpha:       float          = 0.0,   # mixup is tricky with binary labels
        noise_prob:        float          = 0.50,
        gain_prob:         float          = 0.70,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Sub-models: instantiated with num_classes=2 to satisfy torchmetrics
        # (MulticlassAccuracy requires num_classes >= 2), then their classifier
        # heads are immediately replaced with binary (1-output) equivalents.
        net_shared = dict(
            num_classes    = 2,      # placeholder — head replaced below
            class_weights  = None,
            sample_rate    = sample_rate,
            n_mels         = n_mels,
            hop_length     = hop_length,
            mixup_alpha    = 0.0,
            noise_prob     = 0.0,
            gain_prob      = 0.0,
            learning_rate  = learning_rate,
            weight_decay   = weight_decay,
            warmup_epochs  = warmup_epochs,
            max_epochs     = max_epochs,
            focal_gamma    = focal_gamma,
        )
        con_shared = dict(
            num_classes    = 2,
            class_weights  = None,
            sample_rate    = sample_rate,
            n_mels         = n_mels,
            hop_length     = hop_length,
            mixup_alpha    = 0.0,
            learning_rate  = learning_rate,
            weight_decay   = weight_decay,
            warmup_epochs  = warmup_epochs,
            max_epochs     = max_epochs,
            focal_gamma    = focal_gamma,
        )
        self.net = HydroNet(
            channels       = net_channels,
            scale          = net_scale,
            dilation_rates = list(net_dilations),
            dropout        = net_dropout,
            drop_path_rate = net_drop_path,
            **net_shared,
        )
        # Replace HydroNet classifier: Linear(2C+9 → C) → BN → ReLU → Dropout → Linear(C → 1)
        self.net.classifier = nn.Sequential(
            nn.Linear(net_channels * 2 + _N_SUBBAND, net_channels),
            nn.BatchNorm1d(net_channels),
            nn.ReLU(),
            nn.Dropout(net_dropout),
            nn.Linear(net_channels, 1),
        )

        self.conformer = HydroConformer(
            model_dim      = con_model_dim,
            n_blocks       = con_n_blocks,
            n_heads        = con_n_heads,
            conv_kernel    = con_conv_kernel,
            dropout        = con_dropout,
            drop_path_rate = con_drop_path,
            **con_shared,
        )
        # Replace HydroConformer classifier: LN → Linear(D → D/2) → GELU → Dropout → Linear(D/2 → 1)
        self.conformer.classifier = nn.Sequential(
            nn.LayerNorm(con_model_dim),
            nn.Linear(con_model_dim, con_model_dim // 2),
            nn.GELU(),
            nn.Dropout(con_dropout),
            nn.Linear(con_model_dim // 2, 1),
        )

        self.criterion = BinaryFocalLoss(pos_weight=pos_weight, gamma=focal_gamma)

        pfx = target_class_name
        self.train_acc  = BinaryAccuracy()
        self.val_acc    = BinaryAccuracy()
        self.val_f1     = BinaryF1Score()
        self.val_prec   = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_auroc  = BinaryAUROC()

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Returns averaged binary logit, shape (B,)."""
        l_net = self.net(waveform).squeeze(-1)        # (B,)
        l_con = self.conformer(waveform).squeeze(-1)  # (B,)
        return (l_net + l_con) * 0.5

    # ── Augmentation ────────────────────────────────────────────────────

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1).item() < self.hparams.gain_prob:
            gain = 0.6 + 0.8 * torch.rand(x.shape[0], 1, device=x.device)
            x = x * gain
        if torch.rand(1).item() < self.hparams.noise_prob:
            snr_db  = 20.0 + 20.0 * torch.rand(1, device=x.device)
            snr_lin = 10.0 ** (snr_db / 10.0)
            sig_pwr = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-9)
            x = x + torch.randn_like(x) * (sig_pwr / snr_lin).sqrt()
        return x

    def _to_binary(self, y: torch.Tensor) -> torch.Tensor:
        """Remap multiclass labels to binary: target class → 1, rest → 0."""
        return (y == self.hparams.target_class_idx).float()

    # ── Lightning steps ─────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y  = batch
        x     = self._augment(x)
        y_bin = self._to_binary(y)
        logit = self(x)
        loss  = self.criterion(logit, y_bin)
        self.train_acc(torch.sigmoid(logit), y_bin.int())
        self.log("train/loss", loss,           on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y  = batch
        y_bin = self._to_binary(y)
        logit = self(x)
        loss  = self.criterion(logit, y_bin)
        prob  = torch.sigmoid(logit)
        self.val_acc(prob,    y_bin.int())
        self.val_f1(prob,     y_bin.int())
        self.val_prec(prob,   y_bin.int())
        self.val_recall(prob, y_bin.int())
        self.val_auroc(prob,  y_bin.int())
        self.log("val/loss",   loss,            on_epoch=True, prog_bar=True)
        self.log("val/acc",    self.val_acc,    on_epoch=True, prog_bar=True)
        self.log("val/f1",     self.val_f1,     on_epoch=True, prog_bar=True)
        self.log("val/prec",   self.val_prec,   on_epoch=True)
        self.log("val/recall", self.val_recall, on_epoch=True)
        self.log("val/auroc",  self.val_auroc,  on_epoch=True)

    # ── Optimiser (same cosine-warmup schedule) ──────────────────────────

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
    model  = HydroBinaryEnsemble(
        target_class_idx=0, target_class_name="Cargo"
    ).to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"HydroBinaryEnsemble  |  {total:,} params  (target=Cargo)")
    x = torch.randn(2, 32_000, device=device)
    with torch.no_grad():
        logit = model(x)
    print(f"Output shape : {logit.shape}  probs={torch.sigmoid(logit).tolist()}")
