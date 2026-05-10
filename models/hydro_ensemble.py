"""
HydroEnsemble — Joint training of HydroNet + HydroConformer.

Both sub-models are trained from scratch on the same augmented waveform.
Validation and test metrics are computed on the averaged softmax probabilities,
which is where the ensemble benefit materialises.

Architecture summary
---------------------
  Waveform (B, T)
      │
      ├─► HydroNet     (4-ch EnhancedFrontEnd → SE-Res2Blocks → AttPool)
      │        └─► logits_net  (B, C)
      │
      └─► HydroConformer  (2-ch MultiScalePCEN → ConformerBlocks → AttPool)
               └─► logits_con  (B, C)

  Training  : loss = focal(logits_net, y) + focal(logits_con, y)
  Inference : probs = softmax(logits_net) + softmax(logits_con)   (averaged)

Both branches share the same waveform augmentations (applied once at the
ensemble level before forwarding to each sub-model's feature extractor).
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from .hydro_net import HydroNet
from .hydro_conformer import HydroConformer, FocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  Ensemble LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroEnsemble(pl.LightningModule):
    """
    Joint HydroNet + HydroConformer ensemble.

    Args (sub-model architecture)
    ─────────────────────────────
    num_classes        : Number of output classes.
    class_weights      : Inverse-frequency weights for focal loss.
    sample_rate        : Audio sample rate (Hz).
    n_mels             : Mel / gammatone filterbank bins.
    hop_length         : STFT hop in samples.

    HydroNet params
    ───────────────
    net_channels       : SE-Res2Block channel width.
    net_scale          : Res2Net split factor.
    net_dilations      : Dilation per SERes2Block.
    net_dropout        : HydroNet dropout.
    net_drop_path      : HydroNet stochastic-depth rate.

    HydroConformer params
    ──────────────────────
    con_model_dim      : Conformer hidden dimension.
    con_n_blocks       : Number of ConformerBlocks.
    con_n_heads        : Attention heads.
    con_conv_kernel    : Depthwise conv kernel (odd).
    con_dropout        : Conformer dropout.
    con_drop_path      : Conformer stochastic-depth rate.

    Shared training params
    ──────────────────────
    learning_rate      : Peak AdamW LR.
    weight_decay       : AdamW weight decay.
    warmup_epochs      : Linear LR warmup.
    max_epochs         : Total epochs for cosine schedule.
    mixup_alpha        : Waveform Mixup β (0 = off).
    noise_prob         : Probability of additive noise augmentation.
    gain_prob          : Probability of random gain augmentation.
    focal_gamma        : Focal loss γ.
    label_smoothing    : Label smoothing ε.
    """

    def __init__(
        self,
        num_classes:     int            = 3,
        class_weights:   Optional[list] = None,
        sample_rate:     int            = 32_000,
        n_mels:          int            = 128,
        hop_length:      int            = 320,
        # HydroNet
        net_channels:    int            = 128,
        net_scale:       int            = 8,
        net_dilations:   List[int]      = (2, 4, 8),
        net_dropout:     float          = 0.24,
        net_drop_path:   float          = 0.10,
        # HydroConformer
        con_model_dim:   int            = 128,
        con_n_blocks:    int            = 4,
        con_n_heads:     int            = 16,
        con_conv_kernel: int            = 31,
        con_dropout:     float          = 0.10,
        con_drop_path:   float          = 0.10,
        # Shared training
        learning_rate:   float          = 3e-4,
        weight_decay:    float          = 0.012,
        warmup_epochs:   int            = 10,
        max_epochs:      int            = 100,
        mixup_alpha:     float          = 0.20,
        noise_prob:      float          = 0.50,
        gain_prob:       float          = 0.70,
        focal_gamma:     float          = 2.0,
        label_smoothing: float          = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Sub-models ─────────────────────────────────────────────────
        # Pass dummy values for sub-model-specific training params that
        # won't be used — augmentation and mixup are handled here.
        self.net = HydroNet(
            num_classes     = num_classes,
            class_weights   = class_weights,
            sample_rate     = sample_rate,
            n_mels          = n_mels,
            hop_length      = hop_length,
            channels        = net_channels,
            scale           = net_scale,
            dilation_rates  = list(net_dilations),
            dropout         = net_dropout,
            drop_path_rate  = net_drop_path,
            learning_rate   = learning_rate,   # unused (ensemble owns optimizer)
            weight_decay    = weight_decay,
            warmup_epochs   = warmup_epochs,
            max_epochs      = max_epochs,
            mixup_alpha     = 0.0,             # disabled — ensemble handles mixup
            noise_prob      = 0.0,             # disabled — ensemble handles noise
            gain_prob       = 0.0,             # disabled — ensemble handles gain
            focal_gamma     = focal_gamma,
            label_smoothing = label_smoothing,
        )

        self.conformer = HydroConformer(
            num_classes     = num_classes,
            class_weights   = class_weights,
            sample_rate     = sample_rate,
            n_mels          = n_mels,
            hop_length      = hop_length,
            model_dim       = con_model_dim,
            n_blocks        = con_n_blocks,
            n_heads         = con_n_heads,
            conv_kernel     = con_conv_kernel,
            dropout         = con_dropout,
            drop_path_rate  = con_drop_path,
            learning_rate   = learning_rate,
            weight_decay    = weight_decay,
            warmup_epochs   = warmup_epochs,
            max_epochs      = max_epochs,
            mixup_alpha     = 0.0,
            focal_gamma     = focal_gamma,
            label_smoothing = label_smoothing,
        )

        # ── Ensemble-level loss ────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights   = class_weights,
            gamma           = focal_gamma,
            label_smoothing = label_smoothing,
        )

        # ── Metrics (on averaged ensemble probabilities) ───────────────
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

    # ── Ensemble forward ────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Returns averaged softmax probabilities (B, num_classes).
        Used for val / test metrics and inference.
        """
        p_net = F.softmax(self.net(waveform), dim=-1)
        p_con = F.softmax(self.conformer(waveform), dim=-1)
        return (p_net + p_con) * 0.5

    # ── Augmentation helpers ────────────────────────────────────────────

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
            noise   = torch.randn_like(x) * (sig_pwr / snr_lin).sqrt()
            x = x + noise
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
        return (lam * self.criterion(logits, y)
                + (1.0 - lam) * self.criterion(logits, y_perm))

    # ── Lightning steps ─────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self._augment(x)
        x, y, y_p, lam = self._mixup(x, y)

        # Each sub-model computes its own logits and loss
        logits_net = self.net(x)
        logits_con = self.conformer(x)

        loss_net = self._loss(logits_net, y, y_p, lam)
        loss_con = self._loss(logits_con, y, y_p, lam)
        loss = loss_net + loss_con

        # Ensemble accuracy on train (averaged probs, no mixup label)
        with torch.no_grad():
            probs = (F.softmax(logits_net, dim=-1) + F.softmax(logits_con, dim=-1)) * 0.5
        self.train_acc(probs, y)

        self.log("train/loss",     loss,           on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/loss_net", loss_net,       on_step=False, on_epoch=True)
        self.log("train/loss_con", loss_con,       on_step=False, on_epoch=True)
        self.log("train/acc",      self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        probs  = self(x)       # averaged softmax (B, C)

        # Loss on averaged probs (log to get cross-entropy-like scalar)
        log_p = torch.log(probs.clamp(min=1e-9))
        loss  = F.nll_loss(log_p, y)

        self.val_acc(probs, y);       self.val_f1(probs, y)
        self.val_precision(probs, y); self.val_mcc(probs, y)

        self.log("val/loss",      loss,               on_epoch=True, prog_bar=True)
        self.log("val/acc",       self.val_acc,       on_epoch=True, prog_bar=True)
        self.log("val/f1",        self.val_f1,        on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True, prog_bar=True)
        self.log("val/mcc",       self.val_mcc,       on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y  = batch
        probs = self(x)

        self.test_acc(probs, y);   self.test_f1(probs, y)
        self.test_mcc(probs, y);   self.test_auroc(probs, y)
        self.test_cm(probs, y)

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
    model  = HydroEnsemble(num_classes=3).to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    net_p = sum(p.numel() for p in model.net.parameters())
    con_p = sum(p.numel() for p in model.conformer.parameters())
    print(f"HydroEnsemble  |  {total:,} total params")
    print(f"  HydroNet     :  {net_p:,}")
    print(f"  HydroConform :  {con_p:,}")

    x = torch.randn(2, 32_000, device=device)
    with torch.no_grad():
        probs = model(x)
    print(f"Output shape   : {probs.shape}  (sum={probs.sum(-1).tolist()})")
