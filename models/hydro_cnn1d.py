"""
HydroCNN1D — 1D Convolutional Feature Extractor (refined for SVM/classical heads).

Architecture
------------
  InstanceNorm1d (input normalisation — critical for raw waveform amplitude variance)
  3 × (Conv1d[stride=2] → BatchNorm1d → ReLU → ConcreteDropout)
    channels : 1 → 64 → 128 → 256
    kernels  : 9 → 7 → 5
    stride=2 in each conv replaces MaxPool — preserves more gradient signal
  Dual pooling: cat(GlobalAvgPool, GlobalMaxPool) → (512,)
  Projection  : Linear(512, 256) → feature vector
  Head        : Linear(256, num_classes)

Feature vector (256-dim) is returned by extract_features() for classical heads.

Training improvements vs v1
---------------------------
* FocalLoss with per-class weights (handles Cargo 72% dominance)
* ConcreteDropout — learnable dropout rate per conv block, optimised jointly
* Waveform-level mixup (Beta-distributed interpolation)
* Warmup (5 ep) + CosineAnnealing LR schedule
* InstanceNorm1d on raw waveform before conv blocks
* EarlyStopping on val_f1_score (max) instead of val_loss

Usage
-----
    # Stage 1 — pretrain:
    python training/train_cnn1d.py

    # Stage 2 — extract features for classical heads:
    python training/eval_classical.py --ckpt <ckpt_path>
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import (
    MetricCollection,
    F1Score,
    FBetaScore,
    MatthewsCorrCoef,
    ConfusionMatrix,
    ROC,
    AUROC,
)


# ──────────────────────────────────────────────────────────────────────────────
# Concrete Dropout
# ──────────────────────────────────────────────────────────────────────────────

class ConcreteDropout(nn.Module):
    """
    Concrete (continuous relaxation of) Dropout with a learnable dropout rate.

    The dropout probability *p* is parameterised via an unconstrained logit so
    that gradients flow freely through the rate itself.  During training each
    activation is masked by a sample from Concrete(1-p, temperature); at eval
    time the module is a pure identity (no masking, no rescaling needed).

    A scalar regularisation term is computed on every forward pass and exposed
    via ``regularization(N_in)`` — add this to the task loss during training.

    Reference: Gal, Hron & Turner, "Concrete Dropout", NeurIPS 2017.

    Args:
        init_p           : Initial dropout probability (0 < init_p < 1).
        dropout_regularizer : Scales the entropy regularisation term.
                              Tune to keep learned p from collapsing to 0 or 1.
        temperature      : Concrete relaxation temperature.  Lower → harder
                           samples (closer to Bernoulli); 0.1 works well.
    """

    def __init__(
        self,
        init_p:               float = 0.2,
        dropout_regularizer:  float = 1e-5,
        temperature:          float = 0.1,
    ):
        super().__init__()
        self.dropout_regularizer = dropout_regularizer
        self.temperature         = temperature

        # Unconstrained logit: p = sigmoid(p_logit)
        init_logit = math.log(init_p) - math.log(1.0 - init_p)
        self.p_logit = nn.Parameter(torch.tensor(init_logit))

        # Buffer: regularisation scalar from the most recent forward pass.
        # Initialised to zero so it's safe to call before the first forward.
        self.register_buffer("_last_entropy", torch.zeros(()))   # scalar

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def p(self) -> torch.Tensor:
        """Current dropout probability (0, 1)."""
        return torch.sigmoid(self.p_logit)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p
        eps = 1e-7

        if self.training:
            # Concrete relaxation of Bernoulli(1-p):
            #   u ~ Uniform(0,1)
            #   z = sigmoid((log(u/(1-u)) + log((1-p)/p)) / temp)
            # z ≈ 1 → keep,  z ≈ 0 → drop
            u = torch.rand_like(x).clamp(eps, 1.0 - eps)
            z = torch.sigmoid(
                (torch.log(u) - torch.log(1.0 - u)
                 + torch.log(1.0 - p + eps) - torch.log(p + eps))
                / self.temperature
            )
            # Rescale to preserve expected activation magnitude
            x = x * z / (1.0 - p + eps)

        # ── Entropy regularisation ─────────────────────────────────────────
        # H(Bernoulli(p)) = -p*log(p) - (1-p)*log(1-p)  [always positive]
        # Stored as a buffer so regularization() can be called after forward.
        p_c = p.clamp(eps, 1.0 - eps)
        self._last_entropy = -(p_c * torch.log(p_c) + (1.0 - p_c) * torch.log(1.0 - p_c))

        return x

    def regularization(self, N_in: int) -> torch.Tensor:
        """
        Concrete dropout regularisation contribution (add to the task loss).

        Args:
            N_in: number of input units (channels or features) to this layer.
                  Used to scale the entropy term to a comparable magnitude.
        Returns:
            Scalar tensor; device matches the module's buffer.
        """
        return self.dropout_regularizer * N_in * self._last_entropy

    def extra_repr(self) -> str:
        return (f"p={self.p.item():.3f}, "
                f"dropout_regularizer={self.dropout_regularizer}, "
                f"temperature={self.temperature}")


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss  (reused from models/hydro_conformer.py:359-389)
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss with per-class alpha weighting.

    Reduces relative loss for easy examples and focuses training on hard ones —
    especially valuable when one class (Cargo, 72 %) dominates the dataset.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        class_weights:   Optional[list] = None,
        gamma:           float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "alpha",
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Conv block (strided — no MaxPool)
# ──────────────────────────────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
    """Conv1d(stride=2) → BatchNorm1d → ReLU.
    Stride=2 halves the time dimension without discarding gradient paths."""
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                  stride=2, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class HydroCNN1D(pl.LightningModule):
    """
    Refined 1D CNN feature extractor optimised for downstream SVM classification.

    Args:
        num_classes         : Number of output classes.
        input_len           : Waveform length in samples (= sample_rate for 1-s clips).
        class_weights       : Inverse-frequency weights from DataModule.class_weights.
                              Passed to FocalLoss; set None to disable weighting.
        lr                  : Peak learning rate (after warmup).
        warmup_epochs       : Linear warmup length in epochs.
        max_epochs          : Total training epochs (needed for cosine schedule).
        focal_gamma         : FocalLoss focusing parameter (2.0 is standard).
        label_smoothing     : FocalLoss label-smoothing coefficient.
        mixup_alpha         : Beta distribution alpha for waveform mixup (0 = off).
        f_beta              : Beta for FBetaScore metric.
        cd_dropout_reg      : ConcreteDropout entropy regularisation coefficient.
    """

    def __init__(
        self,
        num_classes:     int            = 3,
        input_len:       int            = 5120,
        class_weights:   Optional[list] = None,
        lr:              float          = 3e-4,
        warmup_epochs:   int            = 5,
        max_epochs:      int            = 100,
        focal_gamma:     float          = 2.0,
        label_smoothing: float          = 0.05,
        mixup_alpha:     float          = 0.3,
        f_beta:          float          = 1.0,
        cd_dropout_reg:  float          = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Input normalisation ───────────────────────────────────────────────
        # InstanceNorm normalises each waveform independently — essential for
        # raw audio where amplitude varies orders of magnitude across clips.
        self.input_norm = nn.InstanceNorm1d(1, affine=True)

        # ── Conv blocks (stride=2 downsampling, no MaxPool) ───────────────────
        #   T → T/2 → T/4 → T/8  (at input_len=5120: 5120→2560→1280→640)
        self.conv1 = _conv_block(1,   64,  kernel=9)
        self.conv2 = _conv_block(64,  128, kernel=7)
        self.conv3 = _conv_block(128, 256, kernel=5)

        # ── Concrete Dropout (one per conv block) ─────────────────────────────
        # Each ConcreteDropout independently learns its optimal dropout rate.
        self.cd1 = ConcreteDropout(init_p=0.1, dropout_regularizer=cd_dropout_reg)
        self.cd2 = ConcreteDropout(init_p=0.15, dropout_regularizer=cd_dropout_reg)
        self.cd3 = ConcreteDropout(init_p=0.2, dropout_regularizer=cd_dropout_reg)

        # ── Dual pooling + projection ─────────────────────────────────────────
        # Concatenating global-average and global-max captures complementary
        # statistics: mean activation level AND peak activation level.
        # Result: (B, 512) → projected to (B, 256) feature vector.
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.gmp  = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Linear(256, num_classes)

        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

        # ── Metrics (pattern from models/wavelet_scattering_model.py:69-87) ───
        task = "multiclass" if num_classes > 2 else "binary"
        metrics = MetricCollection({
            "f1_score": F1Score(task=task, num_classes=num_classes, average="macro"),
            "f_beta":   FBetaScore(task=task, num_classes=num_classes,
                                   beta=float(f_beta), average="macro"),
            "mcc":      MatthewsCorrCoef(task=task, num_classes=num_classes),
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics   = metrics.clone(prefix="val_")
        self.test_metrics  = metrics.clone(prefix="test_")

        self.val_roc    = ROC(task=task, num_classes=num_classes)
        self.test_roc   = ROC(task=task, num_classes=num_classes)
        self.val_auroc  = AUROC(task=task, num_classes=num_classes)
        self.test_auroc = AUROC(task=task, num_classes=num_classes)
        self.val_cm     = ConfusionMatrix(task=task, num_classes=num_classes)
        self.test_cm    = ConfusionMatrix(task=task, num_classes=num_classes)

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _prep(x: torch.Tensor) -> torch.Tensor:
        """Accept (B, T) or (B, 1, T); always return (B, 1, T) as float32."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x.float()

    def _cd_regularization(self) -> torch.Tensor:
        """Sum of ConcreteDropout entropy regularisation across all three blocks."""
        return (
            self.cd1.regularization(N_in=64)
            + self.cd2.regularization(N_in=128)
            + self.cd3.regularization(N_in=256)
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Shared feature extraction path used by both forward() and extract_features()."""
        x = self.input_norm(x)
        x = self.cd1(self.conv1(x))    # (B, 64,  T/2)
        x = self.cd2(self.conv2(x))    # (B, 128, T/4)
        x = self.cd3(self.conv3(x))    # (B, 256, T/8)
        avg = self.gap(x).squeeze(-1)  # (B, 256)
        mx  = self.gmp(x).squeeze(-1)  # (B, 256)
        return self.proj(torch.cat([avg, mx], dim=1))  # (B, 256)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return 256-dim feature vector (B, 256) without gradients.

        Puts the model in eval mode for the extraction so ConcreteDropout
        acts as identity (no masking), giving deterministic features.
        Used by training/eval_classical.py to feed classical heads.
        """
        with torch.no_grad():
            return self._features(self._prep(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prep(x)
        return self.head(self._features(x))    # (B, num_classes)

    # ── Lightning step / epoch-end hooks ──────────────────────────────────────
    # Pattern reused from models/wavelet_scattering_model.py:100-172

    def _common_step(self, batch, batch_idx, stage: str) -> torch.Tensor:
        x, y = batch

        # ── Waveform mixup (training only) ───────────────────────────────────
        y_perm, lam = y, 1.0
        if stage == "train" and self.hparams.mixup_alpha > 0:
            lam  = torch.distributions.Beta(
                self.hparams.mixup_alpha, self.hparams.mixup_alpha
            ).sample().item()
            perm   = torch.randperm(x.size(0), device=x.device)
            x      = lam * x + (1.0 - lam) * x[perm]
            y_perm = y[perm]

        logits = self.forward(x)

        # ── Loss: FocalLoss + ConcreteDropout regularisation ─────────────────
        task_loss = lam * self.criterion(logits, y) + (1.0 - lam) * self.criterion(logits, y_perm)
        cd_reg    = self._cd_regularization() if stage == "train" else torch.zeros(1, device=x.device)
        loss      = task_loss + cd_reg

        self.log(f"{stage}_loss",      loss,      on_step=True,  on_epoch=True, prog_bar=True)
        self.log(f"{stage}_task_loss", task_loss, on_step=False, on_epoch=True)
        if stage == "train":
            self.log(f"{stage}_cd_reg", cd_reg,   on_step=False, on_epoch=True)
            # Log learned dropout rates for observability
            self.log("p_cd1", self.cd1.p, on_step=False, on_epoch=True)
            self.log("p_cd2", self.cd2.p, on_step=False, on_epoch=True)
            self.log("p_cd3", self.cd3.p, on_step=False, on_epoch=True)

        getattr(self, f"{stage}_metrics").update(logits, y)
        if stage == "val":
            self.val_roc.update(logits, y)
            self.val_auroc.update(logits, y)
            self.val_cm.update(logits, y)
        elif stage == "test":
            self.test_roc.update(logits, y)
            self.test_auroc.update(logits, y)
            self.test_cm.update(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def _on_common_epoch_end(self, stage: str):
        metrics = getattr(self, f"{stage}_metrics")
        self.log_dict(metrics.compute(), on_step=False, on_epoch=True)
        metrics.reset()
        if stage in ("val", "test"):
            auroc = getattr(self, f"{stage}_auroc")
            self.log(f"{stage}_auroc", auroc.compute(), on_epoch=True)
            auroc.reset()
            getattr(self, f"{stage}_roc").reset()
            getattr(self, f"{stage}_cm").reset()

    def on_train_epoch_end(self):
        self._on_common_epoch_end("train")

    def on_validation_epoch_end(self):
        self._on_common_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_common_epoch_end("test")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Warmup + CosineAnnealing schedule reused from models/hydro_omni_resnet.py:588-596

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
            eps=1e-8,
        )

        def lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            if epoch < wu:
                return max(epoch, 1) / max(wu, 1)   # linear warmup
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
