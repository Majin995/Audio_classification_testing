"""
AcousticOmniResNet — Multi-Head Residual Fusion Network
=======================================================

Architecture
------------
::

    feat_1d (B, 138)                    feat_2d (B, 9, 64, 128)
         ↓                                      ↓
    ResidualMLP                          ResNet2DBranch
         ↓                                      ↓
      (B, 512)                              (B, 2048)
         ↓                                      ↓
         └──────── CrossAttentionFusion ────────┘
                           ↓
                     SE Gate  (B, 512)
                           ↓
                     Classifier → (B, num_classes)

The 1D branch (``ResidualMLP``) processes the concatenated scalar / PSD /
spectral features produced by ``UnderwaterFeatureExtractor`` (138 dims total)
through a stack of pre-norm residual feedforward blocks.

The 2D branch (``ResNet2DBranch``) treats the 9-channel stacked spectrogram
as a multi-spectral image and runs it through a full ResNet-50 backbone.

``CrossAttentionFusion`` projects both embeddings into a shared token space
and lets 2D tokens attend over 1D tokens before a squeeze-excitation gate
integrates the two streams.

Mixup is applied **at the feature level** (both ``feat_1d`` and ``feat_2d``
simultaneously with the same :math:`\\lambda` and permutation), which avoids
the need to modify the waveform-level loader.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as tvm
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassMatthewsCorrCoef,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from .hydro_conformer import FocalLoss, DropPath


# ═══════════════════════════════════════════════════════════════════════
#  1D Branch — ResidualMLP
# ═══════════════════════════════════════════════════════════════════════

class ResidualMLPBlock(nn.Module):
    r"""
    Single pre-norm residual feedforward block.

    .. math::
        \mathbf{x} \leftarrow \mathbf{x}
            + \text{DropPath}\!\left(
                W_2\,\sigma\!\left(W_1\,\text{LN}(\mathbf{x})\right)
              \right)

    where :math:`W_1 \in \mathbb{R}^{4d \times d}`,
    :math:`W_2 \in \mathbb{R}^{d \times 4d}`, and :math:`\sigma` is GELU.

    Parameters
    ----------
    dim : int
        Feature dimension.
    dropout : float
        Dropout probability applied after each linear projection.
    drop_path : float
        Stochastic-depth drop probability.
    """

    def __init__(self, dim: int, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.norm    = nn.LayerNorm(dim)
        self.fc1     = nn.Linear(dim, dim * 4)
        self.act     = nn.GELU()
        self.drop1   = nn.Dropout(dropout)
        self.fc2     = nn.Linear(dim * 4, dim)
        self.drop2   = nn.Dropout(dropout)
        self.dp      = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.drop2(self.fc2(self.drop1(self.act(self.fc1(h)))))
        return x + self.dp(h)


class ResidualMLP(nn.Module):
    r"""
    1-D scalar / vector branch.

    Embeds the concatenated 138-dim feature vector into a 512-d space and
    refines it through three stacked ``ResidualMLPBlock`` s with linearly
    increasing stochastic-depth rates.

    .. math::
        \mathbf{z} = \text{LN}\!\left(
            \text{Block}_3\!\left(
                \text{Block}_2\!\left(
                    \text{Block}_1\!\left(
                        \text{GELU}\!\left(W_\text{in}\,\mathbf{x}\right)
                    \right)
                \right)
            \right)
        \right)

    Parameters
    ----------
    in_dim : int
        Input dimensionality (default: 138 — scalars ∥ PSD ∥ spectral).
    hidden_dim : int
        Projection and hidden dimension (default: 512).
    n_blocks : int
        Number of residual blocks.
    dropout : float
    drop_path_max : float
        Maximum stochastic-depth rate (linearly scaled per block).
    """

    def __init__(
        self,
        in_dim:        int   = 138,
        hidden_dim:    int   = 512,
        n_blocks:      int   = 3,
        dropout:       float = 0.1,
        drop_path_max: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        dp_rates = [
            drop_path_max * i / max(n_blocks - 1, 1)
            for i in range(n_blocks)
        ]
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, dropout=dropout, drop_path=dp_rates[i])
            for i in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)                                     # (B, 512)


# ═══════════════════════════════════════════════════════════════════════
#  2D Branch — ResNet2DBranch
# ═══════════════════════════════════════════════════════════════════════

class ResNet2DBranch(nn.Module):
    r"""
    9-channel spectrogram image branch based on ResNet-50.

    The standard ``conv1`` (3-channel → 64, 7×7, stride 2) is replaced
    with a 9-channel equivalent trained from scratch so that all nine
    spectro-temporal representations are processed jointly in the first
    convolutional layer:

    .. math::
        \hat{W}_\text{conv1} \in \mathbb{R}^{64 \times 9 \times 7 \times 7}

    The final average-pooling and fully-connected layers are stripped;
    the branch outputs a ``(B, 2048)`` feature vector from the last
    residual stage.

    Parameters
    ----------
    n_channels : int
        Number of input spectro-temporal channels (default: 9).
    """

    def __init__(self, n_channels: int = 9):
        super().__init__()
        base = tvm.resnet50(weights=None)
        base.conv1 = nn.Conv2d(
            n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Strip adaptive average pool + classifier; keep up to AdaptiveAvgPool2d
        self.backbone = nn.Sequential(*list(base.children())[:-1])   # → (B,2048,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).flatten(1)                       # (B, 2048)


# ═══════════════════════════════════════════════════════════════════════
#  Fusion — CrossAttentionFusion + SEGate1D
# ═══════════════════════════════════════════════════════════════════════

class SEGate1D(nn.Module):
    r"""
    Squeeze-excitation gate on flat feature vectors.

    Adapted from the channel-wise SE block used in ``hydro_net.SEBlock``
    but operating on 1-D vectors rather than feature maps:

    .. math::
        \mathbf{s} = \sigma\!\left(
            W_2\,\text{ReLU}\!\left(W_1\,\mathbf{x}\right)
        \right), \qquad
        \text{out} = \mathbf{x} \odot \mathbf{s}

    where :math:`W_1 \in \mathbb{R}^{(d/r) \times d}`,
    :math:`W_2 \in \mathbb{R}^{d \times (d/r)}`, and :math:`r` is the
    reduction ratio.

    Parameters
    ----------
    dim : int
        Feature vector width.
    reduction : int
        Reduction ratio for the bottleneck (default: 8).
    """

    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return x * s


class CrossAttentionFusion(nn.Module):
    r"""
    Cross-attention fusion of the 1D and 2D branch embeddings.

    Both embeddings are projected to ``fused_dim`` and then split into
    ``n_tokens`` tokens of dimension ``token_dim = fused_dim / n_tokens``.
    The 2D tokens act as **queries** while the 1D tokens act as
    **keys/values**:

    .. math::
        \mathbf{A} = \text{MultiheadAttn}\!\left(
            \text{LN}_Q\!\left(\mathbf{Q}\right),\,
            \text{LN}_K\!\left(\mathbf{K}\right),\,
            \mathbf{V}
        \right)

    The attended output is reshaped back to ``(B, fused_dim)``, added to
    the 2D projection as a residual, and then gated by ``SEGate1D``:

    .. math::
        \mathbf{z} = \text{SE}\!\left(\text{reshape}(\mathbf{A}) + \mathbf{h}_{2D}\right)

    Parameters
    ----------
    dim_1d : int
        Width of the 1D branch output (default: 512).
    dim_2d : int
        Width of the 2D branch output (default: 2048).
    fused_dim : int
        Shared projection / fusion dimension (default: 512).
    n_tokens : int
        Number of tokens to split the fused dim into (default: 8).
    n_heads : int
        Number of attention heads per token (default: 4).
    dropout : float
        Attention dropout.
    """

    def __init__(
        self,
        dim_1d:    int   = 512,
        dim_2d:    int   = 2048,
        fused_dim: int   = 512,
        n_tokens:  int   = 8,
        n_heads:   int   = 4,
        dropout:   float = 0.1,
    ):
        super().__init__()
        assert fused_dim % n_tokens == 0, "fused_dim must be divisible by n_tokens"
        token_dim = fused_dim // n_tokens

        self.n_tokens  = n_tokens
        self.token_dim = token_dim

        self.proj_2d    = nn.Linear(dim_2d, fused_dim)
        self.proj_1d    = nn.Linear(dim_1d, fused_dim)   # align if dim_1d ≠ fused_dim
        self.norm_q     = nn.LayerNorm(token_dim)
        self.norm_k     = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.se_gate    = SEGate1D(fused_dim)

    def forward(self, feat_1d: torch.Tensor, feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feat_1d : (B, dim_1d)
        feat_2d : (B, dim_2d)

        Returns
        -------
        (B, fused_dim)
        """
        B = feat_1d.size(0)
        h_2d = self.proj_2d(feat_2d)                             # (B, fused_dim)
        h_1d = self.proj_1d(feat_1d)                             # (B, fused_dim)

        q = h_2d.view(B, self.n_tokens, self.token_dim)          # 2D as queries
        k = h_1d.view(B, self.n_tokens, self.token_dim)          # 1D as keys
        v = k

        attn_out, _ = self.cross_attn(
            self.norm_q(q), self.norm_k(k), v,
        )                                                         # (B, n_tokens, token_dim)

        fused = attn_out.reshape(B, -1) + h_2d                   # residual
        return self.se_gate(fused)                                # (B, fused_dim)


# ═══════════════════════════════════════════════════════════════════════
#  Top-level LightningModule
# ═══════════════════════════════════════════════════════════════════════

class AcousticOmniResNet(pl.LightningModule):
    r"""
    Multi-head residual fusion network for underwater vessel classification.

    Ingests the full ``UnderwaterFeatureExtractor`` feature set:

    * ``feat_1d`` — concatenated scalars ∥ PSD ∥ spectral features (B, 138)
    * ``feat_2d`` — stacked spectro-temporal grams (B, 9, 64, 128)

    and fuses them through a cross-attention + SE gating tower before the
    final 4-class classifier.

    Mixup is applied at the **feature level**: both ``feat_1d`` and
    ``feat_2d`` are blended with the same :math:`\lambda \sim
    \text{Beta}(\alpha, \alpha)` and permutation index, so no waveform-
    level modification is required.

    Parameters
    ----------
    num_classes : int
    class_weights : list of float, optional
        Inverse-frequency weights passed to ``FocalLoss``.
    mlp_dim : int
        Hidden dimension for the 1D residual MLP branch.
    mlp_n_blocks : int
    mlp_dropout : float
    mlp_drop_path : float
    fusion_dim : int
        Shared projection dimension inside ``CrossAttentionFusion``.
    fusion_tokens : int
        Token split for cross-attention.
    fusion_heads : int
    fusion_dropout : float
    classifier_dropout : float
    focal_gamma : float
    label_smoothing : float
    mixup_alpha : float
    learning_rate : float
    weight_decay : float
    warmup_epochs : int
    max_epochs : int
    """

    def __init__(
        self,
        num_classes:        int          = 4,
        class_weights:      Optional[List[float]] = None,
        # 1D branch
        mlp_dim:            int          = 512,
        mlp_n_blocks:       int          = 3,
        mlp_dropout:        float        = 0.1,
        mlp_drop_path:      float        = 0.1,
        # 2D branch
        n_gram_channels:    int          = 9,
        # Fusion
        fusion_dim:         int          = 512,
        fusion_tokens:      int          = 8,
        fusion_heads:       int          = 4,
        fusion_dropout:     float        = 0.1,
        # Classifier
        classifier_dropout: float        = 0.2,
        # Loss / augmentation
        focal_gamma:        float        = 2.0,
        label_smoothing:    float        = 0.05,
        mixup_alpha:        float        = 0.3,
        # Optimiser
        learning_rate:      float        = 3e-4,
        weight_decay:       float        = 1e-2,
        warmup_epochs:      int          = 10,
        max_epochs:         int          = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── 1D scalar / vector branch ───────────────────────────────────
        self.mlp_branch = ResidualMLP(
            in_dim        = 138,
            hidden_dim    = mlp_dim,
            n_blocks      = mlp_n_blocks,
            dropout       = mlp_dropout,
            drop_path_max = mlp_drop_path,
        )

        # ── 2D spectrogram branch ───────────────────────────────────────
        self.resnet_branch = ResNet2DBranch(n_channels=n_gram_channels)

        # ── Cross-attention fusion + SE gate ────────────────────────────
        self.fusion = CrossAttentionFusion(
            dim_1d    = mlp_dim,
            dim_2d    = 2048,
            fused_dim = fusion_dim,
            n_tokens  = fusion_tokens,
            n_heads   = fusion_heads,
            dropout   = fusion_dropout,
        )

        # ── Classification head ─────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )

        # ── Loss ────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights  = class_weights,
            gamma          = focal_gamma,
            label_smoothing = label_smoothing,
        )

        # ── Metrics ─────────────────────────────────────────────────────
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

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        feat_1d: torch.Tensor,
        feat_2d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        feat_1d : (B, 138)   scalars ∥ PSD ∥ spectral
        feat_2d : (B, 9, H, W)  stacked spectro-temporal grams

        Returns
        -------
        logits : (B, num_classes)
        """
        z_1d   = self.mlp_branch(feat_1d)          # (B, mlp_dim)
        z_2d   = self.resnet_branch(feat_2d)        # (B, 2048)
        fused  = self.fusion(z_1d, z_2d)            # (B, fusion_dim)
        return self.classifier(fused)               # (B, num_classes)

    # ── Mixup ────────────────────────────────────────────────────────────

    def _mixup(
        self,
        feat_1d: torch.Tensor,
        feat_2d: torch.Tensor,
        y:       torch.Tensor,
    ):
        r"""
        Feature-level mixup with :math:`\lambda \sim \text{Beta}(\alpha, \alpha)`.

        Returns
        -------
        feat_1d_m, feat_2d_m, y, y_perm, lam
        """
        alpha = self.hparams.mixup_alpha
        if not self.training or alpha <= 0.0:
            return feat_1d, feat_2d, y, y, 1.0

        lam  = torch.distributions.Beta(alpha, alpha).sample().to(feat_1d)
        perm = torch.randperm(feat_1d.size(0), device=feat_1d.device)

        f1_m = lam * feat_1d + (1.0 - lam) * feat_1d[perm]
        f2_m = lam * feat_2d + (1.0 - lam) * feat_2d[perm]
        return f1_m, f2_m, y, y[perm], lam

    def _loss(self, logits, y, y_perm=None, lam=1.0):
        if y_perm is None or lam == 1.0:
            return self.criterion(logits, y)
        return (lam * self.criterion(logits, y)
                + (1.0 - lam) * self.criterion(logits, y_perm))

    # ── Lightning steps ──────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        feat_1d, feat_2d, y = batch
        feat_1d, feat_2d, y, y_p, lam = self._mixup(feat_1d, feat_2d, y)

        logits = self(feat_1d, feat_2d)
        loss   = self._loss(logits, y, y_p, lam)

        self.train_acc(logits, y)
        self.log("train/loss", loss,           on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feat_1d, feat_2d, y = batch
        logits = self(feat_1d, feat_2d)
        loss   = self.criterion(logits, y)

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_precision(logits, y)
        self.val_mcc(logits, y)
        self.log("val/loss",      loss,               on_epoch=True, prog_bar=True)
        self.log("val/acc",       self.val_acc,       on_epoch=True, prog_bar=True)
        self.log("val/f1",        self.val_f1,        on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True)
        self.log("val/mcc",       self.val_mcc,       on_epoch=True)

    def test_step(self, batch, batch_idx):
        feat_1d, feat_2d, y = batch
        logits = self(feat_1d, feat_2d)
        probs  = F.softmax(logits, dim=-1)

        self.test_acc(logits,  y)
        self.test_f1(logits,   y)
        self.test_mcc(logits,  y)
        self.test_auroc(probs, y)
        self.test_cm(logits,   y)
        self.log("test/loss",  self.criterion(logits, y), on_epoch=True)
        self.log("test/acc",   self.test_acc,             on_epoch=True)
        self.log("test/f1",    self.test_f1,              on_epoch=True)
        self.log("test/mcc",   self.test_mcc,             on_epoch=True)
        self.log("test/auroc", self.test_auroc,           on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix (rows=true, cols=pred):\n{cm.cpu().numpy()}")
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
            [
                {"params": decay,    "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        def lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
