"""
Large Margin Focal Loss (LMF)
==============================

Extends Focal Loss with an additive angular margin on the target-class logit,
forcing the model to maintain a larger separation between the target class and
the decision boundary.  Particularly effective for BAHTNet where transient
boundary events cause ambiguous soft predictions.

Math
----
  p_y^m = softmax(z - m · e_y)[y]     (margin shrinks the target-class score)
  LMF   = -α_y · (1 - p_y^m)^γ · log(p_y^m + ε)

When margin=0, LMF reduces to standard Focal Loss.

Reference
---------
  Wang et al., "Large Margin Few-Shot Learning" (NeurIPS 2018) — margin concept;
  Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017) — focal weighting.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LargeMarginFocalLoss(nn.Module):
    """
    Focal loss with a subtractive margin applied to the target-class logit.

    Args:
        num_classes     : Number of output classes.
        alpha           : Per-class weight vector (length == num_classes).
                          Pass ``None`` to use uniform weights.
        gamma           : Focal modulation exponent (default 2.0).
        margin          : Margin subtracted from the ground-truth logit
                          before softmax (default 0.35).
        label_smoothing : Cross-entropy label smoothing ε (default 0.0).
        eps             : Numerical stability clamp inside log (default 1e-7).
    """

    def __init__(
        self,
        num_classes:     int,
        alpha:           Optional[list[float]] = None,
        gamma:           float = 2.0,
        margin:          float = 0.35,
        label_smoothing: float = 0.0,
        eps:             float = 1e-7,
    ):
        super().__init__()
        self.num_classes     = num_classes
        self.gamma           = gamma
        self.margin          = margin
        self.label_smoothing = label_smoothing
        self.eps             = eps

        if alpha is not None:
            self.register_buffer(
                "alpha",
                torch.tensor(alpha, dtype=torch.float32),
            )
        else:
            self.register_buffer("alpha", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw unnormalised scores.
            targets : (B,)   integer class indices.
        Returns:
            Scalar loss.
        """
        # ── Apply margin to the ground-truth logit ───────────────────────
        if self.margin > 0.0:
            # one-hot mask: (B, C), True at target position
            one_hot = torch.zeros_like(logits, dtype=torch.bool)
            one_hot.scatter_(1, targets.unsqueeze(1), True)
            logits_m = logits - self.margin * one_hot.float()
        else:
            logits_m = logits

        # ── Per-sample cross-entropy with optional label smoothing ───────
        ce = F.cross_entropy(
            logits_m,
            targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # ── Focal weighting ──────────────────────────────────────────────
        # p_t = exp(-CE_noweight) ≈ probability assigned to the correct class
        p_t = torch.exp(-F.cross_entropy(logits_m, targets, reduction="none"))
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = (focal_weight * ce).mean()
        return loss

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, gamma={self.gamma}, "
            f"margin={self.margin}, label_smoothing={self.label_smoothing}"
        )
