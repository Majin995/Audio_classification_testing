"""
LDAM and Class-Balanced Focal losses for long-tailed classification.

LDAM (Cao et al., NeurIPS 2019)
  Per-class margin m_y = C / n_y^{1/4}, scaled so m_max = max_m.
  Forward: subtract m_y from the target logit, then scale all logits by s,
  then standard cross-entropy with optional class weights.

Class-Balanced Focal (Cui et al., CVPR 2019)
  Effective number of samples E_n = (1 - β^n) / (1 - β); per-class weight
  w_y = 1 / E_n_y, then standard focal loss with these weights.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss.

    Args:
        cls_num_list   : Per-class sample counts (length == num_classes).
        max_m          : Maximum margin (applied to the rarest class).
        s              : Logit scaling factor (LDAM convention is 30).
        weight         : Optional per-class CE weight vector.
        label_smoothing: Cross-entropy label smoothing.
    """

    def __init__(
        self,
        cls_num_list:    Sequence[float],
        max_m:           float = 0.5,
        s:               float = 30.0,
        weight:          Optional[Sequence[float]] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        m_list = 1.0 / (torch.tensor(cls_num_list, dtype=torch.float32) ** 0.25)
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", m_list)
        self.s = s
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        margin = self.m_list[targets].unsqueeze(1)
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        adjusted = logits - margin * one_hot
        return F.cross_entropy(
            self.s * adjusted, targets,
            weight=self.weight, label_smoothing=self.label_smoothing,
        )


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss using effective-number reweighting.

    Args:
        cls_num_list   : Per-class sample counts (length == num_classes).
        beta           : Effective-number hyperparameter, typically 0.999.
        gamma          : Focal modulation exponent.
        label_smoothing: Cross-entropy label smoothing.
    """

    def __init__(
        self,
        cls_num_list:    Sequence[float],
        beta:            float = 0.999,
        gamma:           float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        n = torch.tensor(cls_num_list, dtype=torch.float32)
        eff_n = (1.0 - beta ** n) / (1.0 - beta)
        weight = 1.0 / eff_n
        weight = weight * len(cls_num_list) / weight.sum()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()
