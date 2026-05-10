"""
Smoke tests for LargeMarginFocalLoss.

Run:
    python -m pytest tests/smoke/test_lmf.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import pytest

from processing.losses import LargeMarginFocalLoss
from models.hydro_conformer import FocalLoss


class TestLargeMarginFocalLoss:

    def setup_method(self):
        self.B = 16
        self.C = 4
        self.logits  = torch.randn(self.B, self.C)
        self.targets = torch.randint(0, self.C, (self.B,))

    def test_scalar_output(self):
        loss = LargeMarginFocalLoss(num_classes=self.C)
        out  = loss(self.logits, self.targets)
        assert out.shape == (), f"Expected scalar, got shape {out.shape}"

    def test_finite(self):
        loss = LargeMarginFocalLoss(num_classes=self.C)
        out  = loss(self.logits, self.targets)
        assert torch.isfinite(out), f"Loss is not finite: {out.item()}"

    def test_positive(self):
        loss = LargeMarginFocalLoss(num_classes=self.C)
        out  = loss(self.logits, self.targets)
        assert out.item() > 0.0, "Loss should be positive"

    def test_margin_zero_approx_focal(self):
        """LMF with margin=0 should produce the same value as FocalLoss."""
        # Both with no class weighting, no label smoothing, gamma=2
        lmf   = LargeMarginFocalLoss(num_classes=self.C, gamma=2.0, margin=0.0)
        focal = FocalLoss(class_weights=None, gamma=2.0, label_smoothing=0.0)

        lmf_val   = lmf(self.logits, self.targets)
        focal_val = focal(self.logits, self.targets)
        assert torch.allclose(lmf_val, focal_val, atol=1e-5), (
            f"LMF(margin=0) ≠ FocalLoss: {lmf_val:.6f} vs {focal_val:.6f}"
        )

    def test_margin_increases_loss(self):
        """A positive margin should increase the loss compared to margin=0."""
        lmf0 = LargeMarginFocalLoss(num_classes=self.C, gamma=2.0, margin=0.0)
        lmf1 = LargeMarginFocalLoss(num_classes=self.C, gamma=2.0, margin=0.5)

        l0 = lmf0(self.logits, self.targets)
        l1 = lmf1(self.logits, self.targets)
        # Larger margin → harder targets → higher loss
        assert l1 >= l0 - 1e-4, (
            f"margin=0.5 should ≥ margin=0 loss: {l1:.6f} vs {l0:.6f}"
        )

    def test_gradient_flows(self):
        """Gradients should flow through logits."""
        logits = self.logits.clone().requires_grad_(True)
        loss   = LargeMarginFocalLoss(num_classes=self.C)(logits, self.targets)
        loss.backward()
        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad)), "Gradients contain NaN/Inf"

    def test_class_weights(self):
        """Class weights should affect the loss."""
        w_uniform = [1.0] * self.C
        w_skewed  = [0.1, 5.0, 0.1, 0.1]
        lmf_u = LargeMarginFocalLoss(num_classes=self.C, alpha=w_uniform)
        lmf_s = LargeMarginFocalLoss(num_classes=self.C, alpha=w_skewed)
        lu = lmf_u(self.logits, self.targets)
        ls = lmf_s(self.logits, self.targets)
        assert lu.item() != ls.item(), "Different class weights should give different loss"

    def test_label_smoothing(self):
        lmf0 = LargeMarginFocalLoss(num_classes=self.C, label_smoothing=0.0)
        lmf1 = LargeMarginFocalLoss(num_classes=self.C, label_smoothing=0.1)
        assert lmf0(self.logits, self.targets).item() != \
               lmf1(self.logits, self.targets).item(), \
               "Label smoothing should change the loss"
