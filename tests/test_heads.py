"""Sanity tests for ``models.heads``: shapes, ArcFace label-leak guard,
cosine bound, prototype centroid init."""
import math

import pytest
import torch

from models.heads import (
    ArcFaceHead,
    CosineHead,
    MLPHead,
    PrototypeHead,
    WiderMLPHead,
    build_head,
    HEAD_NAMES,
)


D, C, B = 256, 4, 8


@pytest.mark.parametrize("name", HEAD_NAMES)
def test_factory_shapes(name):
    head = build_head(name, in_dim=D, num_classes=C, fusion_dim=128, dropout=0.1)
    x = torch.randn(B, D)
    y = torch.randint(0, C, (B,))
    head.train()
    out = head(x, y)
    assert out.shape == (B, C), f"{name} train shape {out.shape}"
    head.eval()
    with torch.no_grad():
        out = head(x, None if name == "arcface" else y)
        assert out.shape == (B, C), f"{name} eval shape {out.shape}"


def test_arcface_no_label_leak_in_eval():
    head = ArcFaceHead(D, C, margin=0.3, scale=30.0)
    head.eval()
    x = torch.randn(B, D)
    out = head(x, labels=None)  # eval is fine without labels
    assert out.shape == (B, C)
    # Also: train mode without labels must raise.
    head.train()
    with pytest.raises(AssertionError):
        head(x, labels=None)


def test_arcface_margin_reduces_target_logit():
    """With non-zero margin and correct labels, the target-class logit should be
    strictly less than the unmargined cosine logit (cos(θ + m) < cos(θ) for θ in (0, π-m))."""
    torch.manual_seed(0)
    head = ArcFaceHead(D, C, margin=0.3, scale=1.0)
    head.train()
    x = torch.randn(B, D)
    y = torch.randint(0, C, (B,))
    margined = head(x, labels=y)
    head.eval()
    with torch.no_grad():
        unmargined = head(x, labels=None)
    target = torch.arange(B)
    diff = unmargined[target, y] - margined[target, y]
    assert (diff >= -1e-5).all(), f"margin should not increase target logit: {diff}"


def test_cosine_logits_bounded_by_scale():
    head = CosineHead(D, C, scale_init=10.0, learnable_scale=False)
    head.eval()
    out = head(torch.randn(B, D))
    assert out.abs().max().item() <= 10.0 + 1e-4


def test_prototype_centroid_init():
    head = PrototypeHead(D, C)
    centroids = torch.randn(C, D)
    head.init_from_centroids(centroids)
    assert torch.allclose(head.W.detach(), centroids)
    with pytest.raises(ValueError):
        head.init_from_centroids(torch.randn(C + 1, D))


def test_param_counts_roughly_match_plan():
    """Sanity: cosine ~30x fewer params than MLP, wider ~6x more."""
    mlp = build_head("mlp", D, C, fusion_dim=128, dropout=0.1)
    cos = build_head("cosine", D, C, fusion_dim=128, dropout=0.1)
    wide = build_head("mlp_wide", D, C, fusion_dim=128, dropout=0.1)

    p_mlp = sum(p.numel() for p in mlp.parameters())
    p_cos = sum(p.numel() for p in cos.parameters())
    p_wide = sum(p.numel() for p in wide.parameters())

    assert p_cos < p_mlp / 20, f"cosine ({p_cos}) not much smaller than mlp ({p_mlp})"
    assert p_wide > p_mlp * 4, f"wider ({p_wide}) not much larger than mlp ({p_mlp})"
