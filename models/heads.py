"""Classifier heads for HydroPreciseV2.

Each head implements ``forward(x, labels=None) -> logits`` so the trainer can
swap heads via a single ``--head_type`` flag. Only ArcFace consumes ``labels``
(for the additive angular margin); the others ignore it.

Build via :func:`build_head` to keep call sites uniform.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── MLP (refactor of current head) ──────────────────────────────────────────

class MLPHead(nn.Module):
    """Linear(2D→D) → GELU → Dropout → Linear(D→C). Matches the historical head."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(x)


# ── Wider MLP ───────────────────────────────────────────────────────────────

class WiderMLPHead(nn.Module):
    """Linear(2D→4D)→GELU→Drop→Linear(4D→D)→GELU→Drop→Linear(D→C) with LayerNorm."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        wide = hidden_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, wide),
            nn.LayerNorm(wide),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(wide, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(x)


# ── Cosine head with learnable scale ────────────────────────────────────────

class CosineHead(nn.Module):
    """``logits = scale * L2(x) @ L2(W)^T``. ``scale`` is a learnable scalar."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        scale_init: float = 10.0,
        learnable_scale: bool = True,
    ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        Wn = F.normalize(self.W, dim=1)
        xn = F.normalize(x, dim=1)
        return self.scale * (xn @ Wn.t())


# ── Prototype / NCM head ────────────────────────────────────────────────────

class PrototypeHead(nn.Module):
    """Cosine head whose weight matrix is initialised from class centroids.

    Centroids must be set via :meth:`init_from_centroids` (called by the trainer
    after computing them from a feature cache); otherwise weights are random.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        scale_init: float = 10.0,
        learnable_scale: bool = True,
    ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale_init)))

    @torch.no_grad()
    def init_from_centroids(self, centroids: torch.Tensor) -> None:
        """centroids: (C, D), already in the embedding space."""
        if centroids.shape != self.W.shape:
            raise ValueError(
                f"centroid shape {tuple(centroids.shape)} != W {tuple(self.W.shape)}"
            )
        self.W.copy_(centroids)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        Wn = F.normalize(self.W, dim=1)
        xn = F.normalize(x, dim=1)
        return self.scale * (xn @ Wn.t())


# ── ArcFace head ────────────────────────────────────────────────────────────

class ArcFaceHead(nn.Module):
    """Additive angular margin softmax (Deng et al. 2019).

    Train (``labels`` given): ``cos(θ_y + m)`` for the true class only.
    Eval  (``labels=None``):   plain cosine logits, no margin.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.2,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.scale = float(scale)
        self.margin = float(margin)
        self.easy_margin = bool(easy_margin)
        # Precomputed for numerical stability of the margin transform.
        self._cos_m = math.cos(self.margin)
        self._sin_m = math.sin(self.margin)
        self._th = math.cos(math.pi - self.margin)
        self._mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        Wn = F.normalize(self.W, dim=1)
        xn = F.normalize(x, dim=1)
        cos = xn @ Wn.t()                                    # (B, C)

        if labels is None:
            # No-leak guard: eval/inference must not pass labels.
            assert not self.training, "ArcFaceHead.forward in train mode requires labels"
            return self.scale * cos

        sin = torch.sqrt((1.0 - cos.pow(2)).clamp(min=1e-9))
        cos_m = cos * self._cos_m - sin * self._sin_m         # cos(θ + m)
        if self.easy_margin:
            cos_m = torch.where(cos > 0, cos_m, cos)
        else:
            cos_m = torch.where(cos > self._th, cos_m, cos - self._mm)

        onehot = torch.zeros_like(cos)
        onehot.scatter_(1, labels.long().view(-1, 1), 1.0)
        logits = onehot * cos_m + (1.0 - onehot) * cos
        return self.scale * logits


# ── Sub-Center ArcFace head ─────────────────────────────────────────────────

class SubCenterArcFaceHead(nn.Module):
    """Sub-center ArcFace (Deng et al., ECCV 2020).

    Each class owns ``K`` weight vectors. The per-class cosine is the max over
    its K sub-centers. This tolerates intra-class acoustic heterogeneity
    (e.g. tanker-loaded vs tanker-ballast) which a single-center ArcFace
    cannot represent.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.2,
        sub_centers: int = 2,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.K = int(sub_centers)
        self.W = nn.Parameter(torch.empty(num_classes * self.K, in_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.scale = float(scale)
        self.margin = float(margin)
        self.easy_margin = bool(easy_margin)
        self._cos_m = math.cos(self.margin)
        self._sin_m = math.sin(self.margin)
        self._th = math.cos(math.pi - self.margin)
        self._mm = math.sin(math.pi - self.margin) * self.margin

    def _class_cos(self, x: torch.Tensor) -> torch.Tensor:
        Wn = F.normalize(self.W, dim=1)
        xn = F.normalize(x, dim=1)
        cos_all = xn @ Wn.t()                                           # (B, C*K)
        cos_all = cos_all.view(-1, self.num_classes, self.K)
        cos = cos_all.amax(dim=-1)                                      # (B, C)
        return cos

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        cos = self._class_cos(x)
        if labels is None:
            assert not self.training, "SubCenterArcFaceHead in train mode requires labels"
            return self.scale * cos

        sin = torch.sqrt((1.0 - cos.pow(2)).clamp(min=1e-9))
        cos_m = cos * self._cos_m - sin * self._sin_m
        if self.easy_margin:
            cos_m = torch.where(cos > 0, cos_m, cos)
        else:
            cos_m = torch.where(cos > self._th, cos_m, cos - self._mm)

        onehot = torch.zeros_like(cos)
        onehot.scatter_(1, labels.long().view(-1, 1), 1.0)
        logits = onehot * cos_m + (1.0 - onehot) * cos
        return self.scale * logits


# ── DEMON-MoE head ──────────────────────────────────────────────────────────

class DemonMoEHead(nn.Module):
    """K-expert mixture head over pooled features, soft-gated.

    Inspired by DEMONet (arXiv 2411.02758): different ship classes have
    distinct envelope-modulation character, so distinct classifier "experts"
    can specialise without forcing one set of weights to handle all of them.
    Each expert is a small MLP. The gate is a single Linear over the pooled
    feature; outputs are convex-combined per sample.

    During training, a load-balance auxiliary regulariser (Shazeer et al.)
    is exposed via :attr:`aux_loss` for the trainer to add to the primary
    loss with a small weight (typical ``λ ∈ [0.01, 0.1]``).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        n_experts: int = 4,
        hidden_dim: int = 192,
        dropout: float = 0.1,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.n_experts = int(n_experts)
        self.num_classes = int(num_classes)
        self.gate = nn.Linear(in_dim, n_experts)
        self.gate_temperature = float(gate_temperature)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            for _ in range(n_experts)
        ])
        self.register_buffer("_aux_loss", torch.zeros(()))

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        gate_logits = self.gate(x) / max(self.gate_temperature, 1e-3)
        g = F.softmax(gate_logits, dim=-1)                         # (B, E)
        # Stack expert outputs (B, E, C)
        out = torch.stack([e(x) for e in self.experts], dim=1)
        logits = (g.unsqueeze(-1) * out).sum(dim=1)                # (B, C)

        if self.training:
            # Load-balance loss: encourage uniform expert utilisation by
            # penalising the dot-product of the per-expert mean gate weight
            # and the per-expert fraction of selections (Shazeer 2017).
            with_grad = g
            mean_g = with_grad.mean(dim=0)                         # (E,)
            # soft "fraction selected" ~ same as mean_g for soft routing
            self._aux_loss = (mean_g * mean_g).sum() * float(self.n_experts)
        else:
            self._aux_loss = torch.zeros((), device=x.device)
        return logits


# ── Factory ─────────────────────────────────────────────────────────────────

def build_head(
    name: str,
    in_dim: int,
    num_classes: int,
    fusion_dim: int,
    dropout: float = 0.1,
    arcface_margin: float = 0.2,
    arcface_scale: float = 30.0,
    arcface_subcenters: int = 1,
    cosine_scale_init: float = 10.0,
    moe_n_experts: int = 4,
    moe_gate_temperature: float = 1.0,
) -> nn.Module:
    """Return the head module for ``name``.

    ``fusion_dim`` is used as the hidden dim for MLP heads (matches the
    historical ``Linear(2D → D → C)`` shape, where D = ``fusion_dim``).
    """
    name = name.lower()
    if name == "mlp":
        return MLPHead(in_dim, num_classes, hidden_dim=fusion_dim, dropout=dropout)
    if name == "mlp_wide":
        return WiderMLPHead(in_dim, num_classes, hidden_dim=fusion_dim, dropout=dropout)
    if name == "cosine":
        return CosineHead(in_dim, num_classes, scale_init=cosine_scale_init)
    if name == "prototype":
        return PrototypeHead(in_dim, num_classes, scale_init=cosine_scale_init)
    if name == "arcface":
        if arcface_subcenters > 1:
            return SubCenterArcFaceHead(
                in_dim, num_classes,
                scale=arcface_scale, margin=arcface_margin,
                sub_centers=arcface_subcenters,
            )
        return ArcFaceHead(in_dim, num_classes, scale=arcface_scale, margin=arcface_margin)
    if name == "subcenter_arcface":
        return SubCenterArcFaceHead(
            in_dim, num_classes,
            scale=arcface_scale, margin=arcface_margin,
            sub_centers=max(2, arcface_subcenters),
        )
    if name == "demon_moe":
        return DemonMoEHead(
            in_dim, num_classes,
            n_experts=moe_n_experts,
            hidden_dim=fusion_dim,
            dropout=dropout,
            gate_temperature=moe_gate_temperature,
        )
    raise ValueError(f"Unknown head type: {name!r}")


HEAD_NAMES = ("mlp", "cosine", "prototype", "arcface", "subcenter_arcface",
              "mlp_wide", "demon_moe")
