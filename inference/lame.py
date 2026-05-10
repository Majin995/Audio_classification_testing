"""LAME — Laplacian-Adjusted Maximum-likelihood Estimation for parameter-free
test-time adaptation.

Boudiaf et al., "Parameter-free Online Test-time Adaptation," CVPR 2022
(arXiv:2201.05718). Reference impl: github.com/fiveai/LAME.

LAME refines the model's softmax outputs on a test batch by enforcing
Laplacian smoothness over a similarity graph in feature space. Unlike
TENT it does NOT modify the model — only the per-sample probabilities.
The objective is concave w.r.t. the soft-assignment Z; we solve it via
the concave-convex procedure (CCCP), which converges in ~3-5 iterations.

Inputs
------
features : (N, D) tensor of penultimate embeddings (post-pool, pre-head).
logits   : (N, C) raw logits from the model.
k        : neighbours per node in the kNN graph (default 5).
n_iter   : CCCP iterations (default 5).
sigma    : RBF kernel bandwidth (auto-computed from feature distances if None).

Output
------
refined  : (N, C) refined log-probabilities (interpretable as logits).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _knn_affinity(features: torch.Tensor, k: int = 5,
                  sigma: Optional[float] = None) -> torch.Tensor:
    """Symmetric kNN-RBF affinity matrix W (N, N) with zeros on the diagonal."""
    N, _ = features.shape
    Fn = F.normalize(features, dim=1)
    sim = Fn @ Fn.t()                                              # cosine in [-1, 1]
    sim.fill_diagonal_(-2.0)                                       # exclude self

    # Top-k per row
    vals, idx = sim.topk(k=min(k, N - 1), dim=1)
    W = torch.zeros_like(sim)
    W.scatter_(1, idx, vals.clamp(min=0.0))                        # negative cos → 0
    # Symmetrise: only keep pairs that are mutual.
    W = (W + W.t()) * 0.5

    if sigma is not None:
        # Convert similarity to RBF on Euclidean: d = sqrt(2(1 - sim)).
        d2 = (2.0 * (1.0 - sim.clamp(min=-1.0, max=1.0))).clamp(min=0.0)
        W = W * torch.exp(-d2 / (2.0 * sigma * sigma + 1e-12))
    return W


def lame_refine(
    features: torch.Tensor,
    logits: torch.Tensor,
    k: int = 5,
    n_iter: int = 5,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """Run CCCP iterations of the LAME objective.

    The objective (eq. 6 in the paper, after the unary mass-conservation
    constraint is folded in) is

        max_Z  Σ Z_ij log p_ij + ½ Σ W_ij (Z_i · Z_j)

    s.t. each row of Z is in the simplex. The CCCP linearises the
    quadratic term at the current Z; the resulting concave subproblem has
    a closed-form softmax solution per row.
    """
    N, C = logits.shape
    device = logits.device
    if N <= 1:
        return F.log_softmax(logits, dim=1)

    W = _knn_affinity(features.float(), k=k, sigma=sigma).to(device)
    log_p = F.log_softmax(logits.float(), dim=1)
    Z = log_p.exp()                                                # warm start at p

    for _ in range(n_iter):
        # Linearise the quadratic term: gradient is W @ Z.
        msg = W @ Z                                                # (N, C)
        # Closed-form simplex update: Z_i ∝ p_i * exp(msg_i)
        Z = F.softmax(log_p + msg, dim=1)

    return (Z + 1e-12).log().to(logits.dtype)
