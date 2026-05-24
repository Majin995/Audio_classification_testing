"""HydroGraphProto — 1D frontend → per-source GNN → Perceiver pool →
Sinkhorn-Knopp prototype head.

Pipeline (one source = K 1-s clips @ 5120 Hz):
  1. _ClipEncoder (shared HydroSpark trunk) per clip → z_i ∈ R^D
  2. Concat with frozen 6-ckpt cargo_confirm ensemble log-probs (24 dim),
     project back to D
  3. Build per-source graph: temporal k=2 ring ∪ kNN(k=4) by cosine in
     embedding space; learned edge features (cos, Δt, pair MLP)
  4. 2× GINE-style edge-aware message passing ⊕ global self-attention
     (GraphGPS hybrid)
  5. Perceiver: 4 learned latent queries cross-attend the K node tokens
     → source vector s ∈ R^Dh
  6. Sinkhorn-Knopp prototype head (M prototypes / class, n_classes*M total)
     + 1 deep-gambler abstain logit

Forward returns:
  cls_logits  (N, n_classes)     — soft-min over M prototypes per class
  abstain     (N,)               — gambler abstain logit
  cluster_scores  (N, n_classes*M)  — raw scores for SwAV loss
  s           (N, Dh)            — source embedding (for proto/queue updates)

When K < `min_K_graph` the graph is bypassed (mean-pool over masked tokens
→ Perceiver directly). Matches the proposal's K<6 fallback.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hydro_set_spark import _ClipEncoder


def _safe_softmax(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    x = x.masked_fill(~mask, float("-inf"))
    return F.softmax(x, dim=dim)


class _GINEBlock(nn.Module):
    """One GraphGPS-style block: edge-aware GINE message passing on a dense
    masked graph, fused with global multi-head self-attention.

    Inputs:
      h:        (N, K, D)  node features
      adj:      (N, K, K)  binary adjacency (in {0,1}, includes self-loops)
      edge_f:   (N, K, K, E)  edge features (cos, Δt, etc.)
      pad_mask: (N, K)     True for VALID nodes
    """

    def __init__(self, d_model: int, n_heads: int = 4, edge_dim: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.eps = nn.Parameter(torch.zeros(1))   # GINE-style learnable self-skip
        self.node_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj, edge_f, pad_mask):
        N, K, D = h.shape
        h_i = h.unsqueeze(2).expand(N, K, K, D)
        h_j = h.unsqueeze(1).expand(N, K, K, D)
        msg = self.edge_mlp(torch.cat([h_i, h_j, edge_f], dim=-1))  # (N,K,K,D)
        msg = msg * adj.unsqueeze(-1)                                # mask edges
        agg = msg.sum(dim=2)                                         # (N,K,D)
        h_gnn = self.node_mlp((1.0 + self.eps) * h + agg)
        h = self.norm1(h + self.dropout(h_gnn))

        # Global self-attention (mask out pad)
        key_pad = ~pad_mask                                          # (N,K)
        h_attn, _ = self.attn(h, h, h, key_padding_mask=key_pad,
                              need_weights=False)
        h = self.norm2(h + self.dropout(h_attn))
        h = self.norm3(h + self.dropout(self.node_mlp(h)))
        return h


class _PerceiverPool(nn.Module):
    """Q learned latent queries cross-attend the K node tokens → flat vector."""

    def __init__(self, d_model: int, n_queries: int = 4, n_heads: int = 4,
                 depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross": nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True),
                "self": nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True),
                "mlp": nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(d_model, d_model)),
                "n1": nn.LayerNorm(d_model), "n2": nn.LayerNorm(d_model),
                "n3": nn.LayerNorm(d_model),
            }) for _ in range(depth)
        ])
        self.out_dim = d_model * n_queries

    def forward(self, tokens, pad_mask):
        N = tokens.size(0)
        q = self.q.unsqueeze(0).expand(N, -1, -1)                # (N, Q, D)
        key_pad = ~pad_mask
        for L in self.layers:
            cx, _ = L["cross"](q, tokens, tokens, key_padding_mask=key_pad,
                                need_weights=False)
            q = L["n1"](q + cx)
            sx, _ = L["self"](q, q, q, need_weights=False)
            q = L["n2"](q + sx)
            q = L["n3"](q + L["mlp"](q))
        return q.flatten(1)                                       # (N, Q*D)


@torch.no_grad()
def sinkhorn_knopp(scores: torch.Tensor, n_iters: int = 3,
                   epsilon: float = 0.05) -> torch.Tensor:
    """Balanced soft-assignment of N samples to P prototypes (SwAV).

    scores: (N, P) — typically cosine sim / epsilon, detached.
    returns Q: (N, P), rows sum to 1, columns balanced.
    """
    Q = torch.exp(scores / epsilon).t()        # (P, N)
    Q = Q / Q.sum().clamp_min(1e-12)
    P, N = Q.shape
    for _ in range(n_iters):
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Q = Q / P
        Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        Q = Q / N
    Q = Q * N
    return Q.t()


class HydroGraphProto(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        sample_rate: int = 5_120,
        n_bands: int = 24,
        tcn_dilations=(1, 4, 16, 64),
        embed_dim: int = 48,
        ens_n_ckpts: int = 6,
        ens_n_classes: int = 4,
        gnn_dim: int = 64,
        gnn_depth: int = 2,
        gnn_heads: int = 4,
        kNN: int = 4,
        temporal_ring: int = 2,
        perceiver_queries: int = 4,
        perceiver_depth: int = 2,
        n_prototypes_per_class: int = 16,
        proto_temp: float = 10.0,
        proto_dim: int = 128,
        dropout: float = 0.1,
        band_dropout: float = 0.10,
        min_K_graph: int = 6,
        ens_in_log_space: bool = True,
    ):
        super().__init__()
        self.encoder = _ClipEncoder(
            sample_rate=sample_rate, n_bands=n_bands,
            tcn_dilations=tcn_dilations, embed_dim=embed_dim,
            band_dropout=band_dropout,
        )
        self.ens_in_log_space = bool(ens_in_log_space)
        self.ens_dim = ens_n_ckpts * ens_n_classes
        token_dim = embed_dim + self.ens_dim
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, gnn_dim),
        )
        self.gnn = nn.ModuleList([
            _GINEBlock(gnn_dim, n_heads=gnn_heads, edge_dim=4, dropout=dropout)
            for _ in range(gnn_depth)
        ])
        self.pool = _PerceiverPool(gnn_dim, n_queries=perceiver_queries,
                                   n_heads=gnn_heads, depth=perceiver_depth,
                                   dropout=dropout)
        self.proj_proto = nn.Sequential(
            nn.LayerNorm(self.pool.out_dim),
            nn.Linear(self.pool.out_dim, proto_dim),
        )
        # Prototypes: (n_classes * M, proto_dim), L2-normalized at use-time.
        P = num_classes * n_prototypes_per_class
        self.prototypes = nn.Parameter(torch.randn(P, proto_dim) * 0.02)
        self.abstain_head = nn.Linear(self.pool.out_dim, 1)

        self.num_classes = int(num_classes)
        self.M = int(n_prototypes_per_class)
        self.proto_temp = float(proto_temp)
        self.kNN = int(kNN)
        self.temporal_ring = int(temporal_ring)
        self.min_K_graph = int(min_K_graph)
        self.gnn_dim = int(gnn_dim)
        self.embed_dim = int(embed_dim)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ----- graph construction --------------------------------------------------
    def _build_adj(self, tokens: torch.Tensor, pad_mask: torch.Tensor):
        """Return (adj, edge_f). adj: (N,K,K) {0,1}. edge_f: (N,K,K,4)."""
        N, K, D = tokens.shape
        device = tokens.device
        # Cosine similarity in token space (detached for adjacency choice)
        with torch.no_grad():
            tnorm = F.normalize(tokens, dim=-1)
            cos = torch.bmm(tnorm, tnorm.transpose(1, 2))          # (N,K,K)
            # Mask invalid columns
            valid_col = pad_mask.unsqueeze(1)                       # (N,1,K)
            cos_masked = cos.masked_fill(~valid_col, -2.0)
            # kNN per row
            k_eff = min(self.kNN, K)
            _, knn_idx = cos_masked.topk(k_eff, dim=-1)
            knn_adj = torch.zeros_like(cos)
            knn_adj.scatter_(-1, knn_idx, 1.0)
            # Temporal ring
            idx = torch.arange(K, device=device)
            dt = (idx.view(1, K) - idx.view(K, 1)).abs()           # (K,K)
            ring_adj = (dt <= self.temporal_ring).float().unsqueeze(0).expand(N, K, K)
            # Union + self-loops, mask invalid rows/cols
            adj = ((knn_adj + ring_adj) > 0).float()
            eye = torch.eye(K, device=device).unsqueeze(0)
            adj = ((adj + eye) > 0).float()
            valid = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(1)
            adj = adj * valid.float()

        # Edge features (cos and Δt are differentiable wrt cos; keep simple)
        cos_f = torch.bmm(F.normalize(tokens, dim=-1),
                          F.normalize(tokens, dim=-1).transpose(1, 2))   # (N,K,K)
        dt_norm = (idx.view(1, K) - idx.view(K, 1)).abs().float() / max(K, 1)
        dt_f = dt_norm.unsqueeze(0).expand(N, K, K).to(tokens.dtype)
        # Two onehot-style flags for "is knn edge" / "is temporal edge"
        edge_f = torch.stack([cos_f, dt_f,
                              adj, adj * 0 + 1.0], dim=-1)               # (N,K,K,4)
        return adj, edge_f

    # ----- forward -------------------------------------------------------------
    def forward(
        self,
        clips: torch.Tensor,         # (N, K, T)
        ens: torch.Tensor,           # (N, K, M, C) or (N, K, M*C)
        mask: Optional[torch.Tensor] = None,   # (N, K) True=valid
    ):
        if clips.dim() != 3:
            raise ValueError(f"clips must be (N,K,T); got {clips.shape}")
        N, K, T = clips.shape
        device = clips.device
        if mask is None:
            mask = torch.ones(N, K, dtype=torch.bool, device=device)

        z = self.encoder(clips.reshape(N * K, T)).view(N, K, self.embed_dim)
        ens_flat = ens.reshape(N, K, -1)
        if self.ens_in_log_space:
            ens_flat = torch.log(ens_flat.clamp_min(1e-6))
        tok = self.token_proj(torch.cat([z, ens_flat], dim=-1))         # (N,K,Dg)
        tok = tok * mask.unsqueeze(-1).to(tok.dtype)

        # GNN if enough valid nodes per row; otherwise identity.
        lengths = mask.sum(dim=1)
        use_graph = (lengths.min().item() >= self.min_K_graph)
        if use_graph:
            adj, edge_f = self._build_adj(tok, mask)
            h = tok
            for blk in self.gnn:
                h = blk(h, adj, edge_f, mask)
        else:
            h = tok

        s_pool = self.pool(h, mask)                                     # (N, Q*Dg)
        s = self.proj_proto(s_pool)                                     # (N, Dp)
        s_n = F.normalize(s, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        cluster_scores = s_n @ p_n.t()                                  # (N, P)

        # Class logits via soft-min (logsumexp) over the M prototypes per class
        scaled = cluster_scores * self.proto_temp
        scaled = scaled.view(N, self.num_classes, self.M)
        cls_logits = torch.logsumexp(scaled, dim=-1)                    # (N, C)
        abstain = self.abstain_head(s_pool).squeeze(-1)                 # (N,)
        return cls_logits, abstain, cluster_scores, s


__all__ = ["HydroGraphProto", "sinkhorn_knopp"]
