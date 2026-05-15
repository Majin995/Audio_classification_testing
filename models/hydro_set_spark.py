"""
HydroSetSpark — source-level architecture that consumes ALL clips of a source
in one forward pass and emits ONE source-level prediction.

Built on top of `HydroSpark` (strictly 1D DSP-probe). The per-clip encoder
outputs a small (D-dim) embedding; an attention pool aggregates an arbitrary
number of clip embeddings into a single source embedding; a tiny classifier
head produces the source prediction.

Why this is the right architecture for the active goal
─────────────────────────────────────────────────────
CLAUDE.md documents that the existing 9-ckpt per-clip → log-mean pool tops at
F1=0.7895 on n=83 test sources because 5 Cargo sources are unanimously
confused with Tanker. The hint is: "source-level transformer that re-encodes
all clips of a source captures across-clip patterns the per-clip pool
ensemble cannot."

HydroSetSpark IS that: the attention pool learns which clips of a source are
class-informative (so a Cargo source whose 80% of clips look like Tanker but
20% look like Cargo gets classified as Cargo — log-mean would average it
out).

The whole model has ~5-15k parameters (smaller than a single HydroHydra ckpt
by ~100×).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hydro_spark import (
    ParametricSincBank, EnvelopeDecimator, PCEN, ModTCN,
)


class _ClipEncoder(nn.Module):
    """The HydroSpark feature stack, with the final classifier head replaced
    by a single LayerNorm → Linear that maps to an embedding of size
    ``embed_dim``. Output: (B, embed_dim).
    """

    def __init__(
        self,
        sample_rate: int = 5_120,
        n_bands: int = 24,
        sinc_kernel: int = 257,
        env_lpf_kernel: int = 33,
        env_decim: int = 32,
        tcn_kernel: int = 7,
        tcn_dilations=(1, 4, 16, 64),
        expansion: int = 1,
        embed_dim: int = 48,
        band_dropout: float = 0.0,
        use_delta: bool = True,
        use_coherence: bool = True,
        freeze_sinc: bool = False,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.band_dropout = float(band_dropout)
        self.use_delta = bool(use_delta)
        self.use_coherence = bool(use_coherence)
        self.sinc = ParametricSincBank(
            n_filters=n_bands, kernel_size=sinc_kernel, sample_rate=sample_rate,
        )
        if freeze_sinc:
            for p in self.sinc.parameters():
                p.requires_grad_(False)
        self.env = EnvelopeDecimator(
            n_channels=n_bands, lpf_kernel=env_lpf_kernel, decim=env_decim,
        )
        self.pcen = PCEN(n_channels=n_bands)
        self.tcn = ModTCN(
            n_channels=n_bands, kernel=tcn_kernel,
            dilations=tcn_dilations, expansion=expansion,
        )
        c_after = self.tcn.out_channels
        feat_dim = 2 * c_after + n_bands  # mean + std + log_energy
        if self.use_delta:
            feat_dim += 2 * c_after          # delta over TCN output
        if self.use_coherence:
            feat_dim += (c_after - 1)        # coherence over TCN channels
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, embed_dim),
        )
        self.embed_dim = int(embed_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        x = self.sinc(waveform)                          # (B, C, T)
        if self.training and self.band_dropout > 0:
            B, C, _ = x.shape
            keep = (torch.rand(B, C, 1, device=x.device) > self.band_dropout).float()
            scale = keep.sum(dim=1, keepdim=True).clamp_min(1.0) / C
            x = x * keep / scale.clamp_min(1e-3)
        env = self.env(x)                                # (B, C, T_env)
        h = self.pcen(env)                               # (B, C, T_env)
        h = self.tcn(h)                                  # (B, C', T_env)
        mean = h.mean(dim=-1); std = h.std(dim=-1, unbiased=False)
        log_e = (env.pow(2).mean(dim=-1) + 1e-8).log()
        log_e = log_e - log_e.mean(dim=-1, keepdim=True)
        feats = [mean, std, log_e]
        if self.use_delta:
            d = h[..., 1:] - h[..., :-1]
            feats.append(d.mean(dim=-1)); feats.append(d.std(dim=-1, unbiased=False))
        if self.use_coherence:
            hc = h - h.mean(dim=-1, keepdim=True)
            num = (hc[:, :-1] * hc[:, 1:]).mean(dim=-1)
            den = (hc[:, :-1].pow(2).mean(dim=-1).clamp_min(1e-8)
                   * hc[:, 1:].pow(2).mean(dim=-1).clamp_min(1e-8)).sqrt()
            feats.append(num / den.clamp_min(1e-8))
        z = torch.cat(feats, dim=-1)
        return self.proj(z)                              # (B, embed_dim)


class _AttentivePool(nn.Module):
    """Single learnable query attention over a set of embeddings.

    Input  : (N, K, D), mask (N, K) — True for valid clips
    Output : (N, D)  — source-level embedding (and the attention weights, in
                       case the caller wants to inspect them).
    """

    def __init__(self, embed_dim: int, n_heads: int = 2):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = int(embed_dim)
        self.n_heads = int(n_heads)
        self.head_dim = embed_dim // n_heads
        # Learnable query (one per head)
        self.query = nn.Parameter(torch.randn(1, n_heads, self.head_dim) * 0.02)
        # Key/value projections (kept tiny)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Output linear (post-concat of heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (N, K, D); mask: (N, K) — True for valid
        N, K, D = x.shape
        H, Hd = self.n_heads, self.head_dim
        k = self.k_proj(x).view(N, K, H, Hd).transpose(1, 2)             # (N, H, K, Hd)
        v = self.v_proj(x).view(N, K, H, Hd).transpose(1, 2)             # (N, H, K, Hd)
        q = self.query.unsqueeze(0).expand(N, -1, -1, -1)                # (N, 1, H, Hd)
        q = q.transpose(1, 2)                                            # (N, H, 1, Hd)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)               # (N, H, 1, K)
        if mask is not None:
            scores = scores.masked_fill(~mask.view(N, 1, 1, K), -1e4)
        attn = F.softmax(scores, dim=-1)                                 # (N, H, 1, K)
        out = (attn @ v).squeeze(2)                                      # (N, H, Hd)
        out = out.reshape(N, D)
        return self.out_proj(out), attn.squeeze(2)                       # (N, D), (N, H, K)


class HydroSetSpark(nn.Module):
    """End-to-end source-level classifier.

    Forward inputs:
      clips: (N, K, T)            — N sources, K clips of length T per source
      mask:  (N, K) bool optional — True for valid (real) clips, False for padding

    Output:
      logits: (N, num_classes [+1 gambler]) — source-level prediction
    """

    def __init__(
        self,
        num_classes: int = 4,
        sample_rate: int = 5_120,
        n_bands: int = 24,
        sinc_kernel: int = 257,
        env_decim: int = 32,
        tcn_dilations=(1, 4, 16, 64),
        expansion: int = 1,
        embed_dim: int = 48,
        n_heads: int = 2,
        head_hidden: int = 64,
        dropout: float = 0.20,
        band_dropout: float = 0.0,
        use_delta: bool = True,
        use_coherence: bool = True,
        freeze_sinc: bool = False,
        gambler: bool = True,
    ):
        super().__init__()
        self.encoder = _ClipEncoder(
            sample_rate=sample_rate, n_bands=n_bands,
            sinc_kernel=sinc_kernel, env_decim=env_decim,
            tcn_dilations=tcn_dilations, expansion=expansion,
            embed_dim=embed_dim, band_dropout=band_dropout,
            use_delta=use_delta, use_coherence=use_coherence,
            freeze_sinc=freeze_sinc,
        )
        self.pool = _AttentivePool(embed_dim=embed_dim, n_heads=n_heads)
        self.gambler = bool(gambler)
        out = num_classes + (1 if gambler else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out),
        )
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, clips: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # clips: (N, K, T) or (N*K, T) with mask of shape (N, K)
        if clips.dim() == 3:
            N, K, T = clips.shape
            flat = clips.reshape(N * K, T)
        elif clips.dim() == 2 and mask is not None:
            flat = clips; N, K = mask.shape
        else:
            raise ValueError(f"clips must be (N,K,T) or (N*K,T)+mask; got {clips.shape}")
        z = self.encoder(flat)                                  # (N*K, D)
        z = z.view(N, K, self.embed_dim)
        pooled, _ = self.pool(z, mask=mask)                     # (N, D)
        return self.head(pooled)                                # (N, K_cls[+1])

    def encode_clips(self, clips: torch.Tensor) -> torch.Tensor:
        """Convenience: get per-clip embeddings (B, D)."""
        return self.encoder(clips)


__all__ = ["HydroSetSpark"]
