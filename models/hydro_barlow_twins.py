"""
HydroBarlowTwins — Self-Supervised Pretraining for UATR
========================================================

Barlow Twins SSL pretraining (Zbontar et al., ICML 2021) on top of the
HydroPrecise three-branch encoder.  The classification head of
``HydroPrecise`` is bypassed; the post-pool ``(B, 2*fusion_dim)`` embedding
is fed through a 3-layer MLP projector and the cross-correlation matrix
loss is minimised between two stochastic views of each waveform.

Pipeline
--------

  Waveform (B, T)
      │
      ├──► _TwoViewWaveformAug ──► view 1 ──► HydroPrecise._features ──► embed1 (B, 2D)
      │                                                                          │
      └──► _TwoViewWaveformAug ──► view 2 ──► HydroPrecise._features ──► embed2 (B, 2D)
                                                                                 │
                                                                          projector (3-layer MLP)
                                                                                 │
                                                                          z1, z2  (B, P)
                                                                                 │
                                                            BarlowTwins cross-correlation loss

Loss
----
  C = (BN(z1)ᵀ · BN(z2)) / N        (P × P cross-correlation matrix)
  on_diag  = Σ_i (C_ii − 1)²
  off_diag = λ · Σ_{i≠j} C_ij²
  L        = on_diag + off_diag

The diagonal target is the identity matrix: perfect feature redundancy
reduction means C = I.

Reference
---------
  Zbontar, Jing, Misra, LeCun & Deny.
  "Barlow Twins: Self-Supervised Learning via Redundancy Reduction".
  ICML 2021. arXiv:2103.03230
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.hydro_precise import HydroPrecise


# ═══════════════════════════════════════════════════════════════════════
#  Two-view waveform augmentation
# ═══════════════════════════════════════════════════════════════════════

class _TwoViewWaveformAug(nn.Module):
    """Stochastic waveform augmentation producing two correlated views.

    Each call returns ``(view1, view2)`` where each view independently
    samples Gaussian noise (15-30 dB SNR), random ±gain, and a small
    random circular time-shift.  Views differ even when the same
    waveform is passed twice.
    """

    def __init__(
        self,
        noise_prob:    float = 0.8,
        noise_snr_min: float = 10.0,
        noise_snr_max: float = 30.0,
        gain_prob:     float = 0.8,
        gain_range:    float = 0.4,
        shift_prob:    float = 0.5,
        shift_max:     float = 0.1,
    ):
        super().__init__()
        self.noise_prob    = noise_prob
        self.noise_snr_min = noise_snr_min
        self.noise_snr_max = noise_snr_max
        self.gain_prob     = gain_prob
        self.gain_range    = gain_range
        self.shift_prob    = shift_prob
        self.shift_max     = shift_max

    def _one_view(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.noise_prob:
            snr = self.noise_snr_min + (self.noise_snr_max - self.noise_snr_min) * torch.rand(1).item()
            sig_pow   = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12)
            noise_pow = sig_pow / (10.0 ** (snr / 10.0))
            x = x + torch.randn_like(x) * noise_pow.sqrt()
        if torch.rand(1).item() < self.gain_prob:
            g = 1.0 + (2.0 * torch.rand(1, device=x.device).item() - 1.0) * self.gain_range
            x = x * g
        if torch.rand(1).item() < self.shift_prob:
            T = x.size(-1)
            max_shift = max(1, int(T * self.shift_max))
            s = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            if s != 0:
                x = torch.roll(x, shifts=s, dims=-1)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._one_view(x), self._one_view(x)


# ═══════════════════════════════════════════════════════════════════════
#  Barlow Twins projection head
# ═══════════════════════════════════════════════════════════════════════

class _BarlowProjector(nn.Module):
    """3-layer MLP projector per Zbontar et al. §3.2.

    ``Linear → BN → ReLU`` for the first two layers; final ``Linear`` has
    no normalisation or non-linearity (matches the reference implementation).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
#  HydroBarlowTwins LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroBarlowTwins(pl.LightningModule):
    """Self-supervised Barlow Twins pretrainer for hydrophone waveforms.

    The encoder is a frozen-or-trainable ``HydroPrecise`` backbone whose
    classification head is unused; the post-pool embedding is projected
    through a 3-layer MLP and the cross-correlation matrix loss is
    minimised across two stochastic views of each batch.

    The downstream classifier loads ``self.encoder.state_dict()`` as
    initialisation for fine-tuning.
    """

    def __init__(
        self,
        num_classes:     int   = 4,
        sample_rate:     int   = 5_120,
        # ── Projector ────────────────────────────────────────────────────
        projection_hidden: int = 2048,
        projection_dim:    int = 2048,
        # ── Loss ─────────────────────────────────────────────────────────
        lambda_param:    float = 5e-3,
        # ── Two-view augmentation ────────────────────────────────────────
        noise_prob:      float = 0.8,
        noise_snr_min:   float = 10.0,
        noise_snr_max:   float = 30.0,
        gain_prob:       float = 0.8,
        gain_range:      float = 0.4,
        shift_prob:      float = 0.5,
        shift_max:       float = 0.1,
        # ── Encoder kwargs (forwarded to HydroPrecise) ───────────────────
        gabor_n_filters: int   = 64,
        gabor_kernel:    int   = 257,
        gabor_ch:        int   = 128,
        cqt_n_bins:      int   = 84,
        cqt_bpo:         int   = 12,
        cqt_hop:         int   = 64,
        cqt_ch:          int   = 128,
        demon_hop:       int   = 64,
        demon_ch:        int   = 64,
        demon_n_fft:     int   = 2048,
        fusion_T:        int   = 64,
        fusion_dim:      int   = 192,
        n_heads:         int   = 2,
        dropout:         float = 0.15,
        # ── Optimiser ────────────────────────────────────────────────────
        learning_rate:   float = 3e-4,
        weight_decay:    float = 1e-2,
        warmup_epochs:   int   = 10,
        max_epochs:      int   = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = HydroPrecise(
            num_classes=num_classes,
            sample_rate=sample_rate,
            gabor_n_filters=gabor_n_filters,
            gabor_kernel=gabor_kernel,
            gabor_ch=gabor_ch,
            cqt_n_bins=cqt_n_bins,
            cqt_bpo=cqt_bpo,
            cqt_hop=cqt_hop,
            cqt_ch=cqt_ch,
            demon_hop=demon_hop,
            demon_ch=demon_ch,
            demon_n_fft=demon_n_fft,
            fusion_T=fusion_T,
            fusion_dim=fusion_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        self._embed_dim = fusion_dim * 2
        self.projector = _BarlowProjector(
            in_dim=self._embed_dim,
            hidden_dim=projection_hidden,
            out_dim=projection_dim,
        )

        self.two_view_aug = _TwoViewWaveformAug(
            noise_prob=noise_prob,
            noise_snr_min=noise_snr_min,
            noise_snr_max=noise_snr_max,
            gain_prob=gain_prob,
            gain_range=gain_range,
            shift_prob=shift_prob,
            shift_max=shift_max,
        )

        self.bn = nn.BatchNorm1d(projection_dim, affine=False)
        self.lambda_param = lambda_param

    # ── Forward ──────────────────────────────────────────────────────────

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run waveform through the encoder up to (but excluding) the head.

        Args:
            waveform: ``(B, T)`` mono float32 waveform at ``sample_rate``.

        Returns:
            ``(B, 2 * fusion_dim)`` pooled embedding.
        """
        return self.encoder._features(waveform)

    def project(self, embedding: torch.Tensor) -> torch.Tensor:
        """Run encoder embedding through the 3-layer projector."""
        return self.projector(embedding)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return the projected embedding ``(B, projection_dim)`` for one view."""
        return self.project(self.encode(waveform))

    # ── Barlow Twins loss ────────────────────────────────────────────────

    def barlow_twins_loss(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the Barlow Twins cross-correlation loss.

        Args:
            z1: ``(B, P)`` projected embedding from view 1.
            z2: ``(B, P)`` projected embedding from view 2.

        Returns:
            ``(loss, on_diag, off_diag)`` — the on-diagonal and off-diagonal
            terms are returned for logging.
        """
        N = z1.size(0)
        z1n = self.bn(z1)
        z2n = self.bn(z2)

        c = (z1n.T @ z2n) / float(N)            # (P, P)

        diag = torch.diagonal(c)
        on_diag = (diag - 1.0).pow(2).sum()

        off_diag_mask = ~torch.eye(c.size(0), dtype=torch.bool, device=c.device)
        off_diag = c[off_diag_mask].pow(2).sum()

        loss = on_diag + self.lambda_param * off_diag
        return loss, on_diag.detach(), off_diag.detach()

    # ── Lightning steps ──────────────────────────────────────────────────

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            waveform = batch[0]
        else:
            waveform = batch

        view1, view2 = self.two_view_aug(waveform)
        z1 = self.project(self.encode(view1))
        z2 = self.project(self.encode(view2))

        loss, on_diag, off_diag = self.barlow_twins_loss(z1, z2)

        self.log(f"{stage}/bt_loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True)
        self.log(f"{stage}/on_diag",  on_diag,  on_epoch=True)
        self.log(f"{stage}/off_diag", off_diag, on_epoch=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    # ── Optimiser ────────────────────────────────────────────────────────

    def configure_optimizers(self) -> Dict[str, Any]:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [{"params": decay,    "weight_decay": self.hparams.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.learning_rate, betas=(0.9, 0.98), eps=1e-8,
        )

        def lr_lambda(epoch: int) -> float:
            wu, total = self.hparams.warmup_epochs, self.hparams.max_epochs
            if epoch < wu:
                return (epoch + 1) / max(wu, 1)
            p = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
