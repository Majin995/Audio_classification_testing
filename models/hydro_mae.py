"""
HydroMAE — Audio Masked Autoencoder for LOFAR Spectrogram Classification
=========================================================================

Implements the MAE paradigm (He et al., 2022) adapted for 2-D LOFAR
spectrograms with a two-phase training workflow:

  Phase 1 — Self-supervised pre-training (--pretrain)
  ────────────────────────────────────────────────────
  Randomly mask 75 % of spectrogram patches.  The encoder (ViT-small)
  processes only the 25 % visible patches — intentionally sparse so it
  must learn robust representations from incomplete views.  A lightweight
  decoder reconstructs the full spectrogram from encoder outputs + learnable
  [MASK] tokens.  Loss: MSE on normalised pixel values of the masked patches.

  Phase 2 — Supervised fine-tuning  (--finetune)
  ───────────────────────────────────────────────
  Load the pre-trained encoder.  Discard the decoder.  All N tokens (no
  masking) are passed through the encoder.  Global average pooling + a
  linear classification head is trained with FocalLoss.  The full model
  (encoder + head) is fine-tuned end-to-end with a small peak LR.

Architecture
------------
  Raw waveform (5 120 Hz, 1 s)
      ↓
  LofarFrontend → (B, 1, 32, 256)  [time_bins × freq_bins]
      ↓
  Patchify: (patch_h=4, patch_w=16) → (B, 128, 64)   [128 patches, 64-d each]
      ↓
  Linear patch projection → (B, 128, enc_dim=384)
  + sinusoidal 2-D positional embeddings

  ┌─── Pre-train branch ──────────────────────────────────────────────┐
  │  Random mask 75 % → visible tokens only (B, ~32, 384)             │
  │  MAE Encoder: 6 × TransformerBlock(d=384, heads=6)                │
  │      ↓                                                            │
  │  Full sequence reconstruction:                                    │
  │    encode visible → insert learnable MASK tokens at masked positions│
  │    + decoder positional embeddings                                 │
  │  MAE Decoder: 4 × TransformerBlock(d=192, heads=4)                │
  │      ↓                                                            │
  │  Linear(192 → patch_dim=64) → MSE on masked patches               │
  └───────────────────────────────────────────────────────────────────┘

  ┌─── Fine-tune branch ──────────────────────────────────────────────┐
  │  All 128 tokens → MAE Encoder (no masking)                        │
  │  Global average pool → (B, 384)                                   │
  │  Classifier: LayerNorm → Linear(384 → num_classes)                │
  │      ↓                                                            │
  │  FocalLoss (class-weighted)                                       │
  └───────────────────────────────────────────────────────────────────┘

Why MAE for underwater acoustics?
----------------------------------
Labelled underwater recordings are scarce and expensive (expert annotation).
MAE allows pre-training on large volumes of *unlabelled* passive sonar data
and then fine-tuning on the small labelled set, recovering the benefit of
self-supervised representation learning without requiring paired ground truth.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from .hydro_conformer import FocalLoss, DropPath
from .hydro_lofar_resnet import LofarFrontend, LofarSpecAugment


# ═══════════════════════════════════════════════════════════════════════
#  2-D Sinusoidal Positional Embedding
# ═══════════════════════════════════════════════════════════════════════

def sinusoidal_2d_pe(H: int, W: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Build a (1, H*W, dim) 2-D sinusoidal positional embedding.

    The first dim//2 channels encode row (time) position;
    the second dim//2 encode column (frequency) position.
    """
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2

    def _pe1d(length, d, dev):
        pe  = torch.zeros(length, d, device=dev)
        pos = torch.arange(length, device=dev).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d, 2, device=dev).float() * (-math.log(10_000.0) / d)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe                             # (length, d)

    row_pe = _pe1d(H, half, device)          # (H, dim//2)
    col_pe = _pe1d(W, half, device)          # (W, dim//2)

    # Broadcast to (H, W, dim)
    row_pe = row_pe.unsqueeze(1).expand(H, W, half)
    col_pe = col_pe.unsqueeze(0).expand(H, W, half)
    pe     = torch.cat([row_pe, col_pe], dim=-1)  # (H, W, dim)
    return pe.reshape(1, H * W, dim)              # (1, N, dim)


# ═══════════════════════════════════════════════════════════════════════
#  Transformer Block  (pre-norm, used by both encoder and decoder)
# ═══════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Standard pre-LayerNorm transformer block.

        LN → MHSA → residual
        LN → MLP  → residual

    With stochastic depth (DropPath) on both residual branches.
    """

    def __init__(
        self,
        dim:        int,
        n_heads:    int,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.0,
        drop_path:  float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        a, _ = self.attn(xn, xn, xn)
        x = x + self.dp1(a)
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  MAE Encoder
# ═══════════════════════════════════════════════════════════════════════

class MAEEncoder(nn.Module):
    """
    ViT-small encoder.  In pre-train mode receives only visible patches;
    in fine-tune mode receives the full token sequence.
    """

    def __init__(
        self,
        patch_dim:  int,
        enc_dim:    int   = 384,
        n_blocks:   int   = 6,
        n_heads:    int   = 6,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.0,
        drop_path:  float = 0.1,
    ):
        super().__init__()
        self.projection = nn.Linear(patch_dim, enc_dim)
        dp_rates = [drop_path * i / max(n_blocks - 1, 1)
                    for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            TransformerBlock(enc_dim, n_heads, mlp_ratio, dropout, dp_rates[i])
            for i in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(enc_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, N_vis, patch_dim)  — already masked/selected externally
        Returns:
            (B, N_vis, enc_dim)
        """
        x = self.projection(patches)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════════
#  MAE Decoder
# ═══════════════════════════════════════════════════════════════════════

class MAEDecoder(nn.Module):
    """
    Lightweight transformer decoder for patch reconstruction.

    Takes the full token sequence (encoded visible + MASK placeholders)
    and predicts the raw (normalised) pixel values of every patch.
    Only the masked positions contribute to the MSE loss.
    """

    def __init__(
        self,
        enc_dim:    int,
        patch_dim:  int,
        dec_dim:    int   = 192,
        n_blocks:   int   = 4,
        n_heads:    int   = 4,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(enc_dim, dec_dim)
        self.blocks     = nn.ModuleList([
            TransformerBlock(dec_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])
        self.norm    = nn.LayerNorm(dec_dim)
        self.pred    = nn.Linear(dec_dim, patch_dim)   # reconstruct pixels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, enc_dim)  — full sequence (visible + mask tokens)
        Returns:
            (B, N, patch_dim)   — reconstructed pixel values for all patches
        """
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.pred(self.norm(x))


# ═══════════════════════════════════════════════════════════════════════
#  HydroMAE  LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroMAE(pl.LightningModule):
    """
    Masked Autoencoder for LOFAR underwater acoustic classification.

    Supports two training modes controlled by the `mode` argument:

    mode = "pretrain"
        Self-supervised pre-training.  No labels required.  The model learns
        to reconstruct randomly masked LOFAR spectrogram patches.  Save the
        best checkpoint (lowest val/recon_loss) and use its path as
        `encoder_ckpt` when switching to fine-tune mode.

    mode = "classify"
        Supervised fine-tuning.  Set `encoder_ckpt` to the pre-train
        checkpoint path; the encoder weights will be loaded automatically
        and fine-tuned end-to-end with FocalLoss.  If `encoder_ckpt` is
        empty, the encoder is trained from scratch.

    Args:
        num_classes     : Number of vessel classes (only used in classify mode).
        class_weights   : Inverse-frequency weights for focal loss.
        mode            : "pretrain" | "classify".
        encoder_ckpt    : Path to a pre-trained HydroMAE checkpoint (optional).
        sample_rate     : Audio sample rate in Hz.
        n_fft / hop     : LOFAR spectrogram parameters.
        time_bins / freq_bins : Fixed spectrogram image dimensions.
        patch_h / patch_w    : Patch size (must divide time_bins / freq_bins).
        mask_ratio      : Fraction of patches masked during pre-training.
        enc_dim         : Encoder hidden dimension.
        enc_depth       : Encoder transformer blocks.
        enc_heads       : Encoder attention heads.
        dec_dim         : Decoder hidden dimension (lighter than encoder).
        dec_depth       : Decoder transformer blocks.
        dec_heads       : Decoder attention heads.
        learning_rate   : Peak AdamW LR.
        weight_decay    : AdamW weight decay.
        warmup_epochs   : Linear LR warmup.
        max_epochs      : Total epochs.
        focal_gamma     : Focal loss γ (classify mode only).
        label_smoothing : Label smoothing ε (classify mode only).
    """

    def __init__(
        self,
        num_classes:     int            = 3,
        class_weights:   Optional[list] = None,
        mode:            str            = "pretrain",  # "pretrain" | "classify"
        encoder_ckpt:    str            = "",
        sample_rate:     int            = 5_120,
        n_fft:           int            = 4_096,
        hop_length:      int            = 160,
        time_bins:       int            = 32,
        freq_bins:       int            = 256,
        patch_h:         int            = 4,
        patch_w:         int            = 16,
        mask_ratio:      float          = 0.75,
        enc_dim:         int            = 384,
        enc_depth:       int            = 6,
        enc_heads:       int            = 6,
        dec_dim:         int            = 192,
        dec_depth:       int            = 4,
        dec_heads:       int            = 4,
        learning_rate:   float          = 1e-4,
        weight_decay:    float          = 5e-2,
        warmup_epochs:   int            = 10,
        max_epochs:      int            = 200,
        focal_gamma:     float          = 2.0,
        label_smoothing: float          = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert time_bins % patch_h == 0, "patch_h must divide time_bins"
        assert freq_bins % patch_w == 0, "patch_w must divide freq_bins"
        assert mode in ("pretrain", "classify"), "mode must be 'pretrain' or 'classify'"

        self.mode       = mode
        self.n_patches_h = time_bins  // patch_h   # rows of patches
        self.n_patches_w = freq_bins  // patch_w   # cols of patches
        self.n_patches   = self.n_patches_h * self.n_patches_w
        self.patch_dim   = patch_h * patch_w        # pixels per patch

        # ── Frontend ────────────────────────────────────────────────────
        self.frontend = LofarFrontend(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            time_bins=time_bins, freq_bins=freq_bins,
        )
        self.spec_aug = LofarSpecAugment(
            n_time_masks=2, time_mask_max=6,
            n_freq_masks=2, freq_mask_max=24,
        )

        # ── Encoder ─────────────────────────────────────────────────────
        self.encoder = MAEEncoder(
            patch_dim=self.patch_dim,
            enc_dim=enc_dim, n_blocks=enc_depth, n_heads=enc_heads,
        )

        # Learnable MASK token (pre-train only; occupies masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, enc_dim))

        # ── Decoder (pre-train only — not needed in classify mode) ──────
        if mode == "pretrain":
            self.decoder = MAEDecoder(
                enc_dim=enc_dim, patch_dim=self.patch_dim,
                dec_dim=dec_dim, n_blocks=dec_depth, n_heads=dec_heads,
            )
        else:
            self.decoder = None

        # ── Classification head (classify mode only) ────────────────────
        if mode == "classify":
            self.head = nn.Sequential(
                nn.LayerNorm(enc_dim),
                nn.Linear(enc_dim, num_classes),
            )
            self.criterion = FocalLoss(
                class_weights=class_weights,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
        else:
            self.head      = None
            self.criterion = None

        # ── Metrics (classify mode only) ────────────────────────────────
        if mode == "classify":
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

        self._init_weights()

        # Load pre-trained encoder if provided
        if encoder_ckpt:
            self._load_encoder(encoder_ckpt)

    # ── Weight initialisation ────────────────────────────────────────────

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_encoder(self, ckpt_path: str):
        """Transfer encoder weights from a pre-train checkpoint."""
        ckpt       = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        enc_state  = {
            k.removeprefix("encoder."): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
        print(f"[HydroMAE] Loaded encoder from {ckpt_path}")
        if missing:
            print(f"  Missing  : {missing}")
        if unexpected:
            print(f"  Unexpected: {unexpected}")

    # ── Patchify / Unpatchify ────────────────────────────────────────────

    def _patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        (B, 1, H, W) → (B, n_patches, patch_dim)

        Patches are extracted in row-major order (top-left first, then
        left-to-right across columns, then top-to-bottom across rows).
        """
        B = imgs.shape[0]
        ph, pw = self.hparams.patch_h, self.hparams.patch_w
        nh, nw = self.n_patches_h, self.n_patches_w
        # (B, 1, nh, ph, nw, pw) → (B, nh*nw, ph*pw)
        x = imgs.reshape(B, 1, nh, ph, nw, pw)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, nh * nw, ph * pw)
        return x

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        (B, n_patches, patch_dim) → (B, 1, H, W)
        """
        B    = patches.shape[0]
        ph, pw = self.hparams.patch_h, self.hparams.patch_w
        nh, nw = self.n_patches_h, self.n_patches_w
        x    = patches.reshape(B, nh, nw, 1, ph, pw)
        x    = x.permute(0, 3, 1, 4, 2, 5).reshape(
            B, 1, nh * ph, nw * pw
        )
        return x

    def _random_mask(self, B: int, device: torch.device) -> tuple:
        """
        Generate a random mask for a batch.

        Returns
        -------
        visible_idx : (B, N_vis) — indices of unmasked patches
        masked_idx  : (B, N_msk) — indices of masked patches
        restore_idx : (B, N)     — argsort of shuffle, for restoring order
        """
        N      = self.n_patches
        noise  = torch.rand(B, N, device=device)
        ids_shuffle   = torch.argsort(noise, dim=1)
        ids_restore   = torch.argsort(ids_shuffle, dim=1)
        n_keep         = int(N * (1.0 - self.hparams.mask_ratio))
        visible_idx   = ids_shuffle[:, :n_keep]
        masked_idx    = ids_shuffle[:, n_keep:]
        return visible_idx, masked_idx, ids_restore

    # ── Pre-train forward ────────────────────────────────────────────────

    def _pretrain_forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs: (B, 1, time_bins, freq_bins)
        Returns:
            recon_loss: scalar MSE on masked patches (normalised targets)
        """
        B = imgs.shape[0]
        N = self.n_patches
        H, W = self.n_patches_h, self.n_patches_w

        # Patchify
        patches = self._patchify(imgs)      # (B, N, patch_dim)

        # Normalise patch targets: mean=0, std=1 per patch
        mean   = patches.mean(dim=-1, keepdim=True)
        std    = patches.std(dim=-1, keepdim=True).clamp(min=1e-6)
        target = (patches - mean) / std     # (B, N, patch_dim)

        # Random mask
        vis_idx, msk_idx, restore_idx = self._random_mask(B, imgs.device)
        n_vis  = vis_idx.shape[1]
        n_msk  = msk_idx.shape[1]

        # Gather visible patches and their positional embeddings
        vis_patches = torch.gather(
            patches, 1,
            vis_idx.unsqueeze(-1).expand(B, n_vis, self.patch_dim)
        )  # (B, n_vis, patch_dim)

        pe = sinusoidal_2d_pe(H, W, self.hparams.enc_dim, imgs.device)  # (1, N, enc_dim)
        vis_pe  = torch.gather(
            pe.expand(B, -1, -1), 1,
            vis_idx.unsqueeze(-1).expand(B, n_vis, self.hparams.enc_dim)
        )  # (B, n_vis, enc_dim)

        # Encode visible patches (projection + PE added inside encoder)
        vis_tokens = self.encoder.projection(vis_patches) + vis_pe  # (B, n_vis, enc_dim)
        for blk in self.encoder.blocks:
            vis_tokens = blk(vis_tokens)
        vis_tokens = self.encoder.norm(vis_tokens)       # (B, n_vis, enc_dim)

        # Build full-sequence decoder input:
        # sorted positions with encoded visible tokens at correct indices,
        # learnable MASK token at masked positions + full PE
        full_tokens = self.mask_token.expand(B, N, -1).clone()  # (B, N, enc_dim)
        full_tokens = full_tokens.scatter(
            1,
            vis_idx.unsqueeze(-1).expand(B, n_vis, self.hparams.enc_dim),
            vis_tokens,
        )
        dec_pe = sinusoidal_2d_pe(H, W, self.hparams.enc_dim, imgs.device)
        full_tokens = full_tokens + dec_pe.expand(B, -1, -1)

        # Decode & predict
        recon = self.decoder(full_tokens)                # (B, N, patch_dim)

        # MSE on masked patches only
        msk_recon  = torch.gather(
            recon, 1,
            msk_idx.unsqueeze(-1).expand(B, n_msk, self.patch_dim)
        )  # (B, n_msk, patch_dim)
        msk_target = torch.gather(
            target, 1,
            msk_idx.unsqueeze(-1).expand(B, n_msk, self.patch_dim)
        )
        return F.mse_loss(msk_recon, msk_target)

    # ── Classify forward ─────────────────────────────────────────────────

    def _classify_forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs: (B, 1, time_bins, freq_bins)
        Returns:
            logits: (B, num_classes)
        """
        B = imgs.shape[0]
        H, W = self.n_patches_h, self.n_patches_w

        patches = self._patchify(imgs)                            # (B, N, patch_dim)
        pe      = sinusoidal_2d_pe(H, W, self.hparams.enc_dim,
                                   imgs.device).expand(B, -1, -1)
        tokens  = self.encoder.projection(patches) + pe
        for blk in self.encoder.blocks:
            tokens = blk(tokens)
        tokens = self.encoder.norm(tokens)                        # (B, N, enc_dim)
        feat   = tokens.mean(dim=1)                               # (B, enc_dim)
        return self.head(feat)                                    # (B, num_classes)

    # ── Public forward ───────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor):
        """
        Args:
            waveform: (B, T) float32 waveform
        Returns:
            pretrain mode : recon_loss scalar
            classify mode : (B, num_classes) logits
        """
        imgs = self.frontend(waveform)       # (B, 1, time_bins, freq_bins)
        imgs = self.spec_aug(imgs)
        if self.mode == "pretrain":
            return self._pretrain_forward(imgs)
        return self._classify_forward(imgs)

    # ── Lightning steps — pre-train ──────────────────────────────────────

    def _pretrain_step(self, batch, stage: str):
        x = batch[0]                        # ignore labels
        loss = self(x)
        self.log(f"{stage}/recon_loss", loss, on_epoch=True, prog_bar=True,
                 on_step=(stage == "train"))
        return loss

    # ── Lightning steps — classify ───────────────────────────────────────

    def _mixup(self, x, y):
        alpha = 0.3
        if not self.training or alpha <= 0.0:
            return x, y, y, 1.0
        lam  = torch.distributions.Beta(alpha, alpha).sample().to(x)
        perm = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam

    def _classify_loss(self, logits, y, y_perm=None, lam=1.0):
        if y_perm is None or lam == 1.0:
            return self.criterion(logits, y)
        return (lam * self.criterion(logits, y)
                + (1.0 - lam) * self.criterion(logits, y_perm))

    # ── Unified dispatch ─────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        if self.mode == "pretrain":
            return self._pretrain_step(batch, "train")
        # classify
        x, y           = batch
        x, y, y_p, lam = self._mixup(x, y)
        logits         = self(x)
        loss           = self._classify_loss(logits, y, y_p, lam)
        self.train_acc(logits, y)
        self.log("train/loss", loss,           on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == "pretrain":
            return self._pretrain_step(batch, "val")
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        self.val_acc(logits, y);       self.val_f1(logits, y)
        self.val_precision(logits, y); self.val_mcc(logits, y)
        self.log("val/loss",      loss,               on_epoch=True, prog_bar=True)
        self.log("val/acc",       self.val_acc,       on_epoch=True, prog_bar=True)
        self.log("val/f1",        self.val_f1,        on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True)
        self.log("val/mcc",       self.val_mcc,       on_epoch=True)

    def test_step(self, batch, batch_idx):
        if self.mode == "pretrain":
            return  # no test in pretrain
        x, y   = batch
        logits = self(x)
        probs  = F.softmax(logits, dim=-1)
        self.test_acc(logits, y);    self.test_f1(logits, y)
        self.test_mcc(logits, y);    self.test_auroc(probs, y)
        self.test_cm(logits, y)
        self.log("test/acc",   self.test_acc,   on_epoch=True)
        self.log("test/f1",    self.test_f1,    on_epoch=True, prog_bar=True)
        self.log("test/mcc",   self.test_mcc,   on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_epoch=True)

    def on_test_epoch_end(self):
        if self.mode != "classify":
            return
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix (rows=true, cols=pred):\n{cm.cpu().numpy()}")
        self.test_cm.reset()

    # ── Optimiser ───────────────────────────────────────────────────────

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
            [{"params": decay,    "weight_decay": self.hparams.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.learning_rate, betas=(0.9, 0.95), eps=1e-8,
        )

        def lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- pre-train smoke ---
    ptm = HydroMAE(mode="pretrain").to(device).train()
    x   = torch.randn(2, 5_120, device=device)
    loss = ptm(x)
    print(f"HydroMAE pretrain  |  recon_loss={loss.item():.4f}")
    total = sum(p.numel() for p in ptm.parameters())
    print(f"  Parameters: {total:,}")
    print(f"  Patches   : {ptm.n_patches}  ({ptm.n_patches_h}×{ptm.n_patches_w}), "
          f"patch_dim={ptm.patch_dim}")

    # --- classify smoke ---
    clm = HydroMAE(mode="classify", num_classes=3).to(device).eval()
    with torch.no_grad():
        logits = clm(x)
    print(f"\nHydroMAE classify  |  output {logits.shape}")
