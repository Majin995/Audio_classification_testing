"""
HydroEAT — Efficient Audio Transformer for LOFAR Spectrogram Classification
=============================================================================

Efficiency design choices vs a plain ViT
-----------------------------------------
1. Convolutional patch embedding — captures local temporal/spectral context
   before the global attention layers; reduces the number of tokens via strided
   convolutions rather than non-overlapping patches.

2. Axial attention (alternating frequency-axis and time-axis MHSA) — attends
   independently along each axis, reducing complexity from O((H·W)²) to
   O(H·W·(H+W)) per block pair. This is critical for the high-frequency-
   resolution LOFAR spectrogram (256 freq × 32 time → 8 192 token-pairs if
   treated as 2-D flat; axial cuts that to ~8k per axis separately).

3. Depthwise-separable FFN — replaces the standard MLP sublayer with a
   depthwise conv (models local cross-channel dependencies along the sequence)
   followed by pointwise projections, reducing FLOPs by ~d_model×.

4. Rotary positional embeddings (RoPE) applied per-axis — no extra learned
   parameters; relative-position extrapolates naturally to new sequence lengths.

Architecture
------------
  Raw waveform (5 120 Hz, 1 s)
      ↓
  LofarFrontend  (shared with HydroLofarResNet)
      n_fft=4096, hop=160  →  (B, 1, 32, 256)
      ↓
  LofarSpecAugment  (train only)
      ↓
  ConvPatchEmbed
      4 depth-wise + pointwise conv blocks, stride (2,2) each
      (B, 1, 32, 256) → (B, d_model, 8, 16)  →  (B, 128, d_model)
      Splits into H=8 freq tokens × W=16 time tokens
      ↓
  EATBlock × n_blocks
      each block:  AxialAttn(freq) → AxialAttn(time) → DWSepFFN
      with pre-LayerNorm and stochastic depth residuals
      ↓
  CLS token  (prepended after embedding, participates in all attn)
  Mean-pool CLS + mean-pool all tokens  → concat  → (B, 2·d_model)
      ↓
  Classifier  Linear(2·d_model, num_classes)
      ↓
  FocalLoss
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
#  Rotary Positional Embeddings (RoPE)
# ═══════════════════════════════════════════════════════════════════════

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by 90° in complex space."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """
    Apply rotary embeddings to queries and keys.

    Args:
        q, k : (B, heads, L, head_dim)
        cos, sin : (1, 1, L, head_dim)
    """
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


def build_rope_cache(seq_len: int, head_dim: int,
                     device: torch.device) -> tuple:
    """Pre-compute cos/sin tables for a given sequence length."""
    theta = 1.0 / (10_000 ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim
    ))
    pos   = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)          # (L, head_dim//2)
    emb   = torch.cat([freqs, freqs], dim=-1)  # (L, head_dim)
    cos   = emb.cos()[None, None]            # (1, 1, L, head_dim)
    sin   = emb.sin()[None, None]
    return cos, sin


# ═══════════════════════════════════════════════════════════════════════
#  Convolutional Patch Embedding
# ═══════════════════════════════════════════════════════════════════════

class ConvPatchEmbed(nn.Module):
    """
    Progressive down-sampling via strided depthwise-separable convolutions.

    (B, 1, H, W)  →  (B, d_model, H//stride^n, W//stride^n)  →  (B, N, d_model)

    Using 2 stages of (3×3 DW conv, stride=2, BN, GELU, PW conv) halves the
    spatial dimensions twice, reducing the token count 4× while building
    local frequency-time features before the global attention.
    """

    def __init__(self, in_channels: int = 1, d_model: int = 192, n_stages: int = 2):
        super().__init__()
        channels = [in_channels] + [d_model // (2 ** (n_stages - 1 - i))
                                     for i in range(n_stages)]
        layers = []
        for i in range(n_stages):
            c_in, c_out = channels[i], channels[i + 1]
            layers += [
                # Depthwise spatial conv
                nn.Conv2d(c_in,  c_in,  3, stride=2, padding=1,
                          groups=c_in, bias=False),
                nn.BatchNorm2d(c_in),
                nn.GELU(),
                # Pointwise channel expansion
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns:
            tokens : (B, N, d_model)   flattened token sequence
            H, W   : spatial dims after down-sampling (for axial attention)
        """
        x = self.net(x)                 # (B, d_model, H', W')
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H'*W', d_model)
        return tokens, H, W


# ═══════════════════════════════════════════════════════════════════════
#  Axial Attention
# ═══════════════════════════════════════════════════════════════════════

class AxialAttention(nn.Module):
    """
    Multi-head self-attention along a single spatial axis.

    Given tokens arranged as (H, W), this layer attends along either the
    frequency (H) axis or the time (W) axis independently.  Tokens on the
    other axis are treated as batch dimensions, limiting context but keeping
    the quadratic cost to O(N × axis_len²) instead of O(N²).

    RoPE is applied along the attended axis.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(d_model)
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, H: int, W: int, axis: str) -> torch.Tensor:
        """
        Args:
            x    : (B, N, D)   N = H*W tokens (excluding CLS if prepended)
            H, W : spatial grid dimensions
            axis : "freq" (attend along H, treat W as batch)
                   "time" (attend along W, treat H as batch)
        Returns:
            x    : (B, N, D)  same shape
        """
        B, N, D = x.shape

        # Separate CLS token if present (N > H*W)
        has_cls = (N == H * W + 1)
        if has_cls:
            cls, x_grid = x[:, :1, :], x[:, 1:, :]   # (B,1,D), (B, H*W, D)
        else:
            x_grid = x

        # Reshape to spatial grid
        grid = x_grid.reshape(B, H, W, D)             # (B, H, W, D)

        if axis == "freq":
            # Attend along H for each W column: merge (B, W) as batch
            seq = grid.permute(0, 2, 1, 3).reshape(B * W, H, D)  # (B*W, H, D)
            seq_len = H
        else:
            # Attend along W for each H row: merge (B, H) as batch
            seq = grid.reshape(B * H, W, D)                       # (B*H, W, D)
            seq_len = W

        # Pre-norm + QKV
        seq_n = self.norm(seq)
        qkv   = self.qkv(seq_n).reshape(-1, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                  # each: (Bb, L, heads, hd)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

        # RoPE
        cos, sin = build_rope_cache(seq_len, self.head_dim, seq.device)
        q, k = apply_rope(q, k, cos, sin)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out  = torch.matmul(attn, v)                 # (Bb, heads, L, hd)
        out  = out.transpose(1, 2).reshape(-1, seq_len, D)  # (Bb, L, D)
        out  = self.proj(out)

        # Residual
        seq = seq + self.drop(out)

        # Restore spatial shape → flatten back to token sequence
        if axis == "freq":
            grid_out = seq.reshape(B, W, H, D).permute(0, 2, 1, 3)
        else:
            grid_out = seq.reshape(B, H, W, D)

        x_grid_out = grid_out.reshape(B, H * W, D)

        if has_cls:
            return torch.cat([cls, x_grid_out], dim=1)
        return x_grid_out


# ═══════════════════════════════════════════════════════════════════════
#  Depthwise-Separable FFN
# ═══════════════════════════════════════════════════════════════════════

class DWSepFFN(nn.Module):
    """
    Feed-forward sublayer with a depthwise conv in the middle.

    Structure:
        LN  →  Linear(D→4D)  →  DW-Conv1d(4D, k=3)  →  GELU  →  Linear(4D→D)  →  Dropout

    The 1-D depthwise convolution over the token sequence gives the model a
    local inductive bias at a fraction of the cost of a global attention layer.
    """

    def __init__(self, d_model: int, expansion: int = 4,
                 kernel: int = 3, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * expansion
        self.norm  = nn.LayerNorm(d_model)
        self.fc1   = nn.Linear(d_model, hidden, bias=False)
        self.dw    = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2,
                               groups=hidden, bias=False)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(hidden, d_model, bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        h = self.fc1(self.norm(x))          # (B, N, 4D)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)   # local conv
        h = self.act(h)
        h = self.drop(self.fc2(h))
        return x + h


# ═══════════════════════════════════════════════════════════════════════
#  EAT Block  (freq-axis attn + time-axis attn + DWSep FFN)
# ═══════════════════════════════════════════════════════════════════════

class EATBlock(nn.Module):
    """
    Single Efficient Audio Transformer block.

    Order:  AxialAttn(freq) → DropPath
          → AxialAttn(time) → DropPath
          → DWSepFFN        → DropPath
    """

    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        dropout:    float = 0.0,
        drop_path:  float = 0.0,
        ffn_expand: int   = 4,
    ):
        super().__init__()
        self.freq_attn = AxialAttention(d_model, n_heads, dropout)
        self.time_attn = AxialAttention(d_model, n_heads, dropout)
        self.ffn       = DWSepFFN(d_model, ffn_expand, dropout=dropout)
        self.dp_f      = DropPath(drop_path)
        self.dp_t      = DropPath(drop_path)
        self.dp_ff     = DropPath(drop_path)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.dp_f(self.freq_attn(x, H, W, axis="freq") - x)
        x = x + self.dp_t(self.time_attn(x, H, W, axis="time") - x)
        x = self.ffn(x)   # DWSepFFN already has residual internally
        return x


# ═══════════════════════════════════════════════════════════════════════
#  HydroEAT  LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroEAT(pl.LightningModule):
    """
    Efficient Audio Transformer for LOFAR underwater vessel classification.

    Args:
        num_classes     : Number of vessel classes.
        class_weights   : Inverse-frequency weights for focal loss.
        sample_rate     : Audio sample rate in Hz.
        n_fft           : STFT window size for LOFAR frontend.
        hop_length      : STFT hop in samples.
        time_bins       : LOFAR image height (time axis).
        freq_bins       : LOFAR image width  (frequency axis).
        d_model         : Hidden dimension throughout the transformer.
        n_blocks        : Number of EAT blocks.
        n_heads         : Attention heads (d_model must be divisible).
        dropout         : Attention and FFN dropout.
        drop_path_rate  : Max stochastic-depth rate (linearly scheduled).
        ffn_expand      : Feed-forward expansion ratio.
        learning_rate   : Peak AdamW LR.
        weight_decay    : AdamW weight decay.
        warmup_epochs   : Linear LR warmup length.
        max_epochs      : Total epochs for cosine schedule.
        mixup_alpha     : Waveform Mixup β distribution α (0 = off).
        focal_gamma     : Focal loss γ.
        label_smoothing : Label smoothing ε.
    """

    def __init__(
        self,
        num_classes:     int            = 3,
        class_weights:   Optional[list] = None,
        sample_rate:     int            = 5_120,
        n_fft:           int            = 4_096,
        hop_length:      int            = 160,
        time_bins:       int            = 32,
        freq_bins:       int            = 256,
        d_model:         int            = 192,
        n_blocks:        int            = 8,
        n_heads:         int            = 6,
        dropout:         float          = 0.1,
        drop_path_rate:  float          = 0.15,
        ffn_expand:      int            = 4,
        learning_rate:   float          = 5e-4,
        weight_decay:    float          = 1e-2,
        warmup_epochs:   int            = 10,
        max_epochs:      int            = 100,
        mixup_alpha:     float          = 0.3,
        focal_gamma:     float          = 2.0,
        label_smoothing: float          = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Frontend ────────────────────────────────────────────────────
        self.frontend = LofarFrontend(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            time_bins=time_bins, freq_bins=freq_bins,
        )
        self.spec_aug = LofarSpecAugment(
            n_time_masks=2, time_mask_max=8,
            n_freq_masks=2, freq_mask_max=32,
        )

        # ── Patch embedding (2 stages of /2 downsampling) ───────────────
        # (B,1,32,256) → (B,d_model,8,64) → (B,512,d_model) tokens
        self.patch_embed = ConvPatchEmbed(in_channels=1, d_model=d_model, n_stages=2)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Transformer blocks ──────────────────────────────────────────
        dp_rates = [drop_path_rate * i / max(n_blocks - 1, 1)
                    for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            EATBlock(
                d_model=d_model, n_heads=n_heads,
                dropout=dropout, drop_path=dp_rates[i],
                ffn_expand=ffn_expand,
            )
            for i in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ── Classifier ──────────────────────────────────────────────────
        # CLS token + mean of all tokens
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # ── Loss ────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

        # ── Metrics ─────────────────────────────────────────────────────
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

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) float32 at sample_rate Hz
        Returns:
            logits: (B, num_classes)
        """
        x = self.frontend(waveform)           # (B, 1, 32, 256)
        x = self.spec_aug(x)                  # (B, 1, 32, 256)

        tokens, H, W = self.patch_embed(x)    # (B, N, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, d_model)

        for block in self.blocks:
            tokens = block(tokens, H, W)      # (B, N+1, d_model)

        tokens = self.norm(tokens)

        # Readout: CLS + mean of spatial tokens
        cls_out  = tokens[:, 0]               # (B, d_model)
        mean_out = tokens[:, 1:].mean(dim=1)  # (B, d_model)
        feat     = torch.cat([cls_out, mean_out], dim=-1)  # (B, 2*d_model)

        return self.classifier(feat)          # (B, num_classes)

    # ── Mixup ───────────────────────────────────────────────────────────

    def _mixup(self, x, y):
        alpha = self.hparams.mixup_alpha
        if not self.training or alpha <= 0.0:
            return x, y, y, 1.0
        lam  = torch.distributions.Beta(alpha, alpha).sample().to(x)
        perm = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam

    def _loss(self, logits, y, y_perm=None, lam=1.0):
        if y_perm is None or lam == 1.0:
            return self.criterion(logits, y)
        return (lam * self.criterion(logits, y)
                + (1.0 - lam) * self.criterion(logits, y_perm))

    # ── Lightning steps ─────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y           = batch
        x, y, y_p, lam = self._mixup(x, y)
        logits         = self(x)
        loss           = self._loss(logits, y, y_p, lam)
        self.train_acc(logits, y)
        self.log("train/loss", loss,           on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
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
            lr=self.hparams.learning_rate, betas=(0.9, 0.98), eps=1e-8,
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
    model  = HydroEAT(num_classes=3).to(device).eval()
    total  = sum(p.numel() for p in model.parameters())
    print(f"HydroEAT  |  {total:,} parameters")
    print(f"  d_model={model.hparams.d_model}, "
          f"n_blocks={model.hparams.n_blocks}, "
          f"n_heads={model.hparams.n_heads}")

    x = torch.randn(2, 5_120, device=device)
    with torch.no_grad():
        tokens, H, W = model.patch_embed(model.spec_aug(model.frontend(x)))
        logits       = model(x)
    print(f"Token grid  : {H} × {W} = {H*W} tokens (+ 1 CLS)")
    print(f"Output shape: {logits.shape}")
