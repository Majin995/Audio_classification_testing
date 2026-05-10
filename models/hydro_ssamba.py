"""
HydroSSAMBA — Self-Supervised Audio Mamba for Vessel Classification.

Architecture
------------
  Raw waveform (32 kHz, 1 s)
      ↓
  Log-Mel Spectrogram  (128 mel bins, hop 320)
      ↓
  SpecAugment
      ↓
  2-D Patch Embedding  (patch_f=16, patch_t=8) → (B, n_patches, d_model)
  + learnable positional embedding
      ↓
  N × BidirMambaBlock  (forward Mamba + backward Mamba, outputs summed)
      ↓
  LayerNorm
      ↓
  Attentive Statistics Pool  → (B, 2·d_model)
      ↓
  Classifier  Linear(2D→D) → BN → ReLU → Dropout → Linear(D→C)
      ↓
  FocalLoss

Mamba vs S4
-----------
Unlike S4 where A, B, C are fixed (linear time-invariant), Mamba's S6 layer
makes B, C and the step size Δ *input-dependent* (selective):

    Δ = softplus( dt_proj( linear(x) ) )      # (B, L, d_inner)
    B = linear(x)[..., :N]                    # (B, L, N)
    C = linear(x)[..., N:2N]                  # (B, L, N)
    Ā[t] = exp(Δ[t] · A)                     # ZOH discretization
    B̄[t] = Δ[t] · B[t]                       # approximate ZOH for B
    h[t] = Ā[t] · h[t-1] + B̄[t] · x[t]     # state update
    y[t] = (C[t] · h[t]).sum(-1)              # readout

This selective gating allows the model to focus on relevant acoustic
events and filter out background noise — critical for vessel fingerprinting.

Bidirectional Processing
------------------------
SSAMBA processes patches in both temporal directions and sums the outputs,
giving each token access to its full left and right context without
quadratic attention complexity.

Pure-PyTorch Implementation
----------------------------
The reference SSAMBA uses custom CUDA kernels (mamba-ssm package) for the
parallel associative scan.  Here we use a sequential loop over patch positions
(L ≈ 100 for 1 s audio), which is functionally identical and fast enough for
our short sequences without requiring custom kernels.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchaudio
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from .hydro_conformer import SpecAugment, DropPath, FocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  Mamba S6 — Selective State Space (pure PyTorch, sequential scan)
# ═══════════════════════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """
    Mamba-style selective SSM block (no CUDA extensions required).

    Key differences from S4D in HydroS4:
      - B, C, Δ are *functions of the input*, not fixed parameters
      - This selectivity allows the block to filter irrelevant inputs
      - A is fixed diagonal (negative real) — same stability guarantee as S4D
      - Uses a causal depthwise Conv1d for local context before the SSM

    Architecture (single direction):
        x (B, L, d_model)
          → in_proj  → [x_branch, z]  (B, L, d_inner) each
          → causal Conv1d + SiLU
          → x_proj  → [Δ_raw, B_ssm, C_ssm]
          → dt_proj → Δ expanded to (B, L, d_inner)
          → selective_scan → y  (B, L, d_inner)
          → y * SiLU(z)
          → out_proj → (B, L, d_model)
    """

    def __init__(
        self,
        d_model:  int   = 128,
        d_state:  int   = 16,
        d_conv:   int   = 4,
        expand:   int   = 2,
        dropout:  float = 0.0,
    ):
        super().__init__()
        self.d_state  = d_state
        d_inner       = int(expand * d_model)
        self.d_inner  = d_inner

        # Input projection: x → (x_branch ‖ z)
        self.in_proj  = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise Conv1d — local context mixing
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        # Input-dependent SSM parameters: [Δ_raw (1), B (N), C (N)]
        self.x_proj   = nn.Linear(d_inner, 1 + 2 * d_state, bias=False)
        # Expand rank-1 Δ to all d_inner channels
        self.dt_proj  = nn.Linear(1, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)

        # Fixed diagonal A (initialized to -n, ensures Re(A) << 0)
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_init = A_init.unsqueeze(0).expand(d_inner, -1)  # (d_inner, N)
        self.log_A = nn.Parameter(torch.log(A_init))      # fixed during pre-training; fine-tunable

        # Skip connection scale
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()

    def _selective_scan(
        self,
        x:     torch.Tensor,   # (B, L, d_inner)
        delta: torch.Tensor,   # (B, L, d_inner)  softplus-activated step size
        B_ssm: torch.Tensor,   # (B, L, N)
        C_ssm: torch.Tensor,   # (B, L, N)
    ) -> torch.Tensor:         # (B, L, d_inner)
        B_size, L, D = x.shape
        N = self.d_state

        # Fixed A: negative real diagonal  (D, N)
        A = -torch.exp(self.log_A)   # (D, N)

        # Discretize A: Ā[t] = exp(Δ[t] · A)  → (B, L, D, N)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # Discretize B (simplified ZOH): B̄[t] · x[t] = Δ[t] · B[t] · x[t]
        # combined to avoid storing (B, L, D, N) intermediate
        dBx = delta.unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, D, N)

        # Sequential scan over L (fast enough for L ≈ 100 tokens)
        h   = torch.zeros(B_size, D, N, device=x.device, dtype=x.dtype)
        ys  = []
        for t in range(L):
            h  = dA[:, t] * h + dBx[:, t]                    # (B, D, N)
            y  = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)       # (B, D)
            ys.append(y)

        y_seq = torch.stack(ys, dim=1)                         # (B, L, D)
        return y_seq + self.D * x                              # skip connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)  [no residual, applied outside]"""
        B_size, L, _ = x.shape

        # Split projection
        xz      = self.in_proj(x)               # (B, L, 2·d_inner)
        x_br, z = xz.chunk(2, dim=-1)           # (B, L, d_inner) each

        # Causal conv (trim extra padding)
        x_conv  = self.conv1d(x_br.transpose(1, 2))[..., :L]
        x_conv  = F.silu(x_conv.transpose(1, 2))   # (B, L, d_inner)

        # Compute input-dependent SSM params
        xbc     = self.x_proj(x_conv)               # (B, L, 1 + 2N)
        delta_r = xbc[..., :1]                       # (B, L, 1)
        B_ssm   = xbc[..., 1 : 1 + self.d_state]    # (B, L, N)
        C_ssm   = xbc[..., 1 + self.d_state:]        # (B, L, N)
        delta   = F.softplus(self.dt_proj(delta_r))  # (B, L, d_inner)

        # Selective scan
        y = self._selective_scan(x_conv, delta, B_ssm, C_ssm)  # (B, L, d_inner)

        # Gate and project
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.drop(y)


# ═══════════════════════════════════════════════════════════════════════
#  Bidirectional Mamba Block
# ═══════════════════════════════════════════════════════════════════════

class BidirMambaBlock(nn.Module):
    """
    SSAMBA-style bidirectional Mamba block.

    Processes the patch sequence in both directions simultaneously using
    separate Mamba instances, then sums the outputs.  The pre-norm and
    residual wrap the bidirectional SSM pair.

        x
        ├─ LayerNorm ─► MambaBlock_fwd(z)                   → y_fwd
        └─ LayerNorm ─► MambaBlock_bwd(z.flip(1)).flip(1)   → y_bwd
                                                              + DropPath
                                                              + x  (residual)
    """

    def __init__(
        self,
        d_model:   int   = 128,
        d_state:   int   = 16,
        d_conv:    int   = 4,
        expand:    int   = 2,
        dropout:   float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm     = nn.LayerNorm(d_model)
        self.mamba_f  = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.mamba_b  = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.dp       = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z     = self.norm(x)
        y_fwd = self.mamba_f(z)
        y_bwd = self.mamba_b(z.flip(1)).flip(1)
        return x + self.dp(y_fwd + y_bwd)


# ═══════════════════════════════════════════════════════════════════════
#  2-D Patch Embedding
# ═══════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """
    Splits a 2-D spectrogram into fixed-size patches and projects each
    patch to a d_model-dimensional token.

    Input : (B, n_mels, T_frames)
    Output: (B, n_patches, d_model)
    """

    def __init__(
        self,
        n_mels:  int = 128,
        patch_f: int = 16,
        patch_t: int = 8,
        d_model: int = 128,
    ):
        super().__init__()
        self.patch_f = patch_f
        self.patch_t = patch_t

        # Number of patches along each axis (may require padding)
        self.n_patches_f = math.ceil(n_mels / patch_f)
        patch_dim        = patch_f * patch_t
        self.proj        = nn.Linear(patch_dim, d_model)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_mels, T)"""
        B, n_freq, T = x.shape
        pf, pt       = self.patch_f, self.patch_t

        # Pad T so it is divisible by patch_t
        T_pad = math.ceil(T / pt) * pt
        if T_pad > T:
            x = F.pad(x, (0, T_pad - T))

        # Pad freq so it is divisible by patch_f (n_mels=128 is exact for pf=16)
        F_pad = math.ceil(n_freq / pf) * pf
        if F_pad > n_freq:
            x = F.pad(x, (0, 0, 0, F_pad - n_freq))

        n_t = T_pad // pt
        n_f = F_pad // pf

        # Rearrange into patches: (B, n_f, pf, n_t, pt)
        x = x.reshape(B, n_f, pf, n_t, pt)
        # → (B, n_f * n_t, pf * pt)  by scanning frequency then time
        x = x.permute(0, 1, 3, 2, 4).reshape(B, n_f * n_t, pf * pt)

        return self.norm(self.proj(x))   # (B, n_patches, d_model)


# ═══════════════════════════════════════════════════════════════════════
#  Attentive Statistics Pooling (sequence-first layout)
# ═══════════════════════════════════════════════════════════════════════

class AttentiveStatisticsPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H)
        w    = F.softmax(self.attn(x), dim=1)           # (B, L, H)
        mean = (x * w).sum(dim=1)                        # (B, H)
        var  = ((x ** 2) * w).sum(dim=1) - mean ** 2
        std  = var.clamp(min=1e-9).sqrt()
        return torch.cat([mean, std], dim=-1)            # (B, 2H)


# ═══════════════════════════════════════════════════════════════════════
#  HydroSSAMBA LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroSSAMBA(pl.LightningModule):
    """
    SSAMBA-inspired bidirectional Mamba model for vessel acoustic classification.

    Uses a standard log-Mel spectrogram front-end (matching the SSAMBA paper),
    2-D patch tokenisation, and stacked bidirectional Mamba blocks.

    Args
    ----
    num_classes     : Number of vessel classes.
    class_weights   : Inverse-frequency weights for FocalLoss (None = uniform).
    sample_rate     : Audio sample rate in Hz.
    n_mels          : Log-Mel filterbank bins.
    hop_length      : STFT hop in samples.
    patch_f         : Patch height in frequency axis (must divide n_mels).
    patch_t         : Patch width in time axis.
    d_model         : Token/residual channel width.
    d_state         : SSM state dimension N per Mamba block.
    n_layers        : Number of bidirectional Mamba blocks.
    expand          : Mamba inner expansion ratio.
    d_conv          : Causal conv kernel size inside Mamba.
    dropout         : Dropout inside Mamba blocks and classifier.
    drop_path_rate  : Max stochastic-depth rate (linearly scheduled).
    learning_rate   : Peak AdamW LR.
    weight_decay    : AdamW weight decay.
    warmup_epochs   : Linear LR warmup epochs.
    max_epochs      : Total epochs (cosine schedule).
    mixup_alpha     : Waveform Mixup β (0 = off).
    noise_prob      : Additive Gaussian noise augmentation probability.
    gain_prob       : Random gain augmentation probability.
    focal_gamma     : Focal loss γ.
    label_smoothing : Label smoothing ε.
    """

    def __init__(
        self,
        num_classes:     int            = 3,
        class_weights:   Optional[list] = None,
        sample_rate:     int            = 32_000,
        n_mels:          int            = 128,
        hop_length:      int            = 320,
        patch_f:         int            = 16,
        patch_t:         int            = 8,
        d_model:         int            = 128,
        d_state:         int            = 16,
        n_layers:        int            = 6,
        expand:          int            = 2,
        d_conv:          int            = 4,
        dropout:         float          = 0.24,
        drop_path_rate:  float          = 0.10,
        learning_rate:   float          = 3e-4,
        weight_decay:    float          = 0.012,
        warmup_epochs:   int            = 10,
        max_epochs:      int            = 100,
        mixup_alpha:     float          = 0.20,
        noise_prob:      float          = 0.50,
        gain_prob:       float          = 0.70,
        focal_gamma:     float          = 2.0,
        label_smoothing: float          = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Log-Mel front-end ────────────────────────────────────────────
        n_fft       = hop_length * 4
        self.mel    = torchaudio.transforms.MelSpectrogram(
            sample_rate    = sample_rate,
            n_fft          = n_fft,
            hop_length     = hop_length,
            n_mels         = n_mels,
            f_min          = 50.0,
            f_max          = sample_rate / 2.0,
        )
        self.to_db  = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.spec_aug = SpecAugment(
            n_freq_masks=2, freq_mask_max=16,
            n_time_masks=2, time_mask_max=20,
        )

        # ── Patch embedding + positional ─────────────────────────────────
        self.patch_embed = PatchEmbedding(
            n_mels=n_mels, patch_f=patch_f, patch_t=patch_t, d_model=d_model,
        )
        # Compute max number of patches for a 1 s clip at given hop_length
        _T = math.ceil(sample_rate / hop_length)      # ≈ 100 frames
        _n = math.ceil(_T / patch_t) * self.patch_embed.n_patches_f + 32  # +buffer
        self.pos_embed = nn.Parameter(torch.zeros(1, _n, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Bidirectional Mamba blocks ────────────────────────────────────
        dp_rates = [
            drop_path_rate * i / max(n_layers - 1, 1) for i in range(n_layers)
        ]
        self.blocks = nn.ModuleList([
            BidirMambaBlock(
                d_model   = d_model,
                d_state   = d_state,
                d_conv    = d_conv,
                expand    = expand,
                dropout   = dropout,
                drop_path = dp_rates[i],
            )
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── Pooling + classifier ─────────────────────────────────────────
        self.pool       = AttentiveStatisticsPool(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # ── Loss + metrics ───────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights   = class_weights,
            gamma           = focal_gamma,
            label_smoothing = label_smoothing,
        )
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

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform : (B, T_samples) float32
        returns  : (B, num_classes) logits
        """
        # ── Spectrogram ──────────────────────────────────────────────────
        x = self.mel(waveform)        # (B, n_mels, T_frames)
        x = self.to_db(x)             # log scale
        if self.training:
            # SpecAugment expects (B, C, F, T) — add dummy channel
            x = x.unsqueeze(1)
            x = self.spec_aug(x)
            x = x.squeeze(1)

        # Normalize to zero-mean unit-variance (per sample)
        mu  = x.mean(dim=(1, 2), keepdim=True)
        sig = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        x   = (x - mu) / sig

        # ── Patch embedding ───────────────────────────────────────────────
        x = self.patch_embed(x)           # (B, n_patches, d_model)

        # Add positional embedding (truncate / extend as needed)
        L = x.size(1)
        x = x + self.pos_embed[:, :L, :]

        # ── Bidirectional Mamba blocks ────────────────────────────────────
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)            # (B, n_patches, d_model)

        # ── Pool + classify ───────────────────────────────────────────────
        x = self.pool(x)                  # (B, 2·d_model)
        return self.classifier(x)         # (B, num_classes)

    # ── Augmentation ─────────────────────────────────────────────────────

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1).item() < self.hparams.gain_prob:
            gain = 0.6 + 0.8 * torch.rand(x.shape[0], 1, device=x.device)
            x    = x * gain
        if torch.rand(1).item() < self.hparams.noise_prob:
            snr_db  = 20.0 + 20.0 * torch.rand(1, device=x.device)
            snr_lin = 10.0 ** (snr_db / 10.0)
            sig_pwr = x.pow(2).mean(-1, keepdim=True).clamp(min=1e-9)
            x       = x + torch.randn_like(x) * (sig_pwr / snr_lin).sqrt()
        return x

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
        return lam * self.criterion(logits, y) + (1.0 - lam) * self.criterion(logits, y_perm)

    # ── Lightning steps ──────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y = batch
        x    = self._augment(x)
        x, y, y_p, lam = self._mixup(x, y)
        logits = self(x)
        loss   = self._loss(logits, y, y_p, lam)
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
        self.log("val/precision", self.val_precision, on_epoch=True, prog_bar=True)
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

    # ── Optimiser ────────────────────────────────────────────────────────

    def configure_optimizers(self):
        # Separate A / dt parameters (SSM poles) with lower LR
        ssm_params, decay, no_decay = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "log_A" in name or "dt_proj" in name:
                ssm_params.append(p)
            elif p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        ssm_lr    = self.hparams.learning_rate * 0.1
        optimizer = torch.optim.AdamW(
            [
                {"params": decay,      "weight_decay": self.hparams.weight_decay,
                 "lr": self.hparams.learning_rate},
                {"params": no_decay,   "weight_decay": 0.0,
                 "lr": self.hparams.learning_rate},
                {"params": ssm_params, "weight_decay": 0.0,
                 "lr": ssm_lr},
            ],
            lr=self.hparams.learning_rate, betas=(0.9, 0.98), eps=1e-8,
        )

        def lr_lambda(epoch: int) -> float:
            wu       = self.hparams.warmup_epochs
            total    = self.hparams.max_epochs
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

    model  = HydroSSAMBA(num_classes=3).to(device).eval()
    total  = sum(p.numel() for p in model.parameters())
    ssm_p  = sum(p.numel() for n, p in model.named_parameters()
                 if "mamba_f" in n or "mamba_b" in n)
    pe_p   = sum(p.numel() for n, p in model.named_parameters()
                 if "patch_embed" in n or "pos_embed" in n)

    x_dummy = torch.randn(2, 32_000, device=device)
    with torch.no_grad():
        logits = model(x_dummy)

    hp = model.hparams
    print(f"\nHydroSSAMBA  |  {total:,} total params")
    print(f"  Mamba blocks : {ssm_p:,}  ({hp.n_layers} × BidirMambaBlock, d_state={hp.d_state})")
    print(f"  Patch embed  : {pe_p:,}  (patch {hp.patch_f}×{hp.patch_t}, d_model={hp.d_model})")
    print(f"  Other        : {total - ssm_p - pe_p:,}  (mel + pool + classifier)")
    print(f"\nOutput shape : {list(logits.shape)}")
