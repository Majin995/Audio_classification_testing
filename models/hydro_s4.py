"""
HydroS4 — Continuous-Time State Space Model (S4 / SaShiMi) for Vessel Classification.

Theory
------
Structured State Space Sequence Models (S4) model sequences via a continuous-time
linear system:

    x'(t) = A x(t) + B u(t)        state update
    y(t)  = C x(t) + D u(t)        readout

where A ∈ ℝ^(N×N), B ∈ ℝ^(N×1), C ∈ ℝ^(1×N), D ∈ ℝ.

HiPPO Framework (High-order Polynomial Projection Operators)
------------------------------------------------------------
HiPPO gives principled initializations for A that make the state x(t) a
compressed representation of the function history u(·).  The LegS variant
(Legendre Scaled) projects onto Legendre polynomials on a sliding window:

    A[n,k] = -(2n+1)^0.5 * (2k+1)^0.5    if n > k
    A[n,n] = -(n+1)
    B[n]   = (2n+1)^0.5

This structure enables the model to remember uniform-weight history and gives
superior initialization for audio tasks.

S4D Diagonal Approximation (Gu et al. 2022, "On the Parameterization...")
--------------------------------------------------------------------------
The full HiPPO-LegS matrix can be diagonalized as A ≈ V Λ V^{-1}, where Λ
are complex eigenvalues.  S4D parameterises A directly as diagonal complex:

    A_n = -(1/2 + i π n)    [S4D-LegS: poles uniformly spaced on imaginary axis]

with Re(A_n) < 0 ensuring stability.  The diagonal structure enables O(N)
per-step complexity and O(L log L) kernel computation via FFT.

Discretization (ZOH)
--------------------
Given step size Δ (learnable, one per channel):

    Ā_n = exp(Δ · A_n)                 (diagonal complex matrix)
    B̄_n = (Ā_n - 1) / A_n · B_n

The SSM reduces to a length-L convolution kernel:

    K[t] = Re( Σ_n  C_n · Ā_n^t · B̄_n )

which is applied as K * u via FFT.

SaShiMi Architecture (Goel et al. 2022, "It's Raw!")
----------------------------------------------------
SaShiMi is a hierarchical audio model built from S4 blocks:

    S4Block:
        Pre-Norm (LayerNorm)
        Linear up-projection  d → 2d
        S4D layer  (applied channel-wise)
        GELU-gated activation (GLU splits 2d → 2×d, gates with sigmoid)
        Linear down-projection  d → d
        Residual add
        +
        Pre-Norm FFN  (d → 4d → GELU → 4d → d)  with DropPath

Architecture here (classification adaptation)
---------------------------------------------
  Raw waveform (32 kHz, 1 s)
      ↓
  MultiScalePCEN       — 2-channel dual-resolution mel + trainable PCEN
      ↓                  (B, 2, n_mels, T_frames)
  SpecAugment
      ↓
  Flatten + Project    — (B, 2·n_mels, T) → permute → (B, T, 2·n_mels)
      ↓                  → Linear(2·n_mels → d_model) → (B, T, d_model)
  S4Block × n_layers   — each: pre-norm S4D with GLU gate + FFN + DropPath
      ↓
  Attentive Stats Pool — learned-weight mean + std over T → (B, 2·d_model)
      ↓
  Classifier           — Linear(2D→D) → BN → ReLU → Dropout → Linear(D→classes)
      ↓
  FocalLoss
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from .hydro_conformer import MultiScalePCEN, SpecAugment, DropPath, FocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  HiPPO utilities
# ═══════════════════════════════════════════════════════════════════════

def hippo_legs_matrix(N: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the HiPPO-LegS (Legendre Scaled) matrices A ∈ ℝ^(N×N), B ∈ ℝ^N.

    These matrices project onto Legendre polynomial coefficients with a
    uniform sliding-window measure.  Using them as initializations for the
    SSM state gives the model principled long-range memory.

    Returns (A, B) as float32 tensors.
    """
    n = torch.arange(N, dtype=torch.float64)
    k = torch.arange(N, dtype=torch.float64)

    # A[n,k] = -(2n+1)^0.5 * (2k+1)^0.5  for n > k
    #         = -(n+1)                      for n == k
    #         = 0                           for n < k
    A = torch.zeros(N, N, dtype=torch.float64)
    nn_grid, kk_grid = torch.meshgrid(n, k, indexing="ij")
    A[nn_grid > kk_grid] = -(
        (2 * nn_grid + 1).sqrt() * (2 * kk_grid + 1).sqrt()
    )[nn_grid > kk_grid]
    A.diagonal().copy_(-(n + 1))

    B = (2 * n + 1).sqrt()
    return A.float(), B.float()


def hippo_diag_init(N: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute S4D-LegS diagonal initialization for A and B.

    Rather than diagonalizing HiPPO-LegS exactly (expensive), we use the
    closed-form S4D-LegS approximation from:
        Gu et al. "On the Parameterization and Initialization of
        Diagonal State Space Models", NeurIPS 2022.

    A eigenvalues: A_n = -(1/2 + i π n)  for n = 0, ..., N-1
    B eigenvalues: B_n = i^n / sqrt(N)   (approximately uniform modulus)

    Returns:
        A_real : (N,) float  — real part of eigenvalues  (< 0, ensures stability)
        A_imag : (N,) float  — imaginary part
        B_init : (N, 2) float  — complex B as (real, imag)
    """
    n = torch.arange(N, dtype=torch.float32)

    # Poles uniformly spaced on imaginary axis, all with Re(A) = -0.5
    A_real = -0.5 * torch.ones(N)
    A_imag = math.pi * n

    # B: alternating real/imag columns matching i^n pattern
    angle = math.pi / 2 * n          # 0, π/2, π, 3π/2, ...
    B_real = torch.cos(angle) / math.sqrt(N)
    B_imag = torch.sin(angle) / math.sqrt(N)
    B_init = torch.stack([B_real, B_imag], dim=-1)   # (N, 2)

    return A_real, A_imag, B_init


# ═══════════════════════════════════════════════════════════════════════
#  S4D Layer (Diagonal State Space)
# ═══════════════════════════════════════════════════════════════════════

class S4DLayer(nn.Module):
    """
    Diagonal S4 layer — the core building block of HydroS4.

    Each of the `d_model` channels has an independent diagonal SSM with
    `d_state` complex poles, initialized via the HiPPO-LegS diagonalization.

    The output y = K * u + D·u is computed as an FFT convolution where

        K[t] = Re( Σ_n  C_n · Ā_n^t · B̄_n )

    is the discretized SSM convolution kernel of length L.

    Parameters
    ----------
    d_model : int
        Number of parallel channels (= model width).
    d_state : int
        State space dimension N (number of poles per channel).
    dt_min / dt_max : float
        Range for uniform log-initialization of per-channel step sizes Δ.
    """

    def __init__(
        self,
        d_model:  int   = 128,
        d_state:  int   = 64,
        dt_min:   float = 1e-3,
        dt_max:   float = 1e-1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # ── HiPPO-LegS diagonal initialization ──────────────────────────
        A_real_init, A_imag_init, B_init = hippo_diag_init(d_state)

        # A_real: parameterized as softplus so that Re(A) = -(softplus(p) + 0.5) <= -0.5.
        # The +0.5 offset hard-floors the damping, keeping |Ā| <= exp(-0.5*dt_min).
        # Init: softplus(p) ≈ 0 when p << 0, so init p = -5 → Re(A) ≈ -0.507.
        self.log_A_real = nn.Parameter(
            torch.full((d_model, d_state), -5.0)
        )                                              # (d_model, d_state)
        self.A_imag = nn.Parameter(
            A_imag_init.unsqueeze(0).expand(d_model, -1).clone()
        )                                              # (d_model, d_state)

        # B: complex (d_model, d_state, 2)
        self.B = nn.Parameter(
            B_init.unsqueeze(0).expand(d_model, -1, -1).clone()
        )                                              # (d_model, d_state, 2)

        # C: complex, random init  (d_model, d_state, 2)
        C = torch.randn(d_model, d_state, 2) / math.sqrt(d_state)
        self.C = nn.Parameter(C)

        # D: skip connection (one per channel)
        self.D = nn.Parameter(torch.ones(d_model))

        # Δ: per-channel log step size, uniform in log-space
        log_dt = torch.rand(d_model) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)             # (d_model,)

    # ── Kernel computation ───────────────────────────────────────────────

    def _kernel(self, L: int) -> torch.Tensor:
        """
        Build the length-L SSM convolution kernel K ∈ ℝ^(L, d_model).

        Steps
        -----
        1. Reconstruct A:  Re(A) = -(softplus(p) + 0.5) ≤ -0.5  (always stable)
        2. ZOH discretize: Ā = exp(Δ·A),  B̄ = (Ā-1)/A · B
        3. Compute K[t] = Re(Σ_n  C_n · Ā_n^t · B̄_n) via cumprod
           — avoids torch.log and its branch-cut discontinuities entirely.
           cumprod([1, Ā, Ā, ..., Ā], dim=0) gives [Ā^0, Ā^1, ..., Ā^{L-1}].
        """
        device = self.log_A_real.device

        # ── Reconstruct A with hard stability floor ──────────────────────
        # softplus ensures the learned offset > 0; +0.5 guarantees Re(A) <= -0.5
        A_real = -(F.softplus(self.log_A_real) + 0.5)    # (H, N) always <= -0.5
        A      = torch.complex(A_real, self.A_imag)       # (H, N) complex

        # ── Discretize ──────────────────────────────────────────────────
        dt     = torch.exp(self.log_dt).unsqueeze(-1)     # (H, 1) real
        A_bar  = torch.exp(dt * A)                        # (H, N) complex
        B_cplx = torch.view_as_complex(self.B)            # (H, N) complex
        B_bar  = (A_bar - 1.0) / A * B_cplx              # (H, N) ZOH B̄

        # ── Kernel via cumprod (no log, no branch cuts) ──────────────────
        # Build: [1, Ā, Ā, Ā, ...] shape (L, H, N), then cumprod → [Ā^0, Ā^1, ...]
        C_cplx = torch.view_as_complex(self.C)            # (H, N) complex
        CB     = C_cplx * B_bar                           # (H, N) combined coeff

        # First element is Ā^0 = 1; remaining L-1 are Ā (to be cumproded)
        base      = A_bar.unsqueeze(0).expand(L, -1, -1).clone()  # (L, H, N)
        base[0]   = 1.0                                   # t=0: Ā^0 = 1
        A_pow     = torch.cumprod(base, dim=0)            # (L, H, N)

        K = (A_pow * CB.unsqueeze(0)).sum(dim=-1).real    # (L, H) real
        return K

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, L, H)  input sequence
        Returns:
            y : (B, L, H)  output sequence

        Note: complex tensor ops (FFT, exp) require float32; the S4D kernel
        is computed in float32 and cast back to the input dtype on return.
        """
        in_dtype = x.dtype
        x32      = x.float()                              # ensure float32 for complex ops

        B_size, L, H = x32.shape
        K = self._kernel(L)                               # (L, H) float32

        # FFT convolution: y = K ∗ u  (linear convolution, padded to 2L)
        x_t = x32.transpose(1, 2)                         # (B, H, L)
        K_t = K.transpose(0, 1)                           # (H, L)

        xf  = torch.fft.rfft(x_t, n=2 * L, dim=-1)       # (B, H, L+1) complex64
        Kf  = torch.fft.rfft(K_t, n=2 * L, dim=-1)       # (H,  L+1) complex64

        yf  = xf * Kf.unsqueeze(0)                        # (B, H, L+1)
        y   = torch.fft.irfft(yf, n=2 * L, dim=-1)[..., :L]  # (B, H, L)

        y = y.transpose(1, 2)                             # (B, L, H)

        # Skip / feedthrough; cast back to original dtype for AMP compatibility
        return (y + self.D * x32).to(in_dtype)


# ═══════════════════════════════════════════════════════════════════════
#  SaShiMi Block
# ═══════════════════════════════════════════════════════════════════════

class SaShiMiBlock(nn.Module):
    """
    SaShiMi-style residual block (adapted for classification).

    From Goel et al. "It's Raw! Audio Generation with State-Space Models"
    (ICML 2022).  The block applies S4D with a GLU gate, followed by a
    position-wise FFN, both with LayerNorm + residual connection.

        x  ──► LayerNorm ──► Linear(d→2d) ──► S4D ──► GLU ──► Linear(d→d) ──► + ──► ...
        │                                                                        │
        └────────────────────────────────────────────────────────────────────────┘

    The GLU gate: split (B, L, 2d) into two halves; output = val * sigmoid(gate).
    This allows the S4D output to selectively suppress uninformative features.
    After the S4+gate branch, a standard pre-norm FFN (SiLU, 4× expansion) refines.

    Args:
        d_model    : Residual channel width.
        d_state    : SSM state dimension (poles per channel).
        expansion  : FFN expansion ratio.
        dropout    : Dropout rate in FFN.
        drop_path  : Stochastic depth drop probability.
    """

    def __init__(
        self,
        d_model:   int   = 128,
        d_state:   int   = 64,
        expansion: int   = 4,
        dropout:   float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dp = DropPath(drop_path)

        # ── S4 branch ───────────────────────────────────────────────────
        self.norm_s4   = nn.LayerNorm(d_model)
        self.up_proj   = nn.Linear(d_model, 2 * d_model)   # expand before S4
        self.s4        = S4DLayer(2 * d_model, d_state)    # operate on wider dim
        self.down_proj = nn.Linear(d_model, d_model)       # after GLU halves width

        # ── FFN branch ──────────────────────────────────────────────────
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── S4 + GLU branch ─────────────────────────────────────────────
        res = x
        z   = self.up_proj(self.norm_s4(x))   # (B, L, 2d)
        z   = self.s4(z)                       # (B, L, 2d)
        val, gate = z.chunk(2, dim=-1)         # (B, L, d) each
        z   = val * torch.sigmoid(gate)        # GLU  (B, L, d)
        z   = self.down_proj(z)
        x   = res + self.dp(z)

        # ── FFN branch ──────────────────────────────────────────────────
        x   = x + self.dp(self.ff(self.norm_ff(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Pooling
# ═══════════════════════════════════════════════════════════════════════

class AttentiveStatisticsPool1D(nn.Module):
    """
    Learned-weight mean + std pooling over the time axis for (B, T, H) tensors.
    Outputs (B, 2H).  Mirrors AttentiveStatisticsPool from hydro_resnet.py but
    operates on sequence-last layout.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        w    = F.softmax(self.attn(x), dim=1)        # (B, T, H)
        mean = (x * w).sum(dim=1)                    # (B, H)
        var  = (x ** 2 * w).sum(dim=1) - mean ** 2
        std  = var.clamp(min=1e-9).sqrt()
        return torch.cat([mean, std], dim=-1)        # (B, 2H)


# ═══════════════════════════════════════════════════════════════════════
#  HydroS4 LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroS4(pl.LightningModule):
    """
    S4 + SaShiMi for vessel acoustic classification.

    Uses a MultiScalePCEN front-end (identical to HydroConformer) to convert
    the raw waveform to a 2-channel PCEN spectrogram, then processes the
    time-frequency sequence with stacked SaShiMi blocks containing S4D layers
    initialized via the HiPPO-LegS framework.

    Args
    ----
    num_classes     : Number of vessel classes.
    class_weights   : Inverse-frequency weights for focal loss (None = uniform).
    sample_rate     : Audio sample rate (Hz).
    n_mels          : Mel filterbank bins for PCEN front-end.
    hop_length      : STFT hop in samples.
    d_model         : Model (residual) width throughout S4 blocks.
    d_state         : SSM state dimension N (poles per channel in S4D).
    n_layers        : Number of SaShiMi blocks.
    expansion       : FFN expansion ratio inside each SaShiMi block.
    dropout         : Dropout rate.
    drop_path_rate  : Max stochastic-depth rate (linearly scheduled).
    dt_min / dt_max : Log-uniform init range for SSM step sizes Δ.
    learning_rate   : Peak AdamW LR.
    weight_decay    : AdamW weight decay.
    warmup_epochs   : Linear LR warmup epochs.
    max_epochs      : Total epochs for cosine schedule.
    mixup_alpha     : Waveform Mixup β  (0 = off).
    noise_prob      : Additive noise augmentation probability.
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
        d_model:         int            = 128,
        d_state:         int            = 64,
        n_layers:        int            = 6,
        expansion:       int            = 4,
        dropout:         float          = 0.10,
        drop_path_rate:  float          = 0.05,
        dt_min:          float          = 1e-3,
        dt_max:          float          = 1e-1,
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

        # ── PCEN front-end (reused from HydroConformer) ─────────────────
        self.features = MultiScalePCEN(
            sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length,
        )
        self.spec_aug = SpecAugment(
            n_freq_masks=2, freq_mask_max=16,
            n_time_masks=2, time_mask_max=20,
        )

        # ── Input projection: (B, T, 2·n_mels) → (B, T, d_model) ────────
        self.input_proj = nn.Sequential(
            nn.Linear(2 * n_mels, d_model),
            nn.LayerNorm(d_model),
        )

        # ── SaShiMi blocks with linearly increasing DropPath ─────────────
        dp_rates = [
            drop_path_rate * i / max(n_layers - 1, 1)
            for i in range(n_layers)
        ]
        self.blocks = nn.ModuleList([
            SaShiMiBlock(
                d_model   = d_model,
                d_state   = d_state,
                expansion = expansion,
                dropout   = dropout,
                drop_path = dp_rates[i],
            )
            for i in range(n_layers)
        ])

        # ── Pooling + classifier ─────────────────────────────────────────
        self.pool       = AttentiveStatisticsPool1D(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # ── Loss ─────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights   = class_weights,
            gamma           = focal_gamma,
            label_smoothing = label_smoothing,
        )

        # ── Metrics ──────────────────────────────────────────────────────
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
        Args:
            waveform : (B, T) float32 at sample_rate Hz
        Returns:
            logits   : (B, num_classes)
        """
        # ── Front-end ────────────────────────────────────────────────────
        x = self.features(waveform)           # (B, 2, n_mels, T_frames)
        x = self.spec_aug(x)                  # (B, 2, n_mels, T_frames)

        B, C_in, F, T = x.shape
        # Flatten frequency channels, then transpose to sequence format
        x = x.reshape(B, C_in * F, T)        # (B, 2·n_mels, T_frames)
        x = x.transpose(1, 2)                 # (B, T_frames, 2·n_mels)

        # ── Input projection ─────────────────────────────────────────────
        x = self.input_proj(x)                # (B, T, d_model)

        # ── S4 / SaShiMi blocks ─────────────────────────────────────────
        for block in self.blocks:
            x = block(x)                      # (B, T, d_model)

        # ── Pooling + classification ──────────────────────────────────────
        x = self.pool(x)                      # (B, 2·d_model)
        return self.classifier(x)             # (B, num_classes)

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
            sig_pwr = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-9)
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
        return (lam           * self.criterion(logits, y)
                + (1.0 - lam) * self.criterion(logits, y_perm))

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
        # Separate weight decay groups: no decay on 1D params, biases, or SSM params
        ssm_params, decay, no_decay = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # SSM parameters (A, B, C, D, log_dt) get low/no weight decay
            if any(k in name for k in ("log_A_real", "A_imag", ".B", ".C", ".D", "log_dt")):
                ssm_params.append(p)
            elif p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        # SSM poles are sensitive — give them 10× lower LR to avoid large jumps
        # that flip the imaginary part across the branch cut or blow up cumprod.
        ssm_lr = self.hparams.learning_rate * 0.1

        optimizer = torch.optim.AdamW(
            [
                {"params": decay,      "weight_decay": self.hparams.weight_decay,
                 "lr": self.hparams.learning_rate},
                {"params": no_decay,   "weight_decay": 0.0,
                 "lr": self.hparams.learning_rate},
                {"params": ssm_params, "weight_decay": 0.0,
                 "lr": ssm_lr},         # 10× lower LR for SSM params
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

    # Print HiPPO-LegS matrix for verification
    A_hippo, B_hippo = hippo_legs_matrix(8)
    print("HiPPO-LegS A (8×8):")
    print(A_hippo.numpy().round(2))
    print(f"B: {B_hippo.numpy().round(3)}\n")

    model  = HydroS4(num_classes=3).to(device).eval()
    total  = sum(p.numel() for p in model.parameters())
    s4_p   = sum(
        p.numel() for n, p in model.named_parameters()
        if "blocks" in n and "s4" in n
    )
    print(f"HydroS4  |  {total:,} total params")
    print(f"  S4D params : {s4_p:,}")
    print(f"  Other      : {total - s4_p:,}")
    print(f"  Blocks     : {len(model.blocks)}  ×  SaShiMiBlock")
    print(f"  d_model={model.hparams.d_model}, d_state={model.hparams.d_state}\n")

    x = torch.randn(2, 32_000, device=device)
    with torch.no_grad():
        logits = model(x)
    print(f"Output shape : {logits.shape}")
    print(f"Logits       : {logits.tolist()}")
