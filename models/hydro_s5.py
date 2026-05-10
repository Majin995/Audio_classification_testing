"""
HydroS5 — S5 (Simplified State Space) Vessel Acoustic Classifier
==================================================================

S5 (Smith et al., "Simplified State Space Layers for Sequence Modeling",
ICLR 2023) improves over HydroS4's S4D design in four concrete ways:

  1. MIMO SSM — one shared diagonal A and one B matrix (d_state × d_model)
     replaces HydroS4's H independent SISO SSMs.  The state captures all
     input channels simultaneously rather than treating each independently.

  2. Bidirectional by default — forward and backward causal scans are run
     in parallel and summed.  HydroS4 uses a unidirectional causal scan.

  3. Per-layer Δ (scalar) — one step size per S5 block (not H per layer).
     Fewer discretisation parameters; expressiveness stays in C and B.

  4. FFT causal scan in N-state space — the same O(L log L) FFT trick as
     S4D but now over N states total (not H × N), leveraging the MIMO B.

How HydroS5 differs from HydroS4
----------------------------------
  ┌──────────────────────┬──────────────────────────┬────────────────────────────┐
  │ Aspect               │ HydroS4                  │ HydroS5 (this file)        │
  ├──────────────────────┼──────────────────────────┼────────────────────────────┤
  │ SSM type             │ H × SISO S4D             │ Single MIMO S5             │
  │ A parameters         │ H × N (per channel)      │ N (shared diagonal)        │
  │ B parameters         │ H × N (per channel)      │ N × H (MIMO, shared)       │
  │ Directionality       │ Unidirectional (causal)  │ Bidirectional (fwd + bwd)  │
  │ Step size Δ          │ H scalars per layer      │ 1 scalar per layer         │
  │ Scan complexity      │ O(H · L log L) FFT       │ O(L log L) FFT over N      │
  │ SSM params/layer*    │ H(6N + 2) = 98,818       │ 4HN + N + H + 1 = 49,537  │
  │ Residual scaling     │ DropPath only            │ LayerScale + DropPath      │
  │ Default SR / length  │ 32 000 Hz / 1.0 s        │ 5 120 Hz / 0.5 s          │
  │ Total params (default)│ ~1.76 M                 │ ~940 K                     │
  └──────────────────────┴──────────────────────────┴────────────────────────────┘
  *H = d_model = 128, N = d_state = 64 (defaults)

Architecture
------------
  Raw waveform (5 120 Hz, 0.5 s = 2 560 samples)
      ↓
  MultiScalePCEN   — dual-resolution trainable PCEN front-end
                     wideband (n_fft=128, ~25 ms) + narrowband (n_fft=512, ~100 ms)
                     → (B, 2, n_mels, T_frames)
      ↓
  SpecAugment (train only)
      ↓
  Flatten + Input projection
      (B, 2·n_mels, T) → permute → (B, T, 2·n_mels)
      → Linear(2·n_mels → d_model) + LayerNorm  → (B, T, d_model)
      + learnable positional embedding (T_max × d_model)
      ↓
  S5Block × n_layers:
      ├── pre-LayerNorm → S5Layer (MIMO bidir parallel scan) → LayerScale → + residual
      └── pre-LayerNorm → FFN (SiLU, 4 × expansion) → LayerScale → + residual
          + DropPath on both branches
      ↓
  LayerNorm
      ↓
  AttentiveStatisticsPool   — learned-weight mean + std → (B, 2·d_model)
      ↓
  Classifier: Linear(2D→D) → BatchNorm → ReLU → Dropout → Linear(D→C)
      ↓
  FocalLoss (class-weighted, label-smoothing)

S5 Theory
---------
MIMO linear SSM:
    h'(t) = A h(t) + B u(t)        h ∈ ℂ^N,  u ∈ ℝ^H
    y(t)  = Re(C h(t)) + D u(t)    y ∈ ℝ^H

With diagonal A = diag(a_0, …, a_{N-1}), a_n = A_real_n + i A_imag_n:

ZOH discretisation at step Δ:
    Ā_n = exp(Δ · a_n)                 (elementwise on diagonal)
    B̄_n,h = (Ā_n - 1) / a_n · B_n,h  (broadcast over H columns)

Discrete recurrence (causal, per time step):
    h_t = Ā ⊙ h_{t-1} + B̄ u_t     (⊙ = elementwise, Ā constant)
    y_t = Re(C h_t) + D u_t

Since Ā is constant the recurrence is a causal vector convolution:
    h_t = Σ_{s≤t} Ā^{t-s} ⊙ (B̄ u_s)

Computed via FFT in O(L log L) over N-dimensional states.

Stability: A_real = -(softplus(p) + 0.5) ≤ -0.5  →  |Ā| ≤ exp(-0.5·Δ_min) < 1.
"""

from __future__ import annotations

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

from .hydro_conformer import MultiScalePCEN, SpecAugment, DropPath, FocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  HiPPO-LegS S4D initialization for shared S5 diagonal A and B
# ═══════════════════════════════════════════════════════════════════════

def _s5_hippo_init(d_state: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    S4D-LegS diagonal initialization adapted for S5's MIMO B matrix.

    Following Gu et al. (2022) "On the Parameterization and Initialization of
    Diagonal State Space Models" and Smith et al. (2023) "S5":

      A_n = -(1/2 + i π n)   for n = 0, …, N-1
      B_col_n = i^n / sqrt(N) (each column of B starts with this pattern)

    The B matrix in S5 is (N, H) complex; we initialise all H columns with
    the same SISO-pattern so each input channel gets a principled start.

    Returns
    -------
    A_real_init : (d_state,)   — imaginary axis poles, Re(A) = -0.5
    A_imag_init : (d_state,)   — imaginary parts π n
    B_col       : (d_state, 2) — one column of B (real, imag) — replicate for H cols
    """
    n = torch.arange(d_state, dtype=torch.float32)

    A_real_init = -0.5 * torch.ones(d_state)
    A_imag_init = math.pi * n

    angle  = (math.pi / 2.0) * n
    B_real = torch.cos(angle) / math.sqrt(d_state)
    B_imag = torch.sin(angle) / math.sqrt(d_state)
    B_col  = torch.stack([B_real, B_imag], dim=-1)   # (N, 2)

    return A_real_init, A_imag_init, B_col


# ═══════════════════════════════════════════════════════════════════════
#  S5Layer — MIMO diagonal SSM with bidirectional parallel scan
# ═══════════════════════════════════════════════════════════════════════

class S5Layer(nn.Module):
    """
    S5 MIMO Diagonal State Space Layer.

    Concrete improvements over S4DLayer used in HydroS4
    -----------------------------------------------------
    1. Shared A — (d_state,) complex diagonal, same poles for all H channels.
       S4D: separate (d_model, d_state) — H independent sets of poles.

    2. MIMO B — (d_state, d_model) maps all H input channels into state space
       simultaneously.  S4D: H independent (d_state,) vectors, no cross-channel
       interaction in B.

    3. Bidirectional — if bidirectional=True:
         y = Re(C_fwd h_fwd) + Re(C_bwd h_bwd_flipped)
       S4D: causal-only.  Capturing both past and future context in one layer
       without doubling parameters (C_bwd reuses the same B/A kernel).

    4. Per-layer Δ (scalar) — one log_dt for the entire layer.
       Expressiveness over time scales lives in C, not in duplicated Δ's.

    5. FFT parallel scan in N-state space — the time-domain causal scan
       h_t = Σ_{s≤t} Ā^{t-s} ⊙ (B̄ u_s)
       is a vector convolution computed via FFT for all N state dims at once.

    Parameters
    ----------
    d_model      : Input / output feature dim H.
    d_state      : SSM state dim N (shared diagonal poles).
    dt_min/max   : Log-uniform range for Δ initialisation.
    bidirectional: Run fwd + bwd scans and sum (default True).
    """

    def __init__(
        self,
        d_model:       int   = 128,
        d_state:       int   = 64,
        dt_min:        float = 1e-3,
        dt_max:        float = 1e-1,
        bidirectional: bool  = True,
    ):
        super().__init__()
        self.d_model       = d_model
        self.d_state       = d_state
        self.bidirectional = bidirectional

        A_real_init, A_imag_init, B_col = _s5_hippo_init(d_state)

        # ── Shared diagonal A ─────────────────────────────────────────────
        # Re(A) = -(softplus(log_A_real) + 0.5) ≤ -0.5 always → stable
        self.log_A_real = nn.Parameter(torch.full((d_state,), -5.0))
        self.A_imag     = nn.Parameter(A_imag_init.clone())

        # ── MIMO B : (N, H, 2) complex — all H cols init with same HiPPO col ──
        B_init = B_col.unsqueeze(1).expand(-1, d_model, -1).contiguous()  # (N, H, 2)
        self.B = nn.Parameter(B_init)

        # ── C: forward readout (H, N, 2) ─────────────────────────────────
        C_scale = 1.0 / math.sqrt(d_state)
        self.C_fwd = nn.Parameter(
            torch.randn(d_model, d_state, 2) * C_scale
        )
        if bidirectional:
            self.C_bwd = nn.Parameter(
                torch.randn(d_model, d_state, 2) * C_scale
            )

        # ── D: per-channel skip ───────────────────────────────────────────
        self.D = nn.Parameter(torch.ones(d_model))

        # ── Δ: single scalar log step size per layer ──────────────────────
        log_dt = torch.FloatTensor(1).uniform_(math.log(dt_min), math.log(dt_max))
        self.log_dt = nn.Parameter(log_dt)

    # ── Discretisation ────────────────────────────────────────────────────

    def _discretise(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ZOH discretisation.

        Returns
        -------
        A_bar : (N,)    complex  Ā = exp(Δ · A)
        B_bar : (N, H)  complex  B̄[n,h] = (Ā_n - 1) / A_n · B[n,h]
        dt_A  : (N,)    complex  Δ · A  (used for power series in scan)
        """
        A_real = -(F.softplus(self.log_A_real) + 0.5)   # (N,) ≤ -0.5
        A      = torch.complex(A_real, self.A_imag)      # (N,)
        dt     = self.log_dt.exp()                       # scalar

        dt_A   = dt * A                                  # (N,)
        A_bar  = torch.exp(dt_A)                         # (N,)

        B_cplx = torch.view_as_complex(self.B.contiguous())   # (N, H)
        # B̄[n,h] = (Ā_n - 1) / A_n * B[n,h]
        B_bar  = ((A_bar - 1.0) / A).unsqueeze(-1) * B_cplx  # (N, H)

        return A_bar, B_bar, dt_A

    # ── Parallel causal scan ──────────────────────────────────────────────

    def _causal_scan_fft(
        self, b: torch.Tensor, dt_A: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel causal convolution scan for constant A_bar.

            h_t = Σ_{s=0}^{t} Ā^{t-s} ⊙ b_s

        Since Ā is constant this is a length-L causal vector convolution
        computed via FFT in O(L log L) for all N state dims simultaneously.

        Uses exp(t · Δ·A) to compute Ā^t without complex log (avoids
        branch-cut issues entirely).

        Args
        ----
        b    : (B, L, N) complex — B̄ u_t at each step
        dt_A : (N,)      complex — Δ · A (used to build Ā^t = exp(t·Δ·A))

        Returns
        -------
        h : (B, L, N) complex
        """
        B_size, L, N = b.shape
        device       = b.device

        # Ā^t for t = 0, …, L-1 via exp(t · Δ·A), no complex log required
        t     = torch.arange(L, device=device, dtype=torch.float32)
        t_c   = t.to(dtype=torch.complex64).unsqueeze(-1)  # (L, 1)
        A_pow = torch.exp(t_c * dt_A.unsqueeze(0))         # (L, N)

        # FFT causal convolution per state dim
        b_t   = b.permute(0, 2, 1)                        # (B, N, L)
        K_t   = A_pow.T                                    # (N, L)

        # b_t is complex (MIMO B̄ projects real u into complex state)
        # A_pow is complex → must use fft (not rfft which requires real input)
        Bf = torch.fft.fft(b_t, n=2 * L, dim=-1)          # (B, N, 2L)
        Kf = torch.fft.fft(K_t, n=2 * L, dim=-1)          # (N,  2L)

        hf = Bf * Kf.unsqueeze(0)                          # (B, N, 2L)
        h  = torch.fft.ifft(hf, n=2 * L, dim=-1)[..., :L] # (B, N, L) complex
        return h.permute(0, 2, 1)                          # (B, L, N)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args  : x (B, L, H)  — input sequence (any dtype; cast internally)
        Returns: y (B, L, H) — output sequence (same dtype as input)
        """
        in_dtype = x.dtype
        x32      = x.float()
        B_size, L, H = x32.shape

        A_bar, B_bar, dt_A = self._discretise()   # (N,), (N, H), (N,) complex

        # Project input to state: b_t = B̄ u_t → (B, L, N) complex
        # x_c cast: imaginary part = 0; B_bar.T is (H, N)
        x_c = x32.to(torch.complex64)             # (B, L, H)
        b   = x_c @ B_bar.T                       # (B, L, N)  B_bar.T: (H, N)

        # ── Forward causal scan ──────────────────────────────────────────
        h_fwd = self._causal_scan_fft(b, dt_A)    # (B, L, N) complex

        C_fwd = torch.view_as_complex(self.C_fwd.contiguous())  # (H, N)
        y     = (h_fwd @ C_fwd.T).real             # (B, L, H)

        # ── Backward causal scan (flipped) ───────────────────────────────
        if self.bidirectional:
            b_bwd = x_c.flip(1) @ B_bar.T         # (B, L, N)
            h_bwd = self._causal_scan_fft(b_bwd, dt_A).flip(1)  # (B, L, N)
            C_bwd = torch.view_as_complex(self.C_bwd.contiguous())
            y     = y + (h_bwd @ C_bwd.T).real

        y = y + self.D * x32
        return y.to(in_dtype)


# ═══════════════════════════════════════════════════════════════════════
#  S5Block — residual block wrapping S5Layer
# ═══════════════════════════════════════════════════════════════════════

class S5Block(nn.Module):
    """
    Pre-norm residual block with S5Layer + position-wise FFN.

    Compared to SaShiMiBlock in HydroS4:
      - No up_proj (S5 MIMO directly operates on d_model channels)
      - LayerScale on both residual branches for stable deep training
      - SiLU instead of GLU gate (gating is inherent in MIMO C readout)

    Args
    ----
    d_model      : Channel width.
    d_state      : S5 state dimension.
    expansion    : FFN expansion ratio.
    dropout      : Dropout in FFN.
    drop_path    : Stochastic depth rate.
    dt_min/max   : Step size range for S5Layer.
    bidirectional: Forward + backward scan.
    ls_init      : LayerScale initial value (1.0 = identity, < 1.0 suppresses
                   early residuals for stable gradient flow in deep stacks).
    """

    def __init__(
        self,
        d_model:       int   = 128,
        d_state:       int   = 64,
        expansion:     int   = 4,
        dropout:       float = 0.10,
        drop_path:     float = 0.0,
        dt_min:        float = 1e-3,
        dt_max:        float = 1e-1,
        bidirectional: bool  = True,
        ls_init:       float = 1.0,
    ):
        super().__init__()
        self.dp = DropPath(drop_path)

        # ── S5 branch ────────────────────────────────────────────────────
        self.norm_s5 = nn.LayerNorm(d_model)
        self.s5      = S5Layer(d_model, d_state, dt_min, dt_max, bidirectional)
        self.ls_s5   = nn.Parameter(torch.ones(d_model) * ls_init)

        # ── FFN branch ───────────────────────────────────────────────────
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )
        self.ls_ff = nn.Parameter(torch.ones(d_model) * ls_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # S5 + LayerScale + DropPath
        x = x + self.dp(self.ls_s5 * self.s5(self.norm_s5(x)))
        # FFN + LayerScale + DropPath
        x = x + self.dp(self.ls_ff * self.ff(self.norm_ff(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  Pooling (shared with HydroS4, inlined here to be self-contained)
# ═══════════════════════════════════════════════════════════════════════

class _AttentiveStatPool(nn.Module):
    """Learned-weight mean + std pooling over time. Input (B,T,H) → (B,2H)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w    = F.softmax(self.attn(x), dim=1)
        mean = (x * w).sum(dim=1)
        var  = (x ** 2 * w).sum(dim=1) - mean ** 2
        return torch.cat([mean, var.clamp(min=1e-9).sqrt()], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  HydroS5 LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroS5(pl.LightningModule):
    """
    S5 MIMO State Space Model for vessel acoustic classification.

    Defaults differ from HydroS4:
    - sample_rate 5 120 Hz  (vs 32 000 Hz)  — Nyquist-optimal for vessel audio
    - fixed_len   2 560 samples (0.5 s)     — half-second clips reduce compute
    - bidirectional=True                    — S5 natively supports bidir scan
    - n_layers=6, d_model=128, d_state=64   — ~940 K params (vs ~1.76 M for S4)

    Parameters
    ----------
    num_classes     : Output classes (default 3: Cargo, Tanker, Tug).
    class_weights   : Inverse-frequency weights for FocalLoss.
    sample_rate     : Native sample rate. Default 5 120 Hz.
    fixed_len       : Input length in samples. Default 2 560 (0.5 s at 5 120 Hz).
    n_mels          : Mel filterbank bins. Default 64.
    hop_length      : STFT hop (samples). Default 25 (~200 frames/s at 5 120 Hz).
    d_model         : Residual channel width H. Default 128.
    d_state         : SSM state dimension N. Default 64.
    n_layers        : Number of S5Blocks. Default 6.
    expansion       : FFN expansion ratio. Default 4.
    dt_min / dt_max : Step size Δ log-uniform init range.
    bidirectional   : Enable bidirectional scan. Default True.
    ls_init         : LayerScale initial value. 1.0 = standard residual.
    dropout         : Dropout in FFN and classifier. Default 0.10.
    drop_path_rate  : Stochastic depth max rate. Default 0.05.
    learning_rate   : AdamW peak LR. Default 3e-4.
    weight_decay    : AdamW weight decay. Default 0.012.
    warmup_epochs   : Linear LR warmup length. Default 10.
    max_epochs      : Total epochs for cosine schedule. Default 100.
    mixup_alpha     : Waveform Mixup α (0 = off). Default 0.20.
    noise_prob      : Gaussian noise augmentation probability. Default 0.50.
    noise_snr_min   : Min noise SNR (dB). Default 15.0.
    noise_snr_max   : Max noise SNR (dB). Default 35.0.
    gain_prob       : Random gain augmentation probability. Default 0.70.
    focal_gamma     : Focal loss γ. Default 2.0.
    label_smoothing : Label smoothing ε. Default 0.001.
    """

    def __init__(
        self,
        num_classes:     int            = 3,
        class_weights:   Optional[list] = None,
        sample_rate:     int            = 5_120,
        fixed_len:       int            = 2_560,
        n_mels:          int            = 64,
        hop_length:      int            = 25,
        d_model:         int            = 128,
        d_state:         int            = 64,
        n_layers:        int            = 6,
        expansion:       int            = 4,
        dt_min:          float          = 1e-3,
        dt_max:          float          = 1e-1,
        bidirectional:   bool           = True,
        ls_init:         float          = 1.0,
        dropout:         float          = 0.10,
        drop_path_rate:  float          = 0.05,
        learning_rate:   float          = 3e-4,
        weight_decay:    float          = 0.012,
        warmup_epochs:   int            = 10,
        max_epochs:      int            = 100,
        mixup_alpha:     float          = 0.20,
        noise_prob:      float          = 0.50,
        noise_snr_min:   float          = 15.0,
        noise_snr_max:   float          = 35.0,
        gain_prob:       float          = 0.70,
        focal_gamma:     float          = 2.0,
        label_smoothing: float          = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Front-end ─────────────────────────────────────────────────────
        # At 5 120 Hz: wb_n_fft=128 ≈ 25 ms window, nb_n_fft=512 ≈ 100 ms
        self.features  = MultiScalePCEN(
            sample_rate = sample_rate,
            n_mels      = n_mels,
            hop_length  = hop_length,
            wb_n_fft    = 128,
            nb_n_fft    = 512,
        )
        self.spec_aug  = SpecAugment(
            n_freq_masks  = 2, freq_mask_max = 10,
            n_time_masks  = 2, time_mask_max = 16,
        )

        # ── Input projection ──────────────────────────────────────────────
        in_dim = 2 * n_mels
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Positional embedding ──────────────────────────────────────────
        # T_max: generous upper bound (1 s at 5120 Hz, hop=25 → 204 frames)
        T_max = math.ceil(sample_rate / hop_length) + 4
        self.pos_emb = nn.Parameter(torch.zeros(1, T_max, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # ── S5 stack ──────────────────────────────────────────────────────
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            S5Block(
                d_model       = d_model,
                d_state       = d_state,
                expansion     = expansion,
                dropout       = dropout,
                drop_path     = dp_rates[i],
                dt_min        = dt_min,
                dt_max        = dt_max,
                bidirectional = bidirectional,
                ls_init       = ls_init,
            )
            for i in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # ── Pooling + classifier ──────────────────────────────────────────
        self.pool = _AttentiveStatPool(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # ── Loss ──────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights   = class_weights,
            gamma           = focal_gamma,
            label_smoothing = label_smoothing,
        )

        # ── Metrics ───────────────────────────────────────────────────────
        kwargs = dict(num_classes=num_classes, average="macro")
        self.train_acc     = MulticlassAccuracy(**kwargs)
        self.val_acc       = MulticlassAccuracy(**kwargs)
        self.val_f1        = MulticlassF1Score(**kwargs)
        self.val_precision = MulticlassPrecision(**kwargs)
        self.val_mcc       = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_acc      = MulticlassAccuracy(**kwargs)
        self.test_f1       = MulticlassF1Score(**kwargs)
        self.test_mcc      = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_auroc    = MulticlassAUROC(**kwargs)
        self.test_cm       = MulticlassConfusionMatrix(num_classes=num_classes)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args  : x (B, T_samples) raw waveform at self.hparams.sample_rate
        Returns: logits (B, num_classes)
        """
        # Front-end: (B, T_samples) → (B, 2, n_mels, T_frames)
        feat = self.features(x)

        if self.training:
            feat = self.spec_aug(feat)

        # Flatten + project: (B, 2*n_mels, T_frames) → (B, T, d_model)
        B, C, F_bins, T = feat.shape
        feat = feat.reshape(B, C * F_bins, T).permute(0, 2, 1)   # (B, T, 2*n_mels)
        feat = self.input_proj(feat)                               # (B, T, d_model)

        # Positional embedding (truncate / expand if T > T_max)
        if T <= self.pos_emb.shape[1]:
            feat = feat + self.pos_emb[:, :T, :]
        else:
            # Interpolate pos_emb to length T (rare: clips longer than T_max)
            pos = F.interpolate(
                self.pos_emb.permute(0, 2, 1), size=T, mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
            feat = feat + pos

        # S5 stack
        for block in self.blocks:
            feat = block(feat)                    # (B, T, d_model)

        feat = self.norm_out(feat)                # (B, T, d_model)
        feat = self.pool(feat)                    # (B, 2*d_model)
        return self.classifier(feat)              # (B, num_classes)

    # ── Augmentation ──────────────────────────────────────────────────────

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        hp = self.hparams
        if hp.gain_prob > 0 and torch.rand(1).item() < hp.gain_prob:
            gain = 0.6 + 0.8 * torch.rand(x.shape[0], 1, device=x.device)
            x    = x * gain

        if hp.noise_prob > 0 and torch.rand(1).item() < hp.noise_prob:
            snr    = hp.noise_snr_min + (hp.noise_snr_max - hp.noise_snr_min) * torch.rand(
                x.shape[0], 1, device=x.device
            )
            sig_pw = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
            noise  = torch.randn_like(x)
            noise_pw = noise.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
            scale  = (sig_pw / noise_pw / (10 ** (snr / 10.0))).sqrt()
            x      = x + scale * noise

        return x

    def _mixup(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        alpha = self.hparams.mixup_alpha
        if alpha > 0 and self.training:
            lam  = float(torch._C._VariableFunctions.empty(1).uniform_(0, 1).item())
            # Beta(α,α) ≈ Uniform when α small; use symmetric Beta
            import numpy as np
            lam  = float(np.random.beta(alpha, alpha))
            idx  = torch.randperm(x.shape[0], device=x.device)
            x    = lam * x + (1 - lam) * x[idx]
            y_p  = y[idx]
        else:
            lam, y_p = 1.0, y
        return x, y, y_p, lam

    def _loss(self, logits, y, y_p, lam):
        l1 = self.criterion(logits, y)
        if lam < 1.0:
            return lam * l1 + (1 - lam) * self.criterion(logits, y_p)
        return l1

    # ── Training / Validation / Test steps ───────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self._augment(x)
        x, y, y_p, lam = self._mixup(x, y)
        logits = self(x)
        loss   = self._loss(logits, y, y_p, lam)
        preds  = logits.argmax(dim=-1)
        self.train_acc.update(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        preds  = logits.argmax(dim=-1)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_precision.update(preds, y)
        self.val_mcc.update(preds, y)
        self.log("val/loss",      loss,               on_epoch=True, prog_bar=True)
        self.log("val/acc",       self.val_acc,        on_epoch=True, prog_bar=True)
        self.log("val/f1",        self.val_f1,         on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision,  on_epoch=True)
        self.log("val/mcc",       self.val_mcc,        on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        preds  = logits.argmax(dim=-1)
        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.test_mcc.update(preds, y)
        self.test_auroc.update(F.softmax(logits, dim=-1), y)
        self.test_cm.update(preds, y)

    def on_test_epoch_end(self):
        self.log("test/acc",   self.test_acc)
        self.log("test/f1",    self.test_f1)
        self.log("test/mcc",   self.test_mcc)
        self.log("test/auroc", self.test_auroc)
        cm = self.test_cm.compute()
        print(f"\nConfusion matrix:\n{cm.cpu().numpy()}")

    # ── Optimiser ─────────────────────────────────────────────────────────

    def configure_optimizers(self):
        hp = self.hparams

        # SSM parameters (A, B, C, dt) get a reduced LR via a separate group
        ssm_names = {"log_A_real", "A_imag", "B", "C_fwd", "C_bwd", "log_dt", "D"}
        decay_p, no_decay_p, ssm_p = [], [], []

        for name, param in self.named_parameters():
            base = name.split(".")[-1]
            if base in ssm_names:
                ssm_p.append(param)
            elif param.ndim < 2 or base in {"bias", "ls_s5", "ls_ff"}:
                no_decay_p.append(param)
            else:
                decay_p.append(param)

        optimizer = torch.optim.AdamW([
            {"params": decay_p,   "weight_decay": hp.weight_decay},
            {"params": no_decay_p, "weight_decay": 0.0},
            {"params": ssm_p,      "weight_decay": 0.0,
             "lr": hp.learning_rate * 0.1},   # 10× lower LR for SSM params
        ], lr=hp.learning_rate)

        total_steps   = hp.max_epochs
        warmup_steps  = hp.warmup_epochs
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-2, end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6,
                ),
            ],
            milestones=[warmup_steps],
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    model = HydroS5(num_classes=3)
    model.eval()

    total  = sum(p.numel() for p in model.parameters())
    s5_p   = sum(p.numel() for n, p in model.named_parameters() if "blocks" in n and "s5" in n)
    other  = total - s5_p

    print(f"\nHydroS5 architecture (defaults)")
    print(f"  Total params  : {total:,}")
    print(f"  S5 layers     : {s5_p:,}")
    print(f"  Other         : {other:,}  (PCEN + proj + pool + classifier)")
    print(f"\nS5 vs S4D comparison (d_model=128, d_state=64, bidir):")
    print(f"  S5Layer  (bidir=True)  ~{4*128*64 + 64 + 128 + 1:,} params/layer")
    print(f"  S4DLayer (unidir)      ~{128*2*(6*64+2):,} params/layer (at 2×d_model in SaShiMi block)")
    print(f"  S5 is bidirectional at ~S4D unidirectional parameter cost")

    x = torch.randn(2, 2560)
    with torch.no_grad():
        out = model(x)
    print(f"\nForward pass: input {tuple(x.shape)} → output {tuple(out.shape)}")
    assert out.shape == (2, 3), f"Expected (2,3), got {out.shape}"
    print("PASS")
