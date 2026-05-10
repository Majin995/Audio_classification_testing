"""
HydroSpikingLEAF — Learnable Audio Frontend + Spiking Neural Network Backbone.

Combines two biologically-inspired components:

1. LEAF Frontend (learnable Gabor filterbank)
   ─────────────────────────────────────────
   GaborFilterbank → GaussianLowpass → TrainablePCEN → SpecAugment
   The filterbank center frequencies and bandwidths are learnable parameters,
   initialized at mel-spaced / ERB-scale values.  Output: (B, n_filters, T).

2. Spiking Neural Network Backbone
   ─────────────────────────────────
   Neuron model: Leaky Integrate-and-Fire (LIF) with vectorized membrane dynamics.

   LIF Dynamics (continuous-time, discretized per frame)
   ─────────────────────────────────────────────────────
   The membrane potential V at time step t integrates all past inputs with
   exponential decay governed by the membrane time constant τ (or equivalently,
   the decay factor β = exp(-dt/τ)):

       V[t] = Σ_{s≤t}  β^(t-s) · I[s]          (IIR integration)
       spk[t] = H(V[t] − θ)                      (threshold crossing)

   where H is the Heaviside step and θ is the firing threshold.
   This is the "non-resetting" LIF approximation, which is efficient to compute
   as a depthwise causal convolution with a geometric kernel:

       kernel_c = [β_c^(T-1), ..., β_c^1, β_c^0]     (length-T causal filter)
       V = depthwise_conv1d_causal(I, kernel)           (fully vectorized, no loop)

   Both β and θ are learnable per channel, so the model discovers optimal
   membrane time constants and firing thresholds for each feature map.

   Surrogate Gradient (arctangent)
   ────────────────────────────────
   The Heaviside function H has zero gradient almost everywhere, making direct
   backpropagation impossible.  We replace it with the arctangent surrogate:

       Forward:   f(u) = H(u)        (true step, for forward pass)
       Backward:  f'(u) ≈ slope / (π · (1 + (slope·u)²))

   The arctangent surrogate has a smooth, bounded derivative centred at u=0,
   approximating the true gradient of a soft threshold.  slope=10 gives a
   sharp but differentiable approximation (Fang et al. 2021).

Architecture
────────────
  Raw waveform (32 kHz, 1 s)
      ↓
  LEAF Frontend           — 40 learnable Gabor filters + IIR lowpass + PCEN
      ↓                     (B, n_filters=40, T_frames≈100)  continuous-valued
  SpecAugment
      ↓
  Spiking Stem            — BN → Conv1d(40→C) → LIFActivation
      ↓                     (B, C, T)  binary spike trains
  SpikingResBlock1d × N   — BN → Conv1d → LIF → BN → Dropout → Conv1d → LIF
      ↓                     cycling dilations [1,2,4]; DropPath residual
  SpikeRatePool           — mean + std firing rate over T → (B, 2C)
      ↓
  Classifier              — Linear(2C→C) → BN → GELU → Dropout → Linear(C→n)
      ↓
  FocalLoss

Key properties vs continuous models
────────────────────────────────────
  • Binary activations after each LIF: gradients flow only via surrogate
  • Learnable β per channel: model discovers optimal temporal integration window
  • Learnable θ per channel: adaptive firing threshold (not fixed at 1)
  • Spike rate readout: sparse binary representations → interpretable activity
  • Logs val/spike_rate during training to monitor sparsity
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

from .hydro_conformer import SpecAugment, DropPath, FocalLoss
from .hydro_leaf import LEAFFrontend


# ═══════════════════════════════════════════════════════════════════════
#  Surrogate gradient
# ═══════════════════════════════════════════════════════════════════════

class _ArcTanSurrogate(torch.autograd.Function):
    """
    Forward:  Heaviside step  H(u) = 1 if u >= 0 else 0
    Backward: Arctangent surrogate  h'(u) = slope / (π · (1 + (slope·u)²))

    The surrogate is smooth, centred at u=0, and has bounded derivative,
    ensuring stable gradient flow through the threshold operation.
    slope=10 gives a sharp but well-conditioned approximation.
    """
    slope: float = 10.0

    @staticmethod
    def forward(ctx, u: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(u)
        return (u >= 0.0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (u,) = ctx.saved_tensors
        s    = _ArcTanSurrogate.slope
        return grad_output * (s / (math.pi * (1.0 + (s * u).pow(2))))


def spike_fn(u: torch.Tensor) -> torch.Tensor:
    """Apply Heaviside threshold with arctangent surrogate gradient."""
    return _ArcTanSurrogate.apply(u)


# ═══════════════════════════════════════════════════════════════════════
#  Vectorized LIF Activation
# ═══════════════════════════════════════════════════════════════════════

class LIFActivation(nn.Module):
    """
    Vectorized Leaky Integrate-and-Fire activation layer.

    Computes membrane potential as a causal IIR-filtered input, then applies
    a learnable threshold with surrogate-gradient Heaviside:

        V[t]   = Σ_{s≤t}  β^(t-s) · x[s]
        spk[t] = H(V[t] − θ)

    Implemented as a depthwise causal Conv1d with geometric kernel, fully
    vectorized over the batch and time dimensions — no Python for-loop.

    Parameters (learnable, per-channel)
    ────────────────────────────────────
    β  ∈ (0, 1)   membrane decay constant (leak).  β→1: long memory (slow leak);
                  β→0: no memory (instant forget).  Parameterised as sigmoid(p).
    θ  > 0.1      firing threshold.  Lower θ → more spikes (denser code).
                  Parameterised as softplus(p) + 0.1.

    Input / Output
    ──────────────
    x   : (B, C, T)  input current (continuous-valued)
    out : (B, C, T)  binary spike trains  ∈ {0, 1}  (dtype=float)
    """

    def __init__(self, channels: int, beta: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.channels = channels
        # β via logit → sigmoid; init at desired beta value
        self.beta_p   = nn.Parameter(
            torch.full((channels,), math.log(beta / (1.0 - beta)))
        )
        # θ via softplus + 0.1 floor
        self.thresh_p = nn.Parameter(torch.zeros(channels))

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_p)           # (C,) ∈ (0, 1)

    @property
    def threshold(self) -> torch.Tensor:
        return F.softplus(self.thresh_p) + 0.1      # (C,) > 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, T)  continuous input current
        Returns:
            spk : (B, C, T)  binary spike trains
        """
        B, C, T = x.shape
        β   = self.beta.to(x.dtype)                 # (C,)
        θ   = self.threshold.to(x.dtype)            # (C,)

        # ── Build IIR kernel per channel ─────────────────────────────────
        # kernel_c[t] = β_c^(T-1-t), so after causal conv:
        #   V[b,c,t] = Σ_{s=0}^{t}  β_c^(t-s) · x[b,c,s]
        t_idx  = torch.arange(T, device=x.device, dtype=x.dtype)
        # Shape: (C, 1, T) — β_c^0 at rightmost position (current), β_c^{T-1} at leftmost
        kernel = (β.view(C, 1, 1) ** t_idx.view(1, 1, T)).flip(-1)   # (C, 1, T)

        # ── Causal depthwise convolution → membrane potential ─────────────
        x_pad = F.pad(x, (T - 1, 0))               # (B, C, T + T-1)
        V     = F.conv1d(x_pad, kernel, groups=C)  # (B, C, T)

        # ── Threshold with surrogate gradient ────────────────────────────
        return spike_fn(V - θ.view(1, C, 1))        # (B, C, T) ∈ {0,1}


# ═══════════════════════════════════════════════════════════════════════
#  Spiking Residual Block
# ═══════════════════════════════════════════════════════════════════════

class SpikingResBlock1d(nn.Module):
    """
    Pre-activation spiking residual block.

    Identical structure to ResBlock1d (hydro_resnet.py) but with
    LIFActivation replacing the ReLU non-linearity:

        BN → Conv1d → LIF → BN → Dropout → Conv1d → LIF  +  skip (DropPath)

    The Conv1d provides spatial / temporal feature mixing across channels and
    neighbouring time frames; the LIF converts the mixed current into spikes,
    enforcing a binary bottleneck and injecting temporal integration.

    Dilation expands the receptive field without additional parameters.
    """

    def __init__(
        self,
        channels:  int,
        kernel:    int   = 3,
        dilation:  int   = 1,
        dropout:   float = 0.1,
        drop_path: float = 0.0,
        beta:      float = 0.9,
        threshold: float = 1.0,
    ):
        super().__init__()
        pad = dilation * (kernel - 1) // 2

        self.bn1   = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel,
                               dilation=dilation, padding=pad, bias=False)
        self.lif1  = LIFActivation(channels, beta=beta, threshold=threshold)

        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel,
                               dilation=dilation, padding=pad, bias=False)
        self.lif2  = LIFActivation(channels, beta=beta, threshold=threshold)

        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)  spike trains (or continuous for first block)
        # Standard SNN residual: inject skip current in *analog* domain before
        # the final LIF, so the output is always binary {0,1}.
        # Adding binary x directly would produce {0,1,2}, breaking sparsity.
        h = self.lif1(self.conv1(self.bn1(x)))         # (B, C, T) binary spikes
        h = self.conv2(self.drop(self.bn2(h)))          # (B, C, T) analog current
        # Combine: processed current + skip current, then threshold to spikes
        return self.lif2(h + self.dp(x.float()))


# ═══════════════════════════════════════════════════════════════════════
#  Spike Rate Pooling
# ═══════════════════════════════════════════════════════════════════════

class SpikeRatePool(nn.Module):
    """
    Aggregate spike trains (B, C, T) → (B, 2C) via firing rate statistics.

    Mean firing rate:    average spikes per neuron → which features are active
    Firing variability:  std of spike counts      → how regular the firing is

    Analogous to AttentiveStatisticsPool but parameter-free and matched to the
    binary nature of spike trains.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)  spike trains ∈ {0, 1}
        mean = x.mean(dim=-1)                               # (B, C)
        var  = x.var(dim=-1, unbiased=False).clamp(min=1e-8)
        std  = var.sqrt()                                   # (B, C)
        return torch.cat([mean, std], dim=-1)               # (B, 2C)


# ═══════════════════════════════════════════════════════════════════════
#  HydroSpikingLEAF LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroSpikingLEAF(pl.LightningModule):
    """
    Learnable Gabor Frontend + Spiking Neural Network for vessel classification.

    The LEAF frontend replaces the fixed mel spectrogram with learnable Gabor
    filters; the spiking backbone replaces continuous activations with LIF
    neurons that produce binary spike trains.

    Args
    ────
    num_classes     : Number of vessel classes.
    class_weights   : Focal loss class weights (None = uniform).
    sample_rate     : Audio sample rate (Hz).
    n_filters       : LEAF filter count.
    window_size     : Gabor window length (samples, odd).
    lowpass_size    : Gaussian lowpass kernel length (samples, odd).
    hop_length      : Lowpass stride — frame hop (samples).
    min_freq        : Minimum Gabor center frequency (Hz).
    max_freq        : Maximum Gabor center frequency (Hz).
    channels (C)    : Channel width throughout spiking stages.
    n_stages        : Number of spiking residual stages.
    n_blocks        : SpikingResBlock1d per stage.
    kernel_size     : Conv1d kernel width in residual blocks.
    beta_init       : Initial LIF membrane decay constant β ∈ (0,1).
    threshold_init  : Initial LIF firing threshold θ.
    dropout         : Dropout rate in residual blocks.
    drop_path_rate  : Max stochastic depth rate.
    learning_rate   : Peak AdamW LR.
    weight_decay    : AdamW weight decay.
    warmup_epochs   : Linear LR warmup epochs.
    max_epochs      : Total epochs for cosine schedule.
    mixup_alpha     : Waveform Mixup β  (0 = off).
    noise_prob      : Additive noise probability.
    gain_prob       : Random gain probability.
    focal_gamma     : Focal loss γ.
    label_smoothing : Label smoothing ε.
    """

    def __init__(
        self,
        num_classes:     int            = 3,
        class_weights:   Optional[list] = None,
        sample_rate:     int            = 32_000,
        n_filters:       int            = 40,
        window_size:     int            = 401,
        lowpass_size:    int            = 401,
        hop_length:      int            = 320,
        min_freq:        float          = 60.0,
        max_freq:        float          = 16_000.0,
        channels:        int            = 128,
        n_stages:        int            = 3,
        n_blocks:        int            = 3,
        kernel_size:     int            = 3,
        beta_init:       float          = 0.9,
        threshold_init:  float          = 1.0,
        dropout:         float          = 0.10,
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

        # ── LEAF frontend ─────────────────────────────────────────────────
        self.frontend = LEAFFrontend(
            n_filters   = n_filters,
            window_size = window_size,
            kernel_size = lowpass_size,
            stride      = hop_length,
            sample_rate = sample_rate,
            min_freq    = min_freq,
            max_freq    = max_freq,
        )
        self.spec_aug = SpecAugment(
            n_freq_masks=2, freq_mask_max=8,
            n_time_masks=2, time_mask_max=20,
        )

        # ── Spiking stem: continuous PCEN → spike domain ──────────────────
        # BN normalises the PCEN output before the first LIF threshold
        self.stem_bn   = nn.BatchNorm1d(n_filters)
        self.stem_conv = nn.Conv1d(n_filters, channels, kernel_size,
                                   padding=kernel_size // 2, bias=False)
        self.stem_lif  = LIFActivation(channels, beta=beta_init, threshold=threshold_init)

        # ── Spiking residual stages ───────────────────────────────────────
        _DILATIONS   = [1, 2, 4]
        total_blocks = n_stages * n_blocks
        dp_rates     = [
            drop_path_rate * i / max(total_blocks - 1, 1)
            for i in range(total_blocks)
        ]

        self.stages  = nn.ModuleList()
        block_idx    = 0
        for _ in range(n_stages):
            stage = nn.ModuleList([
                SpikingResBlock1d(
                    channels  = channels,
                    kernel    = kernel_size,
                    dilation  = _DILATIONS[i % len(_DILATIONS)],
                    dropout   = dropout,
                    drop_path = dp_rates[block_idx + i],
                    beta      = beta_init,
                    threshold = threshold_init,
                )
                for i in range(n_blocks)
            ])
            self.stages.append(stage)
            block_idx += n_blocks

        # ── Readout: spike rate statistics ────────────────────────────────
        self.pool = SpikeRatePool()

        # ── Classifier: standard (non-spiking) head ───────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
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

        # Track spike rates for monitoring
        self._last_spike_rate: float = 0.0

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (B, T_audio) float32
        Returns:
            logits   : (B, num_classes)
        """
        # ── LEAF frontend → continuous-valued spectrogram ─────────────────
        x = self.frontend(waveform)           # (B, 1, n_filters, T_frames)
        x = self.spec_aug(x)                  # (B, 1, n_filters, T_frames)
        x = x.squeeze(1)                      # (B, n_filters, T_frames)

        # ── Spiking stem: convert to spike domain ─────────────────────────
        x = self.stem_lif(self.stem_conv(self.stem_bn(x)))  # (B, C, T) ∈ {0,1}

        # ── Spiking residual backbone ─────────────────────────────────────
        for stage in self.stages:
            for block in stage:
                x = block(x)                  # (B, C, T) spike trains

        # Track mean spike rate for logging (detached — no gradient cost)
        if self.training or not torch.is_grad_enabled():
            self._last_spike_rate = float(x.detach().mean().item())

        # ── Rate readout + classification ─────────────────────────────────
        x = self.pool(x)                      # (B, 2C)
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
        self.log("train/loss",       loss,                  on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/acc",        self.train_acc,        on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/spike_rate", self._last_spike_rate, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        self.val_acc(logits, y);       self.val_f1(logits, y)
        self.val_precision(logits, y); self.val_mcc(logits, y)
        self.log("val/loss",       loss,                  on_epoch=True, prog_bar=True)
        self.log("val/acc",        self.val_acc,          on_epoch=True, prog_bar=True)
        self.log("val/f1",         self.val_f1,           on_epoch=True, prog_bar=True)
        self.log("val/precision",  self.val_precision,    on_epoch=True, prog_bar=True)
        self.log("val/mcc",        self.val_mcc,          on_epoch=True)
        self.log("val/spike_rate", self._last_spike_rate, on_epoch=True)

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
        # LIF parameters (β, θ) are sensitive — lower weight decay, same LR
        lif_params, decay, no_decay = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ("beta_p", "thresh_p")):
                lif_params.append(p)              # LIF neuron params
            elif p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay,      "weight_decay": self.hparams.weight_decay},
                {"params": no_decay,   "weight_decay": 0.0},
                {"params": lif_params, "weight_decay": 0.0},
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
    model  = HydroSpikingLEAF(num_classes=3).to(device).eval()

    total    = sum(p.numel() for p in model.parameters())
    leaf_p   = sum(p.numel() for p in model.frontend.parameters())
    lif_p    = sum(
        p.numel() for n, p in model.named_parameters()
        if "beta_p" in n or "thresh_p" in n
    )
    print(f"HydroSpikingLEAF  |  {total:,} total params")
    print(f"  LEAF frontend   : {leaf_p:,}")
    print(f"  LIF neurons     : {lif_p:,}  (β + θ, learnable per channel)")
    print(f"  Rest            : {total - leaf_p - lif_p:,}  (conv + bn + classifier)")
    print(f"  Stages          : {model.hparams.n_stages}×{model.hparams.n_blocks} SpikingResBlocks\n")

    x = torch.randn(2, 32_000, device=device)
    with torch.no_grad():
        logits = model(x)
    print(f"Output shape : {logits.shape}")
    print(f"Spike rate   : {model._last_spike_rate:.3f}  (fraction of neurons firing)")
    print(f"Logits       : {logits.tolist()}")
