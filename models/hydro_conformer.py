"""
HydroConformer — Multi-scale Conformer for Underwater Vessel Classification
============================================================================

Architecture
------------
  Raw waveform (32 kHz, 1 s)
      ↓
  MultiScalePCEN          — dual-resolution mel spectrograms with trainable PCEN
      ↓                     [wideband n_fft=1024 + narrowband n_fft=4096] → (B,2,F,T)
  SpecAugment (train)     — frequency & time masking
      ↓
  CNNStem                 — 4-layer strided conv, maps (B,2,F,T) → (B,L,D)
      ↓
  SinusoidalPosEnc        — added to sequence tokens
      ↓
  ConformerBlock × N      — FF(½) → MHSA → DepthwiseConv → FF(½) → LN
      ↓                     with stochastic depth regularisation
  AttentionPool           — weighted aggregation over L tokens → (B,D)
      ↓
  Classifier              — LN → Linear → GELU → Dropout → Linear
      ↓
  FocalLoss (class-weighted)

Training features
-----------------
  - Trainable PCEN (adapts to underwater noise-floor variability)
  - Dual-resolution spectrograms (time-res + freq-res)
  - SpecAugment with independent masking per channel
  - Mixup in waveform space (before spectrogram)
  - Focal loss with inverse-frequency class weights
  - AdamW + cosine annealing with linear warmup
  - AMP-friendly (fp16 safe)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC,
    MulticlassConfusionMatrix,
)


# ═══════════════════════════════════════════════════════════════════════
#  Feature Extraction
# ═══════════════════════════════════════════════════════════════════════

class TrainablePCEN(nn.Module):
    """
    Per-Channel Energy Normalization with learnable parameters.

    Transforms a mel magnitude spectrogram so that the noise floor is
    suppressed and transient sounds are enhanced — critical for underwater
    acoustics where the ambient noise level varies continuously.

    Reference: Wang et al., "Trainable Frontend For Robust and Far-Field
    Keyword Spotting", ICASSP 2017.
    """

    def __init__(self, n_mels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # One scalar per mel band, initialised at sensible defaults
        self.alpha = nn.Parameter(torch.full((n_mels, 1), 0.98))   # smoothing coef
        self.delta = nn.Parameter(torch.full((n_mels, 1), 2.0))    # bias
        self.r     = nn.Parameter(torch.full((n_mels, 1), 0.5))    # compression
        self.s     = nn.Parameter(torch.full((n_mels, 1), 0.025))  # IIR time-const

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Magnitude mel spectrogram (B, n_mels, T), non-negative.
        Returns:
            PCEN-normalised spectrogram (B, n_mels, T).
        """
        alpha = torch.sigmoid(self.alpha)           # (n_mels, 1)  ∈ (0, 1)
        delta = F.softplus(self.delta)              # (n_mels, 1)  > 0
        r     = torch.sigmoid(self.r)               # (n_mels, 1)  ∈ (0, 1)
        s     = torch.sigmoid(self.s)               # (n_mels, 1)  ∈ (0, 1)

        # IIR smoother (EMA across time) — approximates the AGC reference signal
        M = self._ema(x, s)                         # (B, n_mels, T)

        # PCEN formula: (x / (eps + M)^alpha + delta)^r − delta^r
        normed = x / (self.eps + M).pow(alpha)
        return (normed + delta).pow(r) - delta.pow(r)

    def _ema(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T), s: (F, 1) — broadcasts over batch dimension
        frames = x.unbind(dim=-1)          # list of (B, F) tensors
        M = frames[0].unsqueeze(-1)        # (B, F, 1)
        out = [M]
        for f in frames[1:]:
            M = (1.0 - s) * M + s * f.unsqueeze(-1)
            out.append(M)
        return torch.cat(out, dim=-1)      # (B, F, T)


class MultiScalePCEN(nn.Module):
    """
    Computes two mel spectrograms at different FFT resolutions and applies
    trainable PCEN to each, then stacks them as a 2-channel 'image'.

    - Wideband  (n_fft=1024, ~32 ms window): good time resolution
    - Narrowband (n_fft=4096, ~128 ms window): good frequency resolution

    Using the same hop_length keeps the time dimension identical so the
    two channels can be directly stacked.
    """

    def __init__(
        self,
        sample_rate: int = 32_000,
        n_mels:      int = 128,
        hop_length:  int = 320,       # 10 ms at 32 kHz
        wb_n_fft:    int = 1_024,     # ~32 ms window
        nb_n_fft:    int = 4_096,     # ~128 ms window
        f_min:       float = 20.0,    # 20 Hz — captures low-freq vessel noise
        f_max:       Optional[float] = None,
    ):
        super().__init__()
        f_max = f_max or sample_rate / 2.0

        mel_kwargs = dict(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0,           # magnitude (PCEN prefers magnitude over power)
        )
        self.wb_mel  = T.MelSpectrogram(n_fft=wb_n_fft, **mel_kwargs)
        self.nb_mel  = T.MelSpectrogram(n_fft=nb_n_fft, **mel_kwargs)
        self.wb_pcen = TrainablePCEN(n_mels)
        self.nb_pcen = TrainablePCEN(n_mels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T)  raw audio at self.sample_rate
        Returns:
            spectrogram: (B, 2, n_mels, T_spec)
        """
        wb = self.wb_mel(waveform).clamp(min=1e-9)  # (B, n_mels, T)
        nb = self.nb_mel(waveform).clamp(min=1e-9)

        wb = self.wb_pcen(wb)
        nb = self.nb_pcen(nb)

        # Align time axes (narrowband may differ by ≤1 frame due to FFT padding)
        t = min(wb.shape[-1], nb.shape[-1])
        return torch.stack([wb[..., :t], nb[..., :t]], dim=1)  # (B, 2, F, T)


class SpecAugment(nn.Module):
    """
    SpecAugment: independent frequency and time masking applied to each
    spectrogram channel separately.  Active only during training.
    """

    def __init__(
        self,
        n_freq_masks:  int = 2,
        freq_mask_max: int = 16,
        n_time_masks:  int = 2,
        time_mask_max: int = 20,
    ):
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [T.FrequencyMasking(freq_mask_max) for _ in range(n_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [T.TimeMasking(time_mask_max) for _ in range(n_time_masks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        if not self.training:
            return x
        B, C, F, T = x.shape
        x = x.view(B * C, F, T)
        for m in self.freq_masks:
            x = m(x)
        for m in self.time_masks:
            x = m(x)
        return x.view(B, C, F, T)


# ═══════════════════════════════════════════════════════════════════════
#  Building Blocks
# ═══════════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    """Stochastic depth: drop entire residual paths during training."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


class FeedForward(nn.Module):
    """Pre-norm FFN: LN → Linear → SiLU → Dropout → Linear → Dropout."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim * expansion)
        self.act  = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2  = nn.Linear(dim * expansion, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))
        return x


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module:
      LN → Pointwise(2×) → GLU → Depthwise → BN → SiLU → Pointwise → Dropout
    """

    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.norm        = nn.LayerNorm(dim)
        self.pw_in       = nn.Linear(dim, dim * 2)       # for GLU
        self.dw_conv     = nn.Conv1d(dim, dim, kernel_size,
                                     padding=kernel_size // 2, groups=dim)
        self.bn          = nn.BatchNorm1d(dim)
        self.act         = nn.SiLU()
        self.pw_out      = nn.Linear(dim, dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        x = self.norm(x)
        x = F.glu(self.pw_in(x), dim=-1)   # (B, L, D)
        x = self.dw_conv(x.transpose(1, 2)).transpose(1, 2)   # conv along L
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.dropout(self.pw_out(x))
        return x + residual


class ConformerBlock(nn.Module):
    """
    Conformer block:  FF(½) → MHSA → ConvModule → FF(½) → LayerNorm
    with stochastic depth on every residual connection.
    """

    def __init__(
        self,
        dim:        int,
        n_heads:    int   = 8,
        expansion:  int   = 4,
        conv_kernel:int   = 31,
        dropout:    float = 0.1,
        attn_drop:  float = 0.1,
        drop_path:  float = 0.1,
    ):
        super().__init__()
        self.ff1       = FeedForward(dim, expansion, dropout)
        self.ff2       = FeedForward(dim, expansion, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn      = nn.MultiheadAttention(dim, n_heads, dropout=attn_drop,
                                               batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv_mod  = ConvolutionModule(dim, conv_kernel, dropout)
        self.final_norm = nn.LayerNorm(dim)
        self.dp        = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp(0.5 * self.ff1(x))

        residual = x
        x_n = self.attn_norm(x)
        a, _ = self.attn(x_n, x_n, x_n)
        x = residual + self.dp(self.attn_drop(a))

        x = self.conv_mod(x)

        x = x + self.dp(0.5 * self.ff2(x))
        return self.final_norm(x)


class CNNStem(nn.Module):
    """
    4-stage strided CNN that embeds a 2-channel spectrogram (B, 2, F, T)
    into a sequence of D-dimensional feature vectors (B, L, D).

    Stride-2 in stages 2-4 gives an 8× spatial reduction:
      F/8 × T/8 patches, each projected to model_dim.
    """

    def __init__(self, in_channels: int = 2, model_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # Stage 1 — full resolution
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Stage 2 — ½ resolution
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Stage 3 — ¼ resolution
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Stage 4 — ⅛ resolution, project to model_dim
            nn.Conv2d(128, model_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(model_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)                # (B, D, F', T')
        B, D, F, Tt = x.shape
        x = x.view(B, D, F * Tt)      # (B, D, L)
        return x.permute(0, 2, 1)     # (B, L, D)


class AttentionPool(nn.Module):
    """Soft attention over the sequence dimension → single vector per sample."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        w = F.softmax(self.score(x), dim=1)   # (B, L, 1)
        return (x * w).sum(dim=1)             # (B, D)


# ═══════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss with per-class alpha weighting.

    Reduces the relative loss for easy examples (high p_t) and focuses
    training on hard ones — especially useful when one class (Passenger)
    has ~185× fewer samples than the majority class.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.
    """

    def __init__(self, class_weights: Optional[list] = None,
                 gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "alpha",
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard cross-entropy with label smoothing
        ce = F.cross_entropy(logits, targets,
                             weight=self.alpha,
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        # Focal weighting: down-weight easy examples
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ═══════════════════════════════════════════════════════════════════════
#  Sinusoidal Positional Encoding
# ═══════════════════════════════════════════════════════════════════════

def sinusoidal_pe(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """Returns (1, seq_len, dim) sinusoidal positional encoding."""
    pe  = torch.zeros(seq_len, dim, device=device)
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10_000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)   # (1, L, D)


# ═══════════════════════════════════════════════════════════════════════
#  HydroConformer
# ═══════════════════════════════════════════════════════════════════════

class HydroConformer(pl.LightningModule):
    """
    Multi-scale Conformer for underwater vessel acoustic classification.

    Args:
        num_classes     : Number of output classes (e.g. 4).
        class_weights   : Inverse-frequency weights for focal loss
                          (from DALIAudioDataModule.class_weights).
        sample_rate     : Audio sample rate in Hz.
        n_mels          : Number of mel filterbanks.
        hop_length      : STFT hop in samples (10 ms = 320 @ 32 kHz).
        model_dim       : Conformer hidden dimension.
        n_blocks        : Number of Conformer blocks.
        n_heads         : Attention heads (model_dim must be divisible).
        conv_kernel     : Depthwise conv kernel in Conformer (odd).
        dropout         : General dropout rate.
        drop_path_rate  : Stochastic depth rate (linearly scheduled across blocks).
        learning_rate   : Peak LR for AdamW.
        warmup_epochs   : Linear warmup length in epochs.
        max_epochs      : Total training epochs (for cosine schedule).
        mixup_alpha     : Beta distribution α for Mixup (0 = disabled).
        focal_gamma     : Focal loss γ (0 = standard cross-entropy).
        label_smoothing : Label smoothing ε.
    """

    def __init__(
        self,
        num_classes:     int   = 4,
        class_weights:   Optional[list] = None,
        sample_rate:     int   = 32_000,
        n_mels:          int   = 128,
        hop_length:      int   = 320,
        wb_n_fft:        int   = 1_024,   # Wideband FFT size (searchable)
        nb_n_fft:        int   = 4_096,   # Narrowband FFT size (searchable)
        model_dim:       int   = 256,
        n_blocks:        int   = 6,
        n_heads:         int   = 8,
        ff_expansion:    int   = 4,       # FFN expansion ratio in Conformer
        conv_kernel:     int   = 31,
        dropout:         float = 0.1,
        drop_path_rate:  float = 0.15,
        learning_rate:   float = 5e-4,
        weight_decay:    float = 1e-2,
        warmup_epochs:   int   = 10,
        max_epochs:      int   = 100,
        mixup_alpha:     float = 0.3,
        focal_gamma:     float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Feature extraction ──────────────────────────────────────────
        self.features    = MultiScalePCEN(sample_rate=sample_rate,
                                          n_mels=n_mels, hop_length=hop_length,
                                          wb_n_fft=wb_n_fft, nb_n_fft=nb_n_fft)
        self.spec_aug    = SpecAugment(n_freq_masks=2, freq_mask_max=16,
                                       n_time_masks=2, time_mask_max=20)

        # ── Encoder ────────────────────────────────────────────────────
        self.cnn_stem    = CNNStem(in_channels=2, model_dim=model_dim)
        self.pos_dropout = nn.Dropout(dropout)

        # Linearly increasing drop-path rate across blocks (deeper = more)
        dp_rates = [drop_path_rate * i / max(n_blocks - 1, 1)
                    for i in range(n_blocks)]
        self.conformer = nn.ModuleList([
            ConformerBlock(
                dim=model_dim, n_heads=n_heads,
                expansion=ff_expansion,
                conv_kernel=conv_kernel, dropout=dropout,
                drop_path=dp_rates[i],
            )
            for i in range(n_blocks)
        ])

        # ── Pooling & head ──────────────────────────────────────────────
        self.pool        = AttentionPool(model_dim)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

        # ── Loss ────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

        # ── Metrics (macro-averaged — fair under class imbalance) ───────
        m_kw = dict(num_classes=num_classes, average="macro")
        self.train_acc  = MulticlassAccuracy(**m_kw)
        self.val_acc       = MulticlassAccuracy(**m_kw)
        self.val_f1        = MulticlassF1Score(**m_kw)
        self.val_precision = MulticlassPrecision(**m_kw)
        self.val_mcc       = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_acc   = MulticlassAccuracy(**m_kw)
        self.test_f1    = MulticlassF1Score(**m_kw)
        self.test_mcc   = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_auroc = MulticlassAUROC(num_classes=num_classes)
        self.test_cm    = MulticlassConfusionMatrix(num_classes=num_classes)

    # ── Core forward ────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T)  float32 waveform at self.sample_rate
        Returns:
            logits:   (B, num_classes)
        """
        x = self.features(waveform)         # (B, 2, F, T)
        x = self.spec_aug(x)                # (B, 2, F, T)

        x = self.cnn_stem(x)               # (B, L, D)

        # Add sinusoidal positional encoding
        pe = sinusoidal_pe(x.shape[1], x.shape[2], x.device)
        x  = self.pos_dropout(x + pe)

        for block in self.conformer:
            x = block(x)                   # (B, L, D)

        x = self.pool(x)                   # (B, D)
        return self.classifier(x)          # (B, num_classes)

    # ── Mixup ───────────────────────────────────────────────────────────

    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
        """Waveform-level Mixup.  Returns (x_mix, y, y_perm, lam)."""
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
        x, y = batch
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

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_precision(logits, y)
        self.val_mcc(logits, y)
        self.log("val/loss",      loss,               on_epoch=True, prog_bar=True)
        self.log("val/acc",       self.val_acc,       on_epoch=True, prog_bar=True)
        self.log("val/f1",        self.val_f1,        on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True, prog_bar=True)
        self.log("val/mcc",       self.val_mcc,       on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        probs  = F.softmax(logits, dim=-1)

        self.test_acc(logits,  y)
        self.test_f1(logits,   y)
        self.test_mcc(logits,  y)
        self.test_auroc(probs, y)
        self.test_cm(logits,   y)
        self.log("test/loss",  self.criterion(logits, y), on_epoch=True)
        self.log("test/acc",   self.test_acc,   on_epoch=True)
        self.log("test/f1",    self.test_f1,    on_epoch=True)
        self.log("test/mcc",   self.test_mcc,   on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix (rows=true, cols=pred):\n{cm.cpu().numpy()}")
        self.test_cm.reset()

    # ── Optimiser ───────────────────────────────────────────────────────

    def configure_optimizers(self):
        # Separate weight-decay groups: no decay on biases / norms
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
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
        )

        def lr_lambda(epoch: int) -> float:
            wu = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            if epoch < wu:
                return epoch / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
