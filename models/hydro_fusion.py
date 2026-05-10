"""
HydroFusion — Multi-Frontend × Multi-SSM Fusion Vessel Classifier
==================================================================

Architecture
------------
  Raw waveform (B, L) at 32 kHz
      ↓
  [Train-only: Mixup · Additive Gaussian noise · Random gain]
      ↓
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stream A (always) — MultiScalePCEN                                 │
  │    dual-resolution mel  (n_fft=1024 wideband + n_fft=4096 narrow)  │
  │    → (B, 2, n_mels, T_frames)                                       │
  │    → SpecAugment (train) on spectrogram                             │
  │    → flatten + Conv1d → (B, d_model, T_frames)                     │
  └─────────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Stream B (optional, use_xlsr=True) — XLSR-53 CNN extractor        │
  │    facebook/wav2vec2-large-xlsr-53 feature extractor (frozen)       │
  │    resample 32 kHz → 16 kHz                                         │
  │    → (B, 512, T_xlsr) → F.interpolate → (B, 512, T_frames)        │
  │    → Conv1d(512 → d_model) → (B, d_model, T_frames)                │
  └─────────────────────────────────────────────────────────────────────┘
      ↓
  Fusion gate: learned Conv1d([1|2]·d → d) + BN + ReLU
      ↓
  SERes2Block × n_res     [dilations from dilation_rates, channel-first]
  MFALayer                [concat all block outputs → project back to d]
      ↓  permute(0,2,1) ──────────────────────────────────────────────────────
  SaShiMiBlock × n_s4     [S4D long-range, sequence-last (B,T,D)]
      ↓
  BidirMambaBlock × n_mamba  [selective global context, sequence-last]
      ↓  permute(0,2,1) ──────────────────────────────────────────────────────
  AttentiveStatisticsPool → (B, 2·d_model)
      ↓
  Classifier: Linear(2D→D) → BN → ReLU → Dropout → Linear(D→C)
      ↓
  FocalLoss + optional CSSD distillation

Notes
-----
  • CSSD: set cssd_alpha < 1.0 to enable Cross-SR Self-Distillation. With
    use_xlsr=True this doubles VRAM for the teacher copy — requires 80 GB+.
  • Stream B adds ~80 M frozen parameters (only CNN extractor, not transformer).
  • Cluster / DDP: pass strategy="ddp" in train_fusion.py; find_unused_parameters
    is set to False when use_xlsr=False (or True when it is enabled).
"""

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassMatthewsCorrCoef, MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from .hydro_conformer import MultiScalePCEN, SpecAugment, DropPath, FocalLoss
from .hydro_s4 import SaShiMiBlock
from .hydro_ssamba import BidirMambaBlock

try:
    from transformers import Wav2Vec2Model
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  Inline SE-Res2 components (adapted from hydro_net.py)
# ═══════════════════════════════════════════════════════════════════════

class _Res2DilatedConv(nn.Module):
    """Res2Net multi-branch dilated convolution (channel-first)."""

    def __init__(self, channels: int, scale: int = 8,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        assert channels % scale == 0, f"channels ({channels}) must be divisible by scale ({scale})"
        self.scale = scale
        self.width = channels // scale
        pad = dilation * (kernel_size // 2)
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size,
                      dilation=dilation, padding=pad, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.width) for _ in range(scale - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks  = x.chunk(self.scale, dim=1)
        outputs = [chunks[0]]
        carry: Optional[torch.Tensor] = None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            inp   = chunks[i + 1] if carry is None else chunks[i + 1] + carry
            carry = F.relu(bn(conv(inp)))
            outputs.append(carry)
        return torch.cat(outputs, dim=1)


class _SEBlock(nn.Module):
    """Squeeze-Excitation channel attention (channel-first)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.unsqueeze(-1)


class _SERes2Block(nn.Module):
    """SE-Res2Block (channel-first): pointwise → Res2DilatedConv → SE → residual."""

    def __init__(self, channels: int, scale: int = 8, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.pw_in  = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.res2   = _Res2DilatedConv(channels, scale, kernel_size, dilation)
        self.bn     = nn.BatchNorm1d(channels)
        self.pw_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.se      = _SEBlock(channels)
        self.dropout = nn.Dropout(dropout)
        self.dp      = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pw_in(x)
        out = self.res2(out)
        out = self.bn(out)
        out = self.pw_out(out)
        out = self.se(out)
        out = self.dropout(out)
        return x + self.dp(out)


class _MFALayer(nn.Module):
    """Multi-scale Feature Aggregation: concat n feature maps → project to d."""

    def __init__(self, n_inputs: int, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(channels * n_inputs, channels, 1, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )

    def forward(self, *feature_maps: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat(feature_maps, dim=1))


class _AttentiveStatisticsPool(nn.Module):
    """
    Learned-weight mean + std over the time axis for (B, D, T) channel-first tensors.
    Returns (B, 2D).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 1),
            nn.Tanh(),
            nn.Conv1d(channels // 4, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w    = F.softmax(self.attn(x), dim=-1)
        mean = (x * w).sum(dim=-1)
        var  = ((x ** 2) * w).sum(dim=-1) - mean ** 2
        std  = var.clamp(min=1e-9).sqrt()
        return torch.cat([mean, std], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  XLSR-53 CNN-only feature extractor (frozen)
# ═══════════════════════════════════════════════════════════════════════

class XLSRCNNExtractor(nn.Module):
    """
    Loads the wav2vec2-large-xlsr-53 CNN feature extractor only (no transformer).
    Freezes all weights.  Outputs (B, 512, T) from (B, L) at 16 kHz.

    XLSR CNN extractor outputs 512 channels at ~50 Hz regardless of input length,
    making it efficient for long clips without attending over the full sequence.
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-large-xlsr-53"):
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers package is required for XLSR stream. "
                "Install with: pip install transformers"
            )
        w2v = Wav2Vec2Model.from_pretrained(model_name)
        self.feature_extractor = w2v.feature_extractor   # CNN layers only
        del w2v
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) float waveform at 16 kHz.
        Returns:
            (B, 512, T_xlsr) CNN features, channel-first.
        """
        feats = self.feature_extractor(x)      # (B, T_xlsr, 512)
        return feats.transpose(1, 2)           # (B, 512, T_xlsr)


# ═══════════════════════════════════════════════════════════════════════
#  HydroFusion LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroFusion(pl.LightningModule):
    """
    Full-fusion vessel acoustic classifier.

    Args
    ----
    num_classes     : Number of vessel categories.
    class_weights   : (num_classes,) inverse-frequency weights. None = uniform.
    sample_rate     : Input waveform sample rate (Hz). Default 32 000.
    n_mels          : Mel filterbank bins for Stream A. Default 128.
    hop_length      : STFT hop in samples. Default 320.
    d_model         : Feature channel width throughout the backbone. Default 128.
    scale           : Res2Net split factor (must divide d_model). Default 8.
    dilation_rates  : Per-SERes2Block dilation (one per block). Default [2,4,8].
    kernel_size     : Res2DilatedConv kernel. Default 3.
    n_s4            : Number of SaShiMi (S4D) blocks. Default 2.
    s4_d_state      : S4D state size per channel. Default 64.
    n_mamba         : Number of BidirMamba blocks. Default 2.
    mamba_d_state   : Mamba SSM state size. Default 16.
    mamba_expand    : Mamba inner-dim expansion factor. Default 2.
    mamba_d_conv    : Mamba causal conv kernel. Default 4.
    use_xlsr        : Enable XLSR-53 CNN Stream B. Default False.
    xlsr_model_name : HuggingFace model ID for XLSR extractor.
    dropout         : Dropout rate in classifier head. Default 0.24.
    drop_path_rate  : Max stochastic-depth probability. Default 0.10.
    learning_rate   : AdamW base LR. Default 3e-4.
    weight_decay    : AdamW weight decay. Default 0.012.
    warmup_epochs   : Linear LR warmup duration. Default 10.
    max_epochs      : Total training epochs (for cosine schedule). Default 100.
    mixup_alpha     : Beta distribution parameter for waveform Mixup. 0 = off.
    noise_prob      : Probability of additive Gaussian noise augmentation.
    noise_snr_min   : Minimum SNR (dB) for noise augmentation.
    noise_snr_max   : Maximum SNR (dB) for noise augmentation.
    gain_prob       : Probability of random gain augmentation.
    focal_gamma     : Focal loss focusing parameter. Default 2.0.
    label_smoothing : Label smoothing ε. Default 0.001.
    cssd_alpha      : CE weight for CSSD. 1.0 = pure CE (CSSD disabled).
    cssd_temp       : Distillation temperature. Default 4.0.
    cssd_degrade_sr : Target SR (Hz) for audio degradation. Default 8 000.
    cssd_degrade_prob: Probability of applying CSSD per batch. Default 0.5.
    ema_momentum    : EMA teacher update momentum. Default 0.999.
    """

    def __init__(
        self,
        num_classes:      int,
        class_weights:    Optional[torch.Tensor] = None,
        sample_rate:      int   = 32_000,
        n_mels:           int   = 128,
        hop_length:       int   = 320,
        d_model:          int   = 128,
        scale:            int   = 8,
        dilation_rates:   List[int] = None,
        kernel_size:      int   = 3,
        n_s4:             int   = 2,
        s4_d_state:       int   = 64,
        n_mamba:          int   = 2,
        mamba_d_state:    int   = 16,
        mamba_expand:     int   = 2,
        mamba_d_conv:     int   = 4,
        use_xlsr:         bool  = False,
        xlsr_model_name:  str   = "facebook/wav2vec2-large-xlsr-53",
        dropout:          float = 0.24,
        drop_path_rate:   float = 0.10,
        learning_rate:    float = 3e-4,
        weight_decay:     float = 0.012,
        warmup_epochs:    int   = 10,
        max_epochs:       int   = 100,
        mixup_alpha:      float = 0.20,
        noise_prob:       float = 0.50,
        noise_snr_min:    float = 20.0,
        noise_snr_max:    float = 40.0,
        gain_prob:        float = 0.70,
        focal_gamma:      float = 2.0,
        label_smoothing:  float = 0.001,
        cssd_alpha:       float = 1.0,
        cssd_temp:        float = 4.0,
        cssd_degrade_sr:  int   = 8_000,
        cssd_degrade_prob:float = 0.50,
        ema_momentum:     float = 0.999,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        if dilation_rates is None:
            dilation_rates = [2, 4, 8]
        n_res = len(dilation_rates)

        self.learning_rate   = learning_rate
        self.weight_decay    = weight_decay
        self.warmup_epochs   = warmup_epochs
        self.max_epochs      = max_epochs
        self.mixup_alpha     = mixup_alpha
        self.noise_prob      = noise_prob
        self.noise_snr_min   = noise_snr_min
        self.noise_snr_max   = noise_snr_max
        self.gain_prob       = gain_prob
        self.use_xlsr        = use_xlsr
        self.cssd_alpha      = cssd_alpha
        self.cssd_temp       = cssd_temp
        self.cssd_degrade_sr = cssd_degrade_sr
        self.cssd_degrade_prob = cssd_degrade_prob
        self.ema_momentum    = ema_momentum
        self.sample_rate     = sample_rate

        # ── Stream A: MultiScalePCEN + SpecAugment + projection ─────────
        self.pcen      = MultiScalePCEN(
            sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length,
        )
        self.spec_aug  = SpecAugment()
        self.stream_a  = nn.Sequential(
            nn.Conv1d(2 * n_mels, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        # ── Stream B: XLSR-53 CNN extractor (optional) ──────────────────
        _xlsr_dim = 512   # wav2vec2 CNN output dim
        if use_xlsr:
            self.xlsr_cnn   = XLSRCNNExtractor(xlsr_model_name)
            self.stream_b   = nn.Sequential(
                nn.Conv1d(_xlsr_dim, d_model, 1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
            )
            # Resample waveform from input SR to 16 kHz for XLSR
            self._resample_16k = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16_000
            )

        # ── Fusion gate (if both streams) ────────────────────────────────
        n_streams = 2 if use_xlsr else 1
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(n_streams * d_model, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        # ── SE-Res2 backbone (channel-first) ────────────────────────────
        dp_rates = [drop_path_rate * i / max(n_res - 1, 1) for i in range(n_res)]
        self.res_blocks = nn.ModuleList([
            _SERes2Block(
                channels    = d_model,
                scale       = scale,
                kernel_size = kernel_size,
                dilation    = dilation_rates[i],
                dropout     = dropout,
                drop_path   = dp_rates[i],
            )
            for i in range(n_res)
        ])
        self.mfa = _MFALayer(n_inputs=n_res + 1, channels=d_model)

        # ── S4D long-range blocks (sequence-last) ───────────────────────
        self.s4_blocks = nn.ModuleList([
            SaShiMiBlock(
                d_model   = d_model,
                d_state   = s4_d_state,
                dropout   = dropout,
                drop_path = drop_path_rate * 0.5,
            )
            for _ in range(n_s4)
        ])

        # ── BidirMamba selective global blocks (sequence-last) ──────────
        self.mamba_blocks = nn.ModuleList([
            BidirMambaBlock(
                d_model  = d_model,
                d_state  = mamba_d_state,
                expand   = mamba_expand,
                d_conv   = mamba_d_conv,
                drop_path= drop_path_rate * 0.5,
            )
            for _ in range(n_mamba)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── Attentive statistics pool ────────────────────────────────────
        self.pool = _AttentiveStatisticsPool(d_model)

        # ── Classifier head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # ── Loss ─────────────────────────────────────────────────────────
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.criterion = FocalLoss(
            gamma          = focal_gamma,
            class_weights  = self.class_weights,
            label_smoothing= label_smoothing,
        )

        # ── CSSD teacher (EMA copy, frozen) ─────────────────────────────
        self._cssd_enabled = cssd_alpha < 1.0
        if self._cssd_enabled:
            self.teacher = copy.deepcopy(self)
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.teacher._cssd_enabled = False   # prevent recursion

        # ── Metrics ──────────────────────────────────────────────────────
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_acc",
                    MulticlassAccuracy(num_classes=num_classes, average="macro"))
            setattr(self, f"{split}_f1",
                    MulticlassF1Score(num_classes=num_classes, average="macro"))
            setattr(self, f"{split}_prec",
                    MulticlassPrecision(num_classes=num_classes, average="macro"))
            setattr(self, f"{split}_mcc",
                    MulticlassMatthewsCorrCoef(num_classes=num_classes))
            setattr(self, f"{split}_auroc",
                    MulticlassAUROC(num_classes=num_classes))
        self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes)

    # ── Waveform augmentations ──────────────────────────────────────────

    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
        if self.mixup_alpha <= 0 or not self.training:
            return x, y, None
        lam   = float(
            torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
        )
        idx   = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1.0 - lam) * x[idx]
        return x_mix, y, (y[idx], lam)

    def _noise_augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.noise_prob <= 0:
            return x
        mask = torch.rand(x.size(0), device=x.device) < self.noise_prob
        if not mask.any():
            return x
        snr_db = torch.empty(x.size(0), device=x.device).uniform_(
            self.noise_snr_min, self.noise_snr_max
        )
        signal_rms = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-9).sqrt()
        noise      = torch.randn_like(x)
        noise_rms  = noise.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-9).sqrt()
        scale      = signal_rms / noise_rms / (10.0 ** (snr_db.unsqueeze(-1) / 20.0))
        x = torch.where(mask.unsqueeze(-1), x + scale * noise, x)
        return x

    def _gain_augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.gain_prob <= 0:
            return x
        mask  = torch.rand(x.size(0), device=x.device) < self.gain_prob
        gains = torch.empty(x.size(0), device=x.device).uniform_(0.6, 1.4)
        x = torch.where(mask.unsqueeze(-1), x * gains.unsqueeze(-1), x)
        return x

    # ── CSSD degradation ────────────────────────────────────────────────

    @torch.no_grad()
    def _degrade(self, x: torch.Tensor) -> torch.Tensor:
        """Resample waveform to a low SR and back — simulates bandwidth loss."""
        lo  = torchaudio.functional.resample(x, self.sample_rate, self.cssd_degrade_sr)
        return torchaudio.functional.resample(lo, self.cssd_degrade_sr, self.sample_rate)

    @torch.no_grad()
    def _update_teacher(self) -> None:
        m = self.ema_momentum
        for ps, pt in zip(self.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)

    # ── Forward pass ────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) float waveform at self.sample_rate Hz.
        Returns:
            logits (B, num_classes).
        """
        # ── Stream A ─────────────────────────────────────────────────────
        spec = self.pcen(x)                            # (B, 2, n_mels, T)
        if self.training:
            spec = self.spec_aug(spec)                 # (B, 2, n_mels, T)
        B, C, F, T = spec.shape
        feat_a = spec.view(B, C * F, T)                # (B, 2*n_mels, T)
        feat_a = self.stream_a(feat_a)                 # (B, d_model, T)

        # ── Stream B (optional XLSR CNN) ─────────────────────────────────
        if self.use_xlsr:
            x16  = self._resample_16k(x)               # (B, L_16k)
            cnn  = self.xlsr_cnn(x16)                  # (B, 512, T_xlsr)
            feat_b = F.interpolate(
                cnn, size=T, mode="linear", align_corners=False,
            )                                          # (B, 512, T)
            feat_b = self.stream_b(feat_b)             # (B, d_model, T)
            fused  = self.fusion_gate(
                torch.cat([feat_a, feat_b], dim=1)     # (B, 2*d, T)
            )
        else:
            fused = self.fusion_gate(feat_a)            # Conv1d(d→d) identity-like

        # ── SE-Res2 backbone ─────────────────────────────────────────────
        skips = [fused]
        h = fused
        for blk in self.res_blocks:
            h = blk(h)
            skips.append(h)
        h = self.mfa(*skips)                           # (B, d_model, T)

        # ── SSM blocks (sequence-last) ───────────────────────────────────
        h = h.permute(0, 2, 1)                         # (B, T, d_model)
        for blk in self.s4_blocks:
            h = blk(h)
        for blk in self.mamba_blocks:
            h = blk(h)
        h = self.final_norm(h)
        h = h.permute(0, 2, 1)                         # (B, d_model, T)

        # ── Pool + classify ──────────────────────────────────────────────
        h = self.pool(h)                               # (B, 2*d_model)
        return self.classifier(h)                      # (B, num_classes)

    # ── Shared step ─────────────────────────────────────────────────────

    def _compute_loss(self, logits, y, mixup_info=None):
        if mixup_info is None:
            return self.criterion(logits, y)
        y_b, lam = mixup_info
        return lam * self.criterion(logits, y) + (1.0 - lam) * self.criterion(logits, y_b)

    def _shared_step(self, batch, split: str):
        x, y = batch

        if split == "train":
            x = self._gain_augment(self._noise_augment(x))
            x, y, mixup_info = self._mixup(x, y)
        else:
            mixup_info = None

        # ── CSSD (optional, train only) ──────────────────────────────────
        if split == "train" and self._cssd_enabled and \
                torch.rand(1).item() < self.cssd_degrade_prob:
            with torch.no_grad():
                x_deg    = self._degrade(x)
                t_logits = self.teacher(x_deg)
                soft_tgt = F.softmax(t_logits / self.cssd_temp, dim=-1)
            logits   = self(x)
            ce_loss  = self._compute_loss(logits, y, mixup_info)
            kl_loss  = F.kl_div(
                F.log_softmax(logits / self.cssd_temp, dim=-1),
                soft_tgt, reduction="batchmean",
            ) * (self.cssd_temp ** 2)
            loss = self.cssd_alpha * ce_loss + (1.0 - self.cssd_alpha) * kl_loss
        else:
            logits = self(x)
            loss   = self._compute_loss(logits, y, mixup_info)

        preds = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)

        getattr(self, f"{split}_acc")(preds, y)
        getattr(self, f"{split}_f1")(preds, y)
        getattr(self, f"{split}_prec")(preds, y)
        getattr(self, f"{split}_mcc")(preds, y)
        getattr(self, f"{split}_auroc")(probs, y)
        if split == "test":
            self.test_cm(preds, y)

        self.log(f"{split}/loss",  loss,                       prog_bar=True,
                 on_step=(split=="train"), on_epoch=True, sync_dist=True)
        self.log(f"{split}/f1",    getattr(self, f"{split}_f1"),
                 prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{split}/prec",  getattr(self, f"{split}_prec"),
                 on_epoch=True, sync_dist=True)
        self.log(f"{split}/acc",   getattr(self, f"{split}_acc"),
                 on_epoch=True, sync_dist=True)
        self.log(f"{split}/mcc",   getattr(self, f"{split}_mcc"),
                 on_epoch=True, sync_dist=True)
        self.log(f"{split}/auroc", getattr(self, f"{split}_auroc"),
                 on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        if self._cssd_enabled:
            self._update_teacher()
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print("\nConfusion matrix (rows=true, cols=pred):\n", cm.cpu().numpy())
        self.test_cm.reset()

    # ── Optimiser + scheduler ────────────────────────────────────────────

    def configure_optimizers(self):
        # SSM params (S4D poles, Mamba dt_proj) get 0.1× LR
        ssm_params, base_params = [], []
        ssm_keys = {"log_a_real", "log_a_imag", "log_dt", "dt_proj"}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ssm_keys):
                ssm_params.append(p)
            else:
                base_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": self.learning_rate},
                {"params": ssm_params,  "lr": self.learning_rate * 0.1},
            ],
            weight_decay = self.weight_decay,
        )

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / max(
                self.max_epochs - self.warmup_epochs, 1
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler,
                                                          "interval": "epoch"}}


# ═══════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    use_xlsr = "--xlsr" in sys.argv

    model = HydroFusion(
        num_classes    = 3,
        d_model        = 128,
        n_s4           = 2,
        n_mamba        = 2,
        use_xlsr       = use_xlsr,
        max_epochs     = 100,
    )
    model.eval()

    x = torch.randn(2, 32_000)
    with torch.no_grad():
        out = model(x)

    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHydroFusion (use_xlsr={use_xlsr})")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable        : {trainable:,}")
    print(f"  Output shape     : {list(out.shape)}")
    print(f"  CSSD enabled     : {model._cssd_enabled}")
    print()
