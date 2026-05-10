"""
HydroPrecise — High-Precision Multi-Stream UATR Classifier
==========================================================

Designed for macro-precision (low false-positive rate) on a 4-class
underwater vessel dataset at 5120 Hz, 1 s clips.

Three parallel branches share a cross-attention fuser and a margin head
with an abstention logit:

  A. LearnableGabor → PCEN → stride-4 stem → 2× SE-Res2 blocks
     (captures transients, cavitation, blade-rate bursts)
  B. CQT (fmin=20 Hz, 84 bins, 12 bpo → 2560 Hz = Nyquist)
     → log → InstanceNorm → 2D CNN → collapse freq
     (captures shaft harmonics and continuous tonal lines)
  C. DEMON envelope (≥800 Hz bandpass → square → mel+PCEN)
     → 1D CNN → GRU
     (captures blade-rate amplitude modulation)

All three are resampled to a common T, concatenated, projected,
passed through a 2-head cross-attention block, then pooled with
AttentiveStatisticsPool.  The head outputs (num_classes + 1) logits:
the last one is an abstention logit used by the Deep-Gamblers
auxiliary loss — dropped at inference.

Primary loss: LargeMarginFocalLoss (margin = 0.3, γ = 2.0, smoothing = 0.05;
              grid-search winners, val_macro_precision = 0.7269).
Auxiliary:    Deep-Gamblers:  -log(p_y + o · p_abstain), weight 0.1.

Targets ~2 M trainable parameters.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TA
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassRecall, MulticlassAUROC, MulticlassConfusionMatrix,
    MulticlassMatthewsCorrCoef,
)

from models.hydro_catfish   import LearnableGaborFilterbank, SpecAugment1D
from models.hydro_conformer import TrainablePCEN, FocalLoss
from models.hydro_net       import DEMONChannel
from models.hydro_fusion    import _SERes2Block, _AttentiveStatisticsPool
from processing.losses      import LargeMarginFocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  Waveform augmentations (applied inside training_step)
# ═══════════════════════════════════════════════════════════════════════

class _WaveformAug(nn.Module):
    """Gaussian noise at 15-30 dB SNR + random ±30 % gain.  Train-only."""

    def __init__(self, noise_prob: float = 0.5,
                 noise_snr_min: float = 15.0, noise_snr_max: float = 30.0,
                 gain_prob: float = 0.5, gain_range: float = 0.3):
        super().__init__()
        self.noise_prob, self.noise_snr_min, self.noise_snr_max = noise_prob, noise_snr_min, noise_snr_max
        self.gain_prob,  self.gain_range = gain_prob, gain_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1).item() < self.noise_prob:
            snr  = self.noise_snr_min + (self.noise_snr_max - self.noise_snr_min) * torch.rand(1).item()
            sig_pow   = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12)
            noise_pow = sig_pow / (10.0 ** (snr / 10.0))
            x = x + torch.randn_like(x) * noise_pow.sqrt()
        if torch.rand(1).item() < self.gain_prob:
            g = 1.0 + (2.0 * torch.rand(1, device=x.device).item() - 1.0) * self.gain_range
            x = x * g
        return x


# ═══════════════════════════════════════════════════════════════════════
#  CQT branch — handles nnAudio optional import with mel fallback
# ═══════════════════════════════════════════════════════════════════════

class _CQTFrontend(nn.Module):
    """CQT fmin=20 Hz → fmax≈Nyquist. Falls back to MelSpectrogram if nnAudio missing."""

    def __init__(self, sample_rate: int = 5_120, n_bins: int = 84,
                 bins_per_octave: int = 12, hop_length: int = 64, fmin: float = 20.0):
        super().__init__()
        self.n_bins = n_bins
        try:
            from nnAudio.Spectrogram import CQT1992v2
            self.cqt = CQT1992v2(
                sr=sample_rate, hop_length=hop_length, fmin=fmin,
                n_bins=n_bins, bins_per_octave=bins_per_octave,
                output_format="Magnitude", verbose=False,
            )
            self._use_cqt = True
        except Exception:
            self.cqt = TA.MelSpectrogram(
                sample_rate=sample_rate, n_fft=512, hop_length=hop_length,
                n_mels=n_bins, f_min=fmin, f_max=sample_rate / 2.0, power=1.0,
            )
            self._use_cqt = False
        self.norm = nn.InstanceNorm1d(n_bins, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.cqt(x)                # (B, F, T)
        spec = torch.log1p(spec.abs())
        return self.norm(spec)            # (B, F, T)


class _CQT2DBackbone(nn.Module):
    """Two residual 2D conv stages → collapse freq into channels → (B, C_out, T)."""

    def __init__(self, in_freq: int, base_ch: int = 32, out_ch: int = 128):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1,       base_ch,     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.GELU(),
            nn.Conv2d(base_ch, base_ch,     kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_ch,     base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2), nn.GELU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2), nn.GELU(),
        )
        freq_after = math.ceil(in_freq / 4)
        self.project = nn.Conv1d(base_ch * 2 * freq_after, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T) → (B, 1, F, T)
        x = x.unsqueeze(1)
        x = self.stage2(self.stage1(x))
        B, C, Fm, T = x.shape
        x = x.reshape(B, C * Fm, T)
        return self.project(x)            # (B, out_ch, T)


# ═══════════════════════════════════════════════════════════════════════
#  Branch wrappers
# ═══════════════════════════════════════════════════════════════════════

class _GaborBranch(nn.Module):
    def __init__(self, sample_rate: int, n_filters: int = 64, out_ch: int = 128,
                 kernel_size: int = 257, downsample: int = 4, dropout: float = 0.1):
        super().__init__()
        self.filterbank = LearnableGaborFilterbank(
            n_filters=n_filters, kernel_size=kernel_size, sample_rate=sample_rate,
        )
        self.spec_aug = SpecAugment1D(n_freq_masks=2, freq_mask_max=6,
                                      n_time_masks=2, time_mask_max=80)
        self.stem = nn.Sequential(
            nn.Conv1d(n_filters, out_ch, kernel_size=7, stride=downsample,
                      padding=3, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=downsample,
                      padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        # out_ch divisible by scale=8 required
        self.blocks = nn.Sequential(
            _SERes2Block(out_ch, scale=8, kernel_size=3, dilation=2, dropout=dropout),
            _SERes2Block(out_ch, scale=8, kernel_size=3, dilation=4, dropout=dropout),
        )

    def forward(self, waveform: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.filterbank(waveform)             # (B, n_filters, T)
        if training:
            x = self.spec_aug(x)
        x = self.stem(x)                          # (B, out_ch, T/ds²)
        return self.blocks(x)                     # (B, out_ch, T')


class _CQTBranch(nn.Module):
    def __init__(self, sample_rate: int, n_bins: int = 84, bins_per_octave: int = 12,
                 hop_length: int = 64, out_ch: int = 128):
        super().__init__()
        self.front = _CQTFrontend(
            sample_rate=sample_rate, n_bins=n_bins,
            bins_per_octave=bins_per_octave, hop_length=hop_length,
        )
        self.back = _CQT2DBackbone(in_freq=n_bins, base_ch=32, out_ch=out_ch)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.back(self.front(waveform))    # (B, out_ch, T)


class _DEMONBranch(nn.Module):
    def __init__(self, sample_rate: int, hop_length: int = 64,
                 out_ch: int = 64, gru_hidden: int = 64,
                 n_fft: int = 2048, mod_f_min: float = 0.0,
                 mod_f_max: float = 50.0):
        super().__init__()
        self.demon = DEMONChannel(
            sample_rate=sample_rate, hop_length=hop_length,
            n_fft=n_fft, f_cav=800.0,
            mod_f_min=mod_f_min, mod_f_max=mod_f_max,
        )
        in_dim = self.demon.n_bins
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim,    out_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.gru = nn.GRU(input_size=out_ch, hidden_size=gru_hidden,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.out_ch = gru_hidden * 2

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.demon(waveform)                  # (B, n_bins, T)
        x = self.conv(x)                          # (B, out_ch, T)
        x = x.transpose(1, 2)                     # (B, T, out_ch)
        x, _ = self.gru(x)                        # (B, T, 2*gru_hidden)
        return x.transpose(1, 2)                  # (B, 2*gru_hidden, T)


# ═══════════════════════════════════════════════════════════════════════
#  Cross-attention fusion block
# ═══════════════════════════════════════════════════════════════════════

class _CrossAttnFuser(nn.Module):
    """Simple post-norm MHSA block on concatenated branch features."""

    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) sequence-last convention for attention
        h, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  HydroPrecise LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroPrecise(pl.LightningModule):
    """
    Three-branch high-precision UATR classifier.

    The output layer produces ``num_classes + 1`` logits; the last one is an
    abstention logit used only by the Deep-Gamblers auxiliary loss.  Inference
    uses ``logits[:, :num_classes]`` softmax — the abstention channel is never
    returned as a prediction.
    """

    def __init__(
        self,
        num_classes:     int = 4,
        class_weights:   Optional[list] = None,
        sample_rate:     int = 5_120,
        # ── Branch widths ────────────────────────────────────────────────
        gabor_n_filters: int = 64,
        gabor_kernel:    int = 257,
        gabor_ch:        int = 128,
        cqt_n_bins:      int = 84,
        cqt_bpo:         int = 12,
        cqt_hop:         int = 64,
        cqt_ch:          int = 128,
        demon_hop:       int = 64,
        demon_ch:        int = 64,
        demon_n_fft:     int = 2048,
        demon_mod_f_min: float = 0.0,
        demon_mod_f_max: float = 50.0,
        fusion_T:        int = 64,
        fusion_dim:      int = 192,
        n_heads:         int = 2,
        dropout:         float = 0.15,
        # ── Loss ─────────────────────────────────────────────────────────
        # Defaults below are the winners of the 4×2×2×2 grid search over
        # (lmf_margin, lmf_gamma, label_smoothing, gambler_weight).
        # Best: m=0.30 g=2.0 s=0.05 gw=0.1 → val_macro_precision = 0.7269
        # See lightning_logs/grid_precise/summary.csv.
        loss:            str   = "lmf",             # "lmf" | "focal"
        lmf_gamma:       float = 2.0,
        lmf_margin:      float = 0.3,
        label_smoothing: float = 0.05,
        gambler_o:       float = 0.3,
        gambler_weight:  float = 0.1,
        # ── Augmentation ─────────────────────────────────────────────────
        noise_prob:      float = 0.5,
        noise_snr_min:   float = 15.0,
        noise_snr_max:   float = 30.0,
        gain_prob:       float = 0.5,
        gain_range:      float = 0.3,
        # ── Optimiser ────────────────────────────────────────────────────
        learning_rate:   float = 3e-4,
        weight_decay:    float = 1e-2,
        warmup_epochs:   int   = 10,
        max_epochs:      int   = 100,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.num_classes = num_classes
        self.fusion_T    = fusion_T
        self.gambler_o      = gambler_o
        self.gambler_weight = gambler_weight

        # ── Waveform augmentation (train-only) ───────────────────────────
        self.wave_aug = _WaveformAug(
            noise_prob=noise_prob, noise_snr_min=noise_snr_min,
            noise_snr_max=noise_snr_max,
            gain_prob=gain_prob, gain_range=gain_range,
        )

        # ── Branches ─────────────────────────────────────────────────────
        self.branch_a = _GaborBranch(
            sample_rate=sample_rate, n_filters=gabor_n_filters,
            out_ch=gabor_ch, kernel_size=gabor_kernel, dropout=dropout,
        )
        self.branch_b = _CQTBranch(
            sample_rate=sample_rate, n_bins=cqt_n_bins,
            bins_per_octave=cqt_bpo, hop_length=cqt_hop, out_ch=cqt_ch,
        )
        self.branch_c = _DEMONBranch(
            sample_rate=sample_rate,
            hop_length=demon_hop, out_ch=demon_ch,
            n_fft=demon_n_fft, mod_f_min=demon_mod_f_min, mod_f_max=demon_mod_f_max,
        )

        # ── Fusion ───────────────────────────────────────────────────────
        cat_ch = gabor_ch + cqt_ch + self.branch_c.out_ch
        self.fuse_proj = nn.Sequential(
            nn.Conv1d(cat_ch, fusion_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(fusion_dim), nn.GELU(),
        )
        self.cross_attn = _CrossAttnFuser(
            d_model=fusion_dim, n_heads=n_heads, dropout=dropout,
        )
        self.pool = _AttentiveStatisticsPool(fusion_dim)

        # ── Head (includes +1 abstention logit) ──────────────────────────
        self.head = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes + 1),
        )

        # ── Primary classification loss (only the first num_classes cols) ─
        if loss == "lmf":
            self.criterion = LargeMarginFocalLoss(
                num_classes=num_classes, alpha=class_weights,
                gamma=lmf_gamma, margin=lmf_margin,
                label_smoothing=label_smoothing,
            )
        else:
            self.criterion = FocalLoss(
                class_weights=class_weights,
                gamma=lmf_gamma, label_smoothing=label_smoothing,
            )

        # ── Metrics ──────────────────────────────────────────────────────
        m_macro = dict(num_classes=num_classes, average="macro")
        m_micro = dict(num_classes=num_classes, average="micro")
        self.train_acc           = MulticlassAccuracy(**m_macro)
        self.val_acc             = MulticlassAccuracy(**m_macro)
        self.val_f1              = MulticlassF1Score(**m_macro)
        self.val_recall          = MulticlassRecall(**m_macro)
        self.val_precision_macro = MulticlassPrecision(**m_macro)
        self.val_precision_micro = MulticlassPrecision(**m_micro)
        self.val_precision_per   = MulticlassPrecision(num_classes=num_classes, average=None)
        self.val_mcc             = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.val_auroc           = MulticlassAUROC(num_classes=num_classes)
        self.test_acc            = MulticlassAccuracy(**m_macro)
        self.test_f1             = MulticlassF1Score(**m_macro)
        self.test_precision      = MulticlassPrecision(**m_macro)
        self.test_precision_micro = MulticlassPrecision(**m_micro)
        self.test_recall         = MulticlassRecall(**m_macro)
        self.test_mcc            = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_auroc          = MulticlassAUROC(num_classes=num_classes)
        self.test_cm             = MulticlassConfusionMatrix(num_classes=num_classes)

    # ── Forward ──────────────────────────────────────────────────────────

    def _features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return the (B, 2*fusion_dim) post-pool embedding (pre-head)."""
        a = self.branch_a(waveform, self.training)          # (B, Ca, Ta)
        b = self.branch_b(waveform)                         # (B, Cb, Tb)
        c = self.branch_c(waveform)                         # (B, Cc, Tc)

        Tf = self.fusion_T
        a = F.adaptive_avg_pool1d(a, Tf)
        b = F.adaptive_avg_pool1d(b, Tf)
        c = F.adaptive_avg_pool1d(c, Tf)

        z = torch.cat([a, b, c], dim=1)                     # (B, Ca+Cb+Cc, Tf)
        z = self.fuse_proj(z)                               # (B, D, Tf)
        z = z.transpose(1, 2)                               # (B, Tf, D)
        z = self.cross_attn(z)                              # (B, Tf, D)
        z = z.transpose(1, 2)                               # (B, D, Tf)
        return self.pool(z)                                 # (B, 2D)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) float32 mono audio at self.hparams.sample_rate
        Returns:
            logits: (B, num_classes + 1) — last column is the abstention logit.
        """
        return self.head(self._features(waveform))          # (B, num_classes + 1)

    # ── Losses ───────────────────────────────────────────────────────────

    def _split(self, logits: torch.Tensor):
        """Separate class logits from abstention logit; return (class_logits, abstain_prob)."""
        class_logits = logits[:, :self.num_classes]
        # softmax over the full (C+1) distribution to get p_abstain properly normalised
        full = F.softmax(logits, dim=-1)
        p_abstain = full[:, -1]
        return class_logits, p_abstain, full

    def _gambler_loss(self, full: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Deep-Gamblers auxiliary: -log(p_y + o · p_abstain)."""
        p_y       = full.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_abstain = full[:, -1]
        return -torch.log(p_y + self.gambler_o * p_abstain + 1e-8).mean()

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        class_logits, _, full = self._split(logits)
        primary  = self.criterion(class_logits, y)
        if self.gambler_weight > 0.0:
            aux = self._gambler_loss(full, y)
            return primary + self.gambler_weight * aux
        return primary

    # ── Lightning steps ──────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.wave_aug(x)
        logits = self(x)
        loss   = self._compute_loss(logits, y)
        class_logits = logits[:, :self.num_classes]
        self.train_acc(class_logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss   = self._compute_loss(logits, y)
        class_logits = logits[:, :self.num_classes]
        probs = F.softmax(class_logits, dim=-1)

        self.val_acc(class_logits, y)
        self.val_f1(class_logits, y)
        self.val_recall(class_logits, y)
        self.val_precision_macro(class_logits, y)
        self.val_precision_micro(class_logits, y)
        self.val_precision_per(class_logits, y)
        self.val_mcc(class_logits, y)
        self.val_auroc(probs, y)

        self.log("val/loss",            loss,                     on_epoch=True, prog_bar=True)
        self.log("val/acc",             self.val_acc,             on_epoch=True, prog_bar=True)
        self.log("val/f1",              self.val_f1,              on_epoch=True, prog_bar=True)
        self.log("val/recall",          self.val_recall,          on_epoch=True)
        self.log("val/macro_precision", self.val_precision_macro, on_epoch=True, prog_bar=True)
        self.log("val/micro_precision", self.val_precision_micro, on_epoch=True, prog_bar=True)
        self.log("val/mcc",             self.val_mcc,             on_epoch=True)
        self.log("val/auroc",           self.val_auroc,           on_epoch=True)

    def on_validation_epoch_end(self):
        per = self.val_precision_per.compute()
        for i, v in enumerate(per):
            self.log(f"val/precision_c{i}", v, prog_bar=False)
        self.val_precision_per.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss   = self._compute_loss(logits, y)
        class_logits = logits[:, :self.num_classes]
        probs = F.softmax(class_logits, dim=-1)

        self.test_acc(class_logits, y)
        self.test_f1(class_logits, y)
        self.test_precision(class_logits, y)
        self.test_precision_micro(class_logits, y)
        self.test_recall(class_logits, y)
        self.test_mcc(class_logits, y)
        self.test_auroc(probs, y)
        self.test_cm(class_logits, y)

        self.log("test/loss",            loss,                on_epoch=True)
        self.log("test/acc",             self.test_acc,       on_epoch=True)
        self.log("test/f1",              self.test_f1,        on_epoch=True)
        self.log("test/macro_precision", self.test_precision, on_epoch=True)
        self.log("test/micro_precision", self.test_precision_micro, on_epoch=True)
        self.log("test/recall",          self.test_recall,    on_epoch=True)
        self.log("test/mcc",             self.test_mcc,       on_epoch=True)
        self.log("test/auroc",           self.test_auroc,     on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix:\n{cm.cpu().numpy()}")
        self.test_cm.reset()

    # ── Optimiser ────────────────────────────────────────────────────────

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

        def lr_lambda(epoch):
            wu, total = self.hparams.warmup_epochs, self.hparams.max_epochs
            if epoch < wu:
                return (epoch + 1) / max(wu, 1)
            p = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
