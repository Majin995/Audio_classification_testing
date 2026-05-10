"""
SuperModel1D — Multi-Stream Hybrid 1D Raw-Waveform USW Acoustic Classifier
===========================================================================

Architecture overview
---------------------
Takes a raw 1D waveform (B, T=5120) @ 5120 Hz and computes five parallel
streams entirely in-graph (no external feature extraction):

  Stream G  — Learnable Gabor filterbank (log-magnitude, per-sample rate)
  Stream S  — EnhancedFrontEnd: wide-mel + narrow-mel + Gammatone + DEMON,
               all PCEN-normalised; CepstralLifter → CNNInput compression
  Stream L  — LofarFrontend: bandlimited log-power spec → fixed 32×256 image
  Stream F  — SubBandEnvelope: 9 scalar DSP statistics (appended at head)
  Onset     — SpectralFluxOnset on Gabor envelope → binary transient patch mask

Backbone (channel-first → sequence-last transition mirrored from HydroFusion):
  Fusion gate Conv1d
  → 3× _SERes2Block (Res2Net + SE, dilations 2/4/8)
  → _MFALayer (multi-scale aggregation)
  → [permute (B,D,T)→(B,T,D)]
  → n_dart× BAHTBlock (BoundaryAwareAttention with onset mask + SwiGLUFFN)
  → n_s4× SaShiMiBlock (S4D, long-range tonal tracking)
  → n_mamba× BidirMambaBlock (selective bidirectional context)
  → LayerNorm
  → [permute (B,T,D)→(B,D,T)]
  → _AttentiveStatisticsPool → concat(Stream F) → classifier

Loss / training
---------------
  FocalLoss(class_weights, γ=2, label_smoothing=0.05)
  + optional CSSD (EMA teacher + cross-SR degradation to 2048 Hz)
  Augmentations: mixup, Gaussian noise @SNR 20–40 dB, random gain ±40%,
                 SpecAugment1D on Stream G

Design notes
------------
  * S4D poles (log_a_real, log_a_imag, log_dt) and Mamba dt_proj receive a
    0.1× LR to prevent instability — see configure_optimizers.
  * SpecAugment1D mutates in-place; onset mask is computed BEFORE augmentation.
  * PCEN is NOT applied after the Gabor filterbank (Gabor already log-compresses
    via log1p; double-compressing flattens dynamic range — risk #2 in the plan).
  * cssd_degrade_sr defaults to 2048 (not 8000) because 5120 Hz source Nyquist
    is 2560 Hz; 8000 Hz would upsample, not degrade.
  * use_stream_s and use_stream_l flags allow ablation studies via Optuna.
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
)

from models.hydro_conformer    import FocalLoss, DropPath          # noqa: F401 (DropPath used by sub-modules)
from models.hydro_catfish      import LearnableGaborFilterbank, SpecAugment1D
from models.hydro_net          import (
    EnhancedFrontEnd,
    CepstralLifter,
    SubBandEnvelope,
    CNNInput,
)
from models.hydro_lofar_resnet import LofarFrontend
from models.hydro_bahtnet      import BAHTBlock, SpectralFluxOnset
from models.hydro_fusion       import (
    _SERes2Block,
    _MFALayer,
    _AttentiveStatisticsPool,
)
from models.hydro_s4           import SaShiMiBlock
from models.hydro_ssamba       import BidirMambaBlock

try:
    import optuna as _optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  SuperModel1D
# ═══════════════════════════════════════════════════════════════════════

class SuperModel1D(pl.LightningModule):
    """
    Unified 1D raw-waveform USW classifier — single cohesive graph, no ensembling.

    All spectrogram representations are computed in-graph from the input waveform.

    Args
    ----
    num_classes    : Number of target vessel/event classes.
    class_weights  : Per-class inverse-frequency weights for FocalLoss.
    sample_rate    : Input sample rate in Hz (default 5120).

    -- Stream G (raw 1D, Gabor) --
    gabor_n_filters: Number of learnable Gabor filters.
    gabor_kernel   : Gabor filter length in samples (must be odd).
    g_ch           : Channel width after the strided stem Conv1d.
    stem_stride    : Stride of the Gabor stem (controls T_shared; 4 → T=1280, 8 → T=640).

    -- Stream S (EnhancedFrontEnd spectrogram) --
    spec_n_mels    : Mel / Gammatone bands for EnhancedFrontEnd.
    spec_hop       : STFT hop in samples for EnhancedFrontEnd.
    wb_n_fft       : Wide n_fft (should be <= T//2 for a good time resolution).
    nb_n_fft       : Narrow n_fft (longer window for shaft harmonics).
    s_ch           : Channel width after CNNInput projection.
    ceps_low_q     : CepstralLifter lower quefrency limit.
    ceps_high_q    : CepstralLifter upper quefrency limit.
    use_stream_s   : Toggle Stream S (for ablation via Optuna).

    -- Stream L (LofarFrontend) --
    lofar_n_fft    : STFT window for LOFAR spectrogram.
    lofar_hop      : STFT hop for LOFAR spectrogram.
    lofar_time_bins: Fixed time dimension of LOFAR image.
    lofar_freq_bins: Fixed frequency dimension of LOFAR image (used as input channels).
    l_ch           : Channel width after LOFAR frequency projection.
    use_stream_l   : Toggle Stream L (for ablation via Optuna).

    -- Backbone --
    d_model        : Feature channel width throughout the backbone.
    scale          : Res2Net split factor (must divide d_model).
    dilation_rates : Per-SERes2Block dilation (one per block); default [2,4,8].
    n_dart         : Number of BAHTBlocks (onset-biased attention).
    dart_heads     : Attention heads per BAHTBlock.
    dart_dropout   : Dropout inside BAHTBlock.
    onset_patch    : SpectralFluxOnset patch size.
    n_s4           : Number of SaShiMi S4D blocks.
    s4_d_state     : S4D state dimension.
    n_mamba        : Number of BidirMamba blocks.
    mamba_d_state  : Mamba state dimension.
    mamba_expand   : Mamba inner-dim expansion factor.
    mamba_d_conv   : Mamba causal conv kernel size.
    drop_path_rate : Max stochastic-depth probability (ramped linearly).
    dropout        : Dropout inside BAHTBlock and classifier head.

    -- Training schedule --
    learning_rate  : AdamW base LR.
    weight_decay   : AdamW weight decay.
    warmup_epochs  : Linear LR warmup.
    max_epochs     : Total epochs (for cosine decay).

    -- Augmentation --
    mixup_alpha    : Beta(α,α) Mixup strength (0 = off).
    noise_prob     : Probability of additive Gaussian noise augmentation.
    noise_snr_min  : Min SNR (dB) for noise augmentation.
    noise_snr_max  : Max SNR (dB) for noise augmentation.
    gain_prob      : Probability of random gain augmentation.

    -- Loss --
    focal_gamma    : Focal loss γ.
    label_smoothing: Label smoothing ε.

    -- CSSD (Cross-Sampling-Rate Self-Distillation) --
    cssd_alpha     : CE weight; 1.0 = pure CE (CSSD disabled).
    cssd_temp      : Distillation temperature.
    cssd_degrade_sr: Target SR for bandwidth degradation (< sample_rate).
    cssd_degrade_prob: Probability of applying CSSD per batch.
    ema_momentum   : EMA teacher update momentum.

    -- Optuna --
    optuna_trial   : Optuna Trial object for intermediate pruning (optional).
    """

    def __init__(
        self,
        num_classes:      int,
        class_weights:    Optional[list] = None,
        sample_rate:      int   = 5_120,
        # ── Stream G ────────────────────────────────────────────────────
        gabor_n_filters:  int   = 64,
        gabor_kernel:     int   = 257,
        g_ch:             int   = 48,
        stem_stride:      int   = 4,
        # ── Stream S ────────────────────────────────────────────────────
        spec_n_mels:      int   = 32,
        spec_hop:         int   = 51,
        wb_n_fft:         int   = 256,
        nb_n_fft:         int   = 1_024,
        s_ch:             int   = 48,
        ceps_low_q:       int   = 3,
        ceps_high_q:      int   = 15,
        use_stream_s:     bool  = True,
        # ── Stream L ────────────────────────────────────────────────────
        lofar_n_fft:      int   = 1_024,
        lofar_hop:        int   = 51,
        lofar_time_bins:  int   = 32,
        lofar_freq_bins:  int   = 256,
        l_ch:             int   = 32,
        use_stream_l:     bool  = True,
        # ── Backbone ────────────────────────────────────────────────────
        d_model:          int         = 128,
        scale:            int         = 8,
        dilation_rates:   Optional[List[int]] = None,
        n_dart:           int   = 2,
        dart_heads:       int   = 4,
        dart_dropout:     float = 0.1,
        onset_patch:      int   = 4,
        n_s4:             int   = 2,
        s4_d_state:       int   = 64,
        n_mamba:          int   = 2,
        mamba_d_state:    int   = 16,
        mamba_expand:     int   = 2,
        mamba_d_conv:     int   = 4,
        drop_path_rate:   float = 0.10,
        dropout:          float = 0.24,
        # ── Training ────────────────────────────────────────────────────
        learning_rate:    float = 3e-4,
        weight_decay:     float = 0.012,
        warmup_epochs:    int   = 10,
        max_epochs:       int   = 100,
        # ── Augmentation ────────────────────────────────────────────────
        mixup_alpha:      float = 0.3,
        noise_prob:       float = 0.5,
        noise_snr_min:    float = 20.0,
        noise_snr_max:    float = 40.0,
        gain_prob:        float = 0.7,
        # ── Loss ────────────────────────────────────────────────────────
        focal_gamma:      float = 2.0,
        label_smoothing:  float = 0.05,
        # ── CSSD ────────────────────────────────────────────────────────
        cssd_alpha:       float = 1.0,
        cssd_temp:        float = 4.0,
        cssd_degrade_sr:  int   = 2_048,
        cssd_degrade_prob:float = 0.5,
        ema_momentum:     float = 0.999,
        # ── Optuna ──────────────────────────────────────────────────────
        optuna_trial: Optional[object] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights", "optuna_trial"])

        if dilation_rates is None:
            dilation_rates = [2, 4, 8]
        n_res = len(dilation_rates)

        # Store scalars needed at runtime (not accessible via hparams after deepcopy)
        self.learning_rate    = learning_rate
        self.weight_decay     = weight_decay
        self.warmup_epochs    = warmup_epochs
        self.max_epochs       = max_epochs
        self.sample_rate      = sample_rate
        self.mixup_alpha      = mixup_alpha
        self.noise_prob       = noise_prob
        self.noise_snr_min    = noise_snr_min
        self.noise_snr_max    = noise_snr_max
        self.gain_prob        = gain_prob
        self.cssd_alpha       = cssd_alpha
        self.cssd_temp        = cssd_temp
        self.cssd_degrade_sr  = cssd_degrade_sr
        self.cssd_degrade_prob= cssd_degrade_prob
        self.ema_momentum     = ema_momentum
        self.use_stream_s     = use_stream_s
        self.use_stream_l     = use_stream_l
        self.optuna_trial     = optuna_trial

        # ── Stream G: Gabor front-end ─────────────────────────────────
        self.gabor        = LearnableGaborFilterbank(
            n_filters=gabor_n_filters,
            kernel_size=gabor_kernel,
            sample_rate=sample_rate,
        )
        self.onset_detect = SpectralFluxOnset(patch_size=onset_patch)
        self.spec_aug_1d  = SpecAugment1D(
            n_freq_masks=2, freq_mask_max=8,
            n_time_masks=2, time_mask_max=20,
        )
        # Strided stem: downsamples T to T_shared = T // stem_stride
        self.gabor_stem   = nn.Sequential(
            nn.Conv1d(gabor_n_filters, g_ch, kernel_size=7,
                      stride=stem_stride, padding=3, bias=False),
            nn.BatchNorm1d(g_ch),
            nn.GELU(),
        )

        # ── Stream S: EnhancedFrontEnd + CepstralLifter + CNNInput ───
        if use_stream_s:
            self.enhanced_fe = EnhancedFrontEnd(
                sample_rate=sample_rate,
                n_mels=spec_n_mels,
                hop_length=spec_hop,
                wb_n_fft=wb_n_fft,
                nb_n_fft=nb_n_fft,
            )
            self.cepstral    = CepstralLifter(
                n_mels=spec_n_mels,
                low_q=ceps_low_q,
                high_q=ceps_high_q,
            )
            self.cnn_input   = CNNInput(channels=s_ch, in_channels=4)

        # ── Stream L: LofarFrontend ───────────────────────────────────
        if use_stream_l:
            self.lofar_fe   = LofarFrontend(
                sample_rate=sample_rate,
                n_fft=lofar_n_fft,
                hop_length=lofar_hop,
                time_bins=lofar_time_bins,
                freq_bins=lofar_freq_bins,
            )
            # Project freq_bins channels → l_ch, keeping time_bins temporal steps
            self.lofar_proj = nn.Sequential(
                nn.Conv1d(lofar_freq_bins, l_ch, 1, bias=False),
                nn.BatchNorm1d(l_ch),
                nn.GELU(),
            )

        # ── Stream F: SubBandEnvelope (scalar, no temporal axis) ─────
        self.subbandenv   = SubBandEnvelope(sample_rate=sample_rate, n_fft=512)

        # ── Fusion gate ───────────────────────────────────────────────
        total_ch = g_ch
        if use_stream_s:
            total_ch += s_ch
        if use_stream_l:
            total_ch += l_ch
        self.fusion_gate  = nn.Sequential(
            nn.Conv1d(total_ch, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # ── SE-Res2Block backbone ─────────────────────────────────────
        dp_rates  = [drop_path_rate * i / max(n_res - 1, 1) for i in range(n_res)]
        self.res_blocks = nn.ModuleList([
            _SERes2Block(
                channels=d_model,
                scale=scale,
                dilation=dilation_rates[i],
                dropout=dropout,
                drop_path=dp_rates[i],
            )
            for i in range(n_res)
        ])
        # n_res blocks + initial fused features = n_res+1 inputs
        self.mfa          = _MFALayer(n_inputs=n_res + 1, channels=d_model)

        # ── DART-style blocks (sequence-last, onset-biased attention) ─
        self.dart_blocks  = nn.ModuleList([
            BAHTBlock(
                dim=d_model,
                n_heads=dart_heads,
                dropout=dart_dropout,
                drop_path=drop_path_rate,
            )
            for _ in range(n_dart)
        ])

        # ── S4D blocks (sequence-last) ────────────────────────────────
        self.s4_blocks    = nn.ModuleList([
            SaShiMiBlock(
                d_model=d_model,
                d_state=s4_d_state,
                dropout=dropout,
                drop_path=drop_path_rate,
            )
            for _ in range(n_s4)
        ])

        # ── Bidirectional Mamba blocks (sequence-last) ────────────────
        self.mamba_blocks = nn.ModuleList([
            BidirMambaBlock(
                d_model=d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=dropout,
                drop_path=drop_path_rate,
            )
            for _ in range(n_mamba)
        ])

        self.final_norm   = nn.LayerNorm(d_model)

        # ── Attentive Statistics Pooling (channel-first) ─────────────
        self.pool         = _AttentiveStatisticsPool(d_model)

        # ── Classifier head ───────────────────────────────────────────
        head_in_dim = 2 * d_model + 9   # 9 = SubBandEnvelope output dim
        self.classifier   = nn.Sequential(
            nn.Linear(head_in_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # ── Loss ─────────────────────────────────────────────────────
        self.criterion    = FocalLoss(
            class_weights=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )

        # ── CSSD teacher (EMA frozen copy, only if enabled) ──────────
        self._cssd_enabled = cssd_alpha < 1.0
        if self._cssd_enabled:
            self.teacher   = copy.deepcopy(self)
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.teacher._cssd_enabled = False   # no recursion

        # ── Metrics ──────────────────────────────────────────────────
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

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T) float waveform at self.sample_rate Hz.
        Returns:
            logits: (B, num_classes)
        """
        # ── Stream F: scalar DSP features (computed before any augmentation
        #   to the filterbank output — waveform-level augs already applied
        #   by _shared_step before self(x) is called) ─────────────────
        scalars = self.subbandenv(x)   # (B, 9)

        # ── Stream G: Gabor filterbank ────────────────────────────────
        # Compute envelope BEFORE spec augment (SpecAugment1D mutates in-place)
        envelope_G = self.gabor(x)                         # (B, F_gabor, T)
        onset_mask = self.onset_detect(envelope_G)          # (B, N_patches) bool
        if self.training:
            envelope_G = self.spec_aug_1d(envelope_G.clone())  # clone avoids polluting onset_mask
        stream_G   = self.gabor_stem(envelope_G)            # (B, g_ch, T_shared)
        T_shared   = stream_G.shape[-1]

        # ── Stream S: 4-channel spectrogram ──────────────────────────
        streams = [stream_G]
        if self.use_stream_s:
            spec_4ch  = self.enhanced_fe(x)                # (B, 4, spec_n_mels, T_spec)
            spec_4ch  = self.cepstral(spec_4ch)             # lifter on ch 0-2; (B,4,F,T)
            stream_S  = self.cnn_input(spec_4ch)            # (B, s_ch, T_spec)
            stream_S  = F.interpolate(
                stream_S, size=T_shared, mode="linear", align_corners=False,
            )                                               # (B, s_ch, T_shared)
            streams.append(stream_S)

        # ── Stream L: LOFAR bandlimited log-power ────────────────────
        if self.use_stream_l:
            # LofarFrontend: (B, T) → (B, 1, time_bins, freq_bins)
            lofar_out  = self.lofar_fe(x)                   # (B, 1, 32, 256)
            lofar_out  = lofar_out.squeeze(1)               # (B, 32, 256) = (B, time, freq)
            lofar_out  = lofar_out.transpose(1, 2)          # (B, freq=256, time=32)
            lofar_out  = self.lofar_proj(lofar_out)         # (B, l_ch, 32)
            lofar_out  = F.interpolate(
                lofar_out, size=T_shared, mode="linear", align_corners=False,
            )                                               # (B, l_ch, T_shared)
            streams.append(lofar_out)

        # ── Fuse all channel-first streams ────────────────────────────
        merged = torch.cat(streams, dim=1)                  # (B, total_ch, T_shared)
        h      = self.fusion_gate(merged)                   # (B, d_model, T_shared)

        # ── SE-Res2 backbone + MFA (channel-first) ────────────────────
        skips = [h]
        for blk in self.res_blocks:
            h = blk(h)
            skips.append(h)
        h = self.mfa(*skips)                                # (B, d_model, T_shared)

        # ── Transition to sequence-last for attention + SSM ───────────
        h = h.permute(0, 2, 1)                              # (B, T_shared, d_model)

        # DART blocks: onset-biased boundary-aware attention
        for blk in self.dart_blocks:
            h = blk(h, onset_mask)                          # (B, T_shared, d_model)

        # S4D: long-range narrow-band tonal tracking
        for blk in self.s4_blocks:
            h = blk(h)

        # Bidirectional Mamba: selective global context
        for blk in self.mamba_blocks:
            h = blk(h)

        h = self.final_norm(h)
        h = h.permute(0, 2, 1)                              # (B, d_model, T_shared)

        # ── Attentive statistics pooling ──────────────────────────────
        pooled  = self.pool(h)                              # (B, 2*d_model)

        # ── Concat Stream F scalars and classify ─────────────────────
        head_in = torch.cat([pooled, scalars.to(pooled.dtype)], dim=-1)
        return self.classifier(head_in)                     # (B, num_classes)

    # ── Waveform augmentations ────────────────────────────────────────

    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
        if self.mixup_alpha <= 0.0 or not self.training:
            return x, y, None
        lam   = float(
            torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
        )
        idx   = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1.0 - lam) * x[idx]
        return x_mix, y, (y[idx], lam)

    def _noise_augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.noise_prob <= 0.0:
            return x
        mask = torch.rand(x.size(0), device=x.device) < self.noise_prob
        if not mask.any():
            return x
        snr_db     = torch.empty(x.size(0), device=x.device).uniform_(
            self.noise_snr_min, self.noise_snr_max
        )
        sig_rms    = x.pow(2).mean(-1, keepdim=True).clamp(min=1e-9).sqrt()
        noise      = torch.randn_like(x)
        noise_rms  = noise.pow(2).mean(-1, keepdim=True).clamp(min=1e-9).sqrt()
        scale      = sig_rms / noise_rms / (10.0 ** (snr_db.unsqueeze(-1) / 20.0))
        return torch.where(mask.unsqueeze(-1), x + scale * noise, x)

    def _gain_augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.gain_prob <= 0.0:
            return x
        mask  = torch.rand(x.size(0), device=x.device) < self.gain_prob
        gains = torch.empty(x.size(0), device=x.device).uniform_(0.6, 1.4)
        return torch.where(mask.unsqueeze(-1), x * gains.unsqueeze(-1), x)

    # ── CSSD helpers ──────────────────────────────────────────────────

    @torch.no_grad()
    def _degrade(self, x: torch.Tensor) -> torch.Tensor:
        lo = torchaudio.functional.resample(x, self.sample_rate, self.cssd_degrade_sr)
        return torchaudio.functional.resample(lo, self.cssd_degrade_sr, self.sample_rate)

    @torch.no_grad()
    def _update_teacher(self) -> None:
        m = self.ema_momentum
        for ps, pt in zip(self.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)

    # ── Shared step ───────────────────────────────────────────────────

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

        # CSSD: degrade input for teacher, train student on original
        if (split == "train" and self._cssd_enabled
                and torch.rand(1).item() < self.cssd_degrade_prob):
            with torch.no_grad():
                x_deg    = self._degrade(x)
                t_logits = self.teacher(x_deg)
                soft_tgt = F.softmax(t_logits / self.cssd_temp, dim=-1)
            logits  = self(x)
            ce_loss = self._compute_loss(logits, y, mixup_info)
            kl_loss = F.kl_div(
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

        self.log(f"{split}/loss",  loss,
                 prog_bar=True, on_step=(split == "train"), on_epoch=True, sync_dist=True)
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

    # ── Lightning steps ───────────────────────────────────────────────

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

    def on_validation_epoch_end(self):
        """Report intermediate val/f1 to Optuna for MedianPruner."""
        if self.optuna_trial is not None and _OPTUNA_AVAILABLE:
            val_f1 = self.trainer.callback_metrics.get("val/f1")
            if val_f1 is not None:
                self.optuna_trial.report(float(val_f1), self.current_epoch)
                if self.optuna_trial.should_prune():
                    raise _optuna.TrialPruned()

    # ── Optimiser + scheduler ─────────────────────────────────────────

    def configure_optimizers(self):
        # SSM parameters (S4D poles + Mamba dt_proj) get 0.1× LR to prevent
        # instability — omitting this split is the #1 cause of S4D divergence.
        ssm_keys  = {"log_a_real", "log_a_imag", "log_dt", "dt_proj"}
        ssm_params, base_params = [], []
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
            weight_decay=self.weight_decay,
        )

        def lr_lambda(epoch: int) -> float:
            if epoch < self.warmup_epochs:
                return (epoch + 1) / max(self.warmup_epochs, 1)
            progress = (epoch - self.warmup_epochs) / max(
                self.max_epochs - self.warmup_epochs, 1
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ═══════════════════════════════════════════════════════════════════════
#  Standalone smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    num_classes = 4
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Building SuperModel1D (num_classes={num_classes}) …")

    model = SuperModel1D(
        num_classes     = num_classes,
        # Smaller defaults to run fast on CPU
        gabor_n_filters = 32,
        g_ch            = 32,
        s_ch            = 32,
        l_ch            = 16,
        d_model         = 64,
        n_dart          = 1,
        n_s4            = 1,
        n_mamba         = 1,
        stem_stride     = 8,
    ).to(device).eval()

    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")

    x = torch.randn(2, 5_120, device=device)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, num_classes), \
        f"Expected (2, {num_classes}), got {list(out.shape)}"
    assert out.isfinite().all(), "Output contains NaN or Inf"
    print(f"  Output shape    : {list(out.shape)}  ✓")

    # Default-size model
    print("\nBuilding default-size SuperModel1D …")
    big = SuperModel1D(num_classes=num_classes).to(device).eval()
    total_big     = sum(p.numel() for p in big.parameters())
    trainable_big = sum(p.numel() for p in big.parameters() if p.requires_grad)
    print(f"  Total params    : {total_big:,}")
    print(f"  Trainable params: {trainable_big:,}")
    with torch.no_grad():
        out_big = big(x)
    assert out_big.shape == (2, num_classes)
    assert out_big.isfinite().all()
    print(f"  Output shape    : {list(out_big.shape)}  ✓")
