"""
HydroEnsembleTrio — Joint training of HydroNet + HydroConformer + HydroS5.

All three sub-models are trained from scratch on the same augmented waveform.
Validation and test metrics are computed on the averaged softmax probabilities.

Each branch operates at its own native sample rate:
  - HydroNet + HydroConformer : 32 000 Hz / 1.0 s  (their designed operating point)
  - HydroS5                   : 5 120 Hz / 0.5 s   (its designed operating point)

The ensemble data loader runs at 32 000 Hz / 1.0 s.  The S5Branch wrapper
downsamples 32 kHz → 5 120 Hz and center-crops to 2 560 samples before
forwarding into HydroS5 — no information is lost for HydroNet/HydroConformer.

Architecture summary
---------------------
  Waveform (B, 32 000)  @32 000 Hz
      │
      ├─► HydroNet        (4-ch EnhancedFrontEnd → SE-Res2Blocks → AttPool)
      │         └─► logits_net  (B, C)
      │
      ├─► HydroConformer  (2-ch MultiScalePCEN → ConformerBlocks → AttPool)
      │         └─► logits_con  (B, C)
      │
      └─► S5Branch        resample 32k→5120 + center-crop 2560
                └─► HydroS5  (2-ch MultiScalePCEN → MIMO S5 Blocks → ASPool)
                          └─► logits_s5  (B, C)

  Training  : loss = focal(net) + focal(con) + focal(s5)
  Inference : probs = mean(softmax(net), softmax(con), softmax(s5))

Augmentation (gain, noise, mixup) is applied once at the ensemble level
on the 32 kHz waveform before dispatching to each branch.
"""

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

from .hydro_net import HydroNet
from .hydro_conformer import HydroConformer, FocalLoss
from .hydro_s5 import HydroS5


# ═══════════════════════════════════════════════════════════════════════
#  S5 branch wrapper — handles downsampling from ensemble SR → S5 SR
# ═══════════════════════════════════════════════════════════════════════

class _S5Branch(nn.Module):
    """
    Wraps HydroS5 to accept the ensemble's native 32 kHz waveform.

    Applies:
      1. torchaudio.functional.resample(src_sr → tgt_sr)
      2. Center-crop (or zero-pad) to exactly tgt_len samples
      3. Forward through HydroS5
    """

    def __init__(self, s5: HydroS5, src_sr: int, tgt_sr: int, tgt_len: int):
        super().__init__()
        self.s5      = s5
        self.src_sr  = src_sr
        self.tgt_sr  = tgt_sr
        self.tgt_len = tgt_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T) at src_sr
        x = torchaudio.functional.resample(x, self.src_sr, self.tgt_sr)
        L = x.shape[-1]
        if L > self.tgt_len:
            start = (L - self.tgt_len) // 2
            x = x[..., start : start + self.tgt_len]
        elif L < self.tgt_len:
            x = F.pad(x, (0, self.tgt_len - L))
        return self.s5(x)

    # Expose sub-module parameters for named_parameters / param groups
    def named_parameters(self, *args, **kwargs):
        return self.s5.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.s5.parameters(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  Ensemble LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroEnsembleTrio(pl.LightningModule):
    """
    Three-branch ensemble: HydroNet + HydroConformer + HydroS5.

    Sub-model params are prefixed: net_, con_, s5_.
    Shared audio / training params are unprefixed and apply to the
    32 kHz HydroNet / HydroConformer branches.
    HydroS5 always runs at 5 120 Hz / 2 560 samples (internal resample).
    """

    # Fixed S5 operating point — not user-configurable
    _S5_SR  = 5_120
    _S5_LEN = 2_560

    def __init__(
        self,
        num_classes:     int            = 3,
        class_weights:   Optional[list] = None,
        # ── Shared audio (HydroNet + HydroConformer) ────────────────────
        sample_rate:     int            = 32_000,
        fixed_len:       int            = 32_000,
        n_mels:          int            = 128,
        hop_length:      int            = 320,
        # ── Shared STFT windows (derived from sample_rate if not set) ───
        wb_n_fft:        Optional[int]  = None,   # ~25 ms wideband window
        nb_n_fft:        Optional[int]  = None,   # ~100 ms narrowband window
        # ── HydroNet ────────────────────────────────────────────────────
        net_channels:    int            = 128,
        net_scale:       int            = 8,
        net_dilations:   List[int]      = (2, 4, 8),
        net_dropout:     float          = 0.24,
        net_drop_path:   float          = 0.10,
        # ── HydroConformer ──────────────────────────────────────────────
        con_model_dim:   int            = 128,
        con_n_blocks:    int            = 4,
        con_n_heads:     int            = 16,
        con_conv_kernel: int            = 31,
        con_dropout:     float          = 0.10,
        con_drop_path:   float          = 0.10,
        # ── HydroS5 (Optuna best-trial params) ──────────────────────────
        s5_d_model:      int            = 256,
        s5_d_state:      int            = 128,
        s5_n_layers:     int            = 3,
        s5_expansion:    int            = 4,
        s5_dt_min:       float          = 4.25e-3,
        s5_dt_max:       float          = 3.05e-1,
        s5_bidirectional:bool           = False,
        s5_ls_init:      float          = 0.65,
        s5_dropout:      float          = 0.141,
        s5_drop_path:    float          = 0.049,
        # ── Shared training ─────────────────────────────────────────────
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

        # Derive SR-appropriate FFT sizes if not explicitly provided
        def _next_pow2(n: float) -> int:
            return 2 ** math.ceil(math.log2(max(n, 4.0)))

        if wb_n_fft is None:
            wb_n_fft = _next_pow2(sample_rate * 0.025)
        if nb_n_fft is None:
            nb_n_fft = _next_pow2(sample_rate * 0.100)

        self.save_hyperparameters()

        # ── HydroNet ───────────────────────────────────────────────────
        self.net = HydroNet(
            num_classes     = num_classes,
            class_weights   = class_weights,
            sample_rate     = sample_rate,
            n_mels          = n_mels,
            hop_length      = hop_length,
            wb_n_fft        = wb_n_fft,
            nb_n_fft        = nb_n_fft,
            channels        = net_channels,
            scale           = net_scale,
            dilation_rates  = list(net_dilations),
            dropout         = net_dropout,
            drop_path_rate  = net_drop_path,
            learning_rate   = learning_rate,
            weight_decay    = weight_decay,
            warmup_epochs   = warmup_epochs,
            max_epochs      = max_epochs,
            mixup_alpha     = 0.0,
            noise_prob      = 0.0,
            gain_prob       = 0.0,
            focal_gamma     = focal_gamma,
            label_smoothing = label_smoothing,
        )

        # ── HydroConformer ─────────────────────────────────────────────
        self.conformer = HydroConformer(
            num_classes     = num_classes,
            class_weights   = class_weights,
            sample_rate     = sample_rate,
            n_mels          = n_mels,
            hop_length      = hop_length,
            wb_n_fft        = wb_n_fft,
            nb_n_fft        = nb_n_fft,
            model_dim       = con_model_dim,
            n_blocks        = con_n_blocks,
            n_heads         = con_n_heads,
            conv_kernel     = con_conv_kernel,
            dropout         = con_dropout,
            drop_path_rate  = con_drop_path,
            learning_rate   = learning_rate,
            weight_decay    = weight_decay,
            warmup_epochs   = warmup_epochs,
            max_epochs      = max_epochs,
            mixup_alpha     = 0.0,
            focal_gamma     = focal_gamma,
            label_smoothing = label_smoothing,
        )

        # ── HydroS5 (wrapped with downsampler) ─────────────────────────
        _s5 = HydroS5(
            num_classes     = num_classes,
            class_weights   = class_weights,
            sample_rate     = self._S5_SR,
            fixed_len       = self._S5_LEN,
            n_mels          = 64,
            hop_length      = 25,
            d_model         = s5_d_model,
            d_state         = s5_d_state,
            n_layers        = s5_n_layers,
            expansion       = s5_expansion,
            dt_min          = s5_dt_min,
            dt_max          = s5_dt_max,
            bidirectional   = s5_bidirectional,
            ls_init         = s5_ls_init,
            dropout         = s5_dropout,
            drop_path_rate  = s5_drop_path,
            learning_rate   = learning_rate,
            weight_decay    = weight_decay,
            warmup_epochs   = warmup_epochs,
            max_epochs      = max_epochs,
            mixup_alpha     = 0.0,
            noise_prob      = 0.0,
            noise_snr_min   = noise_snr_min,
            noise_snr_max   = noise_snr_max,
            gain_prob       = 0.0,
            focal_gamma     = focal_gamma,
            label_smoothing = label_smoothing,
        )
        self.s5_branch = _S5Branch(
            s5      = _s5,
            src_sr  = sample_rate,
            tgt_sr  = self._S5_SR,
            tgt_len = self._S5_LEN,
        )

        # ── Ensemble-level focal loss ──────────────────────────────────
        self.criterion = FocalLoss(
            class_weights   = class_weights,
            gamma           = focal_gamma,
            label_smoothing = label_smoothing,
        )

        # ── Metrics (on averaged ensemble probabilities) ───────────────
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

    # ── Ensemble forward ────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:    waveform (B, T) at self.hparams.sample_rate  (32 kHz)
        Returns: averaged softmax probabilities (B, num_classes)
        """
        p_net = F.softmax(self.net(waveform),            dim=-1)
        p_con = F.softmax(self.conformer(waveform),      dim=-1)
        p_s5  = F.softmax(self.s5_branch(waveform),      dim=-1)
        return (p_net + p_con + p_s5) / 3.0

    # ── Augmentation ────────────────────────────────────────────────────

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1).item() < self.hparams.gain_prob:
            gain = 0.6 + 0.8 * torch.rand(x.shape[0], 1, device=x.device)
            x = x * gain
        if torch.rand(1).item() < self.hparams.noise_prob:
            snr_db  = (self.hparams.noise_snr_min
                       + (self.hparams.noise_snr_max - self.hparams.noise_snr_min)
                       * torch.rand(1, device=x.device))
            snr_lin = 10.0 ** (snr_db / 10.0)
            sig_pwr = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-9)
            noise   = torch.randn_like(x) * (sig_pwr / snr_lin).sqrt()
            x = x + noise
        return x

    def _mixup(self, x: torch.Tensor, y: torch.Tensor):
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
        x = self._augment(x)
        x, y, y_p, lam = self._mixup(x, y)

        logits_net = self.net(x)
        logits_con = self.conformer(x)
        logits_s5  = self.s5_branch(x)

        loss_net = self._loss(logits_net, y, y_p, lam)
        loss_con = self._loss(logits_con, y, y_p, lam)
        loss_s5  = self._loss(logits_s5,  y, y_p, lam)
        loss     = loss_net + loss_con + loss_s5

        with torch.no_grad():
            probs = (F.softmax(logits_net, dim=-1)
                     + F.softmax(logits_con, dim=-1)
                     + F.softmax(logits_s5,  dim=-1)) / 3.0
        self.train_acc(probs, y)

        self.log("train/loss",     loss,           on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/loss_net", loss_net,       on_step=False, on_epoch=True)
        self.log("train/loss_con", loss_con,       on_step=False, on_epoch=True)
        self.log("train/loss_s5",  loss_s5,        on_step=False, on_epoch=True)
        self.log("train/acc",      self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y  = batch
        probs = self(x)

        log_p = torch.log(probs.clamp(min=1e-9))
        loss  = F.nll_loss(log_p, y)

        self.val_acc(probs, y);       self.val_f1(probs, y)
        self.val_precision(probs, y); self.val_mcc(probs, y)

        self.log("val/loss",      loss,               on_epoch=True, prog_bar=True)
        self.log("val/acc",       self.val_acc,       on_epoch=True, prog_bar=True)
        self.log("val/f1",        self.val_f1,        on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True)
        self.log("val/mcc",       self.val_mcc,       on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y  = batch
        probs = self(x)

        self.test_acc(probs, y);   self.test_f1(probs, y)
        self.test_mcc(probs, y);   self.test_auroc(probs, y)
        self.test_cm(probs, y)

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
        # Three groups: weight-decayed, non-decayed, SSM poles (0.1× LR)
        decay, no_decay, ssm = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "s5_branch" in name and any(
                k in name for k in ("log_A_real", "A_imag", "log_dt")
            ):
                ssm.append(p)
            elif p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
                {"params": ssm,      "weight_decay": 0.0,
                 "lr": self.hparams.learning_rate * 0.1},
            ],
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
#  Smoke test  (python -m models.hydro_ensemble_trio)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = HydroEnsembleTrio(num_classes=3).to(device).eval()

    net_p = sum(p.numel() for p in model.net.parameters())
    con_p = sum(p.numel() for p in model.conformer.parameters())
    s5_p  = sum(p.numel() for p in model.s5_branch.parameters())
    total = net_p + con_p + s5_p
    print(f"HydroEnsembleTrio  |  {total:,} total params")
    print(f"  HydroNet         :  {net_p:,}  (32 kHz native)")
    print(f"  HydroConformer   :  {con_p:,}  (32 kHz native)")
    print(f"  HydroS5          :  {s5_p:,}  (32k→5120 Hz internal resample)")

    x = torch.randn(2, 32_000, device=device)
    with torch.no_grad():
        probs = model(x)
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(-1), torch.ones(2, device=device), atol=1e-4)
    print(f"Output             :  {probs.shape}  sums={probs.sum(-1).tolist()}")
    print("Smoke test passed.")
