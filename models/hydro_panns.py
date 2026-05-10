"""
HydroPANNs — CNN14 (PANNs) with Cross-Sampling-Rate Self-Distillation (CSSD).

Architecture
------------
  PANNs (Pretrained Audio Neural Networks) are CNN-based models trained on
  AudioSet.  CNN14 is the highest-accuracy variant with 6 ConvBlocks.

  CSSD (Cross-Sampling-Rate Self-Distillation) adds robustness to audio
  quality degradation (low-sample-rate recordings, bandwidth-limited
  hydrophone sensors) by training the model against a teacher that always
  sees the clean, high-quality signal.

  Architecture (CNN14 backbone):
    Raw waveform (32 kHz, 1 s)
        ↓
    Log-Mel Spectrogram  (64 mel bins, hop 320)  [CNN14 uses 64 bins]
        ↓
    SpecAugment
        ↓
    BN(1)
        ↓
    ConvBlock(1→64)  + AvgPool(2,2)
    ConvBlock(64→128) + AvgPool(2,2)
    ConvBlock(128→256) + AvgPool(2,2)
    ConvBlock(256→512) + AvgPool(2,2)
    ConvBlock(512→1024) + AvgPool(2,2)
    ConvBlock(1024→2048) + AvgPool(1,1)
        ↓
    Global AvgPool (mean over time×freq) → (B, 2048)
        ↓
    FC(2048→2048) + Dropout + ReLU  (CNN14 embedding layer)
        ↓
    FC(2048→num_classes)  (classifier head)
        ↓
    FocalLoss

CSSD Training
-------------
At each training step with probability `degrade_prob`, the input batch is
"degraded" by resampling to a lower sampling rate and back (simulating
bandwidth-limited / low-quality recordings).  An EMA teacher (momentum=0.999)
always sees the original clean signal.  The total loss is:

    L = α · CE(student_logits, hard_labels)
      + (1 - α) · KL( softmax(student/T) ‖ softmax(teacher/T) )

where T is the distillation temperature.

EMA Teacher
-----------
The teacher is an Exponential Moving Average of the student weights.  This
bootstraps the teacher from random initialisation and progressively improves
the soft targets it provides, avoiding the need for pre-trained weights.
At inference only the student is used (teacher is discarded).

Optional Pretrained Weights
---------------------------
CNN14 pretrained weights (AudioSet, mAP=0.431) can be loaded via
`--pretrained_path` in the training script.  The weights must be in the
format from the official PANNS repository:
    https://zenodo.org/record/3987831/files/Cnn14_mAP=0.431.pth
"""

import copy
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

from .hydro_conformer import SpecAugment, FocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  CNN14 building blocks
# ═══════════════════════════════════════════════════════════════════════

class ConvBlock2d(nn.Module):
    """
    CNN14 convolutional block: two 3×3 convolutions with BN + ReLU.
    Downsampling (AvgPool2d) is applied in the forward call.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(
        self, x: torch.Tensor, pool: tuple[int, int] = (2, 2)
    ) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool != (1, 1):
            x = F.avg_pool2d(x, pool)
        return x


class CNN14Backbone(nn.Module):
    """
    CNN14 backbone from Kong et al. 2020 "PANNs: Large-Scale Pretrained
    Audio Neural Networks for Audio Pattern Recognition".

    Input : (B, 1, T_frames, n_mels)   — time first, freq second (PANNS convention)
    Output: (B, d_embed)               — global average pooled embedding
    """

    def __init__(self, n_mels: int = 64, d_embed: int = 2048, dropout: float = 0.2):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)   # normalize (B, 1, T, F) spectrogram

        self.conv1 = ConvBlock2d(1,    64)
        self.conv2 = ConvBlock2d(64,   128)
        self.conv3 = ConvBlock2d(128,  256)
        self.conv4 = ConvBlock2d(256,  512)
        self.conv5 = ConvBlock2d(512,  1024)
        self.conv6 = ConvBlock2d(1024, 2048)

        self.fc1     = nn.Linear(2048, d_embed, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F)
        x = self.bn0(x)                                  # normalize
        x = self.conv1(x, pool=(2, 2)); x = self.dropout(x)
        x = self.conv2(x, pool=(2, 2)); x = self.dropout(x)
        x = self.conv3(x, pool=(2, 2)); x = self.dropout(x)
        x = self.conv4(x, pool=(2, 2)); x = self.dropout(x)
        x = self.conv5(x, pool=(2, 2)); x = self.dropout(x)
        x = self.conv6(x, pool=(1, 1)); x = self.dropout(x)

        # Aggregate over time and frequency
        x = x.mean(dim=[2, 3])                           # (B, 2048)
        x = F.relu_(self.fc1(x))                         # (B, d_embed)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  HydroPANNs LightningModule (CNN14 + CSSD)
# ═══════════════════════════════════════════════════════════════════════

class HydroPANNs(pl.LightningModule):
    """
    CNN14-based vessel classifier with Cross-Sampling-Rate Self-Distillation.

    Args
    ----
    num_classes     : Number of vessel classes.
    class_weights   : Inverse-frequency weights for FocalLoss (None = uniform).
    sample_rate     : Input audio sample rate in Hz.
    n_mels          : Number of mel filterbank bins (CNN14 default = 64).
    hop_length      : STFT hop in samples.
    d_embed         : CNN14 embedding dimension.
    dropout         : Dropout in CNN14 backbone and classifier.
    degrade_prob    : Probability of applying CSSD degradation to a batch.
    degrade_sr      : Target sampling rate for degradation (e.g. 8000).
    distill_alpha   : Weight of CE vs KL loss.  1.0 = pure CE (no CSSD).
    distill_temp    : Temperature for knowledge distillation.
    ema_momentum    : EMA momentum for teacher network update.
    pretrained_path : Optional path to a CNN14 .pth weight file.
    learning_rate   : Peak AdamW LR.
    weight_decay    : AdamW weight decay.
    warmup_epochs   : Linear LR warmup epochs.
    max_epochs      : Total epochs (cosine schedule).
    mixup_alpha     : Waveform Mixup β (0 = off).
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
        n_mels:          int            = 64,
        hop_length:      int            = 320,
        d_embed:         int            = 2048,
        dropout:         float          = 0.24,
        degrade_prob:    float          = 0.50,
        degrade_sr:      int            = 8_000,
        distill_alpha:   float          = 0.70,
        distill_temp:    float          = 4.0,
        ema_momentum:    float          = 0.999,
        pretrained_path: Optional[str]  = None,
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
            sample_rate = sample_rate,
            n_fft       = n_fft,
            hop_length  = hop_length,
            n_mels      = n_mels,
            f_min       = 50.0,
            f_max       = sample_rate / 2.0,
        )
        self.to_db  = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.spec_aug = SpecAugment(
            n_freq_masks=2, freq_mask_max=8,
            n_time_masks=2, time_mask_max=20,
        )

        # ── Student network ──────────────────────────────────────────────
        self.backbone   = CNN14Backbone(n_mels=n_mels, d_embed=d_embed, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_embed, num_classes),
        )

        # Load optional pretrained CNN14 weights
        if pretrained_path is not None:
            self._load_panns_weights(pretrained_path)

        # ── EMA teacher (copy of student, no grad) ───────────────────────
        self.teacher = copy.deepcopy(self.backbone)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

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

    # ── Pretrained weight loading ─────────────────────────────────────────

    def _load_panns_weights(self, path: str) -> None:
        """
        Load pretrained CNN14 weights from the official PANNS checkpoint.
        Strips the 'model.' prefix if present and ignores mismatched heads.
        """
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)

        # Strip prefix and collect only backbone keys
        new_state = {}
        for k, v in state.items():
            k = k.replace("model.", "")
            # PANNS uses 'conv_block' naming; remap to our 'convN' naming
            for i, name in enumerate(
                ["conv_block1", "conv_block2", "conv_block3",
                 "conv_block4", "conv_block5", "conv_block6"], start=1
            ):
                k = k.replace(name, f"conv{i}")
            new_state[k] = v

        missing, unexpected = self.backbone.load_state_dict(new_state, strict=False)
        print(f"[PANNs] Loaded pretrained weights from {path}")
        if missing:
            print(f"  Missing keys   : {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    # ── Spectrogram helper ────────────────────────────────────────────────

    def _to_spec(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform (B, T) → spectrogram (B, 1, T_frames, n_mels)"""
        x = self.mel(waveform)       # (B, n_mels, T_frames)
        x = self.to_db(x)
        # CNN14 convention: (B, 1, T_frames, n_mels) — time first
        x = x.unsqueeze(1).transpose(2, 3)   # (B, 1, T, F)
        return x

    # ── CSSD degradation ─────────────────────────────────────────────────

    def _degrade(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Downsample waveform to degrade_sr Hz then upsample back to sample_rate.
        This removes high-frequency content, simulating bandwidth-limited sensors.
        """
        sr_orig = self.hparams.sample_rate
        sr_low  = self.hparams.degrade_sr
        if sr_low >= sr_orig:
            return waveform
        wf_low   = torchaudio.functional.resample(waveform, sr_orig, sr_low)
        wf_clean = torchaudio.functional.resample(wf_low,   sr_low,  sr_orig)
        # Match original length (resampling may add/remove 1–2 samples)
        T = waveform.shape[-1]
        if wf_clean.shape[-1] < T:
            wf_clean = F.pad(wf_clean, (0, T - wf_clean.shape[-1]))
        else:
            wf_clean = wf_clean[..., :T]
        return wf_clean

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform (B, T) → logits (B, num_classes)"""
        x = self._to_spec(waveform)          # (B, 1, T_frames, n_mels)
        if self.training:
            # SpecAugment on (B, 1, T, F) — treat F as "freq" axis
            x = self.spec_aug(x)
        emb = self.backbone(x)               # (B, d_embed)
        return self.classifier(emb)          # (B, num_classes)

    # ── EMA teacher update ────────────────────────────────────────────────

    @torch.no_grad()
    def _update_teacher(self) -> None:
        m = self.hparams.ema_momentum
        for p_t, p_s in zip(self.teacher.parameters(), self.backbone.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)

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
        x_clean, y = batch
        x_clean    = self._augment(x_clean)
        x_clean, y, y_p, lam = self._mixup(x_clean, y)

        alpha = self.hparams.distill_alpha
        T     = self.hparams.distill_temp

        # Decide whether to apply CSSD degradation this step
        do_cssd = (
            torch.rand(1).item() < self.hparams.degrade_prob
            and alpha < 1.0
        )

        if do_cssd:
            x_deg    = self._degrade(x_clean)
            logits_s = self(x_deg)         # student on degraded audio

            with torch.no_grad():
                spec_t   = self._to_spec(x_clean)   # teacher on clean audio
                emb_t    = self.teacher(spec_t)
                logits_t = self.classifier(emb_t).detach()

            ce_loss  = self._loss(logits_s, y, y_p, lam)
            kl_loss  = F.kl_div(
                F.log_softmax(logits_s / T, dim=-1),
                F.softmax(logits_t / T,     dim=-1),
                reduction="batchmean",
            ) * (T ** 2)
            loss = alpha * ce_loss + (1.0 - alpha) * kl_loss
            self.log("train/kl_loss", kl_loss, on_step=False, on_epoch=True)
        else:
            logits_s = self(x_clean)
            loss     = self._loss(logits_s, y, y_p, lam)

        self.train_acc(logits_s, y)
        self.log("train/loss", loss,           on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # Update EMA teacher after each student update
        self._update_teacher()

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
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
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

    model  = HydroPANNs(num_classes=3).to(device).eval()
    total  = sum(p.numel() for p in model.parameters())
    bb_p   = sum(p.numel() for p in model.backbone.parameters())

    cl_p  = sum(p.numel() for p in model.classifier.parameters())

    x_dummy = torch.randn(2, 32_000, device=device)
    with torch.no_grad():
        logits = model(x_dummy)

    hp = model.hparams
    student_p = bb_p + cl_p
    print(f"\nHydroPANNs (CNN14+CSSD)  |  student {student_p:,} params")
    print(f"  CNN14 backbone : {bb_p:,}")
    print(f"  Classifier head: {cl_p:,}")
    print(f"  EMA teacher    : {bb_p:,}  (frozen copy, not in training)")
    print(f"  n_mels={hp.n_mels}, d_embed={hp.d_embed}")
    print(f"  degrade_sr={hp.degrade_sr} Hz, T={hp.distill_temp}, α={hp.distill_alpha}")
    print(f"\nOutput shape : {list(logits.shape)}")
