"""
HydroAudioLLM — Whisper-encoder + Llama LLM Classifier
=======================================================

Hybrid Audio-LLM classifier for underwater acoustic intelligence.  A
Whisper-large-v3 audio encoder produces frame-level acoustic embeddings;
a linear adapter projects them into the embedding space of a
Llama-3.2-1B decoder, which is queried via ``inputs_embeds`` (no token
lookup).  The pooled LLM hidden state feeds a small MLP head.

Pipeline
--------

  Waveform (B, T_in @ 5120 Hz)
        ↓ Resample 5120 → 16000 Hz
  Waveform (B, T @ 16 kHz)
        ↓ WhisperFeatureExtractor: log-mel-80
  Mel features (B, 80, 3000)
        ↓ WhisperModel.encoder
  Audio embeddings (B, T_a, d_whisper)
        ↓ Linear adapter + LayerNorm
  Adapter output (B, T_a, d_llama)
        ↓ LlamaModel(inputs_embeds=…)
  LLM hidden state (B, T_a, d_llama)
        ↓ Mean pooling over T_a
  Pooled embedding (B, d_llama)
        ↓ MLP head: Linear → GELU → Dropout → Linear
  Logits (B, num_classes)

Defaults freeze both backbones; only the adapter and the classification
head receive gradients.  Use ``--unfreeze_encoder`` / ``--unfreeze_llm``
for partial fine-tuning.

Loss: ``LargeMarginFocalLoss`` (matches the rest of the repo per
``integration_map.md``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchaudio
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassRecall, MulticlassAUROC, MulticlassMatthewsCorrCoef,
    MulticlassConfusionMatrix,
)

from processing.losses import LargeMarginFocalLoss


# ═══════════════════════════════════════════════════════════════════════
#  Linear adapter (Whisper hidden → Llama hidden)
# ═══════════════════════════════════════════════════════════════════════

class _LinearAdapter(nn.Module):
    """Linear projection + LayerNorm bridging audio and LLM hidden spaces."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.norm(self.proj(x)))


# ═══════════════════════════════════════════════════════════════════════
#  HydroAudioLLM LightningModule
# ═══════════════════════════════════════════════════════════════════════

class HydroAudioLLM(pl.LightningModule):
    """Whisper + Llama hybrid classifier.

    Args:
        num_classes:       Number of output classes.
        class_weights:     Optional per-class weights for the loss.
        sample_rate:       Input waveform sample rate (Hz).  Resampled to 16 kHz.
        whisper_model_id:  HuggingFace model ID for the audio encoder.
        llm_model_id:      HuggingFace model ID for the LLM backbone.
        adapter_type:      ``"linear"`` is implemented; ``"moe"`` is reserved.
        adapter_dropout:   Dropout inside the adapter LayerNorm path.
        freeze_encoder:    If True, freeze the Whisper encoder.
        freeze_llm:        If True, freeze the Llama backbone.
        head_hidden:       Hidden dim of the 2-layer classification MLP head.
        head_dropout:      Dropout inside the head.
        lmf_gamma:         Focal exponent for ``LargeMarginFocalLoss``.
        lmf_margin:        Margin for ``LargeMarginFocalLoss``.
        label_smoothing:   Cross-entropy label smoothing.
        learning_rate:     AdamW learning rate.
        weight_decay:      AdamW weight decay.
        warmup_epochs:     Linear warmup epoch count.
        max_epochs:        Total epochs (cosine schedule).
    """

    def __init__(
        self,
        num_classes:      int            = 4,
        class_weights:    Optional[list] = None,
        sample_rate:      int            = 5_120,
        whisper_model_id: str            = "openai/whisper-large-v3",
        llm_model_id:     str            = "meta-llama/Llama-3.2-1B",
        adapter_type:     str            = "linear",
        adapter_dropout:  float          = 0.1,
        freeze_encoder:   bool           = True,
        freeze_llm:       bool           = True,
        head_hidden:      int            = 512,
        head_dropout:     float          = 0.2,
        lmf_gamma:        float          = 2.0,
        lmf_margin:       float          = 0.3,
        label_smoothing:  float          = 0.05,
        learning_rate:    float          = 3e-4,
        weight_decay:     float          = 1e-2,
        warmup_epochs:    int            = 5,
        max_epochs:       int            = 50,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.num_classes = num_classes

        # ── Audio resampler (no-op when sample_rate == 16000) ────────────
        if sample_rate != 16_000:
            self.resample = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16_000,
            )
        else:
            self.resample = nn.Identity()

        # ── Whisper feature extractor (CPU pre-processor; not a Module) ──
        from transformers import WhisperFeatureExtractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            whisper_model_id,
        )

        # ── Whisper encoder ──────────────────────────────────────────────
        from transformers import WhisperModel
        whisper = WhisperModel.from_pretrained(whisper_model_id)
        self.audio_encoder = whisper.encoder
        whisper_d_model = whisper.config.d_model
        if freeze_encoder:
            for p in self.audio_encoder.parameters():
                p.requires_grad_(False)

        # ── Llama backbone ───────────────────────────────────────────────
        from transformers import LlamaModel
        self.llm = LlamaModel.from_pretrained(llm_model_id)
        llm_hidden = self.llm.config.hidden_size
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)

        # ── Adapter ──────────────────────────────────────────────────────
        if adapter_type == "linear":
            self.adapter = _LinearAdapter(
                in_dim=whisper_d_model, out_dim=llm_hidden,
                dropout=adapter_dropout,
            )
        else:
            raise NotImplementedError(
                f"adapter_type={adapter_type!r} is reserved; only 'linear' is "
                f"implemented in this iteration."
            )

        # ── Classification head ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(llm_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

        # ── Loss ─────────────────────────────────────────────────────────
        self.criterion = LargeMarginFocalLoss(
            num_classes=num_classes, alpha=class_weights,
            gamma=lmf_gamma, margin=lmf_margin,
            label_smoothing=label_smoothing,
        )

        # ── Metrics ──────────────────────────────────────────────────────
        m_macro = dict(num_classes=num_classes, average="macro")
        m_micro = dict(num_classes=num_classes, average="micro")
        self.train_acc            = MulticlassAccuracy(**m_macro)
        self.val_acc              = MulticlassAccuracy(**m_macro)
        self.val_f1               = MulticlassF1Score(**m_macro)
        self.val_recall           = MulticlassRecall(**m_macro)
        self.val_precision_macro  = MulticlassPrecision(**m_macro)
        self.val_precision_micro  = MulticlassPrecision(**m_micro)
        self.val_mcc              = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.val_auroc            = MulticlassAUROC(num_classes=num_classes)
        self.test_acc             = MulticlassAccuracy(**m_macro)
        self.test_f1              = MulticlassF1Score(**m_macro)
        self.test_precision       = MulticlassPrecision(**m_macro)
        self.test_precision_micro = MulticlassPrecision(**m_micro)
        self.test_recall          = MulticlassRecall(**m_macro)
        self.test_mcc             = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_auroc           = MulticlassAUROC(num_classes=num_classes)
        self.test_cm              = MulticlassConfusionMatrix(num_classes=num_classes)

    # ── Audio preprocessing ──────────────────────────────────────────────

    def _whisper_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Resample + log-mel-80 → ``(B, 80, n_frames)`` Whisper input features."""
        wav16 = self.resample(waveform)
        wav_list = [w.detach().float().cpu().numpy() for w in wav16]
        feats = self.feature_extractor(
            wav_list, sampling_rate=16_000, return_tensors="pt",
        )
        return feats.input_features.to(waveform.device)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T)`` waveform to ``(B, num_classes)`` class logits."""
        input_features = self._whisper_features(waveform)
        enc_out = self.audio_encoder(input_features)
        audio_hidden = enc_out.last_hidden_state             # (B, T_a, d_whisper)

        adapted = self.adapter(audio_hidden)                 # (B, T_a, d_llama)
        llm_out = self.llm(inputs_embeds=adapted)
        llm_hidden = llm_out.last_hidden_state               # (B, T_a, d_llama)

        pooled = llm_hidden.mean(dim=1)                      # (B, d_llama)
        return self.head(pooled)                             # (B, num_classes)

    # ── Lightning steps ──────────────────────────────────────────────────

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = F.softmax(logits, dim=-1)

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_recall(logits, y)
        self.val_precision_macro(logits, y)
        self.val_precision_micro(logits, y)
        self.val_mcc(logits, y)
        self.val_auroc(probs, y)

        self.log("val/loss",            loss,                     on_epoch=True, prog_bar=True)
        self.log("val/acc",             self.val_acc,             on_epoch=True, prog_bar=True)
        self.log("val/f1",              self.val_f1,              on_epoch=True, prog_bar=True)
        self.log("val/recall",          self.val_recall,          on_epoch=True)
        self.log("val/macro_precision", self.val_precision_macro, on_epoch=True, prog_bar=True)
        self.log("val/micro_precision", self.val_precision_micro, on_epoch=True, prog_bar=True)
        self.log("val/mcc",             self.val_mcc,             on_epoch=True)
        self.log("val/auroc",           self.val_auroc,           on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = F.softmax(logits, dim=-1)

        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.test_precision(logits, y)
        self.test_precision_micro(logits, y)
        self.test_recall(logits, y)
        self.test_mcc(logits, y)
        self.test_auroc(probs, y)
        self.test_cm(logits, y)

        self.log("test/loss",            loss,                       on_epoch=True)
        self.log("test/acc",             self.test_acc,              on_epoch=True)
        self.log("test/f1",              self.test_f1,               on_epoch=True)
        self.log("test/macro_precision", self.test_precision,        on_epoch=True)
        self.log("test/micro_precision", self.test_precision_micro,  on_epoch=True)
        self.log("test/recall",          self.test_recall,           on_epoch=True)
        self.log("test/mcc",             self.test_mcc,              on_epoch=True)
        self.log("test/auroc",           self.test_auroc,            on_epoch=True)

    def on_test_epoch_end(self) -> None:
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix:\n{cm.cpu().numpy()}")
        self.test_cm.reset()

    # ── Optimiser ────────────────────────────────────────────────────────

    def configure_optimizers(self) -> Dict[str, Any]:
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

        def lr_lambda(epoch: int) -> float:
            wu, total = self.hparams.warmup_epochs, self.hparams.max_epochs
            if epoch < wu:
                return (epoch + 1) / max(wu, 1)
            p = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
