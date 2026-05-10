"""
Training script for HydroUAST3D — Uncompressed Audio Spectrogram Transformer
with 3D Feature Fusion.

Usage:
    python training/train_uast3d.py [options]

Quickstart (defaults match the Split1s dataset at DATA_DIR):
    export DATA_DIR=/path/to/Split1s
    python training/train_uast3d.py

Key design choices:
    - DALI loader with oversampled training set (Passenger: 62 → ~11k samples)
    - FocalLoss with inverse-frequency class weights
    - Mixed precision (bf16 on Ampere+, fp16 otherwise)
    - Model checkpoint on val/f1 (macro) — fair under class imbalance
    - Early stopping on val/loss with patience=20
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_uast3d import HydroUAST3D


def get_args():
    p = argparse.ArgumentParser(description="Train HydroUAST3D")

    # ── Data ────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",    default=os.environ.get("DATA_DIR", ""),
                   help="Path to Split1s root (or set $DATA_DIR)")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_threads", type=int, default=8,
                   help="DALI pipeline threads")
    p.add_argument("--no_oversample", action="store_true",
                   help="Disable minority-class oversampling")

    # ── Audio ────────────────────────────────────────────────────────────
    p.add_argument("--sample_rate", type=int, default=5_120,
                   help="Target sample rate for DALI resampling")
    p.add_argument("--fixed_len",   type=int, default=5_120,
                   help="Fixed waveform length in samples (1 s at 5120 Hz)")
    p.add_argument("--n_mels",      type=int, default=64,
                   help="Mel filterbank bins (freq axis)")
    p.add_argument("--hop_length",  type=int, default=51,
                   help="STFT hop length (~10 ms at 5120 Hz)")

    # ── Patch embedding ──────────────────────────────────────────────────
    p.add_argument("--patch_f",  type=int, default=8,
                   help="Patch height in frequency bins (n_mels must be divisible)")
    p.add_argument("--patch_t",  type=int, default=10,
                   help="Patch width in time frames")

    # ── 3D fusion ────────────────────────────────────────────────────────
    p.add_argument("--fusion_dim", type=int, default=32,
                   help="Inner channels for 3D fusion Conv3d layers")

    # ── Transformer ──────────────────────────────────────────────────────
    p.add_argument("--model_dim",    type=int,   default=256)
    p.add_argument("--n_blocks",     type=int,   default=6)
    p.add_argument("--n_heads",      type=int,   default=8)
    p.add_argument("--ff_expansion", type=int,   default=4)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--attn_drop",    type=float, default=0.1)
    p.add_argument("--drop_path",    type=float, default=0.15)

    # ── Training ─────────────────────────────────────────────────────────
    p.add_argument("--mixup_alpha",     type=float, default=0.3)
    p.add_argument("--focal_gamma",     type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--max_epochs",      type=int,   default=100)
    p.add_argument("--warmup_epochs",   type=int,   default=10)
    p.add_argument("--precision",       default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=42)

    return p.parse_args()


def main():
    args = get_args()

    if not args.data_dir:
        raise ValueError(
            "No data directory specified.\n"
            "  Set the DATA_DIR environment variable:  export DATA_DIR=/path/to/Split1s\n"
            "  Or pass --data_dir /path/to/Split1s"
        )

    if args.n_mels % args.patch_f != 0:
        raise ValueError(
            f"n_mels ({args.n_mels}) must be divisible by patch_f ({args.patch_f})"
        )

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # ── Data ─────────────────────────────────────────────────────────────
    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        oversample_train=not args.no_oversample,
    )
    data.setup()

    # ── Model ─────────────────────────────────────────────────────────────
    model = HydroUAST3D(
        num_classes=data.num_classes,
        class_weights=data.class_weights,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        patch_f=args.patch_f,
        patch_t=args.patch_t,
        fusion_dim=args.fusion_dim,
        model_dim=args.model_dim,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        ff_expansion=args.ff_expansion,
        dropout=args.dropout,
        attn_drop=args.attn_drop,
        drop_path_rate=args.drop_path,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        mixup_alpha=args.mixup_alpha,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHydroUAST3D — {n_params / 1e6:.2f}M trainable parameters")
    print(f"Patch grid    — {model.n_freq_patches} freq × {model.n_time_patches} time"
          f" = {model.n_patches} tokens")
    print(f"Spectral views — WB (n_fft=256)  |  NB (n_fft=1024)  |  Ref (n_fft=512, log-mel)")

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            monitor="val/f1",
            mode="max",
            save_top_k=3,
            filename="uast3d-{epoch:03d}-f1{val/f1:.4f}",
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=20,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = CSVLogger("lightning_logs", name="hydro_uast3d")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        deterministic=False,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.fit(model, data)

    # ── Test ──────────────────────────────────────────────────────────────
    trainer.test(model, data, ckpt_path="best")

    print(f"\nBest checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val/f1:     {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
