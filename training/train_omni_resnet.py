"""
Training script for AcousticOmniResNet — Multi-Head Residual Fusion Network.

Usage:
    python training/train_omni_resnet.py [options]

Quickstart (defaults match the Split1s dataset at DATA_DIR):
    export DATA_DIR=/path/to/Split1s
    python training/train_omni_resnet.py

Key design choices:
    - Pre-computed feature cache (``data/cache/``) eliminates repeated WVD / HHT
      computation across epochs; first run populates the cache in parallel.
    - 9-channel ResNet-50 (2D) + ResidualMLP (1D) fused via cross-attention + SE.
    - WeightedRandomSampler oversampling counteracts 185x Cargo/Passenger imbalance.
    - FocalLoss with inverse-frequency class weights.
    - Feature-level mixup (applied to both feat_1d and feat_2d simultaneously).
    - Mixed precision (bf16 on Ampere+, fp16 otherwise).
    - Checkpoint on val/f1 (macro) — fair under class imbalance.
    - Early stopping on val/loss with patience=20.
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

from data.cached_feature_dataset import OmniFeatureDataModule
from models.hydro_omni_resnet import AcousticOmniResNet


def get_args():
    p = argparse.ArgumentParser(description="Train AcousticOmniResNet")

    # ── Data ────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",  default=os.environ.get("DATA_DIR", ""),
                   help="Path to Split1s root (or set $DATA_DIR)")
    p.add_argument("--cache_dir", default="data/cache",
                   help="Root directory for pre-computed .pt feature cache files")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader worker threads")
    p.add_argument("--precompute_workers", type=int, default=8,
                   help="ProcessPoolExecutor workers for feature pre-computation")
    p.add_argument("--no_oversample", action="store_true",
                   help="Disable minority-class oversampling in training")
    p.add_argument("--force_recompute", action="store_true",
                   help="Overwrite existing .pt cache files")

    # ── Audio / feature extractor ────────────────────────────────────────
    p.add_argument("--sample_rate", type=int, default=5_120,
                   help="Target sample rate for resampling (Hz)")

    # ── 1D MLP branch ────────────────────────────────────────────────────
    p.add_argument("--mlp_dim",       type=int,   default=512,
                   help="Hidden dimension for the ResidualMLP branch")
    p.add_argument("--mlp_n_blocks",  type=int,   default=3,
                   help="Number of ResidualMLPBlock layers")
    p.add_argument("--mlp_dropout",   type=float, default=0.1)
    p.add_argument("--mlp_drop_path", type=float, default=0.1)

    # ── Fusion ───────────────────────────────────────────────────────────
    p.add_argument("--fusion_dim",     type=int,   default=512,
                   help="Shared projection dimension in CrossAttentionFusion")
    p.add_argument("--fusion_tokens",  type=int,   default=8,
                   help="Token split count for cross-attention (fusion_dim must be divisible)")
    p.add_argument("--fusion_heads",   type=int,   default=4,
                   help="Attention heads per token in CrossAttentionFusion")
    p.add_argument("--fusion_dropout", type=float, default=0.1)

    # ── Training ─────────────────────────────────────────────────────────
    p.add_argument("--mixup_alpha",       type=float, default=0.3,
                   help="Beta distribution alpha for feature-level mixup (0=disabled)")
    p.add_argument("--focal_gamma",       type=float, default=2.0)
    p.add_argument("--label_smoothing",   type=float, default=0.05)
    p.add_argument("--classifier_dropout",type=float, default=0.2)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--weight_decay",      type=float, default=1e-2)
    p.add_argument("--max_epochs",        type=int,   default=100)
    p.add_argument("--warmup_epochs",     type=int,   default=10)
    p.add_argument("--precision",         default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",         type=float, default=1.0)
    p.add_argument("--seed",              type=int,   default=42)

    return p.parse_args()


def main():
    args = get_args()

    if not args.data_dir:
        raise ValueError(
            "No data directory specified.\n"
            "  Set the DATA_DIR environment variable:  export DATA_DIR=/path/to/Split1s\n"
            "  Or pass --data_dir /path/to/Split1s"
        )

    if args.fusion_dim % args.fusion_tokens != 0:
        raise ValueError(
            f"fusion_dim ({args.fusion_dim}) must be divisible by "
            f"fusion_tokens ({args.fusion_tokens})"
        )

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # ── Data ─────────────────────────────────────────────────────────────
    print("\n── Feature cache setup ─────────────────────────────────────────")
    data = OmniFeatureDataModule(
        data_dir           = args.data_dir,
        cache_dir          = args.cache_dir,
        batch_size         = args.batch_size,
        num_workers        = args.num_workers,
        precompute_workers = args.precompute_workers,
        sample_rate        = args.sample_rate,
        oversample_train   = not args.no_oversample,
        force_recompute    = args.force_recompute,
    )
    data.setup()

    # ── Model ─────────────────────────────────────────────────────────────
    model = AcousticOmniResNet(
        num_classes         = data.num_classes,
        class_weights       = data.class_weights,
        mlp_dim             = args.mlp_dim,
        mlp_n_blocks        = args.mlp_n_blocks,
        mlp_dropout         = args.mlp_dropout,
        mlp_drop_path       = args.mlp_drop_path,
        fusion_dim          = args.fusion_dim,
        fusion_tokens       = args.fusion_tokens,
        fusion_heads        = args.fusion_heads,
        fusion_dropout      = args.fusion_dropout,
        classifier_dropout  = args.classifier_dropout,
        focal_gamma         = args.focal_gamma,
        label_smoothing     = args.label_smoothing,
        mixup_alpha         = args.mixup_alpha,
        learning_rate       = args.lr,
        weight_decay        = args.weight_decay,
        warmup_epochs       = args.warmup_epochs,
        max_epochs          = args.max_epochs,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nAcousticOmniResNet — {n_params / 1e6:.2f}M trainable parameters")
    print(f"  1D branch  : ResidualMLP  138 → {args.mlp_dim} ({args.mlp_n_blocks} blocks)")
    print(f"  2D branch  : ResNet-50    9-ch → 2048")
    print(f"  Fusion     : CrossAttentionFusion {args.fusion_dim}d "
          f"({args.fusion_tokens} tokens × {args.fusion_dim // args.fusion_tokens}d, "
          f"{args.fusion_heads} heads)")
    print(f"  Classes    : {data.class_to_idx}")

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            monitor="val/f1",
            mode="max",
            save_top_k=3,
            filename="omni-{epoch:03d}-f1{val/f1:.4f}",
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

    logger = CSVLogger("lightning_logs", name="acoustic_omni_resnet")

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
        num_sanity_val_steps=2,
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
