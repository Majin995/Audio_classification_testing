"""
Training script for HydroBAHTNet (Boundary-Aware Hybrid Transformer).

Usage:
    export DATA_DIR=/path/to/Split1s
    python training/train_bahtnet.py [options]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_bahtnet import HydroBAHTNet


def get_args():
    p = argparse.ArgumentParser(description="Train HydroBAHTNet")

    # Data
    p.add_argument("--data_dir",       default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--num_threads",    type=int,   default=8)
    p.add_argument("--no_oversample",  action="store_true")
    p.add_argument("--denoise",        default="off",
                   choices=["off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"])

    # Audio
    p.add_argument("--sample_rate",    type=int,   default=5_120)
    p.add_argument("--fixed_len",      type=int,   default=5_120)
    p.add_argument("--n_mels",         type=int,   default=64)
    p.add_argument("--n_fft",          type=int,   default=512)
    p.add_argument("--hop_length",     type=int,   default=51)

    # Model
    p.add_argument("--patch_size",     type=int,   default=4)
    p.add_argument("--model_dim",      type=int,   default=384)
    p.add_argument("--n_heads",        type=int,   default=6)
    p.add_argument("--n_layers",       type=int,   default=6)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--drop_path",      type=float, default=0.15)
    p.add_argument("--mixup_alpha",    type=float, default=0.3)

    # Loss (LMF specific)
    p.add_argument("--loss",           default="lmf",
                   choices=["focal", "lmf"],
                   help="Use lmf (LargeMarginFocal, default) or focal only")
    p.add_argument("--lmf_gamma",      type=float, default=2.0)
    p.add_argument("--lmf_margin",     type=float, default=0.35)
    p.add_argument("--label_smoothing",type=float, default=0.05)

    # Training
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-2)
    p.add_argument("--max_epochs",     type=int,   default=100)
    p.add_argument("--warmup_epochs",  type=int,   default=10)
    p.add_argument("--patience",       type=int,   default=20)
    p.add_argument("--precision",      default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--run_name",       default="hydro_bahtnet")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)

    return p.parse_args()


def main(args=None):
    if args is None:
        args = get_args()

    if not args.data_dir:
        raise ValueError("Set --data_dir or export DATA_DIR=/path/to/Split1s")

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        oversample_train=not args.no_oversample,
        denoise_method=args.denoise,
    )
    data.setup()

    # Use margin=0 to reduce LMF to standard Focal when --loss focal is passed
    lmf_margin = args.lmf_margin if args.loss == "lmf" else 0.0

    model = HydroBAHTNet(
        num_classes=data.num_classes,
        class_weights=data.class_weights,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        patch_size=args.patch_size,
        model_dim=args.model_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        drop_path=args.drop_path,
        lmf_gamma=args.lmf_gamma,
        lmf_margin=lmf_margin,
        label_smoothing=args.label_smoothing,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        mixup_alpha=args.mixup_alpha,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHydroBAHTNet — {n_params / 1e6:.2f}M trainable parameters")
    print(f"Loss: LMF (margin={lmf_margin:.2f}, γ={args.lmf_gamma})")

    callbacks = [
        ModelCheckpoint(
            monitor="val/f1", mode="max", save_top_k=3,
            filename="bahtnet-{epoch:03d}-f1{val/f1:.4f}", verbose=True,
        ),
        EarlyStopping(monitor="val/loss", patience=args.patience, mode="min", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu", devices=1,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=CSVLogger("lightning_logs", name=args.run_name),
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        deterministic=False,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")

    best_path  = trainer.checkpoint_callback.best_model_path
    best_score = trainer.checkpoint_callback.best_model_score
    print(f"\nBest checkpoint: {best_path}")
    print(f"Best val/f1:     {best_score:.4f}")
    return float(best_score) if best_score is not None else float("nan"), best_path


if __name__ == "__main__":
    main()
