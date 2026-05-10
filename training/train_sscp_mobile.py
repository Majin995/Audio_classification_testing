"""
Training script for HydroSSCPMobile (sub-128kB edge CNN + Knowledge Distillation).

Usage:
    export DATA_DIR=/path/to/Split1s
    # Without KD (standalone):
    python training/train_sscp_mobile.py

    # With BAHTNet teacher:
    python training/train_sscp_mobile.py \\
        --teacher_ckpt lightning_logs/hydro_bahtnet/version_0/checkpoints/best.ckpt
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
from models.hydro_sscp_mobile import HydroSSCPMobile


def get_args():
    p = argparse.ArgumentParser(description="Train HydroSSCPMobile")

    # Data
    p.add_argument("--data_dir",       default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",     type=int,   default=128,
                   help="Larger batch helps small-model stability")
    p.add_argument("--num_threads",    type=int,   default=8)
    p.add_argument("--no_oversample",  action="store_true")
    p.add_argument("--denoise",        default="off",
                   choices=["off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"])

    # Audio
    p.add_argument("--sample_rate",    type=int,   default=5_120)
    p.add_argument("--fixed_len",      type=int,   default=5_120)
    p.add_argument("--n_mels",         type=int,   default=32)
    p.add_argument("--n_fft",          type=int,   default=256)
    p.add_argument("--hop_length",     type=int,   default=51)

    # KD
    p.add_argument("--teacher_ckpt",   default=os.environ.get("BAHTNET_BEST_CKPT", ""),
                   help="Path to BAHTNet checkpoint for KD.  Set $BAHTNET_BEST_CKPT or pass directly.")
    p.add_argument("--kd_alpha",       type=float, default=0.3,
                   help="Weight of CE loss. 1-alpha is the KD weight.")
    p.add_argument("--kd_temp",        type=float, default=4.0,
                   help="KD softmax temperature T.")

    # Model
    p.add_argument("--dropout",        type=float, default=0.05)
    p.add_argument("--mixup_alpha",    type=float, default=0.2)
    p.add_argument("--focal_gamma",    type=float, default=2.0)
    p.add_argument("--label_smoothing",type=float, default=0.05)

    # Training
    p.add_argument("--lr",             type=float, default=1e-3,
                   help="Higher LR works better for very small models")
    p.add_argument("--weight_decay",   type=float, default=1e-3)
    p.add_argument("--max_epochs",     type=int,   default=100)
    p.add_argument("--warmup_epochs",  type=int,   default=5)
    p.add_argument("--patience",       type=int,   default=25)
    p.add_argument("--precision",      default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--run_name",       default="hydro_sscp_mobile")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)

    return p.parse_args()


def main(args=None):
    if args is None:
        args = get_args()

    if not args.data_dir:
        raise ValueError("Set --data_dir or export DATA_DIR=/path/to/Split1s")

    teacher_ckpt = args.teacher_ckpt or None

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

    model = HydroSSCPMobile(
        num_classes=data.num_classes,
        class_weights=data.class_weights,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        teacher_ckpt=teacher_ckpt,
        kd_alpha=args.kd_alpha,
        kd_temp=args.kd_temp,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        mixup_alpha=args.mixup_alpha,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHydroSSCPMobile — {n_params} trainable parameters "
          f"({n_params * 4 / 1024:.1f} kB fp32)")
    if teacher_ckpt:
        print(f"Teacher: {teacher_ckpt}")
        print(f"KD: α={args.kd_alpha}, T={args.kd_temp}")
    else:
        print("No teacher — supervised-only training")

    callbacks = [
        ModelCheckpoint(
            monitor="val/f1", mode="max", save_top_k=3,
            filename="sscp-{epoch:03d}-f1{val/f1:.4f}", verbose=True,
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
