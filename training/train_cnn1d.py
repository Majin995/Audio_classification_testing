"""
Train HydroCNN1D — 1D Convolutional Feature Extractor.

This is Stage 1 of the CNN → Classical pipeline:
  1. Pretrain HydroCNN1D end-to-end on the 3-class vessel dataset.
  2. Save best checkpoint (monitored by val_f1_score).
  3. Run trainer.test() at the end for parity with classical head evaluation.

The trained checkpoint is then consumed by training/eval_classical.py
which freezes the CNN and feeds its 128-dim features into RVM / SVM / boosting.

Usage
-----
    export DATA_DIR=/path/to/Split1s
    python training/train_cnn1d.py

    # Faster sanity check (2 epochs):
    python training/train_cnn1d.py --max_epochs 2 --batch_size 32

    # Use 2560 Hz sample rate (matches user's 2560-sample / 1-s spec):
    python training/train_cnn1d.py --sample_rate 2560

Classes
-------
Cargo + Passenger are merged into a single "Cargo" label via the existing
DALIAudioDataModule.merge_classes kwarg, producing a 3-class task:
  {Cargo (merged), Tanker, Tug}
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_cnn1d import HydroCNN1D


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pretrain HydroCNN1D for classical-head feature extraction"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",    default=os.environ.get("DATA_DIR", ""),
                   help="Path to Split1s root directory (or set $DATA_DIR)")
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--num_workers", type=int,   default=8,
                   help="DALI pipeline threads per loader")
    p.add_argument("--no_oversample", action="store_true",
                   help="Disable minority-class oversampling in training")
    p.add_argument("--merge_cargo_passenger", action="store_true",
                   help="Merge Cargo+Passenger into one class (3-class task). "
                        "Default: all 4 classes.")

    # ── Audio ─────────────────────────────────────────────────────────────────
    p.add_argument("--sample_rate", type=int,   default=5_120,
                   choices=[2560, 5120],
                   help="Target sample rate. 5120 matches repo convention; "
                        "2560 gives exactly 2560 samples per 1-s clip.")

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--lr",              type=float, default=3e-4,
                   help="Peak learning rate (reached after warmup)")
    p.add_argument("--warmup_epochs",   type=int,   default=5,
                   help="Linear warmup length before cosine decay")
    p.add_argument("--focal_gamma",     type=float, default=2.0,
                   help="FocalLoss focusing parameter (0 = standard CE)")
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--mixup_alpha",     type=float, default=0.3,
                   help="Beta distribution alpha for waveform mixup (0 = off)")
    p.add_argument("--cd_dropout_reg",  type=float, default=1e-5,
                   help="ConcreteDropout entropy regularisation coefficient")
    p.add_argument("--f_beta",          type=float, default=1.0,
                   help="Beta for FBetaScore metric (1.0 = standard F1)")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--max_epochs",  type=int,   default=100)
    p.add_argument("--patience",    type=int,   default=15,
                   help="EarlyStopping patience on val_f1_score (no-improvement epochs)")
    p.add_argument("--precision",   default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=42)

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--log_dir",     default="lightning_logs",
                   help="Root directory for CSVLogger output")
    p.add_argument("--ckpt_dir",    default=None,
                   help="Override checkpoint save directory (default: inside log_dir)")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    if not args.data_dir:
        raise ValueError(
            "data_dir is empty. Pass --data_dir or set the DATA_DIR env var."
        )

    # ── DataModule ────────────────────────────────────────────────────────────
    merge = {"Passenger": "Cargo"} if args.merge_cargo_passenger else {}
    dm = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_workers,
        target_sr=args.sample_rate,
        fixed_len=args.sample_rate,
        oversample_train=not args.no_oversample,
        merge_classes=merge,
    )
    dm.setup()
    num_classes = dm.num_classes          # 3

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HydroCNN1D(
        num_classes=num_classes,
        input_len=args.sample_rate,
        class_weights=dm.class_weights,   # FocalLoss per-class alpha
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        cd_dropout_reg=args.cd_dropout_reg,
        f_beta=args.f_beta,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_dir = args.ckpt_dir or None
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_f1_score",
            mode="max",
            save_top_k=1,
            filename="cnn1d-{epoch:02d}-{val_f1_score:.3f}",
            verbose=True,
        ),
        # Monitor val_f1_score (not val_loss) — focal loss values aren't
        # comparable across epochs so loss plateaus are misleading.
        EarlyStopping(
            monitor="val_f1_score",
            patience=args.patience,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = CSVLogger(save_dir=args.log_dir, name="cnn1d")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        devices=1,
        accelerator="gpu",
    )

    trainer.fit(model, datamodule=dm)

    # Run test with the best checkpoint — metrics logged to metrics.csv for
    # direct comparison with classical-head results in eval_classical.py.
    trainer.test(model, datamodule=dm, ckpt_path="best")

    best_path = trainer.checkpoint_callback.best_model_path
    best_model = HydroCNN1D.load_from_checkpoint(best_path)
    print(f"\nBest checkpoint : {best_path}")
    print(f"Learned dropout rates:")
    print(f"  cd1 (after conv1): p = {best_model.cd1.p.item():.4f}")
    print(f"  cd2 (after conv2): p = {best_model.cd2.p.item():.4f}")
    print(f"  cd3 (after conv3): p = {best_model.cd3.p.item():.4f}")
    print(f"\nRun comparison  : python training/eval_classical.py --ckpt {best_path!r}")


if __name__ == "__main__":
    main()
