"""
Training script for HydroQuantumTransfer — frozen UATR backbone + dressed
Quantum-Classifier head.

Quickstart (after training a BAHTNet teacher):
    export DATA_DIR=/path/to/Split1s
    python training/train_quantum_transfer.py \\
        --teacher_arch bahtnet \\
        --teacher_ckpt lightning_logs/hydro_bahtnet/version_X/checkpoints/best.ckpt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_quantum_transfer import HydroQuantumTransfer


def get_args():
    p = argparse.ArgumentParser(description="Train HydroQuantumTransfer")

    # Data
    p.add_argument("--data_dir",    default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--no_oversample", action="store_true")
    p.add_argument("--sample_rate", type=int, default=5_120)
    p.add_argument("--fixed_len",   type=int, default=5_120)

    # Teacher backbone
    p.add_argument("--teacher_arch", default="bahtnet",
                   help="Registry key for the frozen backbone "
                        "(e.g. bahtnet, dart_mt, catfish).")
    p.add_argument("--teacher_ckpt", default=None,
                   help="Path to a Lightning .ckpt for the teacher (REQUIRED "
                        "for meaningful results — without it, the teacher is "
                        "randomly initialised and only smoke-test wiring works).")
    p.add_argument("--teacher_classifier_attr", default="classifier",
                   help="Attribute name on the teacher to replace with Identity.")
    p.add_argument("--teacher_feat_dim", type=int, default=None,
                   help="Manual override for the auto-detected feature dim.")

    # Quantum head
    p.add_argument("--dressed_dim", type=int, default=32)
    p.add_argument("--n_qubits",    type=int, default=8)
    p.add_argument("--q_layers",    type=int, default=4)
    p.add_argument("--n_reuploads", type=int, default=1)

    # Training (head-only — short schedule)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--max_epochs",    type=int,   default=30)
    p.add_argument("--warmup_epochs", type=int,   default=3)
    p.add_argument("--lmf_gamma",     type=float, default=2.0)
    p.add_argument("--lmf_margin",    type=float, default=0.35)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--precision",     default="32",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--accelerator",   default="auto")

    return p.parse_args()


def main():
    args = get_args()

    if not args.data_dir:
        raise ValueError("Set DATA_DIR or pass --data_dir /path/to/Split1s")
    if args.teacher_ckpt is None:
        print("[WARN] --teacher_ckpt not provided — teacher backbone is "
              "RANDOMLY initialised. Use this only for smoke testing.")

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        oversample_train=not args.no_oversample,
    )
    data.setup()

    model = HydroQuantumTransfer(
        num_classes              = data.num_classes,
        class_weights            = data.class_weights,
        teacher_arch             = args.teacher_arch,
        teacher_ckpt             = args.teacher_ckpt,
        teacher_feat_dim         = args.teacher_feat_dim,
        teacher_classifier_attr  = args.teacher_classifier_attr,
        dressed_dim              = args.dressed_dim,
        n_qubits                 = args.n_qubits,
        n_layers                 = args.q_layers,
        n_reuploads              = args.n_reuploads,
        lmf_gamma                = args.lmf_gamma,
        lmf_margin               = args.lmf_margin,
        label_smoothing          = args.label_smoothing,
        learning_rate            = args.lr,
        weight_decay             = args.weight_decay,
        warmup_epochs            = args.warmup_epochs,
        max_epochs               = args.max_epochs,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\nHydroQuantumTransfer — {trainable / 1e3:.1f}k trainable / "
          f"{frozen / 1e6:.2f}M frozen parameters "
          f"({args.n_qubits} qubits × {args.q_layers} layers)")

    callbacks = [
        ModelCheckpoint(monitor="val/f1", mode="max", save_top_k=3,
                        filename="qtransfer-{epoch:03d}-f1{val/f1:.4f}",
                        verbose=True),
        EarlyStopping(monitor="val/loss", patience=10, mode="min", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = CSVLogger("lightning_logs", name="hydro_quantum_transfer")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        deterministic=False,
    )
    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")
    print(f"\nBest checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val/f1:     {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
