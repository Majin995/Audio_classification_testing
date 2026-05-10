"""
Training script for HydroVQCFeatures — Variational Quantum Classifier on
cached 138-D acoustic feature vectors.

Quickstart:
    export DATA_DIR=/path/to/Split1s
    python training/train_vqc_features.py
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

from data.cached_feature_dataset import OmniFeatureDataModule
from models.hydro_vqc_features import HydroVQCFeatures


def get_args():
    p = argparse.ArgumentParser(description="Train HydroVQCFeatures")

    # Data
    p.add_argument("--data_dir",   default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--cache_dir",  default="data/cache")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precompute_workers", type=int, default=8)
    p.add_argument("--sample_rate", type=int, default=5_120)
    p.add_argument("--no_oversample", action="store_true")
    p.add_argument("--force_recompute", action="store_true")

    # Quantum head
    p.add_argument("--compress_dim", type=int, default=16)
    p.add_argument("--n_qubits",     type=int, default=8)
    p.add_argument("--q_layers",     type=int, default=6)
    p.add_argument("--n_reuploads",  type=int, default=1)

    # Training
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--max_epochs",    type=int,   default=100)
    p.add_argument("--warmup_epochs", type=int,   default=5)
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

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data = OmniFeatureDataModule(
        data_dir            = args.data_dir,
        cache_dir           = args.cache_dir,
        batch_size          = args.batch_size,
        num_workers         = args.num_workers,
        precompute_workers  = args.precompute_workers,
        sample_rate         = args.sample_rate,
        oversample_train    = not args.no_oversample,
        force_recompute     = args.force_recompute,
    )
    data.setup()

    model = HydroVQCFeatures(
        num_classes     = data.num_classes,
        class_weights   = data.class_weights,
        compress_dim    = args.compress_dim,
        n_qubits        = args.n_qubits,
        n_layers        = args.q_layers,
        n_reuploads     = args.n_reuploads,
        lmf_gamma       = args.lmf_gamma,
        lmf_margin      = args.lmf_margin,
        label_smoothing = args.label_smoothing,
        learning_rate   = args.lr,
        weight_decay    = args.weight_decay,
        warmup_epochs   = args.warmup_epochs,
        max_epochs      = args.max_epochs,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHydroVQCFeatures — {n_params / 1e3:.1f}k trainable parameters "
          f"({args.n_qubits} qubits × {args.q_layers} layers)")

    callbacks = [
        ModelCheckpoint(monitor="val/f1", mode="max", save_top_k=3,
                        filename="vqc-feat-{epoch:03d}-f1{val/f1:.4f}",
                        verbose=True),
        EarlyStopping(monitor="val/loss", patience=15, mode="min", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = CSVLogger("lightning_logs", name="hydro_vqc_features")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        deterministic=False,
    )
    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")
    print(f"\nBest checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val/f1:     {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
