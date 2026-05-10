"""
Training script for HydroCNNLSTMQC — CNN → Bi-LSTM → Variational Quantum Classifier.

Quickstart:
    export DATA_DIR=/path/to/Split1s
    python training/train_cnnlstm_qc.py
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
from models.hydro_cnnlstm_qc import HydroCNNLSTMQC


def get_args():
    p = argparse.ArgumentParser(description="Train HydroCNNLSTMQC")

    # Data
    p.add_argument("--data_dir",    default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--no_oversample", action="store_true")
    p.add_argument("--sample_rate", type=int, default=5_120)
    p.add_argument("--fixed_len",   type=int, default=5_120)

    # Quantum head
    p.add_argument("--n_qubits",    type=int, default=8)
    p.add_argument("--q_layers",    type=int, default=4)
    p.add_argument("--n_reuploads", type=int, default=1)

    # Classical frontend
    p.add_argument("--lstm_hidden", type=int, default=64)
    p.add_argument("--lstm_layers", type=int, default=2)

    # Training
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--max_epochs",    type=int,   default=100)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--mixup_alpha",   type=float, default=0.3)
    p.add_argument("--lmf_gamma",     type=float, default=2.0)
    p.add_argument("--lmf_margin",    type=float, default=0.35)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--precision",     default="32",
                   choices=["32", "16-mixed", "bf16-mixed"],
                   help="Quantum sim usually requires fp32 — keep default.")
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--accelerator",   default="auto")

    return p.parse_args()


def main():
    args = get_args()

    if not args.data_dir:
        raise ValueError(
            "No data directory specified.\n"
            "  Set DATA_DIR or pass --data_dir /path/to/Split1s"
        )

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

    model = HydroCNNLSTMQC(
        num_classes     = data.num_classes,
        class_weights   = data.class_weights,
        lstm_hidden     = args.lstm_hidden,
        lstm_layers     = args.lstm_layers,
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
        mixup_alpha     = args.mixup_alpha,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHydroCNNLSTMQC — {n_params / 1e6:.3f}M trainable parameters "
          f"({args.n_qubits} qubits, {args.q_layers} variational layers)")

    callbacks = [
        ModelCheckpoint(monitor="val/f1", mode="max", save_top_k=3,
                        filename="cnnlstm-qc-{epoch:03d}-f1{val/f1:.4f}",
                        verbose=True),
        EarlyStopping(monitor="val/loss", patience=20, mode="min", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = CSVLogger("lightning_logs", name="hydro_cnnlstm_qc")

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
