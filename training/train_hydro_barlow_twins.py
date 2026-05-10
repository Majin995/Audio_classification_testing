"""
Training script for HydroBarlowTwins — SSL pretraining via Barlow Twins.

Usage:
    export DATA_DIR=/path/to/Split1s
    python training/train_hydro_barlow_twins.py [options]

Monitors ``val/bt_loss`` for checkpointing.  After training, the encoder
state-dict is saved next to the best checkpoint as ``encoder.pt`` so a
downstream classifier can fine-tune from the SSL-pretrained weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_barlow_twins   import HydroBarlowTwins


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="Pretrain HydroBarlowTwins (Barlow Twins SSL on UATR waveforms)"
    )

    # Data
    p.add_argument("--data_dir",      default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--num_threads",   type=int, default=8)
    p.add_argument("--no_oversample", action="store_true")

    # Audio
    p.add_argument("--sample_rate",   type=int, default=5_120)
    p.add_argument("--fixed_len",     type=int, default=5_120)

    # Projector / loss
    p.add_argument("--projection_hidden", type=int,   default=2048)
    p.add_argument("--projection_dim",    type=int,   default=2048)
    p.add_argument("--lambda_param",      type=float, default=5e-3)

    # Two-view aug
    p.add_argument("--noise_prob",    type=float, default=0.8)
    p.add_argument("--noise_snr_min", type=float, default=10.0)
    p.add_argument("--noise_snr_max", type=float, default=30.0)
    p.add_argument("--gain_prob",     type=float, default=0.8)
    p.add_argument("--gain_range",    type=float, default=0.4)
    p.add_argument("--shift_prob",    type=float, default=0.5)
    p.add_argument("--shift_max",     type=float, default=0.1)

    # Encoder widths (forwarded to HydroPrecise)
    p.add_argument("--gabor_n_filters", type=int, default=64)
    p.add_argument("--gabor_kernel",    type=int, default=257)
    p.add_argument("--gabor_ch",        type=int, default=128)
    p.add_argument("--cqt_n_bins",      type=int, default=84)
    p.add_argument("--cqt_bpo",         type=int, default=12)
    p.add_argument("--cqt_hop",         type=int, default=64)
    p.add_argument("--cqt_ch",          type=int, default=128)
    p.add_argument("--demon_hop",       type=int, default=64)
    p.add_argument("--demon_ch",        type=int, default=64)
    p.add_argument("--demon_n_fft",     type=int, default=2048)
    p.add_argument("--fusion_T",        type=int, default=64)
    p.add_argument("--fusion_dim",      type=int, default=192)
    p.add_argument("--n_heads",         type=int, default=2)
    p.add_argument("--dropout",         type=float, default=0.15)

    # Training
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--max_epochs",    type=int,   default=100)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--precision",     default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--accelerator",   default="auto")
    p.add_argument("--devices",       default=1)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--run_name",      default="hydro_barlow_twins")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

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
    )
    data.setup()

    model = HydroBarlowTwins(
        num_classes=data.num_classes,
        sample_rate=args.sample_rate,
        projection_hidden=args.projection_hidden,
        projection_dim=args.projection_dim,
        lambda_param=args.lambda_param,
        noise_prob=args.noise_prob,
        noise_snr_min=args.noise_snr_min,
        noise_snr_max=args.noise_snr_max,
        gain_prob=args.gain_prob, gain_range=args.gain_range,
        shift_prob=args.shift_prob, shift_max=args.shift_max,
        gabor_n_filters=args.gabor_n_filters, gabor_kernel=args.gabor_kernel,
        gabor_ch=args.gabor_ch,
        cqt_n_bins=args.cqt_n_bins, cqt_bpo=args.cqt_bpo,
        cqt_hop=args.cqt_hop,       cqt_ch=args.cqt_ch,
        demon_hop=args.demon_hop,   demon_ch=args.demon_ch,
        demon_n_fft=args.demon_n_fft,
        fusion_T=args.fusion_T,     fusion_dim=args.fusion_dim,
        n_heads=args.n_heads,       dropout=args.dropout,
        learning_rate=args.lr,      weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHydroBarlowTwins — {n_params / 1e6:.2f}M trainable parameters")
    print(f"  projection: {args.projection_hidden} → {args.projection_dim}, "
          f"lambda = {args.lambda_param}")

    ckpt_cb = ModelCheckpoint(
        monitor="val/bt_loss", mode="min", save_top_k=3, save_last=True,
        filename="bt-{epoch:03d}-{val/bt_loss:.4f}",
        auto_insert_metric_name=False, verbose=True,
    )
    callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="epoch")]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator if args.accelerator != "auto"
                    else ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=args.devices,
        precision=args.precision if torch.cuda.is_available() else 32,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=[
            CSVLogger("lightning_logs", name=args.run_name),
            TensorBoardLogger("lightning_logs", name=args.run_name),
        ],
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        deterministic=False,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    trainer.fit(model, data)

    best_path  = ckpt_cb.best_model_path
    best_score = ckpt_cb.best_model_score
    print(f"\nBest checkpoint:    {best_path}")
    if best_score is not None:
        print(f"Best val/bt_loss:   {best_score:.4f}")

    # ── Export the encoder weights for downstream fine-tuning ────────────
    if best_path:
        out_dir = Path(best_path).parent
        encoder_path = out_dir / "encoder.pt"
        torch.save(model.encoder.state_dict(), encoder_path)
        print(f"Saved encoder state-dict → {encoder_path}")

    return float(best_score) if best_score is not None else float("nan"), best_path


if __name__ == "__main__":
    main()
