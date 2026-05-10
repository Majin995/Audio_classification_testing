"""
Training script for SuperModel1D.

Usage:
    export DATA_DIR=/path/to/Split1s
    python training/train_super_model.py [options]

Mirrors the conventions of training/train_catfish.py:
  - main(args=None) returns (best_val_f1: float, best_ckpt_path: str)
  - CSVLogger under lightning_logs/<run_name>
  - ModelCheckpoint on val/f1, EarlyStopping on val/loss
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from data.audio_lightning_loader import DALIAudioDataModule
from models.super_model import SuperModel1D


def get_args():
    p = argparse.ArgumentParser(description="Train SuperModel1D")

    # ── Data ──────────────────────────────────────────────────────────
    p.add_argument("--data_dir",      default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--num_threads",   type=int,   default=8)
    p.add_argument("--no_oversample", action="store_true")
    p.add_argument("--denoise",       default="off",
                   choices=["off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"])

    # ── Audio ─────────────────────────────────────────────────────────
    p.add_argument("--sample_rate",   type=int,   default=5_120)
    p.add_argument("--fixed_len",     type=int,   default=5_120)

    # ── Stream G (Gabor 1D) ───────────────────────────────────────────
    p.add_argument("--gabor_n_filters", type=int,   default=64)
    p.add_argument("--gabor_kernel",    type=int,   default=257)
    p.add_argument("--g_ch",            type=int,   default=48)
    p.add_argument("--stem_stride",     type=int,   default=4)

    # ── Stream S (EnhancedFrontEnd spectrogram) ───────────────────────
    p.add_argument("--spec_n_mels",   type=int,   default=32)
    p.add_argument("--spec_hop",      type=int,   default=51)
    p.add_argument("--wb_n_fft",      type=int,   default=256)
    p.add_argument("--nb_n_fft",      type=int,   default=1024)
    p.add_argument("--s_ch",          type=int,   default=48)
    p.add_argument("--ceps_low_q",    type=int,   default=3)
    p.add_argument("--ceps_high_q",   type=int,   default=15)
    p.add_argument("--no_stream_s",   action="store_true",
                   help="Disable EnhancedFrontEnd stream (ablation)")

    # ── Stream L (LofarFrontend) ──────────────────────────────────────
    p.add_argument("--lofar_n_fft",       type=int,   default=1024)
    p.add_argument("--lofar_hop",         type=int,   default=51)
    p.add_argument("--lofar_time_bins",   type=int,   default=32)
    p.add_argument("--lofar_freq_bins",   type=int,   default=256)
    p.add_argument("--l_ch",              type=int,   default=32)
    p.add_argument("--no_stream_l",       action="store_true",
                   help="Disable LofarFrontend stream (ablation)")

    # ── Backbone ──────────────────────────────────────────────────────
    p.add_argument("--d_model",         type=int,   default=128)
    p.add_argument("--scale",           type=int,   default=8)
    p.add_argument("--dilations",       default="2,4,8",
                   help="Comma-separated dilation rates, e.g. '2,4,8' or '1,2,4,8'")
    p.add_argument("--n_dart",          type=int,   default=2)
    p.add_argument("--dart_heads",      type=int,   default=4)
    p.add_argument("--dart_dropout",    type=float, default=0.1)
    p.add_argument("--onset_patch",     type=int,   default=4)
    p.add_argument("--n_s4",            type=int,   default=2)
    p.add_argument("--s4_d_state",      type=int,   default=64)
    p.add_argument("--n_mamba",         type=int,   default=2)
    p.add_argument("--mamba_d_state",   type=int,   default=16)
    p.add_argument("--mamba_expand",    type=int,   default=2)
    p.add_argument("--mamba_d_conv",    type=int,   default=4)
    p.add_argument("--drop_path_rate",  type=float, default=0.10)
    p.add_argument("--dropout",         type=float, default=0.24)

    # ── Training ──────────────────────────────────────────────────────
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=0.012)
    p.add_argument("--max_epochs",      type=int,   default=100)
    p.add_argument("--warmup_epochs",   type=int,   default=10)
    p.add_argument("--patience",        type=int,   default=20)
    p.add_argument("--precision",       default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--run_name",        default="super_model")

    # ── Augmentation ─────────────────────────────────────────────────
    p.add_argument("--mixup_alpha",     type=float, default=0.3)
    p.add_argument("--noise_prob",      type=float, default=0.5)
    p.add_argument("--noise_snr_min",   type=float, default=20.0)
    p.add_argument("--noise_snr_max",   type=float, default=40.0)
    p.add_argument("--gain_prob",       type=float, default=0.7)

    # ── Loss ──────────────────────────────────────────────────────────
    p.add_argument("--focal_gamma",     type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)

    # ── CSSD ─────────────────────────────────────────────────────────
    p.add_argument("--cssd_alpha",       type=float, default=1.0,
                   help="CE weight; 1.0 = CSSD disabled")
    p.add_argument("--cssd_temp",        type=float, default=4.0)
    p.add_argument("--cssd_degrade_sr",  type=int,   default=2048,
                   help="Target SR for CSSD degradation (must be < sample_rate)")
    p.add_argument("--cssd_degrade_prob",type=float, default=0.5)
    p.add_argument("--ema_momentum",     type=float, default=0.999)

    # ── Dev flags ─────────────────────────────────────────────────────
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

    # Parse dilations from comma-separated string (or list if already passed by Optuna)
    if isinstance(args.dilations, str):
        dilation_rates = [int(d) for d in args.dilations.split(",")]
    else:
        dilation_rates = list(args.dilations)

    # Build datamodule
    data = DALIAudioDataModule(
        data_dir          = args.data_dir,
        batch_size        = args.batch_size,
        num_threads       = args.num_threads,
        target_sr         = args.sample_rate,
        fixed_len         = args.fixed_len,
        oversample_train  = not args.no_oversample,
        denoise_method    = args.denoise,
    )
    data.setup()

    # Retrieve optional optuna_trial forwarded by tune_super_model.py
    optuna_trial = getattr(args, "optuna_trial", None)

    # Build model
    model = SuperModel1D(
        num_classes       = data.num_classes,
        class_weights     = data.class_weights,
        sample_rate       = args.sample_rate,
        # Stream G
        gabor_n_filters   = args.gabor_n_filters,
        gabor_kernel      = args.gabor_kernel,
        g_ch              = args.g_ch,
        stem_stride       = args.stem_stride,
        # Stream S
        spec_n_mels       = args.spec_n_mels,
        spec_hop          = args.spec_hop,
        wb_n_fft          = args.wb_n_fft,
        nb_n_fft          = args.nb_n_fft,
        s_ch              = args.s_ch,
        ceps_low_q        = args.ceps_low_q,
        ceps_high_q       = args.ceps_high_q,
        use_stream_s      = not args.no_stream_s,
        # Stream L
        lofar_n_fft       = args.lofar_n_fft,
        lofar_hop         = args.lofar_hop,
        lofar_time_bins   = args.lofar_time_bins,
        lofar_freq_bins   = args.lofar_freq_bins,
        l_ch              = args.l_ch,
        use_stream_l      = not args.no_stream_l,
        # Backbone
        d_model           = args.d_model,
        scale             = args.scale,
        dilation_rates    = dilation_rates,
        n_dart            = args.n_dart,
        dart_heads        = args.dart_heads,
        dart_dropout      = args.dart_dropout,
        onset_patch       = args.onset_patch,
        n_s4              = args.n_s4,
        s4_d_state        = args.s4_d_state,
        n_mamba           = args.n_mamba,
        mamba_d_state     = args.mamba_d_state,
        mamba_expand      = args.mamba_expand,
        mamba_d_conv      = args.mamba_d_conv,
        drop_path_rate    = args.drop_path_rate,
        dropout           = args.dropout,
        # Training
        learning_rate     = args.lr,
        weight_decay      = args.weight_decay,
        warmup_epochs     = args.warmup_epochs,
        max_epochs        = args.max_epochs,
        # Augmentation
        mixup_alpha       = args.mixup_alpha,
        noise_prob        = args.noise_prob,
        noise_snr_min     = args.noise_snr_min,
        noise_snr_max     = args.noise_snr_max,
        gain_prob         = args.gain_prob,
        # Loss
        focal_gamma       = args.focal_gamma,
        label_smoothing   = args.label_smoothing,
        # CSSD
        cssd_alpha        = args.cssd_alpha,
        cssd_temp         = args.cssd_temp,
        cssd_degrade_sr   = args.cssd_degrade_sr,
        cssd_degrade_prob = args.cssd_degrade_prob,
        ema_momentum      = args.ema_momentum,
        # Optuna
        optuna_trial      = optuna_trial,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nSuperModel1D — {n_params / 1e6:.2f} M trainable parameters")
    print(f"  Stream S (EnhancedFrontEnd): {'enabled' if not args.no_stream_s else 'DISABLED'}")
    print(f"  Stream L (LofarFrontend):    {'enabled' if not args.no_stream_l else 'DISABLED'}")
    print(f"  CSSD:                        {'enabled (α={:.2f})'.format(args.cssd_alpha) if args.cssd_alpha < 1.0 else 'disabled'}")

    callbacks = [
        ModelCheckpoint(
            monitor   = "val/f1",
            mode      = "max",
            save_top_k= 3,
            filename  = "super-{epoch:03d}-f1{val/f1:.4f}",
            verbose   = True,
        ),
        EarlyStopping(
            monitor  = "val/loss",
            patience = args.patience,
            mode     = "min",
            verbose  = True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs          = args.max_epochs,
        accelerator         = "gpu",
        devices             = 1,
        precision           = args.precision,
        gradient_clip_val   = args.grad_clip,
        callbacks           = callbacks,
        logger              = CSVLogger("lightning_logs", name=args.run_name),
        log_every_n_steps   = 20,
        num_sanity_val_steps= 0,
        deterministic       = False,
        limit_train_batches = args.limit_train_batches,
        limit_val_batches   = args.limit_val_batches,
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
