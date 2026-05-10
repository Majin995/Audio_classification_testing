"""
Training script for I2HOFI in LOFAR (raw STFT) input mode.

The Hz-per-ROI-grid-square ratio (`--hz_per_grid`) is the single knob that
drives both the STFT resolution and the i2-hofi ROI grid height:

    n_fft        = round(sample_rate / hz_per_grid)             # 1 STFT bin == hz_per_grid Hz
    n_freq_bands = round(stft_f_max_hz / hz_per_grid)
    grid_h       = min(n_freq_bands, max_grid_h)                # safety cap on GAT cost
    hop_length   = max(1, (fixed_len - n_fft) // (time_frames - 1))

Usage:
    export DATA_DIR=/path/to/Split1s
    python training/train_i2hofi.py [options]
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
from models.hydro_i2hofi import I2HOFI


def get_args():
    p = argparse.ArgumentParser(description="Train I2HOFI (LOFAR mode)")

    # Data
    p.add_argument("--data_dir",      default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",    type=int, default=16)
    p.add_argument("--num_threads",   type=int, default=8)
    p.add_argument("--no_oversample", action="store_true")
    p.add_argument("--denoise",       default="off",
                   choices=["off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"])

    # Audio
    p.add_argument("--sample_rate",   type=int, default=5_120)
    p.add_argument("--fixed_len",     type=int, default=5_120)

    # LOFAR / grid
    p.add_argument("--hz_per_grid",   type=float, default=1.0,
                   help="Hz per STFT bin == Hz per ROI grid row (search range 0.5–5.0).")
    p.add_argument("--stft_f_max_hz", type=float, default=2_560.0,
                   help="STFT band-limit (Hz). Default 2560 = full Nyquist for 5120 Hz SR.")
    p.add_argument("--time_frames",   type=int,   default=32,
                   help="Target STFT time frames (sets hop_length).")
    p.add_argument("--grid_w",        type=int,   default=4,
                   help="ROI grid width (time axis).")
    p.add_argument("--max_grid_h",    type=int,   default=16,
                   help="Cap on grid_h. Each grid row consumes ~32 spectrogram rows "
                        "(backbone downsampling), so cost scales linearly in grid_h "
                        "for memory and quadratically for the inter-ROI GAT.")

    # Model
    p.add_argument("--backbone",          default="resnet18")
    p.add_argument("--pretrained",        action="store_true", default=True)
    p.add_argument("--no_pretrained",     dest="pretrained", action="store_false")
    p.add_argument("--gcn_out_features",  type=int,   default=256)
    p.add_argument("--gat_out_features",  type=int,   default=256)
    p.add_argument("--appnp_K",           type=int,   default=3)
    p.add_argument("--alpha",             type=float, default=0.3)
    p.add_argument("--gat_heads",         type=int,   default=1)
    p.add_argument("--pool_h",            type=int,   default=3)
    p.add_argument("--pool_w",            type=int,   default=3)

    # Per-layer dropout
    p.add_argument("--dropout",            type=float, default=0.2)
    p.add_argument("--dropout_appnp",      type=float, default=None)
    p.add_argument("--dropout_gat",        type=float, default=None)
    p.add_argument("--dropout_classifier", type=float, default=None)

    # Loss
    p.add_argument("--focal_gamma",     type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--mixup_alpha",     type=float, default=0.0)  # not consumed by I2HOFI

    # Training
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-3)
    p.add_argument("--max_epochs",    type=int,   default=100)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--patience",      type=int,   default=20)
    p.add_argument("--precision",     default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--run_name",      default="i2hofi_lofar")

    # Dev flags
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)

    return p.parse_args()


def _derive_lofar_params(args) -> dict:
    """
    Optimised for 1-second snippets (``fixed_len == sample_rate``).

    The *true* frequency resolution of a window of T seconds is 1/T Hz, so for
    a 1 s clip at 5120 Hz the honest maximum is **1 Hz/bin**.  We pin the
    Hann window to the entire snippet (``win_length = fixed_len``) to hit that
    maximum, and pin ``n_fft = win_length`` — zero-padding higher would only
    interpolate (not resolve) and breaks ``torch.stft(center=True)`` once
    ``n_fft//2 > input_length``.

    ``hz_per_grid`` therefore drives only the band-of-interest slicing
    (``n_freq_bands`` and the ROI grid height) — the FFT itself stays at the
    true 1 Hz/bin maximum.  For sub-Hz ``hz_per_grid`` we round up to the
    nearest 1 Hz, since that is the real ceiling.

    Time axis: with ``center=True`` and ``hop_length = fixed_len / time_frames``
    we get ~``time_frames`` STFT frames spanning the clip.
    """
    win_length     = args.fixed_len
    n_fft          = win_length                                # max real Δf for 1 s
    real_hz_per_bin = args.sample_rate / win_length            # = 1.0 Hz for 1 s @ 5120 Hz

    effective_hpg  = max(real_hz_per_bin, args.hz_per_grid)    # honest floor
    n_freq_bands   = max(1, int(round(args.stft_f_max_hz / effective_hpg)))
    grid_h         = max(1, min(n_freq_bands, args.max_grid_h))

    hop_length     = max(1, args.fixed_len // max(1, args.time_frames))
    return dict(
        n_fft=n_fft, n_freq_bands=n_freq_bands, grid_h=grid_h,
        hop_length=hop_length, win_length=win_length,
        real_hz_per_bin=real_hz_per_bin, effective_hz_per_grid=effective_hpg,
    )


def main(args=None):
    if args is None:
        args = get_args()

    if not args.data_dir:
        raise ValueError("Set --data_dir or export DATA_DIR=/path/to/Split1s")

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    derived = _derive_lofar_params(args)
    note = ""
    if derived["effective_hz_per_grid"] > args.hz_per_grid + 1e-6:
        note = (f"  [note] requested {args.hz_per_grid:.3f} Hz/grid is below the "
                f"true 1-second resolution ({derived['real_hz_per_bin']:.3f} Hz); "
                f"clamped to {derived['effective_hz_per_grid']:.3f}.")
    print(
        "\n[i2hofi/lofar] derived from hz_per_grid="
        f"{args.hz_per_grid:.3f}: "
        f"n_fft={derived['n_fft']}, hop={derived['hop_length']}, "
        f"win={derived['win_length']}, n_freq_bands={derived['n_freq_bands']}, "
        f"grid=({derived['grid_h']}, {args.grid_w}), "
        f"stft_f_max_hz={args.stft_f_max_hz}, "
        f"true_Δf={derived['real_hz_per_bin']:.3f} Hz/bin"
        + (f"\n{note}" if note else "")
    )

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

    model = I2HOFI(
        num_classes      = data.num_classes,
        class_weights    = data.class_weights,
        backbone         = args.backbone,
        pretrained       = args.pretrained,
        grid_size        = (derived["grid_h"], args.grid_w),
        pool_size        = (args.pool_h, args.pool_w),
        gcn_out_features = args.gcn_out_features,
        gat_out_features = args.gat_out_features,
        appnp_K          = args.appnp_K,
        alpha            = args.alpha,
        gat_heads        = args.gat_heads,
        intra_adj_mode   = "full",
        inter_adj_mode   = "full",
        dropout          = args.dropout,
        dropout_appnp      = args.dropout_appnp,
        dropout_gat        = args.dropout_gat,
        dropout_classifier = args.dropout_classifier,
        focal_gamma      = args.focal_gamma,
        label_smoothing  = args.label_smoothing,
        learning_rate    = args.lr,
        weight_decay     = args.weight_decay,
        warmup_epochs    = args.warmup_epochs,
        max_epochs       = args.max_epochs,
        # LOFAR frontend
        lofar_input      = True,
        sample_rate      = args.sample_rate,
        n_fft            = derived["n_fft"],
        hop_length       = derived["hop_length"],
        win_length       = derived["win_length"],
        stft_f_max_hz    = args.stft_f_max_hz,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"I2HOFI (lofar) — {n_params / 1e6:.2f}M trainable parameters\n")

    callbacks = [
        ModelCheckpoint(
            monitor="val/f1", mode="max", save_top_k=2,
            filename="i2hofi-{epoch:03d}-f1{val/f1:.4f}", verbose=True,
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
    print(f"Best val/f1:     {best_score:.4f}" if best_score is not None else "No best score.")
    return float(best_score) if best_score is not None else float("nan"), best_path


if __name__ == "__main__":
    main()
