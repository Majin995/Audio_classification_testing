"""
Training script for HydroWaveScattering — kymatio Scattering1D + 1D backbone.

Same monitor + post-hoc calibration as train_wave1d.py / train_precise.py.
Requires `kymatio>=0.3.0` — the model __init__ raises if not installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from data.audio_lightning_loader   import DALIAudioDataModule
from models.hydro_wave_scattering  import HydroWaveScattering
from training.train_precise        import (
    _collect_logits, _fit_temperature, _search_thresholds,
)


def get_args():
    p = argparse.ArgumentParser(description="Train HydroWaveScattering")

    # Data
    p.add_argument("--data_dir",       default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--num_threads",    type=int, default=8)
    p.add_argument("--no_oversample",  action="store_true")
    p.add_argument("--denoise",        default="off",
                   choices=["off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"])

    # Audio
    p.add_argument("--sample_rate",    type=int, default=5_120)
    p.add_argument("--fixed_len",      type=int, default=5_120)

    # Scattering frontend
    p.add_argument("--J",              type=int, default=6,
                   help="Number of scattering scales (log2 of max scale).")
    p.add_argument("--Q",              type=int, default=12,
                   help="Wavelets per octave.")

    # Model
    p.add_argument("--d_model",         type=int, default=128)
    p.add_argument("--se_res2_blocks",  type=int, default=3)
    p.add_argument("--s4_blocks",       type=int, default=2)
    p.add_argument("--s4_d_state",      type=int, default=64)
    p.add_argument("--dropout",         type=float, default=0.15)
    p.add_argument("--drop_path",       type=float, default=0.10)

    # Loss
    p.add_argument("--loss",            default="lmf", choices=["lmf", "focal"])
    p.add_argument("--lmf_gamma",       type=float, default=2.0)
    p.add_argument("--lmf_margin",      type=float, default=0.5)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--gambler_o",       type=float, default=0.3)
    p.add_argument("--gambler_weight",  type=float, default=0.1)

    # Augmentation
    p.add_argument("--noise_prob",      type=float, default=0.5)
    p.add_argument("--noise_snr_min",   type=float, default=15.0)
    p.add_argument("--noise_snr_max",   type=float, default=30.0)
    p.add_argument("--gain_prob",       type=float, default=0.5)
    p.add_argument("--gain_range",      type=float, default=0.3)

    # Training
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--max_epochs",      type=int,   default=100)
    p.add_argument("--warmup_epochs",   type=int,   default=10)
    p.add_argument("--patience",        type=int,   default=20)
    p.add_argument("--precision",       default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--run_name",        default="hydro_wave_scattering")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)

    # Post-hoc
    p.add_argument("--target_coverage", type=float, default=0.85)
    p.add_argument("--skip_calibration", action="store_true")

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

    model = HydroWaveScattering(
        num_classes=data.num_classes,
        class_weights=data.class_weights,
        sample_rate=args.sample_rate,
        input_size=args.fixed_len, J=args.J, Q=args.Q,
        d_model=args.d_model,
        se_res2_blocks=args.se_res2_blocks, s4_blocks=args.s4_blocks,
        s4_d_state=args.s4_d_state, dropout=args.dropout, drop_path=args.drop_path,
        loss=args.loss, lmf_gamma=args.lmf_gamma, lmf_margin=args.lmf_margin,
        label_smoothing=args.label_smoothing,
        gambler_o=args.gambler_o, gambler_weight=args.gambler_weight,
        noise_prob=args.noise_prob, noise_snr_min=args.noise_snr_min,
        noise_snr_max=args.noise_snr_max,
        gain_prob=args.gain_prob, gain_range=args.gain_range,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHydroWaveScattering — {n_params / 1e6:.2f}M trainable parameters")
    print(f"Scattering: J={args.J} Q={args.Q} → {model.front.out_channels} channels, T={model.front.out_time}")

    ckpt_cb = ModelCheckpoint(
        monitor="val/macro_precision", mode="max", save_top_k=3,
        filename="scat-{epoch:03d}-p{val/macro_precision:.4f}",
        auto_insert_metric_name=False, verbose=True,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val/macro_precision", patience=args.patience,
                      mode="max", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision if torch.cuda.is_available() else 32,
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

    best_path  = ckpt_cb.best_model_path
    best_score = ckpt_cb.best_model_score
    print(f"\nBest checkpoint:       {best_path}")
    print(f"Best val/macro_prec:   {best_score:.4f}" if best_score is not None else "Best val/macro_prec: n/a")

    if not args.skip_calibration and best_path:
        print("\n── Post-hoc calibration + threshold sweep ──")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cal_model = HydroWaveScattering.load_from_checkpoint(
            best_path, map_location=device, strict=False,
        )
        cal_model.to(device).eval()

        logits, targets = _collect_logits(cal_model, data.val_dataloader(), device)
        T = _fit_temperature(logits, targets)
        probs = F.softmax(logits / T, dim=-1)
        print(f"  temperature    = {T:.3f}")

        res = _search_thresholds(probs, targets, num_classes=data.num_classes,
                                 target_coverage=args.target_coverage)
        print(f"  thresholds     = {['%.2f' % t for t in res['thresholds']]}")
        print(f"  macro_precision = {res['macro_precision']:.4f}  "
              f"coverage = {res['coverage']:.4f}  (target >= {res['target_coverage']})")

        out_dir = Path(best_path).parent
        torch.save({"temperature": T}, out_dir / "temperature.pt")
        with open(out_dir / "thresholds.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"  saved → {out_dir / 'temperature.pt'}")
        print(f"  saved → {out_dir / 'thresholds.json'}")

    return float(best_score) if best_score is not None else float("nan"), best_path


if __name__ == "__main__":
    main()
