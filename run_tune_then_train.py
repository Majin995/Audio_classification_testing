"""
Orchestrator: SuperModel1D Optuna sweep → print best params → full training run.

Usage:
    python run_tune_then_train.py [--n_trials N] [--data_dir PATH]
"""

import argparse
import os
import sys
import pprint
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR = os.environ.get("DATA_DIR", "/var/home/damo/Documents/Data/Split1s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials",  type=int, default=40)
    p.add_argument("--data_dir",  default=DATA_DIR)
    p.add_argument("--no_dashboard", action="store_true", default=True,
                   help="Skip launching optuna-dashboard (default: True for headless)")
    return p.parse_args()


# ── Step 1: Hyperparameter sweep ─────────────────────────────────────────────

def run_sweep(data_dir: str, n_trials: int) -> None:
    from training.tune_super_model import run_sweep
    print("\n" + "=" * 60)
    print("  STEP 1 — Optuna TPE sweep for SuperModel1D")
    print(f"  Trials: {n_trials}   Data: {data_dir}")
    print("=" * 60 + "\n")
    run_sweep(data_dir=data_dir, n_trials=n_trials, dry_run=False)


# ── Step 2: Print best params ────────────────────────────────────────────────

def get_best_params() -> dict:
    import optuna
    from scripts.optuna_sweep import STORAGE

    study = optuna.load_study(study_name="uatr_super", storage=STORAGE)
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]

    if not complete:
        print("\nWARNING: No completed trials — cannot extract best params.")
        sys.exit(1)

    best = study.best_trial
    print("\n" + "=" * 60)
    print("  STEP 2 — Best hyperparameters found")
    print("=" * 60)
    print(f"  Trial #:    {best.number}")
    print(f"  val/f1:     {best.value:.4f}")
    print("\n  Params:")
    pprint.pprint(best.params, indent=4)
    print("=" * 60 + "\n")
    return best.params


# ── Step 3: Full training run ────────────────────────────────────────────────

def run_full_training(params: dict, data_dir: str) -> None:
    from argparse import Namespace
    from training.train_super_model import main as train

    print("\n" + "=" * 60)
    print("  STEP 3 — Full training run with optimised params")
    print("=" * 60 + "\n")

    # Map Optuna param names → train_super_model.py arg names
    use_stream_s = params.get("use_stream_s", True)
    use_stream_l = params.get("use_stream_l", True)
    use_cssd     = params.get("use_cssd", False)

    args = Namespace(
        # Data
        data_dir            = data_dir,
        batch_size          = params.get("batch_size", 32),
        num_threads         = 8,
        no_oversample       = False,
        denoise             = params.get("denoise", "off"),
        # Audio
        sample_rate         = 5120,
        fixed_len           = 5120,
        # Stream G
        gabor_n_filters     = params.get("gabor_n_filters", 64),
        gabor_kernel        = params.get("gabor_kernel", 257),
        g_ch                = params.get("g_ch", 48),
        stem_stride         = params.get("stem_stride", 4),
        # Stream S
        no_stream_s         = not use_stream_s,
        spec_n_mels         = params.get("spec_n_mels", 32),
        spec_hop            = 51,
        wb_n_fft            = params.get("wb_n_fft", 256),
        nb_n_fft            = params.get("nb_n_fft", 1024),
        s_ch                = params.get("s_ch", 48),
        ceps_low_q          = params.get("ceps_low_q", 3),
        ceps_high_q         = params.get("ceps_high_q", 15),
        # Stream L
        no_stream_l         = not use_stream_l,
        lofar_n_fft         = params.get("lofar_n_fft", 1024),
        lofar_hop           = 51,
        lofar_time_bins     = 32,
        lofar_freq_bins     = 256,
        l_ch                = params.get("l_ch", 32),
        # Backbone
        d_model             = params.get("d_model", 128),
        scale               = params.get("scale", 8),
        dilations           = params.get("dilations", "2,4,8"),
        n_dart              = params.get("n_dart", 2),
        dart_heads          = 4,
        dart_dropout        = 0.1,
        onset_patch         = 4,
        n_s4                = params.get("n_s4", 2),
        s4_d_state          = 64,
        n_mamba             = params.get("n_mamba", 2),
        mamba_d_state       = 16,
        mamba_expand        = 2,
        mamba_d_conv        = 4,
        drop_path_rate      = params.get("drop_path_rate", 0.10),
        dropout             = params.get("dropout", 0.24),
        # Training
        lr                  = params.get("lr", 3e-4),
        weight_decay        = params.get("weight_decay", 0.012),
        max_epochs          = 100,
        warmup_epochs       = 10,
        patience            = 20,
        precision           = "16-mixed",
        grad_clip           = 1.0,
        seed                = 42,
        run_name            = "super_model_optimised",
        # Augmentation
        mixup_alpha         = params.get("mixup_alpha", 0.3),
        noise_prob          = 0.5,
        noise_snr_min       = params.get("noise_snr_min", 20.0),
        noise_snr_max       = params.get("noise_snr_max", 40.0),
        gain_prob           = 0.7,
        # Loss
        focal_gamma         = params.get("focal_gamma", 2.0),
        label_smoothing     = params.get("label_smoothing", 0.05),
        # CSSD
        cssd_alpha          = params.get("cssd_alpha", 1.0) if use_cssd else 1.0,
        cssd_temp           = params.get("cssd_temp", 4.0),
        cssd_degrade_sr     = params.get("cssd_degrade_sr", 2048),
        cssd_degrade_prob   = 0.5,
        ema_momentum        = 0.999,
        # Full run flags
        limit_train_batches = 1.0,
        limit_val_batches   = 1.0,
    )

    best_f1, best_ckpt = train(args)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best val/f1:     {best_f1:.4f}")
    print(f"  Best checkpoint: {best_ckpt}")
    print("=" * 60 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    run_sweep(data_dir=args.data_dir, n_trials=args.n_trials)
    best_params = get_best_params()
    run_full_training(params=best_params, data_dir=args.data_dir)
