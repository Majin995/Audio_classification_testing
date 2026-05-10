"""
Standalone Optuna TPE sweep for SuperModel1D.

Shares the same SQLite study database as scripts/optuna_sweep.py.
Study name: uatr_super

Usage:
    export DATA_DIR=/path/to/Split1s

    # Run 40 trials
    python training/tune_super_model.py --n_trials 40

    # Dry-run (print trial params without training)
    python training/tune_super_model.py --n_trials 5 --dry_run

    # Resume sweep + launch dashboard
    python training/tune_super_model.py --n_trials 20

    # Dashboard only (no new trials)
    python training/tune_super_model.py --dashboard_only

The same study can also be swept via:
    python scripts/optuna_sweep.py --model super --n_trials 40
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

# ── repo root on path ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Re-use constants and dashboard helper from the central sweep module
from scripts.optuna_sweep import (
    STORAGE,
    DB_PATH,
    STUDY_PREFIX,
    DASHBOARD_PORT,
    launch_dashboard,
    _common_args,
)

# Study name for SuperModel1D
_STUDY_NAME = f"{STUDY_PREFIX}super"


# ── objective ─────────────────────────────────────────────────────────────────

def _objective_super(
    trial: optuna.Trial,
    data_dir: str,
    dry_run: bool,
    limit_train_batches: float = 0.25,
) -> float:
    """Optuna objective for SuperModel1D.

    Builds a Namespace with all SuperModel1D hyperparameters sampled from the
    search space below, then delegates to train_super_model.main().
    """
    from training.train_super_model import main as _train

    # ── shared args (weight_decay, denoise, seed, dropout, …) ────────────────
    base = _common_args(trial, data_dir, "super")

    # ── model-specific search space ───────────────────────────────────────────

    # Core training
    lr         = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # ── Stream G (Gabor 1D) ───────────────────────────────────────────────────
    gabor_n_filters = trial.suggest_categorical("gabor_n_filters", [32, 64])
    gabor_kernel    = trial.suggest_categorical("gabor_kernel", [129, 257])
    stem_stride     = trial.suggest_categorical("stem_stride", [4, 8])
    g_ch            = trial.suggest_categorical("g_ch", [32, 48, 64])

    # ── Stream S (EnhancedFrontEnd spectrogram) ───────────────────────────────
    use_stream_s = trial.suggest_categorical("use_stream_s", [True, False])
    spec_n_mels  = trial.suggest_categorical("spec_n_mels", [32, 48])
    wb_n_fft     = trial.suggest_categorical("wb_n_fft", [256, 512, 1024])
    nb_n_fft     = trial.suggest_categorical("nb_n_fft", [1024, 2048, 4096])
    s_ch         = trial.suggest_categorical("s_ch", [32, 48, 64])
    ceps_low_q   = trial.suggest_categorical("ceps_low_q", [2, 3])
    ceps_high_q  = trial.suggest_categorical("ceps_high_q", [12, 15, 18])

    # ── Stream L (LofarFrontend) ──────────────────────────────────────────────
    use_stream_l = trial.suggest_categorical("use_stream_l", [True, False])
    lofar_n_fft  = trial.suggest_categorical("lofar_n_fft", [1024, 2048, 4096])
    l_ch         = trial.suggest_categorical("l_ch", [16, 32])

    # ── Backbone ──────────────────────────────────────────────────────────────
    d_model        = trial.suggest_categorical("d_model", [96, 128, 160])
    scale          = trial.suggest_categorical("scale", [4, 8])
    dilations      = trial.suggest_categorical("dilations", ["2,4,8", "1,2,4,8", "2,4,8,16"])
    n_dart         = trial.suggest_int("n_dart", 1, 2)
    n_s4           = trial.suggest_int("n_s4", 1, 2)
    n_mamba        = trial.suggest_int("n_mamba", 1, 2)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.05, 0.6)

    # ── Augmentation ─────────────────────────────────────────────────────────
    noise_snr_min = trial.suggest_categorical("noise_snr_min", [0.0, 20.0])
    noise_snr_max = trial.suggest_categorical("noise_snr_max", [30.0, 40.0])

    # ── CSSD ─────────────────────────────────────────────────────────────────
    use_cssd = trial.suggest_categorical("use_cssd", [False, True])
    if use_cssd:
        cssd_alpha      = trial.suggest_categorical("cssd_alpha", [0.5, 0.7, 0.9])
        cssd_temp       = trial.suggest_categorical("cssd_temp", [2.0, 4.0, 8.0])
        cssd_degrade_sr = trial.suggest_categorical("cssd_degrade_sr", [1600, 2048])
    else:
        cssd_alpha      = 1.0   # disabled
        cssd_temp       = 4.0
        cssd_degrade_sr = 2048

    # ── assemble Namespace ────────────────────────────────────────────────────
    # NOTE: _common_args already contains: sample_rate, fixed_len, max_epochs,
    # warmup_epochs, patience, precision, grad_clip, limit_train/val_batches,
    # seed, weight_decay, denoise, dropout, mixup_alpha, focal_gamma,
    # label_smoothing, run_name, data_dir, num_threads, no_oversample.
    # Do NOT repeat those keys here or Namespace() raises "multiple values".
    # Override limit_train_batches from _common_args (which sets 1.0)
    base["limit_train_batches"] = limit_train_batches

    args = Namespace(
        **base,
        # Training
        lr              = lr,
        batch_size      = batch_size,
        # Stream G
        gabor_n_filters = gabor_n_filters,
        gabor_kernel    = gabor_kernel,
        g_ch            = g_ch,
        stem_stride     = stem_stride,
        # Stream S
        no_stream_s     = not use_stream_s,
        spec_n_mels     = spec_n_mels,
        spec_hop        = 51,
        wb_n_fft        = wb_n_fft,
        nb_n_fft        = nb_n_fft,
        s_ch            = s_ch,
        ceps_low_q      = ceps_low_q,
        ceps_high_q     = ceps_high_q,
        # Stream L
        no_stream_l       = not use_stream_l,
        lofar_n_fft       = lofar_n_fft,
        lofar_hop         = 51,
        lofar_time_bins   = 32,
        lofar_freq_bins   = 256,
        l_ch              = l_ch,
        # Backbone
        d_model           = d_model,
        scale             = scale,
        dilations         = dilations,
        n_dart            = n_dart,
        dart_heads        = 4,
        dart_dropout      = 0.1,
        onset_patch       = 4,
        n_s4              = n_s4,
        s4_d_state        = 64,
        n_mamba           = n_mamba,
        mamba_d_state     = 16,
        mamba_expand      = 2,
        mamba_d_conv      = 4,
        drop_path_rate    = drop_path_rate,
        # Augmentation extras (mixup_alpha/focal_gamma/label_smoothing in base)
        noise_prob        = 0.5,
        noise_snr_min     = noise_snr_min,
        noise_snr_max     = noise_snr_max,
        gain_prob         = 0.7,
        # CSSD
        cssd_alpha        = cssd_alpha,
        cssd_temp         = cssd_temp,
        cssd_degrade_sr   = cssd_degrade_sr,
        cssd_degrade_prob = 0.5,
        ema_momentum      = 0.999,
        # Optuna pruning callback
        optuna_trial      = trial,
    )

    if dry_run:
        print(f"  [dry_run] super trial {trial.number}: {trial.params}")
        return 0.0

    try:
        score, _ = _train(args)
    except Exception as exc:
        raise optuna.TrialPruned(f"Training raised: {exc}") from exc
    finally:
        # Free VRAM between trials so the next trial starts clean
        torch.cuda.empty_cache()

    return score


# ── study runner ──────────────────────────────────────────────────────────────

def run_sweep(
    data_dir: str,
    n_trials: int,
    dry_run: bool,
    limit_train_batches: float = 0.25,
) -> None:
    data_dir = str(Path(data_dir).resolve())

    sampler = TPESampler(seed=42, multivariate=True)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study   = optuna.create_study(
        study_name     = _STUDY_NAME,
        storage        = STORAGE,
        direction      = "maximize",
        sampler        = sampler,
        pruner         = pruner,
        load_if_exists = True,
    )

    n_done = len(study.trials)
    print(f"\n{'='*60}")
    print(f"  Study:            {_STUDY_NAME}")
    print(f"  Trials requested: {n_trials}  (already done: {n_done})")
    print(f"  limit_train_batches: {limit_train_batches}")
    print(f"  Storage:          {DB_PATH}")
    print(f"{'='*60}\n")

    def objective(trial: optuna.Trial) -> float:
        return _objective_super(trial, data_dir, dry_run, limit_train_batches)

    _show_bar = sys.stdout.isatty()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=_show_bar)

    # ── post-sweep summary ────────────────────────────────────────────────────
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{'='*60}")
    print(f"  [super] sweep complete — {len(complete)}/{n_trials} trials finished")
    if complete:
        best = study.best_trial
        print(f"  Best trial:  #{best.number}")
        print(f"  Best val/f1: {best.value:.4f}")
        print(f"  Best params: {best.params}")
    else:
        print("  WARNING: all trials failed or were pruned — no best trial.")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna TPE sweep for SuperModel1D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",       default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--n_trials",       type=int, default=40,
                   help="Number of Optuna trials to run")
    p.add_argument("--dry_run",        action="store_true",
                   help="Print trial params without training")
    p.add_argument("--no_dashboard",   action="store_true",
                   help="Skip launching optuna-dashboard")
    p.add_argument("--port",           type=int, default=DASHBOARD_PORT,
                   help="optuna-dashboard port")
    p.add_argument("--dashboard_only", action="store_true",
                   help="Only (re)launch optuna-dashboard, then exit")
    p.add_argument("--limit_train_batches", type=float, default=0.25,
                   help="Fraction of training batches per epoch (keeps trials fast)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.no_dashboard:
        launch_dashboard(port=args.port)

    if args.dashboard_only:
        return

    if not args.data_dir and not args.dry_run:
        print("ERROR: set --data_dir or export DATA_DIR=/path/to/Split1s")
        sys.exit(1)

    run_sweep(
        data_dir             = args.data_dir,
        n_trials             = args.n_trials,
        dry_run              = args.dry_run,
        limit_train_batches  = args.limit_train_batches,
    )


if __name__ == "__main__":
    main()
