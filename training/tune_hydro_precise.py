"""Optuna sweep for HydroPrecise — targets val/micro_precision.

Shares the SQLite study database at optuna_studies.db with scripts/optuna_sweep.py.
Study name: uatr_hydro_precise.

Usage:
    export DATA_DIR=/abs/path/to/Split1s

    # 12-hour grid search (default)
    python training/tune_hydro_precise.py

    # faster smaller-grid sweep
    python training/tune_hydro_precise.py --n_trials 20 --timeout_s 3600

    # dry-run
    python training/tune_hydro_precise.py --dry_run

    # dashboard only
    python training/tune_hydro_precise.py --dashboard_only

Objective: val/micro_precision (matches the monitor in train_precise.py).

The sweep covers the full train_precise.py argument surface (optimiser, losses,
Gabor/CQT/DEMON branches + fusion, augmentation, schedule). Every dimension is
a discrete categorical value so the space can also be traversed with Optuna's
GridSampler (`--sampler grid`). batch_size is constrained to >= 128 for every
trial, per the user's grid-search directive.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from argparse import Namespace
from itertools import product
from pathlib import Path

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import GridSampler, TPESampler

from scripts.optuna_sweep import (  # noqa: E402
    DB_PATH, DASHBOARD_PORT, STORAGE, launch_dashboard,
)

STUDY_NAME = "uatr_hydro_precise_micro_v2_scaleup"
PRIOR_STUDY_NAME = "uatr_hydro_precise_micro_v1"  # v1 trials imported as TPE priors

# ------------------------------------------------------------------ search space

# All values are discrete so GridSampler can traverse them. Keep the grid wide
# enough that a 12-hour budget at ~25 min/trial (~29 trials) only samples a
# fraction — exploration > exhaustion.
SEARCH_SPACE: dict[str, list] = {
    # v2 "scale-up" grid — strict superset of v1, so all 47 prior trials can be
    # imported into this study as TPE priors (see _import_prior_trials below).
    # Based on v1 marginal analysis: more capacity (fusion_dim, cqt/demon/gabor
    # channels, cqt_n_bins, demon_n_mels, gabor_kernel) and softer losses
    # (lmf_gamma below 1.5) and lighter augmentation (lower noise_prob, higher
    # noise_snr_min) were the directions where p rose. emd_wavelet denoise was
    # the dominant variance trap (importance 0.846, mean 0.26) — kept in-space
    # only so v1 trials import; TPE will quickly abandon it.
    # optimiser / schedule
    "lr":              [1e-4, 2e-4, 3e-4, 5e-4, 1e-3],
    "weight_decay":    [1e-3, 5e-3, 1e-2, 3e-2],
    "dropout":         [0.15, 0.20, 0.25, 0.30],
    "warmup_epochs":   [3, 8],
    "batch_size":      [128, 192, 256],          # 128 dominated v1; kept for prior
    "seed":            [42],
    # front-end
    "denoise":         ["off", "emd_wavelet"],   # kept only to let v1 trials import
    # Gabor branch — new ceilings: kernel 321/385, filters 128
    "gabor_n_filters": [64, 96, 128],
    "gabor_kernel":    [193, 257, 321, 385],
    "gabor_ch":        [128, 192],
    # CQT branch — new ceilings: n_bins 108/128, bpo 24, hop 32, ch 192/256
    "cqt_n_bins":      [84, 96, 108, 128],
    "cqt_bpo":         [12, 24],
    "cqt_hop":         [32, 64, 128],
    "cqt_ch":          [128, 192, 256],
    # DEMON branch — linear modulation spectrogram (n_bins is derived from n_fft+mod_f_max)
    "demon_n_fft":     [1024, 2048, 4096],
    "demon_mod_f_min": [0.0],
    "demon_mod_f_max": [50.0, 100.0, 250.0],
    "demon_hop":       [32, 64],
    "demon_ch":        [64, 128, 192, 256],
    # fusion — fusion_dim was the biggest capped axis; push to 512
    "fusion_T":        [64, 80],
    "fusion_dim":      [192, 256, 320, 384, 512],
    "n_heads":         [2, 4, 8],                # divisibility with fusion_dim enforced below
    # loss — push lmf_gamma below 1.5 (marginal says softer is better)
    "loss":            ["lmf", "focal"],
    "lmf_gamma":       [0.5, 1.0, 1.5, 2.5],
    "lmf_margin":      [0.3, 0.5, 0.7, 0.9],
    "label_smoothing": [0.0, 0.03, 0.05],
    "gambler_o":       [0.2, 0.3, 0.5],
    "gambler_weight":  [0.0, 0.1, 0.15, 0.2, 0.3],
    # augmentation — pull back (v1: lower noise_prob and higher noise_snr_min win)
    "noise_prob":      [0.1, 0.2, 0.3, 0.6],
    "noise_snr_min":   [10.0, 15.0, 20.0, 25.0],
    "noise_snr_max":   [30.0, 40.0],
    "gain_prob":       [0.3, 0.4, 0.6, 0.8],
    "gain_range":      [0.2, 0.3, 0.4],
}


# ------------------------------------------------------------------ objective

def _objective(trial: optuna.Trial, data_dir: str, dry_run: bool,
               trial_max_epochs: int, trial_patience: int,
               limit_train_batches: float, limit_val_batches: float) -> float:
    """Maximise val/micro_precision on HydroPrecise."""
    from training.train_precise import main as _train

    params = {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}

    if params["fusion_dim"] % params["n_heads"] != 0:
        raise optuna.TrialPruned("fusion_dim not divisible by n_heads")

    # Enforce ordering to avoid a degenerate SNR range
    noise_snr_min = float(params["noise_snr_min"])
    noise_snr_max = float(params["noise_snr_max"])
    if noise_snr_max <= noise_snr_min + 1.0:
        noise_snr_max = noise_snr_min + 1.0

    args = Namespace(
        # data
        data_dir        = data_dir,
        batch_size      = int(params["batch_size"]),
        num_threads     = 8,
        no_oversample   = False,
        denoise         = params["denoise"],
        sample_rate     = 5_120,
        fixed_len       = 5_120,
        # branches
        gabor_n_filters = int(params["gabor_n_filters"]),
        gabor_kernel    = int(params["gabor_kernel"]),
        gabor_ch        = int(params["gabor_ch"]),
        cqt_n_bins      = int(params["cqt_n_bins"]),
        cqt_bpo         = int(params["cqt_bpo"]),
        cqt_hop         = int(params["cqt_hop"]),
        cqt_ch          = int(params["cqt_ch"]),
        demon_hop       = int(params["demon_hop"]),
        demon_ch        = int(params["demon_ch"]),
        demon_n_fft     = int(params["demon_n_fft"]),
        demon_mod_f_min = float(params["demon_mod_f_min"]),
        demon_mod_f_max = float(params["demon_mod_f_max"]),
        # fusion
        fusion_T        = int(params["fusion_T"]),
        fusion_dim      = int(params["fusion_dim"]),
        n_heads         = int(params["n_heads"]),
        dropout         = float(params["dropout"]),
        # loss
        loss            = params["loss"],
        lmf_gamma       = float(params["lmf_gamma"]),
        lmf_margin      = float(params["lmf_margin"]),
        label_smoothing = float(params["label_smoothing"]),
        gambler_o       = float(params["gambler_o"]),
        gambler_weight  = float(params["gambler_weight"]),
        # augmentation
        noise_prob      = float(params["noise_prob"]),
        noise_snr_min   = noise_snr_min,
        noise_snr_max   = noise_snr_max,
        gain_prob       = float(params["gain_prob"]),
        gain_range      = float(params["gain_range"]),
        # optim / schedule
        lr              = float(params["lr"]),
        weight_decay    = float(params["weight_decay"]),
        max_epochs      = trial_max_epochs,
        warmup_epochs   = min(int(params["warmup_epochs"]), max(1, trial_max_epochs - 2)),
        patience        = trial_patience,
        precision       = "bf16-mixed",
        grad_clip       = 1.0,
        seed            = int(params["seed"]),
        run_name        = f"optuna_precise_t{trial.number:04d}",
        limit_train_batches = limit_train_batches,
        limit_val_batches   = limit_val_batches,
        # post-hoc calibration — skipped during sweep, rerun on best trial
        target_coverage  = 0.85,
        skip_calibration = True,
    )

    if dry_run:
        print(f"  [dry_run] precise trial {trial.number}: {trial.params}")
        return 0.0

    try:
        score, _ = _train(args)
    except optuna.TrialPruned:
        raise
    except Exception as exc:
        raise optuna.TrialPruned(f"Training raised: {exc}") from exc

    if score is None or not (score == score):  # NaN guard
        raise optuna.TrialPruned("Training returned NaN/None score")
    return float(score)


# ------------------------------------------------------------------ helpers

def _grid_size() -> int:
    n = 1
    for v in SEARCH_SPACE.values():
        n *= len(v)
    return n


def _import_prior_trials(study: "optuna.Study", prior_study_name: str) -> tuple[int, int]:
    """Copy completed trials from a prior study into `study` as frozen priors.

    Skips any trial whose param values aren't all inside the current SEARCH_SPACE
    (i.e., axes we've narrowed). The surviving trials become TPE priors so this
    new sweep starts with the knowledge accumulated by the previous invocation.

    Returns (n_imported, n_skipped).
    """
    try:
        prior = optuna.load_study(study_name=prior_study_name, storage=STORAGE)
    except Exception as exc:
        print(f"[prior] no prior study {prior_study_name!r} to import ({exc})")
        return 0, 0

    existing_numbers = {t.number for t in study.get_trials(deepcopy=False)}
    already_imported = {
        (tuple(sorted(t.params.items())), t.value)
        for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE
    }

    imported = 0
    skipped = 0
    for t in prior.get_trials(deepcopy=True):
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            skipped += 1
            continue
        # Ensure every param is within the current search space values
        ok = True
        for k, v in t.params.items():
            if k not in SEARCH_SPACE or v not in SEARCH_SPACE[k]:
                ok = False
                break
        if not ok:
            skipped += 1
            continue
        key = (tuple(sorted(t.params.items())), t.value)
        if key in already_imported:
            continue  # idempotent re-runs don't duplicate

        frozen = optuna.trial.create_trial(
            params=t.params,
            distributions={k: optuna.distributions.CategoricalDistribution(SEARCH_SPACE[k])
                           for k in t.params},
            value=float(t.value),
            state=optuna.trial.TrialState.COMPLETE,
        )
        try:
            study.add_trial(frozen)
            imported += 1
        except Exception as exc:
            print(f"[prior] skip trial #{t.number}: {exc}")
            skipped += 1
    return imported, skipped


# ------------------------------------------------------------------ CLI

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna sweep for HydroPrecise (val/micro_precision)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",       default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--sampler",        choices=["tpe", "grid"], default="tpe",
                   help="tpe: bayesian exploration over the categorical grid; "
                        "grid: deterministic traversal (use --timeout_s to bound)")
    p.add_argument("--n_trials",       type=int, default=200,
                   help="Upper bound on trials; --timeout_s usually ends the run first.")
    p.add_argument("--timeout_s",      type=int, default=43_200,
                   help="Wall-clock budget per invocation (default: 12h)")
    p.add_argument("--trial_max_epochs", type=int, default=10,
                   help="Cap max_epochs per trial (default 10 ~= 25 min at 2:30/epoch)")
    p.add_argument("--trial_patience",   type=int, default=4,
                   help="EarlyStopping patience per trial")
    p.add_argument("--limit_train_batches", type=float, default=1.0,
                   help="<1.0 to subsample train loader each epoch")
    p.add_argument("--limit_val_batches",   type=float, default=1.0,
                   help="<1.0 to subsample val loader each epoch")
    p.add_argument("--port",           type=int, default=DASHBOARD_PORT,
                   help="optuna-dashboard port")
    p.add_argument("--no_dashboard",   action="store_true")
    p.add_argument("--dashboard_only", action="store_true")
    p.add_argument("--dry_run",        action="store_true",
                   help="Suggest params without training")
    p.add_argument("--sampler_seed",   type=int, default=0,
                   help="Seed for the sampler")
    p.add_argument("--no_prior_import", action="store_true",
                   help=f"Skip importing completed trials from {PRIOR_STUDY_NAME!r}")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.no_dashboard:
        launch_dashboard(port=args.port)

    if args.dashboard_only:
        print(f"Dashboard -> http://localhost:{args.port}  (DB: {DB_PATH})")
        return 0

    if not args.data_dir:
        print("ERROR: set --data_dir or export DATA_DIR=/path/to/Split1s", file=sys.stderr)
        return 2

    if args.sampler == "grid":
        sampler = GridSampler(SEARCH_SPACE, seed=args.sampler_seed)
        sampler_name = "GridSampler"
    else:
        sampler = TPESampler(seed=args.sampler_seed, multivariate=True, n_startup_trials=8)
        sampler_name = "TPESampler (multivariate)"

    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    if not args.no_prior_import and PRIOR_STUDY_NAME:
        n_imp, n_skip = _import_prior_trials(study, PRIOR_STUDY_NAME)
        print(f"[prior] imported {n_imp} trials from {PRIOR_STUDY_NAME!r} "
              f"(skipped {n_skip})")

    grid_n = _grid_size()
    print(f"\n{'=' * 60}")
    print(f"  Study:        {STUDY_NAME}")
    print(f"  Sampler:      {sampler_name}")
    print(f"  Grid points:  {grid_n}")
    print(f"  Storage:      {STORAGE}")
    print(f"  Dashboard:    http://localhost:{args.port}")
    print(f"  Trials prior: "
          f"{len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
          f" complete  |  budget: n_trials={args.n_trials}  timeout={args.timeout_s}s")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    def _objective_wrapped(trial: optuna.Trial) -> float:
        return _objective(
            trial, args.data_dir, args.dry_run,
            trial_max_epochs=args.trial_max_epochs,
            trial_patience=args.trial_patience,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )

    show_bar = sys.stdout.isatty()
    study.optimize(
        _objective_wrapped,
        n_trials=args.n_trials,
        timeout=args.timeout_s,
        show_progress_bar=show_bar,
        gc_after_trial=True,
    )

    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    pruned   = [t for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED]
    failed   = [t for t in study.trials
                if t.state == optuna.trial.TrialState.FAIL]
    print(f"\n{'=' * 60}")
    print(f"  HydroPrecise sweep summary "
          f"({time.time() - t0:.0f}s, this invocation)")
    print(f"  complete={len(complete)}  pruned={len(pruned)}  failed={len(failed)}")
    if complete:
        best = study.best_trial
        print(f"  Best trial: #{best.number}")
        print(f"  Best val/micro_precision: {best.value:.4f}")
        print(f"  Best params:")
        for k, v in best.params.items():
            print(f"    {k:22s} = {v}")
    else:
        print("  WARNING: no completed trials this sweep")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
