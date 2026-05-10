"""Optuna sweep — HydroPrecise (v1) DEMON + Gambler-loss focused.

Two-axis-only sweep:
  • DEMON linear modulation spectrogram (new architecture, 2026-04-28):
      demon_n_fft, demon_mod_f_min, demon_mod_f_max
  • Deep-Gamblers auxiliary loss:
      gambler_o, gambler_weight

Everything else is frozen at the v1 grid-search winner from
lightning_logs/grid_precise/summary.csv:
  lmf_margin=0.30, lmf_gamma=2.0, label_smoothing=0.05
plus the v1 default branch / fusion / optimiser / augmentation hparams.

Why HydroPrecise (v1) and not v2: v2 has no Gambler head — its loss surface
is "focal | lmf | ldam | cb_focal" only. Gambler-loss lives in
HydroPrecise.__init__ (gambler_o, gambler_weight) and was an active axis
in the v1 grid search.

Target metric: val/micro_precision (matches train_precise.py monitor).
Study: uatr_hydro_precise_demon_gambler  (no prior import — search space
is disjoint from earlier studies).

Usage:
    export DATA_DIR="/run/media/damo/Lexar M2/Data/Classifier_Dataset"
    python training/tune_hydro_precise_demon_gambler.py
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from argparse import Namespace
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

STUDY_NAME = "uatr_hydro_precise_demon_gambler_v2"
PRIOR_STUDY_NAME = "uatr_hydro_precise_micro_v1"   # top-10 imported as TPE seeds

# ------------------------------------------------------------------ search space
#
# Three groups of axes:
#   • DEMON   (3 × 1 × 4 = 12 cells)
#   • Gambler (4 × 4     = 16 cells)
#   • Non-DEMON axes drawn from the union of values appearing in the top-10
#     trials of `uatr_hydro_precise_micro_v1`. This lets TPE explore around
#     the v1 sweep's known-good neighbourhood instead of being locked at a
#     single config.
SEARCH_SPACE: dict[str, list] = {
    # ── DEMON (linear modulation spectrogram) ──────────────────────────
    # df = SR / n_fft; n_bins = floor(mod_f_max * n_fft / SR) + 1
    # at SR=5120: n_fft=1024 → df=5 Hz; n_fft=2048 → df=2.5 Hz; n_fft=4096 → df=1.25 Hz
    # mod_f_max covers BPF-only (25 Hz), BPF+1st harmonics (50 Hz), mid-range
    # (100 Hz), and the wide test (250 Hz).
    "demon_n_fft":     [1024, 2048, 4096],
    "demon_mod_f_min": [0.0],
    "demon_mod_f_max": [25.0, 50.0, 100.0, 250.0],
    # ── Gambler (Deep-Gamblers auxiliary head) ─────────────────────────
    "gambler_o":       [0.1, 0.2, 0.3, 0.5],     # abstention reservation prior
    "gambler_weight":  [0.0, 0.05, 0.1, 0.2],    # 0.0 disables; v1 grid had {0.0, 0.1}
    # ── Non-DEMON axes (union of top-10 values from v1 sweep) ──────────
    "lr":              [1e-4, 3e-4, 5e-4, 1e-3],
    "weight_decay":    [1e-3, 1e-2],
    "dropout":         [0.15, 0.25],
    "warmup_epochs":   [3, 8],
    "batch_size":      [128, 192],
    "gabor_ch":        [128, 192],
    "cqt_n_bins":      [84, 96],
    "cqt_hop":         [64, 128],
    "fusion_T":        [64, 80],
    "n_heads":         [2, 4],
    "loss":            ["lmf", "focal"],
    "lmf_gamma":       [1.5, 2.5],
    "lmf_margin":      [0.3, 0.7],
    "label_smoothing": [0.0, 0.05],
    "noise_snr_min":   [10.0, 15.0],
    "gain_prob":       [0.3, 0.6],
}

# Singletons frozen at the value common to all top-10 v1 trials.
FROZEN: dict[str, object] = {
    "seed":            42,
    "denoise":         "off",
    "gabor_n_filters": 64,
    "gabor_kernel":    257,
    "cqt_bpo":         12,
    "cqt_ch":          128,
    "fusion_dim":      256,
    "noise_prob":      0.3,
    "noise_snr_max":   30.0,
    "gain_range":      0.3,
}


# ------------------------------------------------------------------ objective

def _objective(trial: optuna.Trial, data_dir: str, dry_run: bool,
               trial_max_epochs: int, trial_patience: int,
               limit_train_batches: float, limit_val_batches: float) -> float:
    """Maximise val/micro_precision on HydroPrecise."""
    from training.train_precise import main as _train

    p = {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}

    if int(p["fusion_T"]) > 0 and 256 % int(p["n_heads"]) != 0:
        raise optuna.TrialPruned("fusion_dim (256) not divisible by n_heads")

    snr_min = float(p["noise_snr_min"])
    snr_max = float(FROZEN["noise_snr_max"])
    if snr_max <= snr_min + 1.0:
        snr_max = snr_min + 1.0

    args = Namespace(
        # ── data ───────────────────────────────────────────────────────
        data_dir        = data_dir,
        batch_size      = int(p["batch_size"]),
        num_threads     = 8,
        no_oversample   = False,
        denoise         = str(FROZEN["denoise"]),
        sample_rate     = 5_120,
        fixed_len       = 5_120,
        # ── branches (top-10 union; singletons from FROZEN) ────────────
        gabor_n_filters = int(FROZEN["gabor_n_filters"]),
        gabor_kernel    = int(FROZEN["gabor_kernel"]),
        gabor_ch        = int(p["gabor_ch"]),
        cqt_n_bins      = int(p["cqt_n_bins"]),
        cqt_bpo         = int(FROZEN["cqt_bpo"]),
        cqt_hop         = int(p["cqt_hop"]),
        cqt_ch          = int(FROZEN["cqt_ch"]),
        # ── DEMON (search axis) ────────────────────────────────────────
        demon_hop       = 64,
        demon_ch        = 64,
        demon_n_fft     = int(p["demon_n_fft"]),
        demon_mod_f_min = float(p["demon_mod_f_min"]),
        demon_mod_f_max = float(p["demon_mod_f_max"]),
        # ── fusion ─────────────────────────────────────────────────────
        fusion_T        = int(p["fusion_T"]),
        fusion_dim      = int(FROZEN["fusion_dim"]),
        n_heads         = int(p["n_heads"]),
        dropout         = float(p["dropout"]),
        # ── loss ───────────────────────────────────────────────────────
        loss            = str(p["loss"]),
        lmf_gamma       = float(p["lmf_gamma"]),
        lmf_margin      = float(p["lmf_margin"]),
        label_smoothing = float(p["label_smoothing"]),
        # ── Gambler (search axis) ──────────────────────────────────────
        gambler_o       = float(p["gambler_o"]),
        gambler_weight  = float(p["gambler_weight"]),
        # ── augmentation ───────────────────────────────────────────────
        noise_prob      = float(FROZEN["noise_prob"]),
        noise_snr_min   = snr_min,
        noise_snr_max   = snr_max,
        gain_prob       = float(p["gain_prob"]),
        gain_range      = float(FROZEN["gain_range"]),
        # ── optim / schedule ───────────────────────────────────────────
        lr              = float(p["lr"]),
        weight_decay    = float(p["weight_decay"]),
        max_epochs      = trial_max_epochs,
        warmup_epochs   = min(int(p["warmup_epochs"]), max(1, trial_max_epochs - 2)),
        patience        = trial_patience,
        precision       = "bf16-mixed",
        grad_clip       = 1.0,
        seed            = int(FROZEN["seed"]),
        run_name        = f"optuna_dem_gam_t{trial.number:04d}",
        limit_train_batches = limit_train_batches,
        limit_val_batches   = limit_val_batches,
        # post-hoc skipped during sweep
        target_coverage  = 0.85,
        skip_calibration = True,
    )

    if dry_run:
        print(f"  [dry_run] dem_gam trial {trial.number}: {trial.params}")
        return 0.0

    try:
        score, _ = _train(args)
    except optuna.TrialPruned:
        raise
    except Exception as exc:
        raise optuna.TrialPruned(f"Training raised: {exc}") from exc

    if score is None or not (score == score):
        raise optuna.TrialPruned("Training returned NaN/None score")
    return float(score)


def _grid_size() -> int:
    n = 1
    for v in SEARCH_SPACE.values():
        n *= len(v)
    return n


# ------------------------------------------------------------------ priors

def _import_v1_priors(study: "optuna.Study", n_top: int = 10) -> tuple[int, int]:
    """Seed TPE with the top-N trials from `uatr_hydro_precise_micro_v1`.

    The v1 sweep predates the linear DEMON refactor — its trials carry
    `demon_n_mels` (now removed) and lack `demon_n_fft` / `demon_mod_f_max`.
    For each imported trial we drop the stale DEMON keys and substitute the
    DEMON config that won the current sweep's first round (n_fft=4096,
    mod_f_max=100.0, mod_f_min=0.0). Trials whose remaining params lie
    outside our SEARCH_SPACE are skipped.
    """
    try:
        prior = optuna.load_study(study_name=PRIOR_STUDY_NAME, storage=STORAGE)
    except Exception as exc:
        print(f"[prior] no prior study {PRIOR_STUDY_NAME!r} to import ({exc})")
        return 0, 0

    DEMON_FILL = {
        "demon_n_fft":     4096,
        "demon_mod_f_min": 0.0,
        "demon_mod_f_max": 100.0,
    }
    STALE_KEYS = {"demon_n_mels", "demon_hop", "demon_ch"}

    completed = [t for t in prior.get_trials(deepcopy=True)
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    completed.sort(key=lambda t: -t.value)
    top = completed[:n_top]

    already = {tuple(sorted(t.params.items()))
               for t in study.get_trials(deepcopy=False)
               if t.state == optuna.trial.TrialState.COMPLETE}

    imported = skipped = 0
    for t in top:
        # Keep only keys that are search axes; drop stale and FROZEN-keys
        params = {k: v for k, v in t.params.items()
                  if k in SEARCH_SPACE and k not in STALE_KEYS}
        params.update(DEMON_FILL)

        # Default any axes the v1 trial lacks
        for k in SEARCH_SPACE:
            params.setdefault(k, SEARCH_SPACE[k][0])

        # Snap any out-of-range values to the closest in-axis value (numeric only)
        ok = True
        for k, v in list(params.items()):
            if v in SEARCH_SPACE[k]:
                continue
            if isinstance(v, (int, float)):
                # snap to nearest categorical value
                axis = SEARCH_SPACE[k]
                params[k] = min(axis, key=lambda a: abs(float(a) - float(v)))
            else:
                ok = False
                break
        if not ok:
            skipped += 1
            continue

        if tuple(sorted(params.items())) in already:
            skipped += 1
            continue

        frozen = optuna.trial.create_trial(
            params=params,
            distributions={
                k: optuna.distributions.CategoricalDistribution(SEARCH_SPACE[k])
                for k in params
            },
            value=float(t.value),
            state=optuna.trial.TrialState.COMPLETE,
        )
        try:
            study.add_trial(frozen)
            imported += 1
        except Exception as exc:
            print(f"[prior] skip v1 trial #{t.number}: {exc}")
            skipped += 1
    return imported, skipped


# ------------------------------------------------------------------ CLI

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna sweep for HydroPrecise (DEMON + Gambler)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",       default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--sampler",        choices=["tpe", "grid"], default="tpe")
    p.add_argument("--n_trials",       type=int, default=10_000,
                   help="Upper bound on trials. Full grid is millions of cells; "
                        "--timeout_s ends the run first.")
    p.add_argument("--timeout_s",      type=int, default=43_200,
                   help="Wall-clock budget (default 12h)")
    p.add_argument("--trial_max_epochs", type=int, default=10,
                   help="Cap max_epochs per trial")
    p.add_argument("--trial_patience",   type=int, default=4)
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)
    p.add_argument("--port",           type=int, default=DASHBOARD_PORT)
    p.add_argument("--no_dashboard",   action="store_true")
    p.add_argument("--dashboard_only", action="store_true")
    p.add_argument("--dry_run",        action="store_true")
    p.add_argument("--sampler_seed",   type=int, default=0)
    p.add_argument("--no_prior_import", action="store_true",
                   help=f"Skip importing top-N trials from {PRIOR_STUDY_NAME!r}")
    p.add_argument("--prior_top_n",    type=int, default=10,
                   help="Number of top v1 trials to import as TPE seeds")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.no_dashboard:
        launch_dashboard(port=args.port)

    if args.dashboard_only:
        print(f"Dashboard -> http://localhost:{args.port}  (DB: {DB_PATH})")
        return 0

    if not args.data_dir:
        print("ERROR: set --data_dir or export DATA_DIR=/path/to/Split1s",
              file=sys.stderr)
        return 2

    if args.sampler == "grid":
        sampler = GridSampler(SEARCH_SPACE, seed=args.sampler_seed)
        sampler_name = "GridSampler"
    else:
        sampler = TPESampler(seed=args.sampler_seed, multivariate=True,
                             n_startup_trials=8)
        sampler_name = "TPESampler (multivariate)"

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    if not args.no_prior_import and PRIOR_STUDY_NAME:
        n_imp, n_skip = _import_v1_priors(study, n_top=args.prior_top_n)
        print(f"[prior] imported {n_imp} v1 trials from {PRIOR_STUDY_NAME!r} "
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
    print(f"  DEMON+Gambler sweep summary "
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
