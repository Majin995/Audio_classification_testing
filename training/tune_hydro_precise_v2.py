"""Optuna sweep for HydroPreciseV2 — targets val/micro_precision.

Study: uatr_hydro_precise_v2
Imports completed trials from uatr_hydro_precise_micro_v2_scaleup as TPE priors
(the v1 search space is a near-subset of v2; non-overlapping axes are skipped).

Adds v2-specific search axes:
  • n_s4d_blocks            ∈ {0, 1, 2}
  • mixup_alpha             ∈ {0.0, 0.1, 0.2, 0.4}
  • aux_supcon_weight       ∈ {0.0, 0.05, 0.1, 0.2}
  • mean_teacher_weight     ∈ {0.0, 0.25, 0.5, 1.0}
  • loss                    ∈ {focal, lmf, ldam, cb_focal}
  • use_gammatone_branch    ∈ {False, True}
  • use_dart_block          ∈ {False, True}
  • use_boundary_attn       ∈ {False, True}

The v1 abstention/Gambler axes (gambler_o, gambler_weight) and v1 denoise axis
are removed — v1 priors that include them are imported with those keys filtered
out by _import_prior_trials.

Usage:
    export DATA_DIR=/abs/path/to/Split1s
    python training/tune_hydro_precise_v2.py
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

STUDY_NAME = "uatr_hydro_precise_v2"
PRIOR_STUDY_NAME = "uatr_hydro_precise_micro_v2_scaleup"

# ------------------------------------------------------------------ search space

SEARCH_SPACE: dict[str, list] = {
    # ── Optimiser / schedule ────────────────────────────────────────────
    "lr":                  [1e-4, 2e-4, 3e-4, 5e-4],
    "weight_decay":        [1e-4, 1e-3, 1e-2],
    "dropout":             [0.15, 0.20, 0.25, 0.30],
    "warmup_epochs":       [3, 5, 8],
    "batch_size":          [64, 128, 192],
    "seed":                [42],
    # ── Branches ────────────────────────────────────────────────────────
    "gabor_n_filters":     [64, 96, 128],
    "gabor_kernel":        [193, 257, 321],
    "gabor_ch":            [128, 192],
    "cqt_n_bins":          [84, 96, 108],
    "cqt_bpo":             [12, 24],
    "cqt_hop":             [32, 64],
    "cqt_ch":              [128, 192],
    "demon_n_fft":         [1024, 2048, 4096],
    "demon_mod_f_min":     [0.0],
    "demon_mod_f_max":     [50.0, 100.0, 250.0],
    "demon_hop":           [32, 64],
    "demon_ch":            [64, 128, 192],
    # ── Fusion ──────────────────────────────────────────────────────────
    "fusion_T":            [64, 80],
    "fusion_dim":          [192, 256, 320],
    "n_heads":             [2, 4, 8],
    "n_attn_blocks":       [1, 2],
    # ── v2-specific architectural axes ──────────────────────────────────
    "n_s4d_blocks":        [0, 1, 2],
    "use_dart_block":      [False, True],
    "use_boundary_attn":   [False, True],
    "use_gammatone_branch": [False, True],
    # ── Loss family ─────────────────────────────────────────────────────
    "loss":                ["focal", "lmf", "ldam", "cb_focal"],
    "focal_gamma":         [0.5, 1.0, 1.5, 2.0, 2.5],
    "lmf_margin":          [0.3, 0.5, 0.7],
    "ldam_max_m":          [0.3, 0.5, 0.7],
    "cb_beta":             [0.99, 0.999, 0.9999],
    "label_smoothing":     [0.0, 0.05],
    "logit_adjust_tau":    [0.0, 0.5, 1.0],
    # ── Auxiliaries ─────────────────────────────────────────────────────
    "mixup_alpha":         [0.0, 0.1, 0.2, 0.4],
    "aux_supcon_weight":   [0.0, 0.05, 0.1, 0.2],
    "mean_teacher_weight": [0.0, 0.25, 0.5, 1.0],
    # ── Augmentation ────────────────────────────────────────────────────
    "noise_prob":          [0.1, 0.3, 0.5],
    "noise_snr_min":       [10.0, 15.0, 20.0],
    "noise_snr_max":       [30.0, 40.0],
    "gain_prob":           [0.3, 0.6, 0.8],
    "gain_range":          [0.2, 0.3, 0.4],
}


# ------------------------------------------------------------------ objective

def _objective(trial: optuna.Trial, data_dir: str, dry_run: bool,
               trial_max_epochs: int, trial_patience: int,
               limit_train_batches: float, limit_val_batches: float) -> float:
    from training.train_precise_v2 import main as _train

    p = {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}

    if p["fusion_dim"] % p["n_heads"] != 0:
        raise optuna.TrialPruned("fusion_dim not divisible by n_heads")

    noise_snr_min = float(p["noise_snr_min"])
    noise_snr_max = float(p["noise_snr_max"])
    if noise_snr_max <= noise_snr_min + 1.0:
        noise_snr_max = noise_snr_min + 1.0

    args = Namespace(
        # data
        data_dir            = data_dir,
        batch_size          = int(p["batch_size"]),
        num_threads         = 8,
        no_oversample       = False,
        denoise             = "off",
        sample_rate         = 5_120,
        fixed_len           = 5_120,
        # branches
        gabor_n_filters     = int(p["gabor_n_filters"]),
        gabor_kernel        = int(p["gabor_kernel"]),
        gabor_ch            = int(p["gabor_ch"]),
        cqt_n_bins          = int(p["cqt_n_bins"]),
        cqt_bpo             = int(p["cqt_bpo"]),
        cqt_hop             = int(p["cqt_hop"]),
        cqt_ch              = int(p["cqt_ch"]),
        no_pcen_on_cqt      = False,
        demon_hop           = int(p["demon_hop"]),
        demon_ch            = int(p["demon_ch"]),
        demon_subbands      = "",
        demon_n_fft         = int(p["demon_n_fft"]),
        demon_mod_f_min     = float(p["demon_mod_f_min"]),
        demon_mod_f_max     = float(p["demon_mod_f_max"]),
        use_gammatone_branch = bool(p["use_gammatone_branch"]),
        gammatone_n_bands   = 64,
        gammatone_ch        = 128,
        seres2_blocks       = "2,2,1,1",
        no_spec_aug_all     = False,
        # fusion
        fusion_T            = int(p["fusion_T"]),
        fusion_dim          = int(p["fusion_dim"]),
        n_heads             = int(p["n_heads"]),
        n_attn_blocks       = int(p["n_attn_blocks"]),
        no_boundary_attn    = not bool(p["use_boundary_attn"]),
        use_dart_block      = bool(p["use_dart_block"]),
        n_s4d_blocks        = int(p["n_s4d_blocks"]),
        s4d_d_state         = 64,
        dropout             = float(p["dropout"]),
        drop_path           = 0.10,
        # loss
        loss                = p["loss"],
        focal_gamma         = float(p["focal_gamma"]),
        lmf_margin          = float(p["lmf_margin"]),
        ldam_max_m          = float(p["ldam_max_m"]),
        ldam_s              = 30.0,
        cb_beta             = float(p["cb_beta"]),
        label_smoothing     = float(p["label_smoothing"]),
        aux_supcon_weight   = float(p["aux_supcon_weight"]),
        supcon_temp         = 0.07,
        logit_adjust_tau    = float(p["logit_adjust_tau"]),
        # mixup
        mixup_alpha         = float(p["mixup_alpha"]),
        # mean-teacher
        mean_teacher_weight = float(p["mean_teacher_weight"]),
        mt_ema_decay        = 0.999,
        mt_rampup_epochs    = max(1, trial_max_epochs // 3),
        # waveform aug
        noise_prob          = float(p["noise_prob"]),
        noise_snr_min       = noise_snr_min,
        noise_snr_max       = noise_snr_max,
        gain_prob           = float(p["gain_prob"]),
        gain_range          = float(p["gain_range"]),
        # optim / schedule
        lr                  = float(p["lr"]),
        weight_decay        = float(p["weight_decay"]),
        max_epochs          = trial_max_epochs,
        warmup_epochs       = min(int(p["warmup_epochs"]), max(1, trial_max_epochs - 2)),
        patience            = trial_patience,
        precision           = "bf16-mixed",
        grad_clip           = 1.0,
        seed                = int(p["seed"]),
        run_name            = f"optuna_precise_v2_t{trial.number:04d}",
        limit_train_batches = limit_train_batches,
        limit_val_batches   = limit_val_batches,
        swa                 = False,
        swa_start_frac      = 0.8,
        swa_lr              = 1e-4,
        target_coverage     = 0.85,
        skip_calibration    = True,
    )

    if dry_run:
        print(f"  [dry_run] precise_v2 trial {trial.number}: {trial.params}")
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


# ------------------------------------------------------------------ helpers

def _grid_size() -> int:
    n = 1
    for v in SEARCH_SPACE.values():
        n *= len(v)
    return n


def _import_prior_trials(study: "optuna.Study", prior_study_name: str) -> tuple[int, int]:
    """Copy completed trials from a prior study into `study` as frozen priors.

    For v2: we drop v1-only keys (gambler_*, denoise, lmf_gamma) before checking
    space membership, so v1 trials whose remaining params live inside v2's grid
    can still seed TPE.
    """
    try:
        prior = optuna.load_study(study_name=prior_study_name, storage=STORAGE)
    except Exception as exc:
        print(f"[prior] no prior study {prior_study_name!r} to import ({exc})")
        return 0, 0

    # v1 keys not in v2; drop them when importing.
    DROP_KEYS = {
        "gambler_o", "gambler_weight", "denoise",
        "lmf_gamma",   # v1 had a single γ for both lmf and focal; v2 splits it
    }
    # v1 → v2 key remap
    REMAP: dict[str, str] = {}

    already = {
        (tuple(sorted(t.params.items())), t.value)
        for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE
    }

    imported = skipped = 0
    for t in prior.get_trials(deepcopy=True):
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            skipped += 1
            continue
        params = {REMAP.get(k, k): v for k, v in t.params.items() if k not in DROP_KEYS}

        # v1 used `gambler_*` but no `loss` family expansion; if loss missing, default to focal.
        # v1 had no `n_s4d_blocks` etc.; fill defaults so the trial completes a valid v2 point.
        defaults = {
            "n_attn_blocks":       1,
            "n_s4d_blocks":        0,
            "use_dart_block":      False,
            "use_boundary_attn":   False,
            "use_gammatone_branch": False,
            "ldam_max_m":          0.5,
            "cb_beta":             0.999,
            "logit_adjust_tau":    0.0,
            "mixup_alpha":         0.0,
            "aux_supcon_weight":   0.0,
            "mean_teacher_weight": 0.0,
            "focal_gamma":         t.params.get("lmf_gamma", 2.0),
        }
        for k, v in defaults.items():
            params.setdefault(k, v)

        ok = True
        for k, v in params.items():
            if k not in SEARCH_SPACE or v not in SEARCH_SPACE[k]:
                ok = False
                break
        if not ok:
            skipped += 1
            continue

        key = (tuple(sorted(params.items())), t.value)
        if key in already:
            continue
        frozen = optuna.trial.create_trial(
            params=params,
            distributions={k: optuna.distributions.CategoricalDistribution(SEARCH_SPACE[k])
                           for k in params},
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
        description="Optuna sweep for HydroPreciseV2 (val/micro_precision)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",            default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--sampler",             choices=["tpe", "grid"], default="tpe")
    p.add_argument("--n_trials",            type=int, default=200)
    p.add_argument("--timeout_s",           type=int, default=43_200)
    p.add_argument("--trial_max_epochs",    type=int, default=10)
    p.add_argument("--trial_patience",      type=int, default=4)
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)
    p.add_argument("--port",                type=int, default=DASHBOARD_PORT)
    p.add_argument("--no_dashboard",        action="store_true")
    p.add_argument("--dashboard_only",      action="store_true")
    p.add_argument("--dry_run",             action="store_true")
    p.add_argument("--sampler_seed",        type=int, default=0)
    p.add_argument("--no_prior_import",     action="store_true")
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

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE,
        direction="maximize", sampler=sampler, pruner=pruner,
        load_if_exists=True,
    )

    if not args.no_prior_import:
        n_imp, n_skip = _import_prior_trials(study, PRIOR_STUDY_NAME)
        print(f"[prior] imported {n_imp} trials from {PRIOR_STUDY_NAME!r} (skipped {n_skip})")

    print(f"\n{'=' * 60}")
    print(f"  Study:        {STUDY_NAME}")
    print(f"  Sampler:      {sampler_name}")
    print(f"  Grid points:  {_grid_size()}")
    print(f"  Storage:      {STORAGE}")
    print(f"  Dashboard:    http://localhost:{args.port}")
    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"  Trials prior: {n_done} complete  |  budget: n_trials={args.n_trials}  timeout={args.timeout_s}s")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    def _wrap(trial: optuna.Trial) -> float:
        return _objective(
            trial, args.data_dir, args.dry_run,
            trial_max_epochs=args.trial_max_epochs,
            trial_patience=args.trial_patience,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )

    show_bar = sys.stdout.isatty()
    study.optimize(_wrap, n_trials=args.n_trials, timeout=args.timeout_s,
                   show_progress_bar=show_bar, gc_after_trial=True)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned   = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed   = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"\n{'=' * 60}")
    print(f"  HydroPreciseV2 sweep summary ({time.time() - t0:.0f}s)")
    print(f"  complete={len(complete)}  pruned={len(pruned)}  failed={len(failed)}")
    if complete:
        best = study.best_trial
        print(f"  Best trial: #{best.number}")
        print(f"  Best val/micro_precision: {best.value:.4f}")
        print(f"  Best params:")
        for k, v in best.params.items():
            print(f"    {k:22s} = {v}")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
