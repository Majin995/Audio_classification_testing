"""Optuna sweep targeting the v2A_lean configuration.

v2A's lean architecture (no S4D, no SupCon, no Mean-Teacher, no boundary
attention, no PCEN-on-CQT, no SpecAug-on-all-branches, no Mixup) won the v2
ablation series at val/micro_precision = 0.7333 — beating v1's ~0.708 and
every more elaborate v2 variant.

Search space is narrowed to the loss surface that v1's 32-cell grid
(lightning_logs/grid_precise/summary.csv) flagged as decisive: lmf_margin,
focal_gamma (= lmf γ), label_smoothing — refined on a finer grid centred on
v1's winner (m=0.30, g=2.0, s=0.05). Architecture, optimiser, and waveform
aug are frozen at v1's winning hparams. logit_adjust_tau (v2A-specific) is
the only axis spanning new ground. Net grid: 81 trials.

Target metric: val/f1_p_score = (val/macro_f1 + val/macro_precision) / 2.
This rewards balanced precision *and* recall instead of micro-precision alone.

Latent diagnostics are enabled in every trial so each trial's logs include
silhouette / Fisher / k-NN / centroid-cosine metrics that help interpret why
a config wins or loses (tighter clusters? more orthogonal centroids?).

Study: uatr_hydro_precise_v2a
Imports prior trials from uatr_hydro_precise_micro_v2_scaleup as TPE seeds.

Usage:
    export DATA_DIR="/run/media/damo/Lexar M2/Data/Classifier_Dataset"
    python training/tune_hydro_precise_v2a.py
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

STUDY_NAME = "uatr_hydro_precise_v2a_lindemon"
PRIOR_STUDY_NAME = "uatr_hydro_precise_micro_v2_scaleup"

# ------------------------------------------------------------------ search space
#
# Narrowed to refine the loss surface around v1's grid-search winner
# (lightning_logs/grid_precise/summary.csv). v1 swept (lmf_margin, lmf_gamma,
# label_smoothing, gambler_weight) and identified m=0.30, g=2.0, s=0.05 as the
# clear optimum (val_macro_precision = 0.7269, top 8/10 runs all m=0.30).
#
# Strategy here:
#   • Loss family locked to "lmf" (v1's family — the only one with empirical evidence).
#   • Loss knobs refined on a finer grid AROUND v1's winner rather than re-explored wide.
#   • Architecture / optimiser / augmentation FROZEN at v1's winning hparams.yaml
#     so search budget concentrates on what actually moved precision in v1.
#   • logit_adjust_tau is v2A-specific (no v1 evidence) → kept as the only axis
#     spanning new ground.
#
# Net grid: 3 × 3 × 3 × 3 = 81 trials (loss × margin × smoothing × tau).
SEARCH_SPACE: dict[str, list] = {
    # ── Optimiser / schedule (frozen at v1 winner) ─────────────────────
    "lr":                  [3e-4],
    "weight_decay":        [1e-2],
    "dropout":             [0.15],
    "warmup_epochs":       [8],
    "batch_size":          [128],
    "seed":                [42],
    # ── Branches (frozen at v1 winner) ─────────────────────────────────
    "gabor_n_filters":     [64],
    "gabor_kernel":        [257],
    "gabor_ch":            [128],
    "cqt_n_bins":          [84],
    "cqt_bpo":             [12],
    "cqt_hop":             [64],
    "cqt_ch":              [128],
    "demon_hop":           [64],
    "demon_ch":            [64],
    "demon_n_fft":         [2048],
    "demon_mod_f_min":     [0.0],
    "demon_mod_f_max":     [50.0],
    # ── Fusion (frozen at v1 winner) ───────────────────────────────────
    "fusion_T":            [64],
    "fusion_dim":          [192],
    "n_heads":             [2],
    "n_attn_blocks":       [1],
    # ── Loss surface (the axes v1 grid said matter — refined near optimum) ─
    "loss":                ["lmf"],                 # v1 winner family
    "focal_gamma":         [1.5, 2.0, 2.5],         # v1 winner = 2.0
    "lmf_margin":          [0.20, 0.30, 0.40],      # v1 winner = 0.30
    "ldam_max_m":          [0.5],                   # unused (loss=lmf)
    "cb_beta":             [0.999],                 # unused (loss=lmf)
    "label_smoothing":     [0.03, 0.05, 0.10],      # v1 winner = 0.05
    "logit_adjust_tau":    [0.0, 0.5, 1.0],         # v2A-specific, not in v1 grid
    # ── Augmentation (frozen at v1 winner) ─────────────────────────────
    "noise_prob":          [0.5],
    "noise_snr_min":       [15.0],
    "noise_snr_max":       [30.0],
    "gain_prob":           [0.5],
    "gain_range":          [0.3],
}


# ------------------------------------------------------------------ objective

def _objective(trial: optuna.Trial, data_dir: str, dry_run: bool,
               trial_max_epochs: int, trial_patience: int,
               limit_train_batches: float, limit_val_batches: float,
               latent_metrics: bool, latent_every_n: int) -> float:
    from training.train_precise_v2 import main as _train

    p = {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}

    if p["fusion_dim"] % p["n_heads"] != 0:
        raise optuna.TrialPruned("fusion_dim not divisible by n_heads")

    snr_min = float(p["noise_snr_min"])
    snr_max = float(p["noise_snr_max"])
    if snr_max <= snr_min + 1.0:
        snr_max = snr_min + 1.0

    args = Namespace(
        # data
        data_dir            = data_dir,
        batch_size          = int(p["batch_size"]),
        num_threads         = 8,
        no_oversample       = False,
        denoise             = "off",
        sample_rate         = 5_120,
        fixed_len           = 5_120,
        # ── v2A architecture (most defaults; vary only inside search space) ─
        gabor_n_filters     = int(p["gabor_n_filters"]),
        gabor_kernel        = int(p["gabor_kernel"]),
        gabor_ch            = int(p["gabor_ch"]),
        cqt_n_bins          = int(p["cqt_n_bins"]),
        cqt_bpo             = int(p["cqt_bpo"]),
        cqt_hop             = int(p["cqt_hop"]),
        cqt_ch              = int(p["cqt_ch"]),
        no_pcen_on_cqt      = True,                # v2A: PCEN-on-CQT OFF
        demon_hop           = int(p["demon_hop"]),
        demon_ch            = int(p["demon_ch"]),
        demon_subbands      = "",
        demon_n_fft         = int(p["demon_n_fft"]),
        demon_mod_f_min     = float(p["demon_mod_f_min"]),
        demon_mod_f_max     = float(p["demon_mod_f_max"]),
        use_gammatone_branch = False,              # v2A: no Gammatone
        gammatone_n_bands   = 64,
        gammatone_ch        = 128,
        seres2_blocks       = "2,2,1,1",
        no_spec_aug_all     = True,                # v2A: SpecAug only on Gabor
        # fusion
        fusion_T            = int(p["fusion_T"]),
        fusion_dim          = int(p["fusion_dim"]),
        n_heads             = int(p["n_heads"]),
        n_attn_blocks       = int(p["n_attn_blocks"]),
        no_boundary_attn    = True,                # v2A: vanilla cross-attn
        use_dart_block      = False,
        n_s4d_blocks        = 0,                   # v2A: no S4D
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
        aux_supcon_weight   = 0.0,                 # v2A: no SupCon
        supcon_temp         = 0.07,
        logit_adjust_tau    = float(p["logit_adjust_tau"]),
        # mixup / mean-teacher
        mixup_alpha         = 0.0,                 # v2A: no Mixup
        mean_teacher_weight = 0.0,                 # v2A: no MT
        mt_ema_decay        = 0.999,
        mt_rampup_epochs    = 10,
        # waveform aug
        noise_prob          = float(p["noise_prob"]),
        noise_snr_min       = snr_min,
        noise_snr_max       = snr_max,
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
        run_name            = f"optuna_v2a_t{trial.number:04d}",
        limit_train_batches = limit_train_batches,
        limit_val_batches   = limit_val_batches,
        swa                 = False,
        swa_start_frac      = 0.8,
        swa_lr              = 1e-4,
        # ── target + diagnostics ───────────────────────────────────────
        monitor             = "val/f1_p_score",
        latent_metrics      = latent_metrics,
        latent_every_n      = latent_every_n,
        latent_sample       = 2000,
        # post-hoc skipped during sweep
        target_coverage     = 0.85,
        skip_calibration    = True,
    )

    if dry_run:
        print(f"  [dry_run] v2a trial {trial.number}: {trial.params}")
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


# ------------------------------------------------------------------ priors

def _import_prior_trials(study: "optuna.Study", prior_study_name: str) -> tuple[int, int]:
    """Copy completed v2-scaleup trials whose params land inside this study's grid.

    The v2-scaleup study had v1's gambler_*/denoise/lmf_gamma keys; we drop those
    and only import the trials whose remaining params live within this v2A grid.
    """
    try:
        prior = optuna.load_study(study_name=prior_study_name, storage=STORAGE)
    except Exception as exc:
        print(f"[prior] no prior study {prior_study_name!r} to import ({exc})")
        return 0, 0

    DROP = {"gambler_o", "gambler_weight", "denoise", "lmf_gamma"}
    DEFAULTS = {
        "n_attn_blocks":   1,
        "ldam_max_m":      0.5,
        "cb_beta":         0.999,
        "logit_adjust_tau": 0.0,
        "focal_gamma":     2.0,
    }

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
        params = {k: v for k, v in t.params.items() if k not in DROP}
        if "focal_gamma" not in params and "lmf_gamma" in t.params:
            params["focal_gamma"] = t.params["lmf_gamma"]
        for k, v in DEFAULTS.items():
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
        description="Optuna sweep for v2A_lean (val/f1_p_score)",
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
    p.add_argument("--latent_metrics",      action="store_true", default=True,
                   help="Enable latent-space diagnostics (default ON for this sweep)")
    p.add_argument("--no_latent_metrics",   dest="latent_metrics", action="store_false")
    p.add_argument("--latent_every_n",      type=int, default=2,
                   help="Compute latent metrics every N val epochs (default 2)")
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
        print("ERROR: set --data_dir or export DATA_DIR=/path/to/Classifier_Dataset",
              file=sys.stderr)
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

    grid = 1
    for v in SEARCH_SPACE.values():
        grid *= len(v)
    print(f"\n{'=' * 60}")
    print(f"  Study:        {STUDY_NAME}")
    print(f"  Target:       val/f1_p_score = (macro_f1 + macro_precision) / 2")
    print(f"  Sampler:      {sampler_name}")
    print(f"  Grid points:  {grid}")
    print(f"  Latent metrics: {'on' if args.latent_metrics else 'off'} "
          f"(every {args.latent_every_n} val epochs)")
    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"  Trials prior: {n_done} complete  |  budget: n_trials={args.n_trials} "
          f"timeout={args.timeout_s}s")
    print(f"  Storage:      {STORAGE}")
    print(f"  Dashboard:    http://localhost:{args.port}")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    def _wrap(trial: optuna.Trial) -> float:
        return _objective(
            trial, args.data_dir, args.dry_run,
            trial_max_epochs=args.trial_max_epochs,
            trial_patience=args.trial_patience,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            latent_metrics=args.latent_metrics,
            latent_every_n=args.latent_every_n,
        )

    show_bar = sys.stdout.isatty()
    study.optimize(_wrap, n_trials=args.n_trials, timeout=args.timeout_s,
                   show_progress_bar=show_bar, gc_after_trial=True)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned   = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed   = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"\n{'=' * 60}")
    print(f"  v2A sweep summary ({time.time() - t0:.0f}s)")
    print(f"  complete={len(complete)}  pruned={len(pruned)}  failed={len(failed)}")
    if complete:
        best = study.best_trial
        print(f"  Best trial: #{best.number}")
        print(f"  Best val/f1_p_score: {best.value:.4f}")
        print(f"  Best params:")
        for k, v in best.params.items():
            print(f"    {k:22s} = {v}")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
