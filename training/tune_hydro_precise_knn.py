"""Optuna sweep for HydroPreciseV2 targeting `latent/knn_acc`.

Optimises the **representation quality** of the post-pool embedding (5-NN
purity on cosine distance, computed by `LatentMetricsCallback`) rather than
the end-classifier metrics already swept by `tune_hydro_precise_v2a.py` and
friends. Useful when the embedding feeds downstream tasks (retrieval,
ensemble heads, transfer) and we want it well-clustered independent of the
final softmax head.

Per-trial budget borrows the active-learning framework's training-time levers
from `train_active.py`: each trial trains on a stratified ~2k subset of the
train pool (via `--train_subset_size`), with shorter `max_epochs/patience`.
Cuts per-trial wall time ~6-10x vs full-data v2 trials. Val/test sets are
unchanged so the metric is comparable across trials.

Search space (TPE-multivariate over a ~10k-cell grid):
  Loss surface : loss, focal_gamma, lmf_margin, label_smoothing, logit_adjust_tau
  Repr-shape   : fusion_dim, n_heads, n_attn_blocks, dropout, drop_path

Optimiser, branches, and waveform aug are frozen at v1 winner hparams (the v2A
fixed point).

Study: uatr_hydro_precise_knn  (cold start — prior studies optimised
different metrics and would mislead TPE if imported as seeds).

Usage:
    export DATA_DIR="/run/media/damo/Lexar M2/Data/Classifier_Dataset"
    python training/tune_hydro_precise_knn.py
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
from optuna.samplers import TPESampler

from scripts.optuna_sweep import (  # noqa: E402
    DB_PATH, DASHBOARD_PORT, STORAGE, launch_dashboard,
)

STUDY_NAME = "uatr_hydro_precise_knn"


# ───────────────────────────────────────────────────── search space ──
# Frozen at the v1 / v2A fixed point: optimiser, branches, augmentation.
# Varied: loss surface (5 axes) + representation-shape head (5 axes).
# Total grid ~3*3*3*3*3 * 3*2*2*3*3 = ~9 700 cells → use TPE, not Grid.
_LR             = 3e-4
_WEIGHT_DECAY   = 1e-2
_BATCH_SIZE     = 64           # smaller than v2A (128) — with 2k samples we
                               # want more steps/epoch.
_WARMUP         = 8
_SEED           = 42
_GABOR_N        = 64
_GABOR_KERNEL   = 257
_GABOR_CH       = 128
_CQT_N_BINS     = 84
_CQT_BPO        = 12
_CQT_HOP        = 64
_CQT_CH         = 128
_DEMON_HOP      = 64
_DEMON_CH       = 64
_DEMON_N_FFT    = 2048
_DEMON_FMIN     = 0.0
_DEMON_FMAX     = 50.0
_FUSION_T       = 64
_NOISE_PROB     = 0.5
_NOISE_SNR_MIN  = 15.0
_NOISE_SNR_MAX  = 30.0
_GAIN_PROB      = 0.5
_GAIN_RANGE     = 0.3


# ────────────────────────────────────────────────────────── objective ──

def _objective(trial: optuna.Trial, *, data_dir: str, dry_run: bool,
               trial_max_epochs: int, trial_patience: int,
               limit_train_batches: float, limit_val_batches: float,
               train_subset_size: int, fixed_subset: bool) -> float:
    from training.train_precise_v2 import main as _train

    # Loss surface
    loss            = trial.suggest_categorical("loss", ["lmf", "focal"])
    focal_gamma     = trial.suggest_categorical("focal_gamma",     [1.5, 2.0, 2.5])
    lmf_margin      = trial.suggest_categorical("lmf_margin",      [0.20, 0.30, 0.40])
    label_smoothing = trial.suggest_categorical("label_smoothing", [0.03, 0.05, 0.10])
    logit_adjust    = trial.suggest_categorical("logit_adjust_tau",[0.0, 0.5, 1.0])

    # Representation-shape head
    fusion_dim      = trial.suggest_categorical("fusion_dim",      [128, 192, 256])
    n_heads         = trial.suggest_categorical("n_heads",         [2, 4])
    n_attn_blocks   = trial.suggest_categorical("n_attn_blocks",   [1, 2])
    dropout         = trial.suggest_categorical("dropout",         [0.10, 0.15, 0.25])
    drop_path       = trial.suggest_categorical("drop_path",       [0.05, 0.10, 0.20])

    if fusion_dim % n_heads != 0:
        raise optuna.TrialPruned("fusion_dim not divisible by n_heads")

    # Per-trial subset draw — different seed per trial keeps one easy draw
    # from biasing TPE; --fixed_subset locks it for stricter comparability.
    subset_seed = 0 if fixed_subset else trial.number

    args = Namespace(
        # data
        data_dir            = data_dir,
        batch_size          = _BATCH_SIZE,
        num_threads         = 8,
        no_oversample       = False,
        denoise             = "off",
        sample_rate         = 5_120,
        fixed_len           = 5_120,
        # train-pool subset (AL-derived speed lever)
        train_subset_size      = train_subset_size,
        train_subset_per_class = 0,
        train_subset_seed      = subset_seed,
        # branches (frozen at v2A winner)
        gabor_n_filters     = _GABOR_N,
        gabor_kernel        = _GABOR_KERNEL,
        gabor_ch            = _GABOR_CH,
        cqt_n_bins          = _CQT_N_BINS,
        cqt_bpo             = _CQT_BPO,
        cqt_hop             = _CQT_HOP,
        cqt_ch              = _CQT_CH,
        no_pcen_on_cqt      = True,
        demon_hop           = _DEMON_HOP,
        demon_ch            = _DEMON_CH,
        demon_subbands      = "",
        demon_n_fft         = _DEMON_N_FFT,
        demon_mod_f_min     = _DEMON_FMIN,
        demon_mod_f_max     = _DEMON_FMAX,
        use_gammatone_branch = False,
        gammatone_n_bands   = 64,
        gammatone_ch        = 128,
        seres2_blocks       = "2,2,1,1",
        no_spec_aug_all     = True,
        # fusion (head-shape — varied)
        fusion_T            = _FUSION_T,
        fusion_dim          = int(fusion_dim),
        n_heads             = int(n_heads),
        n_attn_blocks       = int(n_attn_blocks),
        no_boundary_attn    = True,
        use_dart_block      = False,
        n_s4d_blocks        = 0,
        s4d_d_state         = 64,
        dropout             = float(dropout),
        drop_path           = float(drop_path),
        # loss (varied)
        loss                = loss,
        focal_gamma         = float(focal_gamma),
        lmf_margin          = float(lmf_margin),
        ldam_max_m          = 0.5,
        ldam_s              = 30.0,
        cb_beta             = 0.999,
        label_smoothing     = float(label_smoothing),
        aux_supcon_weight   = 0.0,
        supcon_temp         = 0.07,
        logit_adjust_tau    = float(logit_adjust),
        # mixup / mean-teacher off (matching v2A)
        mixup_alpha         = 0.0,
        mean_teacher_weight = 0.0,
        mt_ema_decay        = 0.999,
        mt_rampup_epochs    = 10,
        # waveform aug (frozen)
        noise_prob          = _NOISE_PROB,
        noise_snr_min       = _NOISE_SNR_MIN,
        noise_snr_max       = _NOISE_SNR_MAX,
        gain_prob           = _GAIN_PROB,
        gain_range          = _GAIN_RANGE,
        # optim / schedule
        lr                  = _LR,
        weight_decay        = _WEIGHT_DECAY,
        max_epochs          = trial_max_epochs,
        warmup_epochs       = min(_WARMUP, max(1, trial_max_epochs - 2)),
        patience            = trial_patience,
        precision           = "bf16-mixed",
        grad_clip           = 1.0,
        seed                = _SEED,
        run_name            = f"optuna_knn_t{trial.number:04d}",
        limit_train_batches = limit_train_batches,
        limit_val_batches   = limit_val_batches,
        swa                 = False,
        swa_start_frac      = 0.8,
        swa_lr              = 1e-4,
        # ── target metric ─────────────────────────────────────────────
        monitor             = "latent/knn_acc",
        latent_metrics      = True,
        latent_every_n      = 1,           # every val epoch — required by
                                           # ModelCheckpoint/EarlyStopping.
        latent_sample       = 2000,
        # post-hoc skipped during sweep
        target_coverage     = 0.85,
        skip_calibration    = True,
    )

    if dry_run:
        print(f"  [dry_run] knn trial {trial.number}: {trial.params}")
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


# ──────────────────────────────────────────────────────────────── CLI ──

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna sweep for HydroPreciseV2 (target: latent/knn_acc)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",            default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--n_trials",            type=int,   default=200)
    p.add_argument("--timeout_s",           type=int,   default=43_200)
    p.add_argument("--trial_max_epochs",    type=int,   default=30,
                   help="AL-derived: 2k subset converges faster than full data, "
                        "but KNN purity needs more epochs than precision.")
    p.add_argument("--trial_patience",      type=int,   default=8,
                   help="Higher than precision sweeps — KNN purity is jitterier.")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)
    p.add_argument("--train_subset_size",   type=int,   default=2000,
                   help="Stratified subset size (AL-style). 0 disables; full pool used.")
    p.add_argument("--fixed_subset",        action="store_true",
                   help="Use the same subset draw across all trials. Default is "
                        "per-trial seed (trial.number).")
    p.add_argument("--port",                type=int,   default=DASHBOARD_PORT)
    p.add_argument("--no_dashboard",        action="store_true")
    p.add_argument("--dashboard_only",      action="store_true")
    p.add_argument("--dry_run",             action="store_true")
    p.add_argument("--sampler_seed",        type=int,   default=0)
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

    sampler = TPESampler(seed=args.sampler_seed, multivariate=True, n_startup_trials=12)
    pruner  = MedianPruner(n_startup_trials=8, n_warmup_steps=4)

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE,
        direction="maximize", sampler=sampler, pruner=pruner,
        load_if_exists=True,
    )

    print(f"\n{'=' * 60}")
    print(f"  Study:        {STUDY_NAME}")
    print(f"  Target:       latent/knn_acc  (5-NN purity, post-pool embedding)")
    print(f"  Sampler:      TPESampler (multivariate, n_startup=12)")
    print(f"  Subset:       {args.train_subset_size} files "
          f"({'fixed' if args.fixed_subset else 'per-trial seed'})")
    print(f"  Per trial:    {args.trial_max_epochs} epochs, patience={args.trial_patience}")
    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"  Trials prior: {n_done} complete  |  budget: n_trials={args.n_trials} "
          f"timeout={args.timeout_s}s")
    print(f"  Storage:      {STORAGE}")
    print(f"  Dashboard:    http://localhost:{args.port}")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    def _wrap(trial: optuna.Trial) -> float:
        return _objective(
            trial,
            data_dir=args.data_dir,
            dry_run=args.dry_run,
            trial_max_epochs=args.trial_max_epochs,
            trial_patience=args.trial_patience,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            train_subset_size=args.train_subset_size,
            fixed_subset=args.fixed_subset,
        )

    show_bar = sys.stdout.isatty()
    study.optimize(_wrap, n_trials=args.n_trials, timeout=args.timeout_s,
                   show_progress_bar=show_bar, gc_after_trial=True)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned   = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed   = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"\n{'=' * 60}")
    print(f"  KNN sweep summary ({time.time() - t0:.0f}s)")
    print(f"  complete={len(complete)}  pruned={len(pruned)}  failed={len(failed)}")
    if complete:
        best = study.best_trial
        print(f"  Best trial: #{best.number}")
        print(f"  Best latent/knn_acc: {best.value:.4f}")
        print(f"  Best params:")
        for k, v in best.params.items():
            print(f"    {k:22s} = {v}")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
