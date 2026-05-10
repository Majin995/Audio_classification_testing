#!/usr/bin/env python3
"""
Optuna TPE sweep for UATR model architectures.

Each model gets its own named study persisted in optuna_studies.db.
optuna-dashboard is launched automatically when the first sweep starts.

Usage:
    # Single model
    python scripts/optuna_sweep.py --model catfish --n_trials 50

    # With SSCP-Mobile (requires BAHTNet teacher)
    python scripts/optuna_sweep.py --model sscp_mobile --teacher_ckpt /path/to/bahtnet.ckpt

    # Dry run (print trial params without training)
    python scripts/optuna_sweep.py --model catfish --dry_run

    # Dashboard URL only (don't run any sweep)
    python scripts/optuna_sweep.py --dashboard_only
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import socket
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

# Suppress BrokenPipeError when stdout is piped and the consumer closes early
# (e.g. tee under bash set -uo pipefail).  SIG_DFL makes Python exit silently
# with the POSIX SIGPIPE exit code instead of printing a traceback.
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# ── repo root on path ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ── constants ─────────────────────────────────────────────────────────────────
DB_PATH       = ROOT / "optuna_studies.db"
STORAGE       = f"sqlite:///{DB_PATH}"
DASHBOARD_PORT = 8080
STUDY_PREFIX  = "uatr_"

# Written after BAHTNet sweep; read by SSCP sweep when no --teacher_ckpt given.
BAHTNET_CKPT_CACHE = ROOT / "optuna_bahtnet_best.txt"

_VALID_MODELS = ("catfish", "alsi", "dcn", "bahtnet", "sscp_mobile", "super", "i2hofi",
                 "hydra", "precise")


# ── dashboard helpers ─────────────────────────────────────────────────────────

def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def launch_dashboard(*, port: int = DASHBOARD_PORT) -> subprocess.Popen | None:
    """
    Start optuna-dashboard in the background (once per machine session).
    Silently skips if the port is already occupied.
    Prints the dashboard URL regardless.
    """
    url = f"http://localhost:{port}"
    if _port_in_use(port):
        print(f"[optuna-dashboard] already running → {url}")
        return None

    # Resolve the dashboard binary relative to the running Python interpreter
    # so it works when called under nohup / without the conda env on PATH.
    _py = Path(sys.executable)
    _dashboard_bin = _py.parent / "optuna-dashboard"
    _dashboard_cmd = str(_dashboard_bin) if _dashboard_bin.exists() else "optuna-dashboard"

    try:
        proc = subprocess.Popen(
            [_dashboard_cmd, STORAGE, "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,   # survive parent exit
        )
        # Give it a moment to bind the port
        for _ in range(10):
            time.sleep(0.5)
            if _port_in_use(port):
                break
        print(f"\n[optuna-dashboard] started (PID {proc.pid}) → {url}")
        print(f"[optuna-dashboard] storage: {DB_PATH}\n")
        return proc
    except FileNotFoundError:
        print(
            "[optuna-dashboard] not found — install with:  pip install optuna-dashboard\n"
            f"  You can run it manually later:  optuna-dashboard {STORAGE} --port {port}"
        )
        return None


# ── per-model objective factories ─────────────────────────────────────────────

def _common_args(trial: optuna.Trial, data_dir: str, model_name: str) -> dict:
    """Shared hyperparameters sampled for every model.

    NOTE: ``batch_size`` and ``lr`` are intentionally absent here — each
    objective function defines them independently because the valid search
    space differs per model.  Including them here and then re-suggesting them
    with different choices in an objective would trigger Optuna's
    ``CategoricalDistribution does not support dynamic value space`` error.
    """
    return dict(
        data_dir       = data_dir,
        num_threads    = 8,
        no_oversample  = False,
        sample_rate    = 5_120,
        fixed_len      = 5_120,
        max_epochs     = 100,
        warmup_epochs  = 10,
        patience       = 20,
        precision      = "16-mixed",
        grad_clip      = 1.0,
        limit_train_batches = 1.0,
        limit_val_batches   = 1.0,
        seed         = trial.suggest_int("seed", 0, 9999),
        weight_decay = trial.suggest_categorical("weight_decay", [1e-3, 1e-2]),
        denoise      = trial.suggest_categorical("denoise", ["off", "emd_wavelet"]),
        dropout      = trial.suggest_float("dropout", 0.05, 0.3),
        mixup_alpha  = 0.3,
        focal_gamma  = 2.0,
        label_smoothing = 0.05,
        run_name = f"optuna_{model_name}_t{trial.number:04d}",
    )


def _objective_catfish(trial: optuna.Trial, data_dir: str, dry_run: bool) -> float:
    from training.train_catfish import main as _train

    args = Namespace(
        **_common_args(trial, data_dir, "catfish"),
        lr         = trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        batch_size = trial.suggest_categorical("batch_size", [32, 64]),
        gabor_n_filters = trial.suggest_categorical("gabor_n_filters", [32, 64]),
        gabor_kernel    = trial.suggest_categorical("gabor_kernel", [129, 257]),
        tcn_channels    = trial.suggest_categorical("tcn_channels", [64, 128]),
        n_tcn_blocks    = trial.suggest_int("n_tcn_blocks", 4, 8),
    )
    if dry_run:
        print(f"  [dry_run] catfish trial {trial.number}: {trial.params}")
        return 0.0
    try:
        score, _ = _train(args)
    except Exception as exc:
        raise optuna.TrialPruned(f"Training raised: {exc}") from exc
    return score


def _objective_alsi(trial: optuna.Trial, data_dir: str, dry_run: bool) -> float:
    from training.train_alsi import main as _train

    args = Namespace(
        **_common_args(trial, data_dir, "alsi"),
        lr         = trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        batch_size = trial.suggest_categorical("batch_size", [16, 32]),
        fusion_dim   = trial.suggest_categorical("fusion_dim", [128, 256]),
        fusion_heads = trial.suggest_categorical("fusion_heads", [4, 8]),
        freeze_wav2vec = trial.suggest_categorical("freeze_wav2vec", [True, False]),
        cqt_bins     = trial.suggest_categorical("cqt_bins", [64, 84]),
    )
    if dry_run:
        print(f"  [dry_run] alsi trial {trial.number}: {trial.params}")
        return 0.0
    try:
        score, _ = _train(args)
    except Exception as exc:
        raise optuna.TrialPruned(f"Training raised: {exc}") from exc
    return score


def _objective_dcn(trial: optuna.Trial, data_dir: str, dry_run: bool) -> float:
    from training.train_dcn import main as _train

    args = Namespace(
        **_common_args(trial, data_dir, "dcn"),
        lr         = trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        batch_size = trial.suggest_categorical("batch_size", [32, 64]),
        n_fft        = trial.suggest_categorical("n_fft", [256, 512]),
        hop_length   = 51,
        dcmf_templates = trial.suggest_categorical("dcmf_templates", [16, 32]),
        complex_depth  = trial.suggest_int("complex_depth", 3, 4),
        base_channels  = trial.suggest_categorical("base_channels", [16, 32]),
    )
    if dry_run:
        print(f"  [dry_run] dcn trial {trial.number}: {trial.params}")
        return 0.0
    try:
        score, _ = _train(args)
    except Exception as exc:
        raise optuna.TrialPruned(f"Training raised: {exc}") from exc
    return score


def _objective_bahtnet(trial: optuna.Trial, data_dir: str, dry_run: bool) -> tuple[float, str]:
    from training.train_bahtnet import main as _train

    loss_type  = trial.suggest_categorical("loss", ["focal", "lmf"])
    lmf_margin = (
        trial.suggest_categorical("lmf_margin", [0.2, 0.35, 0.5])
        if loss_type == "lmf" else 0.0
    )
    args = Namespace(
        **_common_args(trial, data_dir, "bahtnet"),
        lr         = trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        batch_size = trial.suggest_categorical("batch_size", [32, 64]),
        n_mels      = trial.suggest_categorical("n_mels", [64, 128]),
        n_fft       = 512,
        hop_length  = 51,
        patch_size  = trial.suggest_categorical("patch_size", [4, 8]),
        model_dim   = trial.suggest_categorical("model_dim", [256, 384]),
        n_heads     = trial.suggest_categorical("n_heads", [4, 6]),
        n_layers    = trial.suggest_int("n_layers", 4, 8),
        drop_path   = trial.suggest_float("drop_path", 0.05, 0.25),
        loss        = loss_type,
        lmf_gamma   = 2.0,
        lmf_margin  = lmf_margin,
    )
    if dry_run:
        print(f"  [dry_run] bahtnet trial {trial.number}: {trial.params}")
        return 0.0, ""
    try:
        score, ckpt_path = _train(args)
    except Exception as exc:
        raise optuna.TrialPruned(f"Training raised: {exc}") from exc
    return score, ckpt_path


def _objective_super(trial: optuna.Trial, data_dir: str, dry_run: bool) -> float:
    """Delegate to the standalone tune_super_model objective (DRY)."""
    from training.tune_super_model import _objective_super as _obj
    return _obj(trial, data_dir, dry_run)


def _objective_i2hofi(trial: optuna.Trial, data_dir: str, dry_run: bool) -> float:
    """LOFAR-mode i2-hofi sweep — varies hz_per_grid (drives n_fft + grid_h),
    per-layer dropout, lr, batch size, and a lightweight backbone choice.
    grid_size is *derived* inside train_i2hofi.main from hz_per_grid; do not
    sample it here."""
    from training.train_i2hofi import main as _train

    base = _common_args(trial, data_dir, "i2hofi")
    # Per-layer dropout overrides — the shared `dropout` from _common_args
    # remains as the default for any layer not explicitly set.
    args = Namespace(
        **base,
        lr                 = trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        batch_size         = trial.suggest_categorical("batch_size", [16, 32]),
        hz_per_grid        = trial.suggest_float("hz_per_grid", 0.5, 5.0),
        dropout_appnp      = trial.suggest_float("dropout_appnp",      0.0, 0.4),
        dropout_gat        = trial.suggest_float("dropout_gat",        0.0, 0.4),
        dropout_classifier = trial.suggest_float("dropout_classifier", 0.0, 0.5),
        backbone           = trial.suggest_categorical("backbone", ["resnet18", "resnet34"]),
        # Fixed for this study — isolate hz_per_grid as the sole grid driver.
        stft_f_max_hz      = 2_560.0,
        time_frames        = 32,
        grid_w             = 4,
        max_grid_h         = 16,
        # Forwarded fixed defaults
        gcn_out_features   = 256,
        gat_out_features   = 256,
        appnp_K            = 3,
        alpha              = 0.3,
        gat_heads          = 1,
        pool_h             = 3,
        pool_w             = 3,
        pretrained         = True,
    )
    if dry_run:
        print(f"  [dry_run] i2hofi trial {trial.number}: {trial.params}")
        return 0.0
    try:
        score, _ = _train(args)
    except Exception as exc:
        raise optuna.TrialPruned(f"Training raised: {exc}") from exc
    return score


def _objective_hydra(trial: optuna.Trial, data_dir: str, dry_run: bool) -> float:
    """HydroHydra — unified spectrogram-free dual-stream model.
    Optimises val/macro_precision (precision-first objective)."""
    from training.train_hydra import main as _train

    base = _common_args(trial, data_dir, "hydra")
    base["warmup_epochs"] = 8
    base["patience"]      = 15
    base["max_epochs"]    = 80

    # Stream toggles — prune the both-off corner
    use_gabor      = trial.suggest_categorical("use_gabor",      [True, False])
    use_scattering = trial.suggest_categorical("use_scattering", [True, False])
    if not (use_gabor or use_scattering):
        raise optuna.TrialPruned("both streams disabled")

    args = Namespace(
        **base,
        lr               = trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        batch_size       = trial.suggest_categorical("batch_size", [32, 64]),
        use_gabor        = use_gabor,
        use_scattering   = use_scattering,
        gabor_n_filters  = trial.suggest_categorical("gabor_n_filters", [48, 64, 96]),
        gabor_kernel     = 257,
        gabor_ch         = 128,
        scat_J           = trial.suggest_categorical("scat_J", [5, 6, 7]),
        scat_Q           = trial.suggest_categorical("scat_Q", [4, 8, 12]),
        scat_ch          = 128,
        fusion_T         = 80,
        fusion_dim       = 192,
        s4_n_blocks      = trial.suggest_int("s4_n_blocks", 1, 3),
        s4_d_state       = 64,
        loss             = "lmf",
        lmf_gamma        = trial.suggest_categorical("lmf_gamma", [1.0, 2.0, 3.0]),
        lmf_margin       = trial.suggest_categorical("lmf_margin", [0.3, 0.5, 0.7, 0.9]),
        gambler_o        = 0.3,
        gambler_weight   = trial.suggest_categorical("gambler_weight", [0.0, 0.1, 0.2]),
        noise_prob       = 0.5,
        noise_snr_min    = 15.0,
        noise_snr_max    = 30.0,
        gain_prob        = 0.5,
        gain_range       = 0.3,
        target_coverage  = 0.85,
        skip_calibration = False,
    )
    if dry_run:
        print(f"  [dry_run] hydra trial {trial.number}: {trial.params}")
        return 0.0
    try:
        score, _ = _train(args)
    except Exception as exc:
        raise optuna.TrialPruned(f"Training raised: {exc}") from exc
    return score


def _objective_precise(trial: optuna.Trial, data_dir: str, dry_run: bool) -> float:
    """HydroPrecise — thin adapter around the standalone tuner in
    training/tune_hydro_precise.py so `scripts/optuna_sweep.py --model precise`
    shares the same objective and study configuration.

    Sweep epoch/patience caps are conservative here (tractable search); the
    best-trial params can later be trained to completion via train_precise.py.
    """
    from training.tune_hydro_precise import _objective as _obj_precise

    return _obj_precise(
        trial, data_dir, dry_run,
        trial_max_epochs=10,
        trial_patience=4,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
    )


def _objective_sscp_mobile(
    trial: optuna.Trial, data_dir: str, dry_run: bool, teacher_ckpt: str
) -> float:
    from training.train_sscp_mobile import main as _train

    # SSCP uses a wider lr range and larger batches than the other models.
    # All per-model params are set directly here; nothing is re-suggested via
    # _common_args because that would cause a distribution conflict.
    base = _common_args(trial, data_dir, "sscp_mobile")
    # Override fixed values that differ from _common_args defaults — set them
    # directly on the dict rather than as duplicate Namespace kwargs (which
    # raises TypeError: got multiple values for keyword argument).
    base["lr"]            = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    base["batch_size"]    = trial.suggest_categorical("batch_size", [64, 128])
    base["weight_decay"]  = 1e-3       # fixed for the tiny model
    base["warmup_epochs"] = 5
    base["patience"]      = 25
    args = Namespace(
        **base,
        n_mels       = 32,
        n_fft        = 256,
        hop_length   = 51,
        teacher_ckpt = teacher_ckpt,
        kd_alpha     = trial.suggest_categorical("kd_alpha", [0.2, 0.3, 0.5]),
        kd_temp      = trial.suggest_categorical("kd_temp", [2.0, 4.0, 8.0]),
    )
    if dry_run:
        print(f"  [dry_run] sscp_mobile trial {trial.number}: {trial.params}")
        return 0.0
    try:
        score, _ = _train(args)
    except Exception as exc:
        raise optuna.TrialPruned(f"Training raised: {exc}") from exc
    return score


# ── study runner ──────────────────────────────────────────────────────────────

def _study_name(model: str) -> str:
    return f"{STUDY_PREFIX}{model}"


def run_sweep(
    model: str,
    data_dir: str,
    n_trials: int,
    dry_run: bool,
    teacher_ckpt: str = "",
) -> str:
    # Resolve to absolute so DALI workers (which run from /tmp) find files
    data_dir = str(Path(data_dir).resolve())
    """
    Create/resume an Optuna study for *model* and run *n_trials* trials.
    Returns the best checkpoint path (non-empty only for bahtnet).
    """
    sampler = TPESampler(seed=42, multivariate=True)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study   = optuna.create_study(
        study_name    = _study_name(model),
        storage       = STORAGE,
        direction     = "maximize",
        sampler       = sampler,
        pruner        = pruner,
        load_if_exists = True,
    )

    n_done = len(study.trials)   # do DB query before opening the print block
    print(f"\n{'='*60}")
    print(f"  Study: {_study_name(model)}")
    print(f"  Trials requested: {n_trials}  (already done: {n_done})")
    print(f"  Storage: {DB_PATH}")
    print(f"{'='*60}\n")

    best_ckpt  = ""
    best_ckpts: list[tuple[float, str]] = []   # used by bahtnet branch

    if model == "catfish":
        def objective(trial):
            return _objective_catfish(trial, data_dir, dry_run)

    elif model == "alsi":
        def objective(trial):
            return _objective_alsi(trial, data_dir, dry_run)

    elif model == "dcn":
        def objective(trial):
            return _objective_dcn(trial, data_dir, dry_run)

    elif model == "bahtnet":
        def objective(trial):
            score, ckpt = _objective_bahtnet(trial, data_dir, dry_run)
            if ckpt:
                best_ckpts.append((score, ckpt))
            return score

    elif model == "sscp_mobile":
        # Resolve teacher checkpoint
        if not teacher_ckpt and BAHTNET_CKPT_CACHE.exists():
            teacher_ckpt = BAHTNET_CKPT_CACHE.read_text().strip()
            print(f"[sscp_mobile] Using cached BAHTNet ckpt: {teacher_ckpt}")
        if not teacher_ckpt:
            print(
                "[sscp_mobile] WARNING: no teacher_ckpt — training supervised-only.\n"
                "  Run BAHTNet sweep first, or pass --teacher_ckpt."
            )

        def objective(trial):
            return _objective_sscp_mobile(trial, data_dir, dry_run, teacher_ckpt)

    elif model == "super":
        def objective(trial):
            return _objective_super(trial, data_dir, dry_run)

    elif model == "i2hofi":
        def objective(trial):
            return _objective_i2hofi(trial, data_dir, dry_run)

    elif model == "hydra":
        def objective(trial):
            return _objective_hydra(trial, data_dir, dry_run)

    elif model == "precise":
        def objective(trial):
            return _objective_precise(trial, data_dir, dry_run)

    # show_progress_bar only when stdout is a real TTY; tqdm writing to a pipe
    # causes Python to exit 120 (broken pipe on the progress bar flush).
    _show_bar = sys.stdout.isatty()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=_show_bar)

    # ── post-sweep summary ────────────────────────────────────────────────────
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{'='*60}")
    print(f"  [{model}] sweep complete — {len(complete)}/{n_trials} trials finished")
    if complete:
        best = study.best_trial
        print(f"  Best trial:  #{best.number}")
        print(f"  Best val/f1: {best.value:.4f}")
        print(f"  Best params: {best.params}")
    else:
        print("  WARNING: all trials failed or were pruned — no best trial.")
    print(f"{'='*60}\n")

    # For BAHTNet: persist the best checkpoint path so SSCP can pick it up
    if model == "bahtnet" and not dry_run and complete and best_ckpts:
        best_ckpts_sorted = sorted(best_ckpts, key=lambda t: t[0], reverse=True)
        best_ckpt = best_ckpts_sorted[0][1]
        BAHTNET_CKPT_CACHE.write_text(best_ckpt)
        print(f"[bahtnet] Best ckpt saved to {BAHTNET_CKPT_CACHE}:\n  {best_ckpt}\n")

    return best_ckpt


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna TPE sweep for UATR models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", choices=list(_VALID_MODELS),
        help="Which model to sweep (omit with --dashboard_only)",
    )
    p.add_argument("--data_dir",     default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--n_trials",     type=int, default=40,
                   help="Number of Optuna trials to run")
    p.add_argument("--teacher_ckpt", default="",
                   help="BAHTNet checkpoint for SSCP-Mobile KD "
                        "(auto-resolved from optuna_bahtnet_best.txt if omitted)")
    p.add_argument("--port",         type=int, default=DASHBOARD_PORT,
                   help="optuna-dashboard port")
    p.add_argument("--no_dashboard", action="store_true",
                   help="Skip launching optuna-dashboard")
    p.add_argument("--dashboard_only", action="store_true",
                   help="Only (re)launch optuna-dashboard, then exit")
    p.add_argument("--dry_run",      action="store_true",
                   help="Print trial params without actually training")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── launch dashboard ──────────────────────────────────────────────────────
    if not args.no_dashboard:
        launch_dashboard(port=args.port)

    if args.dashboard_only:
        return

    if not args.model:
        print("ERROR: --model is required unless --dashboard_only is set.")
        sys.exit(1)

    if not args.data_dir and not args.dry_run:
        print("ERROR: set --data_dir or export DATA_DIR=/path/to/Split1s")
        sys.exit(1)

    run_sweep(
        model        = args.model,
        data_dir     = args.data_dir,
        n_trials     = args.n_trials,
        dry_run      = args.dry_run,
        teacher_ckpt = args.teacher_ckpt,
    )


if __name__ == "__main__":
    main()
