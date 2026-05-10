"""
tune_svm.py — Hyperparameter optimisation for the cuML GPU-accelerated SVM.

Strategy
--------
* Extract CNN features ONCE (the expensive step), then search over SVM
  hyperparameters without touching the GPU forward pass again.
* Optuna TPE sampler (Bayesian optimisation) with MedianPruner — far more
  efficient than grid/random search for this 3-5 dimensional space.
* 5-fold StratifiedKFold on the training features; macro-F1 objective.
* After the study, the best params are used to refit on the full training set
  and evaluated on the held-out test set.

Outputs  (<out_dir>/)
-------
  best_params.json          — best hyperparameters
  tuning_results.csv        — all trial results
  study.db                  — Optuna SQLite storage (resumable)
  plots/opt_history.png     — objective vs trial number
  plots/param_importance.png — hyperparameter importance
  plots/contour_C_gamma.png  — C vs gamma contour (rbf trials only)

Usage
-----
    python training/tune_svm.py \\
        --ckpt  lightning_logs/cnn1d/version_3/checkpoints/cnn1d-epoch=16-val_f1_score=0.260.ckpt \\
        --data_dir data/Split1s

    # Resume an interrupted study:
    python training/tune_svm.py --ckpt <ckpt> --study_name svm_tune_v1

    # Fewer trials for a quick sanity check:
    python training/tune_svm.py --ckpt <ckpt> --n_trials 20
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# cuML keeps a GPU kernel-cache working set per SVC object. Without explicit
# teardown the C++ layer raises "Working set has already been initialized!" on
# the second fit inside the same process.  Free all cupy memory pools after
# each SVC to force the RAFT allocator to release those resources cleanly.
try:
    import cupy as cp
    def _flush_gpu():
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
except ImportError:
    def _flush_gpu():
        gc.collect()

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_cnn1d import HydroCNN1D

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for cuML GPU SVM on CNN features"
    )
    p.add_argument("--ckpt",        required=True,
                   help="HydroCNN1D checkpoint (.ckpt) to use as feature extractor")
    p.add_argument("--data_dir",    default=os.environ.get("DATA_DIR", ""),
                   help="Path to Split1s root (or set $DATA_DIR)")
    p.add_argument("--sample_rate", type=int, default=5_120,
                   choices=[2560, 5120])
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--merge_cargo_passenger", action="store_true",
                   help="Merge Cargo+Passenger (must match CNN training config)")

    p.add_argument("--n_trials",    type=int, default=150,
                   help="Number of Optuna trials")
    p.add_argument("--n_cv_folds",  type=int, default=5,
                   help="StratifiedKFold splits for cross-validation")
    p.add_argument("--study_name",  default="svm_tune_v1",
                   help="Optuna study name — reuse to resume an interrupted search")
    p.add_argument("--out_dir",     default="reports/svm_tuning",
                   help="Directory for results, plots and study DB")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction  (identical pattern to eval_classical.py)
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(
    model: HydroCNN1D,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(model.extract_features(x.to(device)).cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ──────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────────────────────────────────────

def make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int,
    seed:    int,
):
    """
    Returns a closure over the pre-extracted training features.

    The cuML SVC is imported inside the closure so the module falls back
    gracefully if cuML is unavailable (though that shouldn't happen here).
    """
    try:
        from cuml.svm import SVC
        _backend = "cuml"
    except ImportError:
        from sklearn.svm import SVC
        _backend = "sklearn"
    logger.info("SVM backend for tuning: %s", _backend)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ─────────────────────────────────────────
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        C      = trial.suggest_float("C", 1e-2, 1e3, log=True)
        tol    = trial.suggest_float("tol", 1e-5, 1e-2, log=True)

        params: dict = dict(
            C=C,
            kernel=kernel,
            tol=tol,
            probability=False,           # skip Platt calibration during search (faster)
            output_type="numpy",
            decision_function_shape="ovo",  # OvO is more stable than OvR in cuML multiclass
            random_state=seed,
        )

        if kernel in ("rbf", "poly"):
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)
            params["coef0"]  = trial.suggest_float("coef0", -1.0, 1.0)

        # ── K-fold cross-validation ────────────────────────────────────────
        fold_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            try:
                svc = SVC(**params)
                svc.fit(X_train[tr_idx], y_train[tr_idx])
                y_pred = svc.predict(X_train[val_idx])
            except RuntimeError as e:
                # cuML "Working set has already been initialized" — release GPU
                # state and retry once before giving up on this trial.
                _flush_gpu()
                try:
                    svc = SVC(**params)
                    svc.fit(X_train[tr_idx], y_train[tr_idx])
                    y_pred = svc.predict(X_train[val_idx])
                except RuntimeError:
                    raise optuna.TrialPruned()
            finally:
                try:
                    del svc
                except NameError:
                    pass
                _flush_gpu()

            if hasattr(y_pred, "get"):
                y_pred = y_pred.get()
            score = f1_score(y_train[val_idx], y_pred, average="macro", zero_division=0)
            fold_scores.append(score)

            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


# ──────────────────────────────────────────────────────────────────────────────
# Final evaluation with best params
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_best(
    best_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    num_classes: int,
    seed: int,
) -> dict:
    """Refit on full training set with best params; evaluate on test set."""
    try:
        from cuml.svm import SVC
    except ImportError:
        from sklearn.svm import SVC

    # Enable probability for ROC-AUC
    params = {**best_params, "probability": True, "output_type": "numpy", "random_state": seed}
    svc = SVC(**params)
    logger.info("Fitting final SVM on full training set (%d samples) ...", len(X_train))
    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_test)
    y_prob = svc.predict_proba(X_test)
    if hasattr(y_pred, "get"):
        y_pred = y_pred.get()
    if hasattr(y_prob, "get"):
        y_prob = y_prob.get()

    roc_auc = roc_auc_score(
        y_test, y_prob, multi_class="ovr", average="macro",
        labels=list(range(num_classes)),
    )
    return {
        "accuracy":        round(accuracy_score(y_test, y_pred), 4),
        "f1_macro":        round(f1_score(y_test, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "roc_auc_ovr":     round(roc_auc, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def save_plots(study: optuna.Study, plot_dir: Path) -> None:
    """Save Optuna visualisation plots as PNGs (requires matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")

    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.get_figure().savefig(plot_dir / "opt_history.png", dpi=150, bbox_inches="tight")
    except Exception as e:
        logger.warning("opt_history plot failed: %s", e)

    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.get_figure().savefig(plot_dir / "param_importance.png", dpi=150, bbox_inches="tight")
    except Exception as e:
        logger.warning("param_importance plot failed: %s", e)

    try:
        fig = optuna.visualization.matplotlib.plot_contour(study, params=["C", "gamma"])
        fig.get_figure().savefig(plot_dir / "contour_C_gamma.png", dpi=150, bbox_inches="tight")
    except Exception as e:
        logger.warning("contour plot failed: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    if not args.data_dir:
        raise ValueError("--data_dir is required (or set $DATA_DIR)")

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── DataModule ────────────────────────────────────────────────────────────
    merge = {"Passenger": "Cargo"} if args.merge_cargo_passenger else {}
    dm = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_workers,
        target_sr=args.sample_rate,
        fixed_len=args.sample_rate,
        oversample_train=False,
        merge_classes=merge,
    )
    dm.setup()
    num_classes = dm.num_classes
    logger.info("Classes (%d): %s", num_classes, dm.class_to_idx)

    # ── Load frozen CNN ───────────────────────────────────────────────────────
    logger.info("Loading CNN: %s", args.ckpt)
    cnn = HydroCNN1D.load_from_checkpoint(args.ckpt).to(device).eval()
    for p in cnn.parameters():
        p.requires_grad_(False)

    # ── Extract features ONCE (reused across all trials) ─────────────────────
    logger.info("Extracting train features ...")
    X_train, y_train = extract_features(cnn, dm.train_dataloader(), device)
    logger.info("Extracting test features ...")
    X_test,  y_test  = extract_features(cnn, dm.test_dataloader(),  device)
    logger.info("Train: %s  Test: %s", X_train.shape, X_test.shape)

    # StandardScaler fitted on train only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Optuna study ──────────────────────────────────────────────────────────
    storage  = f"sqlite:///{out_dir / 'study.db'}"
    sampler  = TPESampler(seed=args.seed, multivariate=True)   # multivariate TPE
    pruner   = MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,     # resume if study already exists
    )

    already_done = len(study.trials)
    remaining    = max(0, args.n_trials - already_done)
    logger.info(
        "Study '%s': %d existing trials, running %d more (target %d)",
        args.study_name, already_done, remaining, args.n_trials,
    )

    objective = make_objective(X_train, y_train, args.n_cv_folds, args.seed)

    def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(
                "Trial %3d | F1=%.4f | %s",
                trial.number, trial.value,
                {k: (f"{v:.4g}" if isinstance(v, float) else v)
                 for k, v in trial.params.items()},
            )

    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            callbacks=[trial_callback],
            show_progress_bar=True,
        )

    # ── Results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    logger.info("\nBest trial  : #%d", best.number)
    logger.info("Best CV F1  : %.4f", best.value)
    logger.info("Best params : %s", best.params)

    # Save best params
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=2)
    logger.info("Saved best_params.json")

    # Save all trials table
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "tuning_results.csv", index=False)
    logger.info("Saved tuning_results.csv (%d rows)", len(trials_df))

    # ── Final test-set evaluation with best params ────────────────────────────
    logger.info("\nEvaluating best params on held-out test set ...")
    test_metrics = evaluate_best(
        best.params, X_train, y_train, X_test, y_test, num_classes, args.seed
    )
    logger.info("Test-set results: %s", test_metrics)

    # Append to / create summary JSON
    summary = {
        "study_name":   args.study_name,
        "n_trials":     len(study.trials),
        "best_cv_f1":   round(best.value, 4),
        "best_params":  best.params,
        "test_metrics": test_metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    save_plots(study, plot_dir)

    # ── Final printout ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  SVM Hyperparameter Tuning — {args.study_name}")
    print("=" * 60)
    print(f"  Trials completed : {len(study.trials)}")
    print(f"  Best CV F1 macro : {best.value:.4f}")
    print(f"  Best params      : {json.dumps(best.params, indent=4)}")
    print(f"\n  Test-set metrics (best params, full train fit):")
    for k, v in test_metrics.items():
        print(f"    {k:20s}: {v}")
    print(f"\n  Results saved to : {out_dir.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
