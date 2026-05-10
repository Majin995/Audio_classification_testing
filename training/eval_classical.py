"""
eval_classical.py — Stage 2: Classical head comparison against CNN baseline.

Pipeline
--------
1. Load frozen HydroCNN1D from checkpoint.
2. Extract 128-dim feature vectors from train / val / test splits.
3. Standardise features (StandardScaler fitted on train only).
4. Fit the requested classical heads (RVM, SVM, XGBoost, LightGBM, AdaBoost).
5. Evaluate every head on the test split with a common metric set:
      accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, roc_auc_ovr
6. Write (or merge into existing) results at  <out_dir>/comparison.csv
7. Write t-SNE 2-D latent-space plot to       <out_dir>/latent_tsne.png

Usage
-----
    # Run everything:
    python training/eval_classical.py \\
        --ckpt lightning_logs/cnn1d/version_0/checkpoints/cnn1d-epoch=12-val_f1_score=0.923.ckpt \\
        --out_dir reports/classical_v1

    # Run SVM + boosting first (fast), then RVM separately (sub-sampled):
    python training/eval_classical.py --ckpt <ckpt> --heads SVM XGBoost LightGBM AdaBoost
    python training/eval_classical.py --ckpt <ckpt> --heads RVM --rvm_subsample 5000
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_cnn1d import HydroCNN1D
from models.hydro_classical import build_classical_heads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate classical heads built on frozen HydroCNN1D features"
    )
    p.add_argument("--ckpt",        required=True,
                   help="Path to HydroCNN1D Lightning checkpoint (.ckpt)")
    p.add_argument("--data_dir",    default=os.environ.get("DATA_DIR", ""),
                   help="Path to Split1s root (or set $DATA_DIR)")
    p.add_argument("--sample_rate", type=int, default=5_120,
                   choices=[2560, 5120],
                   help="Must match the sample_rate used during CNN pretraining")
    p.add_argument("--batch_size",  type=int, default=256,
                   help="Batch size for feature extraction (higher = faster)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out_dir",     default="reports/classical_v1",
                   help="Directory to write comparison.csv and latent_tsne.png")
    p.add_argument("--tsne_max_samples", type=int, default=5000,
                   help="Max test-set points in t-SNE plot (subsampled if larger)")
    p.add_argument("--heads", nargs="+",
                   choices=["RVM", "SVM", "XGBoost", "LightGBM", "AdaBoost"],
                   default=None,
                   help="Which classical heads to run. Default: all available. "
                        "Example: --heads SVM XGBoost LightGBM AdaBoost")
    p.add_argument("--rvm_subsample", type=int, default=None,
                   help="If set, sub-sample the training set to this many points "
                        "before fitting RVM (recommended: 3000-5000). "
                        "RVM is O(N²) so the full dataset can take hours.")
    p.add_argument("--merge_cargo_passenger", action="store_true",
                   help="Merge Cargo+Passenger into one class. Must match what "
                        "was used during CNN pretraining.")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(
    model: HydroCNN1D,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pass all batches through model.extract_features() and collect results.

    Returns:
        X : np.ndarray  (N, 128)
        y : np.ndarray  (N,)     integer class labels
    """
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = model.extract_features(x)     # (B, 128)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """
    Compute the standard classification metrics using sklearn.

    Returns a flat dict of metric_name → float.
    """
    roc_auc = roc_auc_score(
        y_true, y_prob,
        multi_class="ovr",
        average="macro",
        labels=list(range(num_classes)),
    )
    return {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":        round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true, y_pred, average="macro",       zero_division=0), 4),
        "roc_auc_ovr":     round(roc_auc, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CNN test via Lightning (for parity row in comparison.csv)
# ──────────────────────────────────────────────────────────────────────────────

def run_cnn_test(
    ckpt_path: str,
    dm: DALIAudioDataModule,
    out_dir: Path,
) -> dict[str, float]:
    """
    Run Lightning trainer.test() on the CNN checkpoint and return metrics dict.

    Creates a throw-away Trainer so it doesn't interfere with the main training
    run's logger state.
    """
    logger.info("Running Lightning test() on CNN checkpoint for baseline row ...")
    model = HydroCNN1D.load_from_checkpoint(ckpt_path)
    tmp_logger = CSVLogger(save_dir=str(out_dir), name="cnn_test_run")
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        logger=tmp_logger,
        enable_progress_bar=True,
    )
    results = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path, verbose=False)
    # results is a list of dicts; flatten and strip the "test_" prefix
    flat = {}
    for d in results:
        for k, v in d.items():
            clean_key = k.replace("test_", "")
            flat[clean_key] = round(float(v), 4)
    return flat


# ──────────────────────────────────────────────────────────────────────────────
# t-SNE latent-space visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_latent_space(
    X: np.ndarray,
    y: np.ndarray,
    idx_to_class: dict[int, str],
    out_path: Path,
    max_samples: int = 5000,
    seed: int = 42,
) -> None:
    """
    Reduce CNN feature vectors to 2-D with t-SNE and save a colour-coded scatter.

    Args:
        X            : (N, 128) feature matrix.
        y            : (N,) integer class labels.
        idx_to_class : Mapping from label int → class name string.
        out_path     : File path for the saved PNG.
        max_samples  : Subsample cap (t-SNE is O(N²) so cap keeps it fast).
        seed         : Random state for reproducibility.
    """
    import matplotlib
    matplotlib.use("Agg")    # headless / server-safe backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE

    # Subsample to keep t-SNE tractable
    if len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[idx], y[idx]

    logger.info("Running t-SNE on %d points (128 → 2 dims) ...", len(X))
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=seed,
        n_jobs=-1,
    )
    Z = tsne.fit_transform(X)   # (N, 2)

    num_classes = len(idx_to_class)
    palette = sns.color_palette("tab10", num_classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, cls_name in sorted(idx_to_class.items()):
        mask = y == cls_idx
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            c=[palette[cls_idx]],
            label=cls_name,
            alpha=0.6,
            s=12,
            linewidths=0,
        )
    ax.set_title("t-SNE of CNN Feature Space (test set)", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(title="Class", markerscale=2.5, framealpha=0.85)
    ax.set_aspect("equal")
    sns.despine(ax=ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved latent-space plot → %s", out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    if not args.data_dir:
        raise ValueError(
            "data_dir is empty. Pass --data_dir or set the DATA_DIR env var."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

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
    num_classes  = dm.num_classes
    idx_to_class = dm.idx_to_class
    logger.info("Classes: %s", dm.class_to_idx)

    # ── Load frozen CNN ───────────────────────────────────────────────────────
    logger.info("Loading CNN checkpoint: %s", args.ckpt)
    cnn = HydroCNN1D.load_from_checkpoint(args.ckpt).to(device)
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad_(False)

    # ── Extract features ──────────────────────────────────────────────────────
    logger.info("Extracting features from train split ...")
    X_train, y_train = extract_features(cnn, dm.train_dataloader(), device)
    logger.info("Extracting features from val split ...")
    X_val,   y_val   = extract_features(cnn, dm.val_dataloader(),   device)
    logger.info("Extracting features from test split ...")
    X_test,  y_test  = extract_features(cnn, dm.test_dataloader(),  device)

    logger.info(
        "Feature shapes — train: %s  val: %s  test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    # ── Standardise (fit on train only) ───────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── CNN baseline via Lightning (torchmetrics) ─────────────────────────────
    # Only run the CNN baseline once — skip if the CSV already has it.
    csv_path = out_dir / "comparison.csv"
    existing_df = pd.read_csv(csv_path, index_col="model") if csv_path.exists() else pd.DataFrame()

    results: dict[str, dict] = {}
    if "CNN_softmax" not in existing_df.index:
        results["CNN_softmax"] = run_cnn_test(args.ckpt, dm, out_dir)
        logger.info("CNN baseline: %s", results["CNN_softmax"])
    else:
        logger.info("CNN_softmax already in %s — skipping.", csv_path)

    # ── Classical heads ───────────────────────────────────────────────────────
    all_heads = build_classical_heads(random_state=args.seed)

    # Filter to the requested subset (--heads); default = all available
    requested = set(args.heads) if args.heads else set(all_heads.keys())
    heads_to_run = {k: v for k, v in all_heads.items() if k in requested}

    if not heads_to_run:
        logger.warning("No heads matched --heads %s. Available: %s",
                       args.heads, list(all_heads.keys()))

    for name, head in heads_to_run.items():
        logger.info("─── %s ──────────────────────────────────", name)

        # RVM sub-sampling: O(N²) kernel matrix — cap training set if requested
        X_fit, y_fit = X_train, y_train
        if name == "RVM" and args.rvm_subsample and len(X_train) > args.rvm_subsample:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(X_train), size=args.rvm_subsample, replace=False)
            X_fit, y_fit = X_train[idx], y_train[idx]
            logger.info("RVM: sub-sampled training set to %d / %d points",
                        args.rvm_subsample, len(X_train))

        head.fit(X_fit, y_fit)
        y_pred = head.predict(X_test)
        y_prob = head.predict_proba(X_test)
        results[name] = compute_metrics(y_test, y_pred, y_prob, num_classes)
        logger.info("%s results: %s", name, results[name])

    # ── Persist / merge comparison table ─────────────────────────────────────
    # Merge new results into existing CSV so partial runs accumulate correctly.
    new_df = pd.DataFrame(results).T
    new_df.index.name = "model"
    if not existing_df.empty:
        merged = pd.concat([existing_df, new_df[~new_df.index.isin(existing_df.index)]])
        merged.update(new_df)          # overwrite any re-run rows
        merged.to_csv(csv_path)
        logger.info("Merged results into %s", csv_path)
    else:
        new_df.to_csv(csv_path)
        logger.info("Saved comparison table → %s", csv_path)

    # Print current state of the full comparison table
    final_df = pd.read_csv(csv_path, index_col="model")
    print("\n" + "=" * 72)
    print("  Classical Head Comparison (test set)")
    print("=" * 72)
    print(final_df.to_string())
    print("=" * 72 + "\n")

    # ── Latent-space t-SNE plot ───────────────────────────────────────────────
    tsne_path = out_dir / "latent_tsne.png"
    if not tsne_path.exists():
        plot_latent_space(
            X_test, y_test,
            idx_to_class=idx_to_class,
            out_path=tsne_path,
            max_samples=args.tsne_max_samples,
            seed=args.seed,
        )
    else:
        logger.info("t-SNE plot already exists at %s — skipping.", tsne_path)


if __name__ == "__main__":
    main()
