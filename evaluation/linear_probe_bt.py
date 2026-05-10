"""
Linear probe for HydroBarlowTwins SSL pretraining.

Loads the best BT checkpoint, extracts the HydroPrecise encoder, freezes it,
computes pooled embeddings for the train/val/test splits, fits a multinomial
logistic regression on the train embeddings, and reports classification
metrics on val and test.

Usage:
    python evaluation/linear_probe_bt.py \
      --ckpt lightning_logs/hydro_barlow_twins_full/version_0/checkpoints/bt-005-11338.4922.ckpt \
      --data_dir "/run/media/damo/Lexar M2/Data/Classifier_Dataset" \
      --batch_size 64 --num_threads 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    classification_report,
)

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_barlow_twins   import HydroBarlowTwins


def get_args():
    p = argparse.ArgumentParser(description="BT encoder linear probe")
    p.add_argument("--ckpt", required=True, help="Path to BT .ckpt")
    p.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--sample_rate", type=int, default=5_120)
    p.add_argument("--fixed_len",   type=int, default=5_120)
    p.add_argument("--lr_max_iter", type=int, default=2000)
    p.add_argument("--lr_C",        type=float, default=1.0)
    p.add_argument("--out_dir",     default=None,
                   help="Where to write report.json. Default: ckpt_dir/linear_probe/")
    return p.parse_args()


@torch.no_grad()
def _embed_split(model: HydroBarlowTwins, loader, device: torch.device,
                 split: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeds, labels = [], []
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        z = model.encode(x)                              # (B, 2*fusion_dim)
        embeds.append(z.float().cpu().numpy())
        labels.append(y.cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  [{split}] batch {i+1}  "
                  f"({(i+1)*loader.batch_size if hasattr(loader,'batch_size') else (i+1)} samples)  "
                  f"elapsed {time.time()-t0:.1f}s")
    E = np.concatenate(embeds, axis=0)
    Y = np.concatenate(labels, axis=0)
    print(f"  [{split}] done — {E.shape[0]} samples, dim={E.shape[1]}, "
          f"elapsed {time.time()-t0:.1f}s")
    return E, Y


def main():
    args = get_args()
    if not args.data_dir:
        raise ValueError("Set --data_dir or DATA_DIR")

    ckpt_path = Path(args.ckpt).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "linear_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    # ── Load BT model ────────────────────────────────────────────────────
    model = HydroBarlowTwins.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False,
    ).to(device)
    print(f"Loaded HydroBarlowTwins, encoder embed dim = {model._embed_dim}")

    # ── Build DataModule with disabled oversampling for probe ────────────
    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        oversample_train=False,
    )
    data.setup()
    classes = [data.idx_to_class[i] for i in range(data.num_classes)]
    print(f"Classes: {classes}")

    # ── Embed all splits ─────────────────────────────────────────────────
    print("\nExtracting embeddings:")
    E_train, Y_train = _embed_split(model, data.train_dataloader(), device, "train")
    E_val,   Y_val   = _embed_split(model, data.val_dataloader(),   device, "val")
    E_test,  Y_test  = _embed_split(model, data.test_dataloader(),  device, "test")

    # ── Fit logistic regression ──────────────────────────────────────────
    print("\nFitting multinomial logistic regression …")
    t0 = time.time()
    clf = LogisticRegression(
        solver="lbfgs", max_iter=args.lr_max_iter, C=args.lr_C, n_jobs=-1,
    )
    clf.fit(E_train, Y_train)
    print(f"  fit took {time.time()-t0:.1f}s, train acc = "
          f"{accuracy_score(Y_train, clf.predict(E_train)):.4f}")

    # ── Evaluate ─────────────────────────────────────────────────────────
    def _eval(E, Y, split_name: str) -> dict:
        pred = clf.predict(E)
        prob = clf.predict_proba(E)
        try:
            auc = roc_auc_score(Y, prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
        cm = confusion_matrix(Y, pred).tolist()
        prec_per_class = precision_score(Y, pred, average=None, zero_division=0).tolist()
        return {
            "split": split_name,
            "n_samples": int(len(Y)),
            "accuracy":         float(accuracy_score(Y, pred)),
            "f1_macro":         float(f1_score(Y, pred, average="macro")),
            "precision_macro":  float(precision_score(Y, pred, average="macro", zero_division=0)),
            "precision_micro":  float(precision_score(Y, pred, average="micro", zero_division=0)),
            "recall_macro":     float(recall_score(Y, pred, average="macro", zero_division=0)),
            "mcc":              float(matthews_corrcoef(Y, pred)),
            "auroc_macro_ovr":  float(auc),
            "precision_per_class": prec_per_class,
            "confusion_matrix":    cm,
            "classification_report":
                classification_report(Y, pred, target_names=classes,
                                      zero_division=0, digits=4),
        }

    val_metrics  = _eval(E_val,  Y_val,  "val")
    test_metrics = _eval(E_test, Y_test, "test")

    # ── Print + save ─────────────────────────────────────────────────────
    print("\n══════ Linear Probe Results ══════")
    for m in (val_metrics, test_metrics):
        print(f"\n[{m['split']}] (n={m['n_samples']})")
        print(f"  accuracy        = {m['accuracy']:.4f}")
        print(f"  f1_macro        = {m['f1_macro']:.4f}")
        print(f"  precision_macro = {m['precision_macro']:.4f}")
        print(f"  precision_micro = {m['precision_micro']:.4f}")
        print(f"  recall_macro    = {m['recall_macro']:.4f}")
        print(f"  mcc             = {m['mcc']:.4f}")
        print(f"  auroc           = {m['auroc_macro_ovr']:.4f}")
        print(f"  precision/class = {[f'{p:.3f}' for p in m['precision_per_class']]}")
        print(f"  confusion_matrix:")
        for row in m["confusion_matrix"]:
            print(f"    {row}")

    report = {
        "ckpt": str(ckpt_path),
        "encoder_embed_dim": int(model._embed_dim),
        "logistic_regression": {"C": args.lr_C, "max_iter": args.lr_max_iter},
        "classes": classes,
        "val":  val_metrics,
        "test": test_metrics,
    }
    out_path = out_dir / "report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")

    # Also dump embeddings for reuse
    np.savez(out_dir / "embeddings.npz",
             E_train=E_train, Y_train=Y_train,
             E_val=E_val, Y_val=Y_val,
             E_test=E_test, Y_test=Y_test)
    print(f"Saved → {out_dir / 'embeddings.npz'}")


if __name__ == "__main__":
    main()
