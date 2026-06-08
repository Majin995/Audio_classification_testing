"""③ rapid_1s LR stacker — MLMC (multi-label) variant.

Reproduces the ``rapid_1s_honest_winner`` feature pipeline (5 HydroHydra
checkpoints' per-clip softmax concatenated → 20-D) but swaps the single
``LogisticRegression`` argmax classifier for a **One-vs-Rest** multi-label
stacker that emits a one-hot / multi-hot row per clip.

Honest protocol (unchanged from the winner):
  • stacker fit on the VAL split (64 clips) only;
  • per-class thresholds tuned on the independent TRAIN split (256 clips);
  • test (128 clips) scored exactly once.

Features come from the cached dump ``campaign/probs_rapid_1s/*.npz``
(train/val/test per-clip probs + labels). Outputs:
``stacker_ovr.joblib``, ``test_onehot.npy``, ``test_y_multihot.npy``,
``thresholds.json``, ``metrics.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from mlmc_common import (
    tune_thresholds_per_class, probs_to_onehot, multilabel_report,
)


def build_split(meta, split):
    """Concat the 5 ckpts' per-clip probs (in meta order) → (N, 20) + labels."""
    feats, y = [], None
    for ck in meta["ckpts"]:
        z = np.load(ck["npz"])
        feats.append(z[f"{split}_probs"])
        y = z[f"{split}_y"]            # identical across ckpts
    X = np.concatenate(feats, axis=1).astype(np.float32)
    return X, y.astype(np.int64)


def onehot(y, n):
    M = np.zeros((len(y), n), dtype=np.float32)
    M[np.arange(len(y)), y] = 1.0
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--single_label", action="store_true",
                    help="strict argmax one-hot (exactly one bit/row) for "
                         "single-label data; ignores per-class thresholds.")
    ap.add_argument("--out", default="lightning_logs/mlmc/rapid_lr_stacker")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    meta = json.load(open("campaign/probs_rapid_1s/_meta.json"))
    classes = meta["classes"]
    nC = len(classes)

    Xtr, ytr = build_split(meta, "train")
    Xva, yva = build_split(meta, "val")
    Xte, yte = build_split(meta, "test")
    print(f"features: train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    # Fit OVR-LR on VAL (matches the winner's "stacker fit on val" protocol).
    clf = OneVsRestClassifier(
        LogisticRegression(C=args.C, max_iter=2000))
    clf.fit(Xva, onehot(yva, nC))

    # Per-class probabilities; tune thresholds on the independent TRAIN split.
    tr_scores = clf.predict_proba(Xtr)
    thr = tune_thresholds_per_class(tr_scores, onehot(ytr, nC))
    print(f"[train] per-class thr={dict(zip(classes, thr.round(3).tolist()))}")
    multilabel_report("TRAIN", onehot(ytr, nC),
                      probs_to_onehot(tr_scores, thr, single_label=args.single_label),
                      classes)

    te_scores = clf.predict_proba(Xte)
    pred_test = probs_to_onehot(te_scores, thr, single_label=args.single_label)
    print("\n--- TEST (touched once) ---")
    m = multilabel_report("TEST", onehot(yte, nC), pred_test, classes)

    import joblib
    joblib.dump(clf, out / "stacker_ovr.joblib")
    np.save(out / "test_onehot.npy", pred_test)
    np.save(out / "test_y_multihot.npy", onehot(yte, nC))
    (out / "thresholds.json").write_text(json.dumps(
        {"per_class": dict(zip(classes, thr.tolist())), "C": args.C}, indent=2))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
