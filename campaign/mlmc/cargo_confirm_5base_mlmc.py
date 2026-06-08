"""① cargo_confirm_5base — MLMC (multi-label) variant.

Same 5-Hydra base pool + complete-044 voucher as
``campaign/cargo_confirm_5base.py``, but emits **multi-hot** per-source
predictions (independent per-class bit) instead of a single argmax label.

How the single-label argmax becomes multi-label
------------------------------------------------
- Each base ckpt's per-clip softmax is log-mean aggregated per source → a
  per-class probability vector. The 5 base vectors are averaged → ``base``.
- The voucher (complete-044) is aggregated the same way → ``voucher``.
- Per-class thresholds ``thr`` are tuned on **val** (one threshold per class,
  maximizing that class's F1). The non-Cargo bits fire when
  ``base[c] >= thr[c]``.
- The Cargo bit keeps the voucher "confirm" mechanism: it fires when the base
  clears its own Cargo threshold OR (voucher Cargo prob >= τ AND Cargo is in the
  base top-2). τ is tuned on val. This recovers the 3 Cargo→Tanker sources the
  base pool misses without paying the voucher's false positives.
- ``force_one`` guarantees a non-empty row (degrades to one-hot argmax when no
  class clears threshold), so single-label sources stay valid one-hot.

Honest contract: thresholds + τ chosen on val only; test scored once.
Outputs (under ``--out``): ``test_onehot.npy``, ``test_y_multihot.npy``,
``thresholds.json``, ``metrics.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mlmc_common import (
    list_files, groupby_source, tune_thresholds_per_class,
    probs_to_onehot, multilabel_report,
)

N_VAL = 50752
N_TST = 14208


def src_scores(P, groups, eps=1e-8):
    """exp(log-mean over a source's clips) → per-class prob per source."""
    out = np.zeros((len(groups), P.shape[-1]), dtype=np.float32)
    for gi, (idx, _) in enumerate(groups):
        out[gi] = np.exp(np.log(P[idx] + eps).mean(0))
    return out


def load_pool():
    pool = {}
    for ck in json.load(open("campaign/probs_classifier_dataset/_meta.json"))["ckpts"]:
        z = np.load(ck["npz"])
        pool[ck["stem"]] = (z["val_probs"][:N_VAL], z["test_probs"][:N_TST])
    for ck in json.load(open("campaign/probs_classifier_dataset_v2/_meta.json"))["ckpts"]:
        z = np.load(ck["npz"])
        pool[ck["stem"]] = (z["val_probs"][:N_VAL], z["test_probs"][:N_TST])
    for fn in sorted(Path("campaign/probs_classifier_dataset_v3").glob("*.npz")):
        z = np.load(fn)
        pool[fn.stem] = (z["val_probs"][:N_VAL], z["test_probs"][:N_TST])
    z = np.load("campaign/probs_classifier_dataset_precise/complete-044-aligned.npz")
    pool["complete-044-p0.7159"] = (z["val_probs"], z["test_probs"])
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_label", action="store_true",
                    help="strict one-hot for single-label data: base = argmax, "
                         "voucher REPLACES the row with Cargo (instead of "
                         "OR-adding the Cargo bit). Exactly one bit per row.")
    ap.add_argument("--out", default="lightning_logs/mlmc/cargo_confirm_5base")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    meta = json.load(open("campaign/probs_classifier_dataset/_meta.json"))
    data_dir = Path(meta["data_dir"])
    classes = meta["classes"]
    CARGO = classes.index("Cargo")
    nC = len(classes)

    pool = load_pool()
    base_stems = ["hydra-026-p0.6908", "hydra-032-p0.7010", "hydra-031-p0.7042",
                  "hydra-069-p0.7279", "hydra-042-p0.7114"]
    voucher_stem = "complete-044-p0.7159"

    GV = groupby_source(list_files(data_dir, "val", classes), N_VAL, nC)
    GT = groupby_source(list_files(data_dir, "test", classes), N_TST, nC)
    val_y = np.stack([g[1] for g in GV])
    test_y = np.stack([g[1] for g in GT])

    val_base = np.mean([src_scores(pool[s][0], GV) for s in base_stems], 0)
    test_base = np.mean([src_scores(pool[s][1], GT) for s in base_stems], 0)
    val_vouch = src_scores(pool[voucher_stem][0], GV)
    test_vouch = src_scores(pool[voucher_stem][1], GT)

    # Per-class thresholds tuned on val base scores.
    thr = tune_thresholds_per_class(val_base, val_y)

    # τ for the Cargo voucher-confirm, tuned on val for Cargo-class F1.
    def confirm_onehot(base, vouch, tau):
        oh = probs_to_onehot(base, thr, force_one=True,
                             single_label=args.single_label)
        top2 = np.argsort(-base, axis=1)[:, :2]
        cargo_top2 = (top2[:, 0] == CARGO) | (top2[:, 1] == CARGO)
        voucher_fire = (vouch[:, CARGO] >= tau) & cargo_top2
        if args.single_label:
            # strict one-hot: voucher REPLACES the predicted row with Cargo,
            # matching the original argmax `np.where(mask, CARGO, base_pred)`.
            oh[voucher_fire] = 0
            oh[voucher_fire, CARGO] = 1
        else:
            oh[:, CARGO] = np.maximum(oh[:, CARGO], voucher_fire.astype(np.int64))
        return oh

    best_tau, best_f1 = 0.5, -1.0
    for tau in np.linspace(0.20, 0.95, 16):
        pred = confirm_onehot(val_base, val_vouch, tau)
        # F1 for the Cargo class specifically (the voucher's target)
        tp = int((pred[:, CARGO] & val_y[:, CARGO].astype(int)).sum())
        fp = int((pred[:, CARGO] & (1 - val_y[:, CARGO].astype(int))).sum())
        fn = int(((1 - pred[:, CARGO]) & val_y[:, CARGO].astype(int)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)

    print(f"[val] per-class thr={dict(zip(classes, thr.round(3).tolist()))} "
          f"cargo-confirm τ={best_tau:.3f} (val Cargo-F1={best_f1:.4f})")
    multilabel_report("VAL", val_y, confirm_onehot(val_base, val_vouch, best_tau), classes)

    pred_test = confirm_onehot(test_base, test_vouch, best_tau)
    print("\n--- TEST (touched once) ---")
    m = multilabel_report("TEST", test_y, pred_test, classes)

    np.save(out / "test_onehot.npy", pred_test)
    np.save(out / "test_y_multihot.npy", test_y)
    (out / "thresholds.json").write_text(json.dumps(
        {"per_class": dict(zip(classes, thr.tolist())), "cargo_tau": best_tau}, indent=2))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
