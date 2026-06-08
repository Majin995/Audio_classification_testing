"""② zero-fit log-mean ensemble — MLMC (multi-label) variant.

Pre-committed aggregation, no learned stacker: for each ckpt, log-mean the
per-clip softmax over a source's clips; arithmetic-mean across the top-5 ckpts
(ranked by val OOF macro-P, exactly as ``zero_fit_ensemble.py``). The only
change is the decision rule: per-class thresholds (tuned on val) → **multi-hot**
output, instead of a single argmax label.

Honest contract: ckpt subset fixed by val_p ranking; thresholds tuned on val;
test scored once. Outputs mirror the other MLMC scripts.
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

# val OOF macro-P per ckpt — selection signal (no test peeking).
VAL_P = {
    "hydra-010-p0.6943": 0.6943, "hydra-012-p0.6929": 0.6929,
    "hydra-026-p0.6908": 0.6908, "hydra-031-p0.7042": 0.7042,
    "hydra-032-p0.7010": 0.7010, "hydra-042-p0.7114": 0.7114,
    "hydra-069-p0.7279": 0.7279,
}


def src_scores(P, groups, eps=1e-8):
    out = np.zeros((len(groups), P.shape[-1]), dtype=np.float32)
    for gi, (idx, _) in enumerate(groups):
        out[gi] = np.exp(np.log(P[idx] + eps).mean(0))
    return out


def load_pool():
    pool = {}
    for ck in json.load(open("campaign/probs_classifier_dataset/_meta.json"))["ckpts"]:
        z = np.load(ck["npz"])
        pool[ck["stem"]] = (z["val_probs"][:N_VAL], z["test_probs"][:N_TST])
    p2 = Path("campaign/probs_classifier_dataset_v2/_meta.json")
    if p2.exists():
        for ck in json.load(open(p2))["ckpts"]:
            z = np.load(ck["npz"])
            pool[ck["stem"]] = (z["val_probs"][:N_VAL], z["test_probs"][:N_TST])
    p3 = Path("campaign/probs_classifier_dataset_v3")
    if p3.exists():
        for fn in sorted(p3.glob("*.npz")):
            z = np.load(fn)
            pool[fn.stem] = (z["val_probs"][:N_VAL], z["test_probs"][:N_TST])
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5, help="#ckpts by val_p rank")
    ap.add_argument("--single_label", action="store_true",
                    help="strict argmax one-hot (exactly one bit/row) for "
                         "single-label datasets; ignores per-class thresholds.")
    ap.add_argument("--out", default="lightning_logs/mlmc/zero_fit_ensemble")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    meta = json.load(open("campaign/probs_classifier_dataset/_meta.json"))
    data_dir = Path(meta["data_dir"])
    classes = meta["classes"]
    nC = len(classes)

    pool = load_pool()
    ranked = sorted([s for s in pool if s in VAL_P], key=lambda s: -VAL_P[s])
    subset = ranked[:args.topk]
    print(f"ranked by val_p: {ranked}\nusing top{args.topk}: {subset}")

    GV = groupby_source(list_files(data_dir, "val", classes), N_VAL, nC)
    GT = groupby_source(list_files(data_dir, "test", classes), N_TST, nC)
    val_y = np.stack([g[1] for g in GV])
    test_y = np.stack([g[1] for g in GT])

    val_scores = np.mean([src_scores(pool[s][0], GV) for s in subset], 0)
    test_scores = np.mean([src_scores(pool[s][1], GT) for s in subset], 0)

    thr = tune_thresholds_per_class(val_scores, val_y)
    mode = "argmax one-hot (single-label)" if args.single_label else "per-class threshold (multi-label)"
    print(f"[val] decision mode: {mode}; thr={dict(zip(classes, thr.round(3).tolist()))}")
    multilabel_report("VAL", val_y,
                      probs_to_onehot(val_scores, thr, single_label=args.single_label),
                      classes)

    pred_test = probs_to_onehot(test_scores, thr, single_label=args.single_label)
    print("\n--- TEST (touched once) ---")
    m = multilabel_report("TEST", test_y, pred_test, classes)

    np.save(out / "test_onehot.npy", pred_test)
    np.save(out / "test_y_multihot.npy", test_y)
    (out / "thresholds.json").write_text(json.dumps(
        {"per_class": dict(zip(classes, thr.tolist())), "subset": subset}, indent=2))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
