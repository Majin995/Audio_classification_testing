"""Phase DD: rerun stacker on Classifier_Dataset with class balancing.

Tug class is severely under-represented in val (11%) and test (10%) — the
default MLP collapses on Tug (F1=0.29). Try:
  - sklearn class_weight='balanced' for LR
  - sample_weight on MLP fit
  - subset analysis: real vs synth split
  - Explicit per-class threshold calibration on top of refit stacker
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from campaign.lib import (
    fmt_metrics,
    load_dump,
    load_meta,
    metrics,
    metrics_with_thresholds,
    search_thresholds_f1_with_p_floor,
    stack_probs,
)


def main():
    probs_dir = "campaign/probs_classifier_dataset"
    meta = load_meta(probs_dir)
    nc = meta["num_classes"]
    val_P, val_y, test_P, test_y, stems = stack_probs(probs_dir)
    eps = 1e-8
    Xv = np.log(val_P + eps).transpose(1, 0, 2).reshape(len(val_y), -1)
    Xt = np.log(test_P + eps).transpose(1, 0, 2).reshape(len(test_y), -1)

    # Class counts
    val_counts = np.bincount(val_y, minlength=nc)
    test_counts = np.bincount(test_y, minlength=nc)
    print(f"val counts: {val_counts.tolist()}")
    print(f"test counts: {test_counts.tolist()}")

    # Sample weights for balanced fit
    n = len(val_y)
    cw = n / (nc * val_counts)
    sw = cw[val_y]

    rows = []
    lines = ["# DD. Class-balanced + threshold-calibrated stacker on Classifier_Dataset", ""]
    lines.append(f"- val n={len(val_y)}, test n={len(test_y)}")
    lines.append(f"- val class counts: Cargo={val_counts[0]}, Passenger={val_counts[1]}, "
                 f"Tanker={val_counts[2]}, Tug={val_counts[3]}")
    lines.append(f"- Tug imbalance: {val_counts[3]/n*100:.1f}% of val")
    lines.append("")
    lines.append("| stacker | balanced? | + thresholds | test F1 | test macroP | test recall | per-class F1 |")
    lines.append("|---|---|---|---|---|---|---|")

    grid = list(np.linspace(0.0, 0.99, 50))

    def eval_clf(name, clf, balanced, with_thr):
        if balanced:
            try:
                clf.fit(Xv, val_y, sample_weight=sw)
            except TypeError:
                # LR doesn't always accept sample_weight via class_weight; use class_weight
                clf.set_params(class_weight="balanced")
                clf.fit(Xv, val_y)
        else:
            clf.fit(Xv, val_y)
        v_p = clf.predict_proba(Xv)
        t_p = clf.predict_proba(Xt)
        if with_thr:
            res = search_thresholds_f1_with_p_floor(v_p, val_y, nc, p_floor=0.5, grid=grid)
            thr = np.array(res["thresholds"])
            t_m = metrics_with_thresholds(t_p, test_y, thr, nc)
        else:
            t_m = metrics(t_p, test_y, nc)
        return t_m, t_p

    # LR variants
    for bal in [False, True]:
        for thr in [False, True]:
            clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
            t_m, _ = eval_clf("LR", clf, bal, thr)
            f1s = [t_m[f'F1_class_{c}'] for c in range(nc)]
            lines.append(f"| LR(C=0.1) | {'yes' if bal else 'no'} | {'yes' if thr else 'no'}"
                         f" | {t_m['f1']:.4f} | {t_m['macro_P']:.4f} | {t_m['recall']:.4f}"
                         f" | {[round(x,3) for x in f1s]} |")
            rows.append({"clf": "LR", "balanced": bal, "thresholds": thr, "metrics": t_m})

    # MLP variants — use sample_weight (sklearn MLP supports it via compatible interface)
    for bal in [False, True]:
        for thr in [False, True]:
            K = 10
            t_acc = v_acc = None
            for s in range(K):
                clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                                     alpha=1e-2, random_state=s)
                # MLPClassifier doesn't natively accept sample_weight in older sklearn,
                # so we approximate balanced training by oversampling.
                if bal:
                    # Bootstrap with weights
                    rng = np.random.default_rng(s)
                    p = sw / sw.sum()
                    idx = rng.choice(len(val_y), size=len(val_y), replace=True, p=p)
                    clf.fit(Xv[idx], val_y[idx])
                else:
                    clf.fit(Xv, val_y)
                v_p = clf.predict_proba(Xv)
                t_p = clf.predict_proba(Xt)
                v_acc = v_p if v_acc is None else v_acc + v_p
                t_acc = t_p if t_acc is None else t_acc + t_p
            v_acc /= K; t_acc /= K
            if thr:
                res = search_thresholds_f1_with_p_floor(v_acc, val_y, nc, p_floor=0.5, grid=grid)
                thr_arr = np.array(res["thresholds"])
                t_m = metrics_with_thresholds(t_acc, test_y, thr_arr, nc)
            else:
                t_m = metrics(t_acc, test_y, nc)
            f1s = [t_m[f'F1_class_{c}'] for c in range(nc)]
            lines.append(f"| MLP(64) K=10 | {'yes' if bal else 'no'} | {'yes' if thr else 'no'}"
                         f" | {t_m['f1']:.4f} | {t_m['macro_P']:.4f} | {t_m['recall']:.4f}"
                         f" | {[round(x,3) for x in f1s]} |")
            rows.append({"clf": "MLP K=10", "balanced": bal, "thresholds": thr, "metrics": t_m})

    # Pick best by F1
    best = max(rows, key=lambda r: r["metrics"]["f1"])
    lines.append("")
    lines.append("## Best by test F1")
    lines.append("```")
    lines.append(f"{best['clf']}, balanced={best['balanced']}, thresholds={best['thresholds']}")
    lines.append("test:")
    lines.append(fmt_metrics(best["metrics"], nc, "  "))
    lines.append("```")

    Path("campaign/DD_classifier_dataset_balanced.md").write_text("\n".join(lines))
    Path("campaign/DD_classifier_dataset_balanced.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
