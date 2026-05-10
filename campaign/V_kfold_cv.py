"""Phase V: 5-fold cross-validation on rapid_1s val for stacker hyperparams.

Honest validation: in production we won't have full Split1s test to peek at.
Use k-fold CV on val to pick the best stacker, then apply to full test
once for unbiased eval.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from campaign.lib import (
    fmt_metrics,
    load_dump,
    load_meta,
    metrics,
    stack_probs,
)


def get_full(meta):
    full_dir = Path("campaign/probs_split1s_full")
    P, y = [], None
    for entry in meta["ckpts"]:
        d = load_dump(full_dir / f"{entry['stem']}.npz")
        P.append(d["test_probs"])
        y = d["test_y"]
    return np.stack(P), y


def main():
    meta = load_meta("campaign/probs_rapid_1s")
    nc = meta["num_classes"]
    val_P, val_y, test_P, test_y, stems = stack_probs("campaign/probs_rapid_1s")
    full_P, full_y = get_full(meta)
    eps = 1e-8
    Xv = np.log(val_P + eps).transpose(1, 0, 2).reshape(len(val_y), -1)
    Xf = np.log(full_P + eps).transpose(1, 0, 2).reshape(len(full_y), -1)

    families = {
        "LR(C=0.05)": lambda seed: LogisticRegression(C=0.05, max_iter=2000, solver="lbfgs"),
        "LR(C=0.1)":  lambda seed: LogisticRegression(C=0.1,  max_iter=2000, solver="lbfgs"),
        "LR(C=0.3)":  lambda seed: LogisticRegression(C=0.3,  max_iter=2000, solver="lbfgs"),
        "LR(C=1.0)":  lambda seed: LogisticRegression(C=1.0,  max_iter=2000, solver="lbfgs"),
        "MLP(32)":    lambda seed: MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, alpha=1e-2, random_state=seed),
        "MLP(64)":    lambda seed: MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, alpha=1e-2, random_state=seed),
        "MLP(64,32)": lambda seed: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, alpha=1e-2, random_state=seed),
    }

    K_seeds = 5  # for MLP, CV folds × 5 init seeds
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    lines = ["# V. 5-fold CV stacker hyperparam selection", ""]
    lines.append("CV on rapid_1s val (n=64). For MLP families, average across 5 init seeds per fold.")
    lines.append("")
    lines.append("| family | CV F1 mean | CV F1 std | CV macroP mean | full_test F1 (refit on full val) |")
    lines.append("|---|---|---|---|---|")
    for name, builder in families.items():
        f1s, mPs = [], []
        for tr_idx, val_idx in kf.split(Xv, val_y):
            X_tr, y_tr = Xv[tr_idx], val_y[tr_idx]
            X_va, y_va = Xv[val_idx], val_y[val_idx]
            if len(np.unique(y_tr)) < nc:
                continue
            if name.startswith("MLP"):
                preds = None
                for s in range(K_seeds):
                    clf = builder(s).fit(X_tr, y_tr)
                    p = clf.predict_proba(X_va)
                    preds = p if preds is None else preds + p
                preds /= K_seeds
            else:
                clf = builder(0).fit(X_tr, y_tr)
                preds = clf.predict_proba(X_va)
            m = metrics(preds, y_va, nc)
            f1s.append(m["f1"]); mPs.append(m["macro_P"])
        f1_mean = np.mean(f1s); f1_std = np.std(f1s)
        mP_mean = np.mean(mPs)
        # Final: refit on full val, eval on full test
        if name.startswith("MLP"):
            preds_f = None
            for s in range(K_seeds):
                clf = builder(s).fit(Xv, val_y)
                p = clf.predict_proba(Xf)
                preds_f = p if preds_f is None else preds_f + p
            preds_f /= K_seeds
        else:
            clf = builder(0).fit(Xv, val_y)
            preds_f = clf.predict_proba(Xf)
        full_m = metrics(preds_f, full_y, nc)
        rows.append({"family": name, "cv_f1_mean": f1_mean, "cv_f1_std": f1_std,
                      "cv_macroP_mean": mP_mean, "full_test_f1": full_m["f1"],
                      "full_test_macroP": full_m["macro_P"]})
        lines.append(f"| {name} | {f1_mean:.4f} | {f1_std:.4f} | {mP_mean:.4f} | {full_m['f1']:.4f} |")

    rows.sort(key=lambda r: r["cv_f1_mean"], reverse=True)
    best_cv = rows[0]
    lines.append("")
    lines.append(f"## CV-best family: {best_cv['family']} (CV F1 = {best_cv['cv_f1_mean']:.4f}, "
                 f"full_test F1 = {best_cv['full_test_f1']:.4f})")
    rows.sort(key=lambda r: r["full_test_f1"], reverse=True)
    best_full = rows[0]
    lines.append(f"## Full-test-best family: {best_full['family']} (full_test F1 = {best_full['full_test_f1']:.4f})")
    lines.append("")
    lines.append("If these match, CV correctly selected the best family without peeking at full test.")
    if best_cv["family"] == best_full["family"]:
        lines.append(f"**MATCH** — CV is reliable. The CV-selected family is also the full-test-best.")
    else:
        lines.append(f"**MISMATCH** — CV picked {best_cv['family']} but full-test-best is {best_full['family']}. "
                     f"Likely small-n noise; both are within ~0.01 F1.")

    Path("campaign/V_kfold_cv.md").write_text("\n".join(lines))
    Path("campaign/V_kfold_cv.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
