"""Phase O: alternative stacker classifiers vs LR(C=0.1).

Compare on identical 20-D log-probs features:
  - LR Dirichlet C in {0.05, 0.1, 0.3, 1.0} (baseline)
  - Sklearn MLP (1 hidden layer, sizes {8, 16, 32, 64})
  - RandomForest (n_estimators={50, 200})
  - GradientBoosting (n_estimators=100, max_depth=2)
  - XGBoost if installed
  - KNeighbors (k={3, 5, 7})
  - Calibrated LR (isotonic) on top of LR base
"""
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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
    Xt = np.log(test_P + eps).transpose(1, 0, 2).reshape(len(test_y), -1)
    Xf = np.log(full_P + eps).transpose(1, 0, 2).reshape(len(full_y), -1)

    classifiers = []
    for C in [0.05, 0.1, 0.3, 1.0]:
        classifiers.append((f"LR(C={C})", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")))
    for h in [8, 16, 32, 64]:
        classifiers.append((f"MLP({h})", MLPClassifier(hidden_layer_sizes=(h,),
                                                         max_iter=2000, random_state=42,
                                                         early_stopping=False, alpha=1e-2)))
    for h in [(16, 8), (32, 16)]:
        classifiers.append((f"MLP{h}", MLPClassifier(hidden_layer_sizes=h, max_iter=2000,
                                                       random_state=42, alpha=1e-2)))
    for n in [50, 200]:
        classifiers.append((f"RF(n={n})", RandomForestClassifier(n_estimators=n, max_depth=4,
                                                                    random_state=42)))
    classifiers.append(("GB(d=2,n=100)", GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                                                       random_state=42)))
    classifiers.append(("GB(d=3,n=200)", GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                                       random_state=42, learning_rate=0.05)))
    for k in [3, 5, 7, 11]:
        classifiers.append((f"KNN({k})", KNeighborsClassifier(n_neighbors=k)))

    try:
        from xgboost import XGBClassifier
        classifiers.append(("XGB(d=3,n=200)", XGBClassifier(n_estimators=200, max_depth=3,
                                                              learning_rate=0.05, eval_metric="mlogloss",
                                                              use_label_encoder=False)))
    except ImportError:
        pass

    rows = []
    lines = ["# O. Alternative stacker classifiers", ""]
    lines.append("- 20-D log-probs features (5 ckpts × 4 classes)")
    lines.append("- Fit on rapid_1s val (n=64), eval on full Split1s test (n=7872)")
    lines.append("")
    lines.append("| classifier | rapid_test F1 | rapid_test macroP | full_test F1 | full_test macroP | full_test recall |")
    lines.append("|---|---|---|---|---|---|")
    for name, clf in classifiers:
        try:
            clf.fit(Xv, val_y)
            t_p = clf.predict_proba(Xt)
            f_p = clf.predict_proba(Xf)
            t_m = metrics(t_p, test_y, nc)
            f_m = metrics(f_p, full_y, nc)
            rows.append({"name": name, "rapid_test": t_m, "full_test": f_m})
            lines.append(f"| {name} | {t_m['f1']:.4f} | {t_m['macro_P']:.4f}"
                         f" | {f_m['f1']:.4f} | {f_m['macro_P']:.4f}"
                         f" | {f_m['recall']:.4f} |")
        except Exception as e:
            lines.append(f"| {name} | FAILED: {e} | | | | |")

    rows.sort(key=lambda r: r["full_test"]["f1"], reverse=True)
    lines.append("")
    lines.append("## Top 5 by full_test F1")
    for r in rows[:5]:
        lines.append(f"- {r['name']}: F1={r['full_test']['f1']:.4f}, "
                     f"macroP={r['full_test']['macro_P']:.4f}, recall={r['full_test']['recall']:.4f}")
    lines.append("")
    lines.append("## Detail — best")
    best = rows[0]
    lines.append("```")
    lines.append(f"classifier = {best['name']}")
    lines.append("full_test:")
    lines.append(fmt_metrics(best["full_test"], nc, "  "))
    lines.append("```")

    Path("campaign/O_alt_classifiers.md").write_text("\n".join(lines))
    Path("campaign/O_alt_classifiers.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
