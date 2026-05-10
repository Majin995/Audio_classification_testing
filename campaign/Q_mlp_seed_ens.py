"""Phase Q: ensemble of MLP stackers across seeds.

Single MLP(64) has high seed variance. Average predictions across K
seed-different MLPs and compare to LR baseline.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
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

    rows = []
    lines = ["# Q. MLP ensemble over seeds", ""]
    lines.append("Average predict_proba over K MLP(64) instances with different random_state.")
    lines.append("")
    lines.append("| K | hidden | full F1 | full macroP | full recall | rapid_test F1 |")
    lines.append("|---|---|---|---|---|---|")

    best = None
    best_f1 = -1.0
    for hidden in [(32,), (64,), (32, 16), (64, 32)]:
        for K in [1, 5, 10, 20, 50]:
            t_acc = None
            f_acc = None
            for s in range(K):
                clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=2000,
                                     alpha=1e-2, random_state=s).fit(Xv, val_y)
                tp = clf.predict_proba(Xt)
                fp = clf.predict_proba(Xf)
                t_acc = tp if t_acc is None else t_acc + tp
                f_acc = fp if f_acc is None else f_acc + fp
            t_acc /= K
            f_acc /= K
            t_m = metrics(t_acc, test_y, nc)
            f_m = metrics(f_acc, full_y, nc)
            rows.append({"K": K, "hidden": str(hidden),
                          "rapid_test": t_m, "full_test": f_m})
            lines.append(f"| {K} | {hidden} | {f_m['f1']:.4f} | {f_m['macro_P']:.4f}"
                         f" | {f_m['recall']:.4f} | {t_m['f1']:.4f} |")
            if f_m["f1"] > best_f1:
                best_f1 = f_m["f1"]
                best = rows[-1]

    # Stacker of stackers — combine LR + MLP(64,K=20) predictions geometrically
    lr = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(Xv, val_y)
    lr_t = lr.predict_proba(Xt)
    lr_f = lr.predict_proba(Xf)

    K = 20
    mlp_t = mlp_f = None
    for s in range(K):
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                             alpha=1e-2, random_state=s).fit(Xv, val_y)
        tp = clf.predict_proba(Xt)
        fp = clf.predict_proba(Xf)
        mlp_t = tp if mlp_t is None else mlp_t + tp
        mlp_f = fp if mlp_f is None else mlp_f + fp
    mlp_t /= K
    mlp_f /= K

    # Geom mean of LR + MLP-ensemble
    eps_p = 1e-8
    geom_t = np.exp(0.5 * (np.log(lr_t + eps_p) + np.log(mlp_t + eps_p)))
    geom_t /= geom_t.sum(axis=1, keepdims=True)
    geom_f = np.exp(0.5 * (np.log(lr_f + eps_p) + np.log(mlp_f + eps_p)))
    geom_f /= geom_f.sum(axis=1, keepdims=True)
    arith_t = (lr_t + mlp_t) / 2
    arith_f = (lr_f + mlp_f) / 2

    lines.append("")
    lines.append("## Stacker of stackers")
    for label, t_arr, f_arr in [
        ("LR alone", lr_t, lr_f),
        ("MLP(64) K=20", mlp_t, mlp_f),
        ("(LR + MLP)/2 arith", arith_t, arith_f),
        ("geom(LR, MLP)", geom_t, geom_f),
    ]:
        t_m = metrics(t_arr, test_y, nc)
        f_m = metrics(f_arr, full_y, nc)
        lines.append(f"- **{label}**: full F1={f_m['f1']:.4f}, macroP={f_m['macro_P']:.4f}, recall={f_m['recall']:.4f}")
        if f_m["f1"] > best_f1:
            best_f1 = f_m["f1"]
            best = {"name": label, "rapid_test": t_m, "full_test": f_m}

    lines.append("")
    lines.append("## Best detail")
    lines.append("```")
    lines.append(f"name = {best.get('name', best.get('K'))}")
    lines.append("full_test:")
    lines.append(fmt_metrics(best["full_test"], nc, "  "))
    lines.append("```")

    Path("campaign/Q_mlp_seed_ens.md").write_text("\n".join(lines))
    Path("campaign/Q_mlp_seed_ens.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
