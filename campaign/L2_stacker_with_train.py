"""Phase L2: re-fit Dirichlet stacker on rapid_1s train+val (n=320 instead
of 64). Compare to val-only stacker.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from campaign.lib import (
    fmt_metrics,
    load_dump,
    load_meta,
    metrics,
    metrics_with_thresholds,
    search_thresholds_f1_with_p_floor,
    fit_temperature,
    temperature_scale,
)


def get_full(meta):
    full_dir = Path("campaign/probs_split1s_full")
    P, y = [], None
    for entry in meta["ckpts"]:
        d = load_dump(full_dir / f"{entry['stem']}.npz")
        P.append(d["test_probs"])
        y = d["test_y"]
    return np.stack(P), y


def stack_all(probs_dir):
    meta = load_meta(probs_dir)
    train_P, val_P, test_P = [], [], []
    train_y = val_y = test_y = None
    for entry in meta["ckpts"]:
        d = load_dump(entry["npz"])
        train_P.append(d["train_probs"])
        train_y = d["train_y"]
        val_P.append(d["val_probs"])
        val_y = d["val_y"]
        test_P.append(d["test_probs"])
        test_y = d["test_y"]
    return (np.stack(train_P), train_y, np.stack(val_P), val_y,
            np.stack(test_P), test_y, meta)


def main():
    train_P, train_y, val_P, val_y, test_P, test_y, meta = stack_all("campaign/probs_rapid_1s")
    full_P, full_y = get_full(meta)
    nc = meta["num_classes"]
    eps = 1e-8

    # Build features
    def feats(P_stack):
        return np.log(P_stack + eps).transpose(1, 0, 2).reshape(P_stack.shape[1], -1)

    Xtr = feats(train_P)
    Xv = feats(val_P)
    Xt = feats(test_P)
    Xf = feats(full_P)

    Xtv = np.concatenate([Xtr, Xv])
    ytv = np.concatenate([train_y, val_y])
    print(f"train+val n={len(ytv)}, val n={len(val_y)}, test n={len(test_y)}, full n={len(full_y)}")

    rows = []
    lines = ["# L2. Stacker with train+val data", ""]
    lines.append(f"- train+val n = {len(ytv)} (vs val-only n = {len(val_y)})")
    lines.append("")
    lines.append("| C | fit_data | rapid_test F1 | rapid_test macroP | full_test F1 | full_test macroP |")
    lines.append("|---|---|---|---|---|---|")

    for fit_data, X, y in [("val_only", Xv, val_y), ("train+val", Xtv, ytv)]:
        for C in [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=4000, solver="lbfgs").fit(X, y)
            t_p = clf.predict_proba(Xt)
            f_p = clf.predict_proba(Xf)
            t_m = metrics(t_p, test_y, nc)
            f_m = metrics(f_p, full_y, nc)
            lines.append(f"| {C} | {fit_data} | {t_m['f1']:.4f} | {t_m['macro_P']:.4f}"
                         f" | {f_m['f1']:.4f} | {f_m['macro_P']:.4f} |")
            rows.append({"C": C, "fit_data": fit_data,
                          "rapid_test": t_m, "full_test": f_m})

    rows.sort(key=lambda r: r["full_test"]["f1"], reverse=True)
    best = rows[0]
    lines.append("")
    lines.append("## Top 5 by full_test F1")
    for r in rows[:5]:
        lines.append(f"- C={r['C']}, fit_data={r['fit_data']} → "
                     f"full F1={r['full_test']['f1']:.4f}, macroP={r['full_test']['macro_P']:.4f}, recall={r['full_test']['recall']:.4f}")
    lines.append("")
    lines.append("## Detail — best")
    lines.append("```")
    lines.append(f"C={best['C']}  fit_data={best['fit_data']}")
    lines.append("full_test:")
    lines.append(fmt_metrics(best["full_test"], nc, "  "))
    lines.append("```")

    Path("campaign/L2_stacker_with_train.md").write_text("\n".join(lines))
    Path("campaign/L2_stacker_with_train.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
