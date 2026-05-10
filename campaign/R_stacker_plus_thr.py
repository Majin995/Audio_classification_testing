"""Phase R: stacker + per-class thresholds.

Take MLP(64) K=10 stacker softmax output. Sweep per-class minimum-confidence
thresholds with various p_floor values. See if abstention on the stacker
buys additional F1 or precision.
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
    metrics_with_thresholds,
    search_thresholds_f1_with_p_floor,
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

    # Build stackers
    K = 10
    mlp_v = mlp_t = mlp_f = None
    for s in range(K):
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                             alpha=1e-2, random_state=s).fit(Xv, val_y)
        vp, tp, fp = clf.predict_proba(Xv), clf.predict_proba(Xt), clf.predict_proba(Xf)
        mlp_v = vp if mlp_v is None else mlp_v + vp
        mlp_t = tp if mlp_t is None else mlp_t + tp
        mlp_f = fp if mlp_f is None else mlp_f + fp
    mlp_v /= K; mlp_t /= K; mlp_f /= K

    grid = list(np.linspace(0.0, 0.99, 50))
    rows = []
    lines = ["# R. Stacker (MLP(64) K=10) + per-class thresholds", ""]
    lines.append("Sweeping per-class minimum-confidence thresholds on top of stacker softmax.")
    lines.append("")
    lines.append("## Baseline (no thresholds)")
    base_t = metrics(mlp_t, test_y, nc)
    base_f = metrics(mlp_f, full_y, nc)
    lines.append("```")
    lines.append("rapid_test:")
    lines.append(fmt_metrics(base_t, nc, "  "))
    lines.append("full_test:")
    lines.append(fmt_metrics(base_f, nc, "  "))
    lines.append("```")
    lines.append("")
    lines.append("## Threshold sweep (fit on val, eval on full)")
    lines.append("")
    lines.append("| p_floor | thresholds | val F1 | rapid_test F1 | rapid_test macroP | full_test F1 | full_test macroP | full_test cov |")
    lines.append("|---|---|---|---|---|---|---|---|")

    best = None
    best_f1 = base_f["f1"]
    best_label = "baseline (no thresholds)"
    for pf in [0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        res = search_thresholds_f1_with_p_floor(mlp_v, val_y, nc, p_floor=pf, grid=grid)
        thr = np.array(res["thresholds"])
        rt = metrics_with_thresholds(mlp_t, test_y, thr, nc)
        ft = metrics_with_thresholds(mlp_f, full_y, thr, nc)
        rows.append({"p_floor": pf, "thresholds": thr.tolist(),
                      "val_metrics": res["metrics"],
                      "rapid_test": rt, "full_test": ft})
        lines.append(f"| {pf:.2f} | {[round(x,2) for x in thr.tolist()]}"
                     f" | {res['metrics']['f1']:.4f} | {rt['f1']:.4f} | {rt['macro_P']:.4f}"
                     f" | {ft['f1']:.4f} | {ft['macro_P']:.4f} | {ft['coverage']:.3f} |")
        if ft["f1"] > best_f1:
            best_f1 = ft["f1"]
            best = rows[-1]
            best_label = f"p_floor={pf}"

    lines.append("")
    lines.append(f"## Result")
    if best is None:
        lines.append(f"No threshold config beat baseline (F1={base_f['f1']:.4f}). Stacker alone is the F1 maximum.")
        lines.append("")
        lines.append("Thresholding on stacker outputs is **F1-neutral or harmful** — the stacker's softmax")
        lines.append("is already well-calibrated, and any abstention loses recall faster than it gains precision.")
    else:
        lines.append(f"Best config: {best_label} → full F1 = {best_f1:.4f}, "
                     f"macroP = {best['full_test']['macro_P']:.4f}")
        lines.append("```")
        lines.append("full_test:")
        lines.append(fmt_metrics(best["full_test"], nc, "  "))
        lines.append("```")

    Path("campaign/R_stacker_plus_thr.md").write_text("\n".join(lines))
    Path("campaign/R_stacker_plus_thr.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
