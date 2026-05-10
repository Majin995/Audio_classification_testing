"""Phase N2: bootstrap MLP(64) stacker since it appears to beat LR.

MLPs are sensitive to init/seed; we want to know if MLP(64)'s 0.879 F1 is
typical or just lucky.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from campaign.lib import (
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
    n = len(val_y)
    rng = np.random.default_rng(42)

    B = 100  # MLP fits slower than LR
    results = {"LR(C=0.1)": [], "MLP(64)": [], "MLP(64,seed=...)": []}

    # 1) Bootstrap each classifier across val resamples
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        Xb, yb = Xv[idx], val_y[idx]
        if len(np.unique(yb)) < nc:
            continue
        # LR
        clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(Xb, yb)
        results["LR(C=0.1)"].append(metrics(clf.predict_proba(Xf), full_y, nc)["f1"])
        # MLP with fixed seed
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, alpha=1e-2,
                             random_state=42).fit(Xb, yb)
        results["MLP(64)"].append(metrics(clf.predict_proba(Xf), full_y, nc)["f1"])

    # 2) Also bootstrap over MLP random seeds (same val)
    for seed in range(50):
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, alpha=1e-2,
                             random_state=seed).fit(Xv, val_y)
        results["MLP(64,seed=...)"].append(metrics(clf.predict_proba(Xf), full_y, nc)["f1"])

    out = {}
    lines = ["# N2. Bootstrap robustness: LR vs MLP(64)", ""]
    lines.append("Two sources of variance: val resampling (rows 1-2) and MLP init seed (row 3).")
    lines.append("")
    lines.append("| classifier | n | mean F1 | std | p05 | p50 | p95 | min | max |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for k, vs in results.items():
        a = np.array(vs)
        s = {"mean": float(a.mean()), "std": float(a.std()),
             "p05": float(np.quantile(a, 0.05)), "p50": float(np.quantile(a, 0.5)),
             "p95": float(np.quantile(a, 0.95)),
             "min": float(a.min()), "max": float(a.max()), "n": len(a)}
        out[k] = s
        lines.append(f"| {k} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f}"
                     f" | {s['p05']:.4f} | {s['p50']:.4f} | {s['p95']:.4f}"
                     f" | {s['min']:.4f} | {s['max']:.4f} |")

    # Probability MLP > LR (paired)
    n_pair = min(len(results["LR(C=0.1)"]), len(results["MLP(64)"]))
    diffs = np.array(results["MLP(64)"][:n_pair]) - np.array(results["LR(C=0.1)"][:n_pair])
    lines.append("")
    lines.append(f"## Paired comparison (same val resamples)")
    lines.append(f"- MLP(64) − LR(C=0.1) F1: mean={diffs.mean():+.4f}, std={diffs.std():.4f}")
    lines.append(f"- P(MLP > LR) = {(diffs > 0).mean()*100:.1f}%")

    Path("campaign/N2_bootstrap_mlp.md").write_text("\n".join(lines))
    Path("campaign/N2_bootstrap_mlp.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
