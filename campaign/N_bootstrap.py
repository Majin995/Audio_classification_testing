"""Phase N: bootstrap robustness of the Dirichlet stacker.

For B=200 bootstrap resamples of rapid_1s val (n=64, with replacement):
  fit LR(C=0.1, log-probs), evaluate on full Split1s test.
Report mean, std, 5/95th percentile of F1 + macroP.

This tests: how lucky was the val n=64 draw? If F1 swings wildly across
bootstraps, the +12pp lift is fragile. If it's tight, the stacker is robust.
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
    Xt = np.log(test_P + eps).transpose(1, 0, 2).reshape(len(test_y), -1)
    n = len(val_y)
    rng = np.random.default_rng(42)

    B = 200
    f1s, mPs, recs, accs = [], [], [], []
    for b in range(B):
        idx = rng.integers(0, n, size=n)  # bootstrap with replacement
        Xb = Xv[idx]
        yb = val_y[idx]
        if len(np.unique(yb)) < nc:
            # Skip resamples missing a class — sklearn LR will degenerate
            continue
        clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(Xb, yb)
        f_p = clf.predict_proba(Xf)
        m = metrics(f_p, full_y, nc)
        f1s.append(m["f1"]); mPs.append(m["macro_P"]); recs.append(m["recall"]); accs.append(m["acc"])

    arr = lambda x: np.asarray(x)
    def stats(x):
        a = arr(x)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "p05": float(np.quantile(a, 0.05)),
                "p50": float(np.quantile(a, 0.50)),
                "p95": float(np.quantile(a, 0.95)),
                "min": float(a.min()), "max": float(a.max())}

    out = {"B": B, "n_used": len(f1s),
           "f1": stats(f1s), "macroP": stats(mPs),
           "recall": stats(recs), "acc": stats(accs)}

    lines = ["# N. Bootstrap robustness of Dirichlet stacker", ""]
    lines.append(f"- B = {B} bootstrap resamples of rapid_1s val (n={n}, with replacement)")
    lines.append(f"- {len(f1s)} resamples retained (others lost a class)")
    lines.append(f"- Stacker: LogisticRegression(C=0.1) on log-probs features (5x4=20 dim)")
    lines.append(f"- Eval: full Split1s test (n=7872)")
    lines.append("")
    lines.append("| metric | mean | std | p05 | p50 | p95 | min | max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k in ["f1", "macroP", "recall", "acc"]:
        s = out[k]
        lines.append(f"| {k} | {s['mean']:.4f} | {s['std']:.4f}"
                     f" | {s['p05']:.4f} | {s['p50']:.4f} | {s['p95']:.4f}"
                     f" | {s['min']:.4f} | {s['max']:.4f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append(f"- Original (no bootstrap): full F1 = 0.8660, macroP = 0.8660")
    lines.append(f"- Bootstrap mean F1 = {out['f1']['mean']:.4f} ± {out['f1']['std']:.4f}")
    lines.append(f"- Bootstrap mean macroP = {out['macroP']['mean']:.4f} ± {out['macroP']['std']:.4f}")
    lines.append(f"- Probability of beating N=5 arith baseline (full F1=0.7432): "
                 f"{(arr(f1s) > 0.7432).mean()*100:.1f}%")
    lines.append(f"- Probability of beating geom-2+thr (full F1=0.8098): "
                 f"{(arr(f1s) > 0.8098).mean()*100:.1f}%")

    Path("campaign/N_bootstrap.md").write_text("\n".join(lines))
    Path("campaign/N_bootstrap.json").write_text(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
