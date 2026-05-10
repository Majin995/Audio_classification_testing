"""Phase T: confusion matrix delta. Where does the stacker actually win?

Compare: N=5 arith (production baseline) vs MLP(64) K=10 stacker.
For each (true_class, predicted_class) cell, report:
  - baseline_count
  - stacker_count
  - delta (stacker - baseline)

Negative delta on off-diagonal cells = stacker fixes those mistakes.
Positive delta on diagonal = stacker recovers those samples.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
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
    classes = meta["classes"]
    nc = meta["num_classes"]
    val_P, val_y, test_P, test_y, stems = stack_probs("campaign/probs_rapid_1s")
    full_P, full_y = get_full(meta)
    eps = 1e-8

    # Baseline: N=5 arith
    base_f = full_P.mean(axis=0)
    base_pred = base_f.argmax(axis=1)
    base_cm = confusion_matrix(full_y, base_pred, labels=list(range(nc)))

    # Stacker: MLP(64) K=10 ensemble
    Xv = np.log(val_P + eps).transpose(1, 0, 2).reshape(len(val_y), -1)
    Xf = np.log(full_P + eps).transpose(1, 0, 2).reshape(len(full_y), -1)
    K = 10
    s_f = None
    for s in range(K):
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                             alpha=1e-2, random_state=s).fit(Xv, val_y)
        fp = clf.predict_proba(Xf)
        s_f = fp if s_f is None else s_f + fp
    s_f /= K
    s_pred = s_f.argmax(axis=1)
    s_cm = confusion_matrix(full_y, s_pred, labels=list(range(nc)))

    # Delta
    delta = s_cm - base_cm

    # Sample-level: which baseline-wrong samples did stacker fix?
    base_wrong = base_pred != full_y
    s_correct = s_pred == full_y
    fixed = base_wrong & s_correct  # baseline wrong, stacker right
    broken = (~base_wrong) & (~s_correct)  # baseline right, stacker wrong

    fixed_by_class = {}
    broken_by_class = {}
    for c in range(nc):
        fixed_by_class[classes[c]] = int((fixed & (full_y == c)).sum())
        broken_by_class[classes[c]] = int((broken & (full_y == c)).sum())

    lines = ["# T. Confusion-matrix delta — Stacker vs N=5 baseline", ""]
    lines.append(f"Eval: full Split1s test (n={len(full_y)})")
    lines.append("")
    lines.append("## Baseline N=5 arith confusion matrix")
    lines.append("(rows = true, cols = predicted)")
    lines.append("```")
    lines.append("       " + " ".join(f"{c:>10}" for c in classes))
    for i, row in enumerate(base_cm):
        lines.append(f"{classes[i]:>6}: " + " ".join(f"{v:>10}" for v in row))
    lines.append("```")
    lines.append("")
    lines.append("## Stacker (MLP(64) K=10) confusion matrix")
    lines.append("```")
    lines.append("       " + " ".join(f"{c:>10}" for c in classes))
    for i, row in enumerate(s_cm):
        lines.append(f"{classes[i]:>6}: " + " ".join(f"{v:>10}" for v in row))
    lines.append("```")
    lines.append("")
    lines.append("## Δ = Stacker - Baseline")
    lines.append("Negative on off-diagonal = stacker FIXED mistakes; positive on diagonal = stacker RECOVERED samples.")
    lines.append("```")
    lines.append("       " + " ".join(f"{c:>10}" for c in classes))
    for i, row in enumerate(delta):
        lines.append(f"{classes[i]:>6}: " + " ".join(f"{v:>+10}" for v in row))
    lines.append("```")
    lines.append("")
    lines.append("## Sample-level deltas")
    lines.append(f"- Total samples baseline got wrong but stacker got right: {int(fixed.sum())}")
    lines.append(f"- Total samples baseline got right but stacker got wrong: {int(broken.sum())}")
    lines.append(f"- **Net improvement**: {int(fixed.sum() - broken.sum())} samples")
    lines.append("")
    lines.append("Per-class breakdown of fixed/broken (by TRUE class):")
    lines.append("")
    lines.append("| class | fixed (baseline wrong → stacker right) | broken (baseline right → stacker wrong) | net |")
    lines.append("|---|---|---|---|")
    for c in classes:
        f, b = fixed_by_class[c], broken_by_class[c]
        lines.append(f"| {c} | {f} | {b} | {f-b:+d} |")
    lines.append("")
    lines.append("## Where the stacker eats baseline mistakes")
    # For each off-diagonal baseline cell, report the share that stacker fixed
    lines.append("")
    lines.append("| true | base_pred | base_count | stacker also wrong here | fixed |")
    lines.append("|---|---|---|---|---|")
    for i in range(nc):
        for j in range(nc):
            if i == j:
                continue
            mask = (full_y == i) & (base_pred == j)
            if mask.sum() == 0:
                continue
            still_wrong = mask & (s_pred != full_y)
            still_wrong_count = int(still_wrong.sum())
            fixed_count = int(mask.sum() - still_wrong_count)
            lines.append(f"| {classes[i]} | {classes[j]} | {int(mask.sum())} "
                         f"| {still_wrong_count} | {fixed_count} |")

    Path("campaign/T_confusion_delta.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
