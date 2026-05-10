"""Phase Y: stacker with abstain-logit features.

Hydra outputs 5 logits: 4 class + 1 abstain. The standard 4-D stacker
discards abstain. This phase uses all 5 → 25-D features (5 ckpts × 5).
Tests whether abstain logits carry useful uncertainty signal.
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
)


def stack_5d(probs_dir):
    meta = load_meta(probs_dir)
    val_P, test_P = [], []
    val_y = test_y = None
    for entry in meta["ckpts"]:
        d = load_dump(entry["npz"])
        if "val_probs" in d:
            val_P.append(d["val_probs"])
            val_y = d["val_y"]
        test_P.append(d["test_probs"])
        test_y = d["test_y"]
    return (np.stack(val_P) if val_P else None, val_y,
            np.stack(test_P), test_y, meta)


def main():
    val_P5, val_y, test_P5, test_y, meta_r = stack_5d("campaign/probs_rapid_full")
    _, _, full_P5, full_y, _ = stack_5d("campaign/probs_split1s_full_5d")
    nc = 4  # number of true classes
    eps = 1e-8

    # 5-D features (per ckpt): include abstain
    Xv_5 = np.log(val_P5 + eps).transpose(1, 0, 2).reshape(len(val_y), -1)
    Xt_5 = np.log(test_P5 + eps).transpose(1, 0, 2).reshape(len(test_y), -1)
    Xf_5 = np.log(full_P5 + eps).transpose(1, 0, 2).reshape(len(full_y), -1)

    # 4-D features (baseline): renormalized 4-class probs, abstain dropped
    val_P4 = val_P5[..., :nc]
    val_P4 = val_P4 / val_P4.sum(axis=-1, keepdims=True)
    test_P4 = test_P5[..., :nc]
    test_P4 = test_P4 / test_P4.sum(axis=-1, keepdims=True)
    full_P4 = full_P5[..., :nc]
    full_P4 = full_P4 / full_P4.sum(axis=-1, keepdims=True)
    Xv_4 = np.log(val_P4 + eps).transpose(1, 0, 2).reshape(len(val_y), -1)
    Xt_4 = np.log(test_P4 + eps).transpose(1, 0, 2).reshape(len(test_y), -1)
    Xf_4 = np.log(full_P4 + eps).transpose(1, 0, 2).reshape(len(full_y), -1)

    print(f"4-D features: {Xv_4.shape[1]} dims; 5-D features: {Xv_5.shape[1]} dims")

    rows = []
    lines = ["# Y. Abstain-logit feature inclusion", ""]
    lines.append("Hydra outputs 5 logits (4 class + 1 abstain). Compare:")
    lines.append("- 4-D: re-normalized 4-class softmax (abstain dropped, then re-softmax)")
    lines.append("- 5-D: full 5-class softmax (abstain kept as feature)")
    lines.append("")
    lines.append("| classifier | 4-D full F1 | 4-D full macroP | 5-D full F1 | 5-D full macroP | Δ F1 |")
    lines.append("|---|---|---|---|---|---|")

    families = {
        "LR(C=0.1)": lambda s: LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"),
        "LR(C=0.3)": lambda s: LogisticRegression(C=0.3, max_iter=2000, solver="lbfgs"),
        "LR(C=1.0)": lambda s: LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
    }
    K = 10
    mlp_factory = lambda s: MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, alpha=1e-2, random_state=s)

    for name, builder in families.items():
        clf = builder(0).fit(Xv_4, val_y)
        m4 = metrics(clf.predict_proba(Xf_4), full_y, nc)
        clf = builder(0).fit(Xv_5, val_y)
        m5 = metrics(clf.predict_proba(Xf_5), full_y, nc)
        d = m5["f1"] - m4["f1"]
        lines.append(f"| {name} | {m4['f1']:.4f} | {m4['macro_P']:.4f}"
                     f" | {m5['f1']:.4f} | {m5['macro_P']:.4f} | {d:+.4f} |")
        rows.append({"clf": name, "m4": m4, "m5": m5, "delta": d})

    # MLP K=10
    p4 = p5 = None
    for s in range(K):
        c4 = mlp_factory(s).fit(Xv_4, val_y)
        c5 = mlp_factory(s).fit(Xv_5, val_y)
        x4, x5 = c4.predict_proba(Xf_4), c5.predict_proba(Xf_5)
        p4 = x4 if p4 is None else p4 + x4
        p5 = x5 if p5 is None else p5 + x5
    p4 /= K; p5 /= K
    m4 = metrics(p4, full_y, nc); m5 = metrics(p5, full_y, nc)
    d = m5["f1"] - m4["f1"]
    lines.append(f"| MLP(64) K=10 | {m4['f1']:.4f} | {m4['macro_P']:.4f}"
                 f" | {m5['f1']:.4f} | {m5['macro_P']:.4f} | {d:+.4f} |")
    rows.append({"clf": "MLP(64) K=10", "m4": m4, "m5": m5, "delta": d})

    # Best of 5-D
    best = max(rows, key=lambda r: r["m5"]["f1"])
    lines.append("")
    lines.append(f"## Best 5-D classifier: {best['clf']} → F1 = {best['m5']['f1']:.4f}, "
                 f"macroP = {best['m5']['macro_P']:.4f}")
    lines.append("")
    lines.append("```")
    lines.append("full_test (5-D):")
    lines.append(fmt_metrics(best["m5"], nc, "  "))
    lines.append("```")
    avg_delta = np.mean([r["delta"] for r in rows])
    lines.append(f"\n**Average Δ F1 across classifiers: {avg_delta:+.4f}**")
    if avg_delta > 0.005:
        lines.append("Including the abstain logit HELPS F1 — it carries uncertainty signal the stacker can exploit.")
    elif avg_delta < -0.005:
        lines.append("Including the abstain logit HURTS F1 — it adds noise without useful signal.")
    else:
        lines.append("Abstain logit is roughly neutral — no clear signal in either direction.")

    Path("campaign/Y_abstain_features.md").write_text("\n".join(lines))
    Path("campaign/Y_abstain_features.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
