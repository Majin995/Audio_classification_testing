"""Phase H: selective-prediction frontier.

Sweep coverage in [0.50, 1.00] using max-confidence rejection (single global
threshold), measure F1 + macroP at each. Map the F1-precision frontier and
identify the operating point that maximizes F1 with macroP held >= a baseline.
"""
import json
from pathlib import Path

import numpy as np

from campaign.lib import (
    fit_temperature,
    fmt_metrics,
    load_dump,
    load_meta,
    metrics,
    metrics_with_thresholds,
    stack_probs,
    temperature_scale,
)


def get_full(meta):
    full_dir = Path("campaign/probs_split1s_full")
    test_P, test_y = [], None
    for entry in meta["ckpts"]:
        d = load_dump(full_dir / f"{entry['stem']}.npz")
        test_P.append(d["test_probs"])
        test_y = d["test_y"]
    return np.stack(test_P), test_y


def selective_metrics(probs, y, threshold, num_classes):
    """Single global confidence threshold (same for all classes)."""
    p_max = probs.max(axis=1)
    accept = p_max >= threshold
    pred = probs.argmax(axis=1)
    pred_full = np.where(accept, pred, -1)
    out = {"coverage": float(accept.mean()), "n_accepted": int(accept.sum()),
           "threshold": float(threshold)}
    if accept.sum() == 0:
        for k in ["acc", "f1", "macro_P", "micro_P", "recall"]:
            out[k] = 0.0
        return out
    pred_a = pred[accept]
    y_a = y[accept]
    from sklearn.metrics import precision_score, recall_score, f1_score
    out["acc"] = float((pred_a == y_a).mean())
    out["macro_P"] = float(precision_score(y_a, pred_a, labels=list(range(num_classes)),
                                            average="macro", zero_division=0))
    out["micro_P"] = float(precision_score(y_a, pred_a, average="micro", zero_division=0))
    rec_per = []
    f1_per = []
    for c in range(num_classes):
        m_t = y == c
        n_t = int(m_t.sum())
        rec = float((pred_full[m_t] == c).mean()) if n_t > 0 else 0.0
        out[f"R_class_{c}"] = rec
        m_p = pred_full == c
        n_p = int(m_p.sum())
        p = float((y[m_p] == c).mean()) if n_p > 0 else float("nan")
        out[f"P_class_{c}"] = p
        if not np.isnan(p) and (p + rec) > 0:
            f1 = 2 * p * rec / (p + rec)
        else:
            f1 = 0.0
        out[f"F1_class_{c}"] = f1
        rec_per.append(rec)
        f1_per.append(f1)
    out["recall"] = float(np.mean(rec_per))
    out["f1"] = float(np.mean(f1_per))
    return out


def main():
    meta = load_meta("campaign/probs_rapid_1s")
    nc = meta["num_classes"]
    val_P, val_y, test_P, test_y, stems = stack_probs("campaign/probs_rapid_1s")
    full_P, full_y = get_full(meta)

    # Use the Phase F winner: geom of best 2
    best_idx = [i for i, s in enumerate(stems) if "026" in s or "031" in s]
    eps = 1e-8
    v = np.exp(np.log(val_P[best_idx] + eps).mean(axis=0))
    v /= v.sum(axis=1, keepdims=True)
    t = np.exp(np.log(test_P[best_idx] + eps).mean(axis=0))
    t /= t.sum(axis=1, keepdims=True)
    f = np.exp(np.log(full_P[best_idx] + eps).mean(axis=0))
    f /= f.sum(axis=1, keepdims=True)
    T = fit_temperature(v, val_y)
    vT = temperature_scale(v, T)
    tT = temperature_scale(t, T)
    fT = temperature_scale(f, T)
    print(f"using geom of {[stems[i] for i in best_idx]}, T={T:.3f}")

    # Sweep many thresholds, plot coverage / F1 / macroP
    thrs = np.linspace(0.0, 0.99, 100)
    rows = []
    for thr in thrs:
        rt = selective_metrics(tT, test_y, thr, nc)
        ft = selective_metrics(fT, full_y, thr, nc)
        rows.append({"threshold": float(thr), "rapid_test": rt, "full_test": ft})

    # Tabulate at canonical coverage targets
    canonical = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    canonical_rows = []
    for cov in canonical:
        # Find threshold that gives nearest coverage on val (oracle: full)
        deltas = [abs(r["full_test"]["coverage"] - cov) for r in rows]
        i = int(np.argmin(deltas))
        canonical_rows.append((cov, rows[i]))

    lines = ["# H. Selective-prediction frontier (geom 026+031 ensemble)", ""]
    lines.append(f"- T={T:.3f}, single global confidence threshold")
    lines.append("")
    lines.append("## Frontier — full Split1s test")
    lines.append("")
    lines.append("| target cov | actual cov | threshold | F1 | macroP | recall |")
    lines.append("|---|---|---|---|---|---|")
    for cov, r in canonical_rows:
        ft = r["full_test"]
        lines.append(f"| {cov:.2f} | {ft['coverage']:.3f} | {r['threshold']:.3f}"
                     f" | {ft['f1']:.4f} | {ft['macro_P']:.4f} | {ft['recall']:.4f} |")
    lines.append("")
    lines.append("## Frontier — rapid_1s test")
    lines.append("")
    lines.append("| target cov | actual cov | threshold | F1 | macroP | recall |")
    lines.append("|---|---|---|---|---|---|")
    for cov, r in canonical_rows:
        rt = r["rapid_test"]
        lines.append(f"| {cov:.2f} | {rt['coverage']:.3f} | {r['threshold']:.3f}"
                     f" | {rt['f1']:.4f} | {rt['macro_P']:.4f} | {rt['recall']:.4f} |")
    lines.append("")
    lines.append("## Pareto frontier (F1 max for each macroP bin)")
    full_rows = [r["full_test"] for r in rows]
    by_p = sorted(full_rows, key=lambda r: r["macro_P"], reverse=True)
    seen_f1 = -1
    pareto = []
    for r in by_p:
        if r["f1"] > seen_f1:
            pareto.append(r)
            seen_f1 = r["f1"]
    lines.append("")
    lines.append("| coverage | F1 | macroP |")
    lines.append("|---|---|---|")
    for r in pareto[:20]:
        lines.append(f"| {r['coverage']:.3f} | {r['f1']:.4f} | {r['macro_P']:.4f} |")

    Path("campaign/H_selective.md").write_text("\n".join(lines))
    Path("campaign/H_selective.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
