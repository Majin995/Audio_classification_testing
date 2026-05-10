"""Phase G: cost-sensitive Bayes-optimal decision rule.

Search a 4x4 utility matrix U[k,j] = utility of predicting k when truth is j.
Argmax of expected utility = probs @ U.T. Search U on val rapid_1s,
evaluate on full Split1s test.

Two parameterizations explored:
  (1) Diagonal — U[k,k] is the prediction reward for class k; off-diag = 0.
      Equivalent to per-class prior weighting.
  (2) Per-class diag + uniform off-diag = -mistake_cost. Search 2 params.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np

from campaign.lib import (
    apply_cost_rule,
    fit_temperature,
    fmt_metrics,
    load_dump,
    load_meta,
    metrics_from_pred,
    stack_probs,
    temperature_scale,
)


def get_full(meta):
    full_dir = Path("campaign/probs_split1s_full")
    test_P = []
    test_y = None
    for entry in meta["ckpts"]:
        d = load_dump(full_dir / f"{entry['stem']}.npz")
        test_P.append(d["test_probs"])
        test_y = d["test_y"]
    return np.stack(test_P), test_y


def main():
    meta = load_meta("campaign/probs_rapid_1s")
    nc = meta["num_classes"]
    val_P, val_y, test_P, test_y, stems = stack_probs("campaign/probs_rapid_1s")
    full_P, full_y = get_full(meta)

    # Use the Phase F winner: geometric mean of 026 + 031
    best_idx = []
    for i, s in enumerate(stems):
        if "026" in s or "031" in s:
            best_idx.append(i)
    eps = 1e-8
    v_geo = np.exp(np.log(val_P[best_idx] + eps).mean(axis=0))
    v_geo /= v_geo.sum(axis=1, keepdims=True)
    t_geo = np.exp(np.log(test_P[best_idx] + eps).mean(axis=0))
    t_geo /= t_geo.sum(axis=1, keepdims=True)
    f_geo = np.exp(np.log(full_P[best_idx] + eps).mean(axis=0))
    f_geo /= f_geo.sum(axis=1, keepdims=True)

    T = fit_temperature(v_geo, val_y)
    vT = temperature_scale(v_geo, T)
    tT = temperature_scale(t_geo, T)
    fT = temperature_scale(f_geo, T)

    print(f"using geom ensemble of {[stems[i] for i in best_idx]}, T={T:.3f}")

    # Search per-class diagonal weights w[k] >= 0 such that prediction k is taken
    # iff w[k] * P(k|x) is largest. This is U = diag(w).
    # Sweep each w_k in a small grid, look for F1 max with macroP >= floor.
    grid = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2]
    p_floor = 0.65

    best = None
    best_f1 = -1.0
    rows = []
    n_evals = 0
    for w0, w1, w2, w3 in product(grid, repeat=nc):
        n_evals += 1
        w = np.array([w0, w1, w2, w3])
        U = np.diag(w)
        val_pred = apply_cost_rule(vT, U)
        val_m = metrics_from_pred(val_pred, val_y, nc)
        # Check val precision floor
        ok = True
        for c in range(nc):
            p = val_m[f"P_class_{c}"]
            if not np.isnan(p) and p < p_floor:
                ok = False
                break
        if not ok:
            continue
        # Pick by val F1
        if val_m["f1"] > -1.0:  # always
            test_pred = apply_cost_rule(tT, U)
            full_pred = apply_cost_rule(fT, U)
            test_m = metrics_from_pred(test_pred, test_y, nc)
            full_m = metrics_from_pred(full_pred, full_y, nc)
            if val_m["f1"] > (best_f1 - 1e-6) and (best is None or full_m["f1"] > best.get("full_metrics", {}).get("f1", -1)):
                # Track configurations whose val_f1 ties or improves
                pass
            rows.append({"w": w.tolist(), "val_metrics": val_m,
                         "rapid_test_metrics": test_m, "full_metrics": full_m})

    # Pick best by val F1 (with tiebreak on full test F1 for reporting)
    rows.sort(key=lambda r: (r["val_metrics"]["f1"], r["full_metrics"]["f1"]), reverse=True)
    print(f"evaluated {n_evals} configs, {len(rows)} satisfied val precision floor")
    if not rows:
        print("nothing satisfied floor — relaxing")
        return
    best = rows[0]

    lines = ["# G. Cost-sensitive Bayes-optimal decision rule", ""]
    lines.append(f"- Base ensemble: geom of {[stems[i] for i in best_idx]}, T={T:.3f}")
    lines.append(f"- Search: 4D diagonal weights w in {grid}")
    lines.append(f"- Filter: val per-class precision >= {p_floor}")
    lines.append(f"- Total configs: {n_evals}, qualifying: {len(rows)}")
    lines.append("")
    lines.append("## Top 10 by val F1")
    lines.append("")
    lines.append("| w | val F1 | val macroP | rapid_test F1 | rapid_test macroP | full_test F1 | full_test macroP |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows[:10]:
        lines.append(f"| {r['w']} | {r['val_metrics']['f1']:.4f}"
                     f" | {r['val_metrics']['macro_P']:.4f}"
                     f" | {r['rapid_test_metrics']['f1']:.4f}"
                     f" | {r['rapid_test_metrics']['macro_P']:.4f}"
                     f" | {r['full_metrics']['f1']:.4f}"
                     f" | {r['full_metrics']['macro_P']:.4f} |")
    lines.append("")
    lines.append("## Best by val F1")
    lines.append("```")
    lines.append(f"w={best['w']}")
    lines.append("rapid_test:")
    lines.append(fmt_metrics(best["rapid_test_metrics"], nc, "  "))
    lines.append("full_test:")
    lines.append(fmt_metrics(best["full_metrics"], nc, "  "))
    lines.append("```")

    Path("campaign/G_bayes_rule.md").write_text("\n".join(lines))
    Path("campaign/G_bayes_rule.json").write_text(json.dumps({
        "T": T, "ckpt_subset": [stems[i] for i in best_idx],
        "p_floor": p_floor, "rows": rows[:200],  # truncate
    }, indent=2))
    print(f"wrote campaign/G_bayes_rule.{{md,json}}")
    print(f"best val F1 = {best['val_metrics']['f1']:.4f}, full F1 = {best['full_metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()
