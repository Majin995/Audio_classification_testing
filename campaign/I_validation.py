"""Phase I: consolidated comparison table on the full Split1s test set.

Re-runs the top configurations end-to-end and ranks them by full_test F1
with macroP held above the baseline (0.7911 on full).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from campaign.lib import (
    fit_temperature,
    fmt_metrics,
    load_dump,
    load_meta,
    metrics,
    metrics_with_thresholds,
    search_thresholds_f1_with_p_floor,
    stack_probs,
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


def main():
    meta = load_meta("campaign/probs_rapid_1s")
    nc = meta["num_classes"]
    val_P, val_y, test_P, test_y, stems = stack_probs("campaign/probs_rapid_1s")
    full_P, full_y = get_full(meta)
    eps = 1e-8
    grid = list(np.linspace(0.0, 0.99, 50))

    # Helper builders
    def arith_ens(P_stack, idx):
        return P_stack[idx].mean(axis=0)

    def geom_ens(P_stack, idx):
        g = np.exp(np.log(P_stack[idx] + eps).mean(axis=0))
        return g / g.sum(axis=1, keepdims=True)

    configs = []

    # 1. Per-model raw (5 entries)
    for i, s in enumerate(stems):
        configs.append({
            "name": f"single_{s.split('-')[1]}",
            "v": val_P[i], "t": test_P[i], "f": full_P[i],
            "calibrate": "none",
        })

    # 2. N=5 arithmetic mean (baseline)
    configs.append({
        "name": "N=5_arith",
        "v": arith_ens(val_P, list(range(5))),
        "t": arith_ens(test_P, list(range(5))),
        "f": arith_ens(full_P, list(range(5))),
        "calibrate": "T_only",
    })

    # 3. N=5 + per-class thresholds (Phase C)
    configs.append({
        "name": "N=5_arith_+thr",
        "v": arith_ens(val_P, list(range(5))),
        "t": arith_ens(test_P, list(range(5))),
        "f": arith_ens(full_P, list(range(5))),
        "calibrate": "T+thr",
    })

    # 4. Geom-2 (Phase F best raw)
    g2 = [i for i, s in enumerate(stems) if "026" in s or "031" in s]
    configs.append({
        "name": "Geom2(s1337+s12345)",
        "v": geom_ens(val_P, g2),
        "t": geom_ens(test_P, g2),
        "f": geom_ens(full_P, g2),
        "calibrate": "T_only",
    })

    # 5. Geom-2 + thresholds (Phase F winner)
    configs.append({
        "name": "Geom2_+thr",
        "v": geom_ens(val_P, g2),
        "t": geom_ens(test_P, g2),
        "f": geom_ens(full_P, g2),
        "calibrate": "T+thr",
    })

    # 6. NNLS-weighted (Phase F)
    from scipy.optimize import nnls
    val_oh = np.zeros((len(val_y), nc))
    val_oh[np.arange(len(val_y)), val_y] = 1.0
    A = val_P.transpose(1, 2, 0).reshape(-1, len(stems))
    b = val_oh.reshape(-1)
    w, _ = nnls(A, b)
    if w.sum() > 0:
        w = w / w.sum()
    nnls_v = np.tensordot(w, val_P, axes=(0, 0))
    nnls_t = np.tensordot(w, test_P, axes=(0, 0))
    nnls_f = np.tensordot(w, full_P, axes=(0, 0))
    nnls_v /= nnls_v.sum(axis=1, keepdims=True)
    nnls_t /= nnls_t.sum(axis=1, keepdims=True)
    nnls_f /= nnls_f.sum(axis=1, keepdims=True)
    configs.append({
        "name": "NNLS_weighted_+thr",
        "v": nnls_v, "t": nnls_t, "f": nnls_f, "calibrate": "T+thr",
    })

    # 7. Stacker — raw probs LR (Phase L)
    Xv_p = val_P.transpose(1, 0, 2).reshape(len(val_y), -1)
    Xt_p = test_P.transpose(1, 0, 2).reshape(len(test_y), -1)
    Xf_p = full_P.transpose(1, 0, 2).reshape(len(full_y), -1)
    clf_p = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(Xv_p, val_y)
    configs.append({
        "name": "Stacker_LR(probs,C=1)",
        "v": clf_p.predict_proba(Xv_p),
        "t": clf_p.predict_proba(Xt_p),
        "f": clf_p.predict_proba(Xf_p),
        "calibrate": "T+thr",
    })

    # 8. Stacker — log-probs LR (Phase L winner)
    Xv_lp = np.log(val_P + eps).transpose(1, 0, 2).reshape(len(val_y), -1)
    Xt_lp = np.log(test_P + eps).transpose(1, 0, 2).reshape(len(test_y), -1)
    Xf_lp = np.log(full_P + eps).transpose(1, 0, 2).reshape(len(full_y), -1)
    clf_lp = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(Xv_lp, val_y)
    configs.append({
        "name": "Stacker_Dirichlet(C=0.1)",
        "v": clf_lp.predict_proba(Xv_lp),
        "t": clf_lp.predict_proba(Xt_lp),
        "f": clf_lp.predict_proba(Xf_lp),
        "calibrate": "T+thr",
    })

    # 9. Stacker on geom-2 only (using just 2 models as stacker input)
    Xv_lp2 = np.log(val_P[g2] + eps).transpose(1, 0, 2).reshape(len(val_y), -1)
    Xt_lp2 = np.log(test_P[g2] + eps).transpose(1, 0, 2).reshape(len(test_y), -1)
    Xf_lp2 = np.log(full_P[g2] + eps).transpose(1, 0, 2).reshape(len(full_y), -1)
    clf_g2 = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(Xv_lp2, val_y)
    configs.append({
        "name": "Stacker_Dirichlet_g2(C=0.1)",
        "v": clf_g2.predict_proba(Xv_lp2),
        "t": clf_g2.predict_proba(Xt_lp2),
        "f": clf_g2.predict_proba(Xf_lp2),
        "calibrate": "T+thr",
    })

    # 10. Stacker + Bayes rule
    # (skip — already in G family)

    # Evaluate each
    rows = []
    for cfg in configs:
        v = cfg["v"]
        t = cfg["t"]
        f = cfg["f"]
        cal = cfg["calibrate"]
        if cal == "none":
            res_full = metrics(f, full_y, nc)
            res_rap = metrics(t, test_y, nc)
            T = None
            thr = None
            cov_full = 1.0
        elif cal == "T_only":
            T = fit_temperature(v, val_y)
            fT = temperature_scale(f, T)
            tT = temperature_scale(t, T)
            res_full = metrics(fT, full_y, nc)
            res_rap = metrics(tT, test_y, nc)
            thr = None
            cov_full = 1.0
        elif cal == "T+thr":
            T = fit_temperature(v, val_y)
            vT = temperature_scale(v, T)
            tT = temperature_scale(t, T)
            fT = temperature_scale(f, T)
            res = search_thresholds_f1_with_p_floor(vT, val_y, nc, p_floor=0.65, grid=grid)
            thr = res["thresholds"]
            res_full = metrics_with_thresholds(fT, full_y, np.array(thr), nc)
            res_rap = metrics_with_thresholds(tT, test_y, np.array(thr), nc)
            cov_full = res_full["coverage"]

        rows.append({
            "name": cfg["name"], "calibrate": cal, "T": T, "thresholds": thr,
            "rapid_test": res_rap, "full_test": res_full,
            "full_F1": res_full["f1"], "full_macroP": res_full["macro_P"],
            "full_cov": cov_full,
        })

    # Rank by full F1
    rows.sort(key=lambda r: r["full_F1"], reverse=True)

    lines = ["# I. Validation — full Split1s test (n=7872) ranked", ""]
    lines.append("All configurations evaluated on FULL Split1s test set (n=7872) — the gold-standard out-of-sample eval.")
    lines.append("")
    lines.append("| rank | config | calib | T | full F1 | full macroP | full recall | full cov | rapid F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        T_str = f"{r['T']:.2f}" if r["T"] is not None else "—"
        lines.append(f"| {i} | {r['name']} | {r['calibrate']} | {T_str}"
                     f" | {r['full_F1']:.4f} | {r['full_macroP']:.4f}"
                     f" | {r['full_test']['recall']:.4f} | {r['full_cov']:.3f}"
                     f" | {r['rapid_test']['f1']:.4f} |")

    # Detail on top 3
    lines.append("")
    lines.append("## Top 3 — detail")
    for r in rows[:3]:
        lines.append("")
        lines.append(f"### {r['name']}")
        lines.append("```")
        if r["T"] is not None:
            lines.append(f"T = {r['T']:.3f}")
        if r["thresholds"] is not None:
            lines.append(f"thresholds = {[round(x,3) for x in r['thresholds']]}")
        lines.append("full_test:")
        lines.append(fmt_metrics(r["full_test"], nc, "  "))
        lines.append("```")

    # Compute deltas vs. N=5_arith baseline
    base = next(r for r in rows if r["name"] == "N=5_arith")
    lines.append("")
    lines.append("## Δ vs N=5 arith baseline (full F1, macroP)")
    lines.append("")
    lines.append("| config | Δ F1 | Δ macroP | Δ recall |")
    lines.append("|---|---|---|---|")
    for r in rows:
        if r["name"] == "N=5_arith":
            continue
        df = r["full_F1"] - base["full_F1"]
        dp = r["full_macroP"] - base["full_macroP"]
        dr = r["full_test"]["recall"] - base["full_test"]["recall"]
        lines.append(f"| {r['name']} | {df:+.4f} | {dp:+.4f} | {dr:+.4f} |")

    Path("campaign/I_validation.md").write_text("\n".join(lines))
    Path("campaign/I_validation.json").write_text(json.dumps(rows, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
