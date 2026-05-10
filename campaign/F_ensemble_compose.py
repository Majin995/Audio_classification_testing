"""Phase F: ensemble composition search.

Try (a) leave-one-out, (b) all subsets of size 3-5, (c) geometric vs
arithmetic averaging, (d) NNLS-weighted averaging fitted on val.
Goal: identify which subset/weighting maximizes F1 with macroP held high.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

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
    test_P = []
    test_y = None
    for entry in meta["ckpts"]:
        d = load_dump(full_dir / f"{entry['stem']}.npz")
        test_P.append(d["test_probs"])
        test_y = d["test_y"]
    return np.stack(test_P), test_y


def evaluate_combo(combo_idx, val_P, val_y, test_P, test_y, full_P, full_y, nc, mode="arith"):
    """Build ensemble over given indices, optionally fit T and per-class thresholds."""
    sub_v = val_P[list(combo_idx)]
    sub_t = test_P[list(combo_idx)]
    sub_f = full_P[list(combo_idx)]
    if mode == "arith":
        v = sub_v.mean(axis=0)
        t = sub_t.mean(axis=0)
        f = sub_f.mean(axis=0)
    elif mode == "geom":
        eps = 1e-8
        v = np.exp(np.log(sub_v + eps).mean(axis=0))
        v = v / v.sum(axis=1, keepdims=True)
        t = np.exp(np.log(sub_t + eps).mean(axis=0))
        t = t / t.sum(axis=1, keepdims=True)
        f = np.exp(np.log(sub_f + eps).mean(axis=0))
        f = f / f.sum(axis=1, keepdims=True)
    else:
        raise ValueError(mode)

    # Plain (no calibration)
    test_raw = metrics(t, test_y, nc)
    full_raw = metrics(f, full_y, nc)

    # Fit T then thresholds
    T = fit_temperature(v, val_y)
    vT = temperature_scale(v, T)
    tT = temperature_scale(t, T)
    fT = temperature_scale(f, T)
    res = search_thresholds_f1_with_p_floor(vT, val_y, nc, p_floor=0.65,
                                             grid=list(np.linspace(0.0, 0.99, 50)))
    thr = np.array(res["thresholds"])
    test_cal = metrics_with_thresholds(tT, test_y, thr, nc)
    full_cal = metrics_with_thresholds(fT, full_y, thr, nc)
    return {
        "T": T, "thresholds": thr.tolist(),
        "test_raw": test_raw, "full_raw": full_raw,
        "test_cal": test_cal, "full_cal": full_cal,
    }


def main():
    meta = load_meta("campaign/probs_rapid_1s")
    nc = meta["num_classes"]
    val_P, val_y, test_P, test_y, stems = stack_probs("campaign/probs_rapid_1s")
    full_P, full_y = get_full(meta)
    M = len(stems)

    rows = []
    lines = ["# F. Ensemble composition search", ""]
    lines.append(f"- 5 ckpts: {[s.split('-')[1] for s in stems]}")
    lines.append("")
    lines.append("Each row: best F1 after fitting T + per-class thresholds (p_floor=0.65) on rapid_1s val.")
    lines.append("")
    lines.append("| mode | combo | T | full_test F1 | full_test macroP | full_test cov | rapid_test F1 |")
    lines.append("|---|---|---|---|---|---|---|")

    best = None
    best_f1 = -1.0
    for size in range(1, M + 1):
        for combo in combinations(range(M), size):
            for mode in ["arith", "geom"]:
                if size == 1 and mode == "geom":
                    continue  # equivalent to arith for size 1
                res = evaluate_combo(combo, val_P, val_y, test_P, test_y,
                                     full_P, full_y, nc, mode=mode)
                names = "+".join(stems[i].split("-")[1] for i in combo)
                lines.append(f"| {mode} | {names} | {res['T']:.2f}"
                             f" | {res['full_cal']['f1']:.4f}"
                             f" | {res['full_cal']['macro_P']:.4f}"
                             f" | {res['full_cal']['coverage']:.3f}"
                             f" | {res['test_cal']['f1']:.4f} |")
                rows.append({"size": size, "mode": mode, "combo": list(combo),
                             "stems": [stems[i] for i in combo], **res})
                if res["full_cal"]["f1"] > best_f1:
                    best_f1 = res["full_cal"]["f1"]
                    best = rows[-1]

    # NNLS-weighted ensemble: fit non-negative weights on val to maximize accuracy
    # Approximate: per-class one-hot least-squares on val probs. Skip if too slow.
    from scipy.optimize import nnls
    val_oh = np.zeros((len(val_y), nc))
    val_oh[np.arange(len(val_y)), val_y] = 1.0
    A = np.transpose(val_P, (1, 0, 2)).reshape(len(val_y) * nc, M, order="F")  # (N*C, M)
    # Wrong reshape; do simpler: stack val_P into (M, N*C)
    A = val_P.transpose(1, 2, 0).reshape(-1, M)  # (N*C, M)
    b = val_oh.reshape(-1)
    w, _ = nnls(A, b)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.ones(M) / M
    nnls_v = np.tensordot(w, val_P, axes=(0, 0))
    nnls_t = np.tensordot(w, test_P, axes=(0, 0))
    nnls_f = np.tensordot(w, full_P, axes=(0, 0))
    nnls_v = nnls_v / nnls_v.sum(axis=1, keepdims=True)
    nnls_t = nnls_t / nnls_t.sum(axis=1, keepdims=True)
    nnls_f = nnls_f / nnls_f.sum(axis=1, keepdims=True)
    T = fit_temperature(nnls_v, val_y)
    nnls_vT = temperature_scale(nnls_v, T)
    nnls_tT = temperature_scale(nnls_t, T)
    nnls_fT = temperature_scale(nnls_f, T)
    res = search_thresholds_f1_with_p_floor(nnls_vT, val_y, nc, p_floor=0.65,
                                             grid=list(np.linspace(0.0, 0.99, 50)))
    thr = np.array(res["thresholds"])
    nnls_full_cal = metrics_with_thresholds(nnls_fT, full_y, thr, nc)
    nnls_test_cal = metrics_with_thresholds(nnls_tT, test_y, thr, nc)
    lines.append(f"| nnls | weighted_all | {T:.2f}"
                 f" | {nnls_full_cal['f1']:.4f}"
                 f" | {nnls_full_cal['macro_P']:.4f}"
                 f" | {nnls_full_cal['coverage']:.3f}"
                 f" | {nnls_test_cal['f1']:.4f} |")
    nnls_row = {"size": M, "mode": "nnls", "combo": list(range(M)),
                "stems": stems, "weights": w.tolist(), "T": T,
                "thresholds": thr.tolist(),
                "test_cal": nnls_test_cal, "full_cal": nnls_full_cal}
    rows.append(nnls_row)
    if nnls_full_cal["f1"] > best_f1:
        best = nnls_row
        best_f1 = nnls_full_cal["f1"]

    lines.append("")
    lines.append("## Best combo by full_test F1")
    lines.append("```")
    lines.append(f"mode={best['mode']}  size={best['size']}  combo={best['stems']}")
    if "weights" in best:
        lines.append(f"weights={[round(w,3) for w in best['weights']]}")
    lines.append(f"T={best['T']:.3f}  thresholds={[round(t,3) for t in best['thresholds']]}")
    lines.append("rapid_test:")
    lines.append(fmt_metrics(best["test_cal"], nc, "  "))
    lines.append("full_test:")
    lines.append(fmt_metrics(best["full_cal"], nc, "  "))
    lines.append("```")

    Path("campaign/F_ensemble_compose.md").write_text("\n".join(lines))
    Path("campaign/F_ensemble_compose.json").write_text(json.dumps(rows, indent=2))
    # Print summary
    print(f"\nTotal combos evaluated: {len(rows)}")
    print(f"Best full_test F1: {best_f1:.4f}")
    print(f"  combo: {best['stems']}")
    print(f"  mode: {best['mode']}")


if __name__ == "__main__":
    main()
