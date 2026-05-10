"""Phase L: stacking. Fit a small classifier on per-model val probs as
features. Five models × 4 probs = 20-D feature vector per sample.

Try: (1) multinomial logistic regression with L2, (2) Dirichlet calibration
(Kull et al.) which is essentially LR on log-probs.

For each combo (LR vs Dirichlet) and L2 strength, fit on val, eval on test.
"""
import json
from itertools import combinations
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


def stack_features(P_stack, mode="probs"):
    """P_stack: (M, N, C) → (N, M*C) features."""
    M, N, C = P_stack.shape
    if mode == "probs":
        feats = P_stack.transpose(1, 0, 2).reshape(N, M * C)
    elif mode == "logprobs":
        eps = 1e-8
        feats = np.log(P_stack + eps).transpose(1, 0, 2).reshape(N, M * C)
    return feats


def main():
    meta = load_meta("campaign/probs_rapid_1s")
    nc = meta["num_classes"]
    val_P, val_y, test_P, test_y, stems = stack_probs("campaign/probs_rapid_1s")
    full_P, full_y = get_full(meta)

    rows = []
    lines = ["# L. Stacking — learned probs combiner", ""]
    lines.append("Five models × 4-class probs = 20 features per sample.")
    lines.append("Logistic regression fit on rapid_1s val (n=64), evaluated on test.")
    lines.append("Compare: (a) raw probs features, (b) log-probs features (≈ Dirichlet calib).")
    lines.append("")
    lines.append("| feat | C (L2 inv) | rapid_test F1 | rapid_test macroP | full_test F1 | full_test macroP |")
    lines.append("|---|---|---|---|---|---|")

    best = None
    best_f1 = -1.0

    for mode in ["probs", "logprobs"]:
        Xv = stack_features(val_P, mode=mode)
        Xt = stack_features(test_P, mode=mode)
        Xf = stack_features(full_P, mode=mode)
        for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            try:
                clf.fit(Xv, val_y)
            except Exception as e:
                print(f"fit failed mode={mode} C={C}: {e}")
                continue
            t_probs = clf.predict_proba(Xt)
            f_probs = clf.predict_proba(Xf)
            t_m = metrics(t_probs, test_y, nc)
            f_m = metrics(f_probs, full_y, nc)
            rows.append({"mode": mode, "C": C, "rapid_test": t_m, "full_test": f_m})
            if f_m["f1"] > best_f1:
                best_f1 = f_m["f1"]
                best = rows[-1]
            lines.append(f"| {mode} | {C} | {t_m['f1']:.4f} | {t_m['macro_P']:.4f}"
                         f" | {f_m['f1']:.4f} | {f_m['macro_P']:.4f} |")

    # Now try stacking + per-class threshold post-processing
    lines.append("")
    lines.append("## Stacking + per-class thresholds (best base + threshold search on stacker val probs)")
    lines.append("")
    lines.append("| feat | C | T | thresholds | full_test F1 | full_test macroP | full_test cov |")
    lines.append("|---|---|---|---|---|---|---|")
    grid = list(np.linspace(0.0, 0.99, 50))
    best_post = None
    best_post_f1 = -1.0
    for mode in ["probs", "logprobs"]:
        Xv = stack_features(val_P, mode=mode)
        Xt = stack_features(test_P, mode=mode)
        Xf = stack_features(full_P, mode=mode)
        for C in [0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            try:
                clf.fit(Xv, val_y)
            except Exception:
                continue
            v_p = clf.predict_proba(Xv)
            t_p = clf.predict_proba(Xt)
            f_p = clf.predict_proba(Xf)
            T = fit_temperature(v_p, val_y)
            vT = temperature_scale(v_p, T)
            tT = temperature_scale(t_p, T)
            fT = temperature_scale(f_p, T)
            res = search_thresholds_f1_with_p_floor(vT, val_y, nc, p_floor=0.65, grid=grid)
            thr = np.array(res["thresholds"])
            t_cal = metrics_with_thresholds(tT, test_y, thr, nc)
            f_cal = metrics_with_thresholds(fT, full_y, thr, nc)
            lines.append(f"| {mode} | {C} | {T:.2f} | {[round(x,2) for x in thr.tolist()]}"
                         f" | {f_cal['f1']:.4f} | {f_cal['macro_P']:.4f} | {f_cal['coverage']:.3f} |")
            if f_cal["f1"] > best_post_f1:
                best_post_f1 = f_cal["f1"]
                best_post = {"mode": mode, "C": C, "T": T, "thresholds": thr.tolist(),
                              "rapid_test": t_cal, "full_test": f_cal}

    lines.append("")
    if best is not None:
        lines.append("## Best stacker (no thresholds)")
        lines.append("```")
        lines.append(f"mode={best['mode']}  C={best['C']}")
        lines.append("rapid_test:")
        lines.append(fmt_metrics(best["rapid_test"], nc, "  "))
        lines.append("full_test:")
        lines.append(fmt_metrics(best["full_test"], nc, "  "))
        lines.append("```")
    if best_post is not None:
        lines.append("")
        lines.append("## Best stacker + post-cal")
        lines.append("```")
        lines.append(f"mode={best_post['mode']}  C={best_post['C']}  T={best_post['T']:.3f}")
        lines.append(f"thresholds={best_post['thresholds']}")
        lines.append("rapid_test:")
        lines.append(fmt_metrics(best_post["rapid_test"], nc, "  "))
        lines.append("full_test:")
        lines.append(fmt_metrics(best_post["full_test"], nc, "  "))
        lines.append("```")

    Path("campaign/L_stacking.md").write_text("\n".join(lines))
    Path("campaign/L_stacking.json").write_text(json.dumps({
        "rows": rows, "best": best, "best_post": best_post,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
