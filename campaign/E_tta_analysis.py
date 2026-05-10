"""Phase E analysis: compare TTA-K=1, K=4, K=8 [, K=16] on rapid + full.

Reads from campaign/probs_rapid_1s (k=1), probs_rapid_tta_k{K}, and the
full equivalents. Reports F1 / macroP at each K, both with and without
per-class thresholds.
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
    search_thresholds_f1_with_p_floor,
    stack_probs,
    temperature_scale,
)


def stack_two_ckpts(probs_dir, target_stems):
    """Load just the 2 best ckpts from a probs_dir."""
    meta = load_meta(probs_dir)
    out_v, out_t = [], []
    val_y = test_y = None
    for entry in meta["ckpts"]:
        if entry["stem"] not in target_stems:
            continue
        d = load_dump(entry["npz"])
        if "val_probs" in d:
            out_v.append(d["val_probs"])
            val_y = d["val_y"]
        if "test_probs" in d:
            out_t.append(d["test_probs"])
            test_y = d["test_y"]
    return (np.stack(out_v) if out_v else None, val_y,
            np.stack(out_t) if out_t else None, test_y)


def geom(P_stack):
    eps = 1e-8
    g = np.exp(np.log(P_stack + eps).mean(axis=0))
    return g / g.sum(axis=1, keepdims=True)


def main():
    target_stems = {"hydra-026-p0.6908", "hydra-031-p0.7042"}
    nc = 4

    Ks = [1, 4, 8, 16]
    rapid_dirs = {1: "campaign/probs_rapid_1s",
                  4: "campaign/probs_rapid_tta_k4",
                  8: "campaign/probs_rapid_tta_k8",
                  16: "campaign/probs_rapid_tta_k16"}
    full_dirs = {1: "campaign/probs_split1s_full",
                 4: "campaign/probs_full_tta_k4",
                 8: "campaign/probs_full_tta_k8"}

    rows = []
    lines = ["# E. TTA sweep on geom-2 ensemble (s1337 + s12345)", ""]
    lines.append("Augmentation source: HydroHydra._WaveformAugV2 with force_train=True.")
    lines.append("Probabilities averaged across K stochastic forwards per sample.")
    lines.append("")
    lines.append("| K | T | rapid_test F1 (raw) | rapid_test F1 (cal) | full_test F1 (raw) | full_test F1 (cal) | full_test macroP (cal) |")
    lines.append("|---|---|---|---|---|---|---|")

    for K in Ks:
        if K not in rapid_dirs or not Path(rapid_dirs[K]).exists():
            continue
        v_stack, val_y, t_stack, test_y = stack_two_ckpts(rapid_dirs[K], target_stems)
        if v_stack is None:
            continue
        v = geom(v_stack)
        t = geom(t_stack)
        T = fit_temperature(v, val_y)
        vT = temperature_scale(v, T)
        tT = temperature_scale(t, T)

        t_raw = metrics(t, test_y, nc)
        # threshold search
        res = search_thresholds_f1_with_p_floor(vT, val_y, nc, p_floor=0.65,
                                                 grid=list(np.linspace(0.0, 0.99, 50)))
        thr = np.array(res["thresholds"])
        t_cal = metrics_with_thresholds(tT, test_y, thr, nc)

        f_raw = f_cal = None
        if K in full_dirs and Path(full_dirs[K]).exists() and (Path(full_dirs[K]) / "_meta.json").exists():
            _, _, ftk_stack, full_y = stack_two_ckpts(full_dirs[K], target_stems)
            if ftk_stack is not None and full_y is not None:
                f = geom(ftk_stack)
                fT = temperature_scale(f, T)
                f_raw = metrics(f, full_y, nc)
                f_cal = metrics_with_thresholds(fT, full_y, thr, nc)

        row = {"K": K, "T": T, "thresholds": thr.tolist(),
               "rapid_test_raw": t_raw, "rapid_test_cal": t_cal,
               "full_test_raw": f_raw, "full_test_cal": f_cal}
        rows.append(row)
        f_raw_f1 = f"{f_raw['f1']:.4f}" if f_raw else "—"
        f_cal_f1 = f"{f_cal['f1']:.4f}" if f_cal else "—"
        f_cal_p = f"{f_cal['macro_P']:.4f}" if f_cal else "—"
        lines.append(f"| {K} | {T:.2f} | {t_raw['f1']:.4f} | {t_cal['f1']:.4f}"
                     f" | {f_raw_f1} | {f_cal_f1} | {f_cal_p} |")

    lines.append("")
    if rows:
        # Identify best by full_test cal F1 if available, else rapid
        has_full = [r for r in rows if r["full_test_cal"]]
        ranked = sorted(has_full or rows,
                        key=lambda r: (r["full_test_cal"] or r["rapid_test_cal"])["f1"],
                        reverse=True)
        best = ranked[0]
        lines.append(f"## Best by full_test_cal F1 (or rapid if no full)")
        lines.append("```")
        lines.append(f"K={best['K']}  T={best['T']:.3f}  thresholds={[round(t,3) for t in best['thresholds']]}")
        if best["full_test_cal"]:
            lines.append("full_test (cal):")
            lines.append(fmt_metrics(best["full_test_cal"], nc, "  "))
        lines.append("rapid_test (cal):")
        lines.append(fmt_metrics(best["rapid_test_cal"], nc, "  "))
        lines.append("```")

    Path("campaign/E_tta_sweep.md").write_text("\n".join(lines))
    Path("campaign/E_tta_sweep.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
