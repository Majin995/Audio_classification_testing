"""Phase C: per-class minimum-confidence threshold sweep.

Fit thresholds on rapid_1s val to maximize macro-F1 subject to per-class
precision floor; evaluate on rapid_1s test AND full Split1s test.
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


def get_full_test_ens(rapid_meta):
    """Stack the full-Split1s probs corresponding to the same ckpt stems."""
    full_dir = Path("campaign/probs_split1s_full")
    test_P, test_y = [], None
    for entry in rapid_meta["ckpts"]:
        npz = full_dir / f"{entry['stem']}.npz"
        d = load_dump(npz)
        test_P.append(d["test_probs"])
        test_y = d["test_y"]
    return np.stack(test_P).mean(axis=0), test_y


def main():
    out_dir = Path("campaign")
    rapid_meta = load_meta("campaign/probs_rapid_1s")
    nc = rapid_meta["num_classes"]

    val_P, val_y, test_P, test_y, stems = stack_probs("campaign/probs_rapid_1s")
    val_ens = val_P.mean(axis=0)
    test_ens = test_P.mean(axis=0)
    full_test_ens, full_test_y = get_full_test_ens(rapid_meta)
    print(f"rapid val n={len(val_y)}  rapid test n={len(test_y)}  full test n={len(full_test_y)}")

    # Apply temperature first (fitted on val)
    T = fit_temperature(val_ens, val_y)
    val_ens_T = temperature_scale(val_ens, T)
    test_ens_T = temperature_scale(test_ens, T)
    full_test_ens_T = temperature_scale(full_test_ens, T)
    print(f"T = {T:.3f}")

    floors = [0.0, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    grid = list(np.linspace(0.0, 0.99, 50))  # finer grid

    rows = []
    lines = ["# C. Per-class threshold sweep (precision-floored F1 max)", ""]
    lines.append(f"- T (val-fitted) = {T:.3f}")
    lines.append(f"- ensemble = mean(softmax) over 5 seeds")
    lines.append(f"- threshold grid: {len(grid)} pts in [0, 0.99]")
    lines.append(f"- fit on rapid_1s val (n={len(val_y)})")
    lines.append(f"- test on rapid_1s test (n={len(test_y)}) + full Split1s test (n={len(full_test_y)})")
    lines.append("")

    # Baseline
    base_rt = metrics(test_ens_T, test_y, nc)
    base_ft = metrics(full_test_ens_T, full_test_y, nc)
    lines.append("## Baseline (post-T, no thresholds)")
    lines.append("```")
    lines.append("rapid_1s_test:")
    lines.append(fmt_metrics(base_rt, nc, "  "))
    lines.append("full_test:")
    lines.append(fmt_metrics(base_ft, nc, "  "))
    lines.append("```")
    lines.append("")

    lines.append("## Threshold sweep results")
    lines.append("")
    lines.append("| p_floor | thresholds (val-fit) | val F1 | rapid_test F1 | rapid_test macroP | full_test F1 | full_test macroP |")
    lines.append("|---|---|---|---|---|---|---|")

    best_full_f1 = -1.0
    best_row = None
    for pf in floors:
        res = search_thresholds_f1_with_p_floor(val_ens_T, val_y, nc, p_floor=pf, grid=grid)
        thr = np.array(res["thresholds"])
        rt = metrics_with_thresholds(test_ens_T, test_y, thr, nc)
        ft = metrics_with_thresholds(full_test_ens_T, full_test_y, thr, nc)
        lines.append(f"| {pf:.2f} | {[round(t,2) for t in thr.tolist()]} "
                     f"| {res['metrics']['f1']:.4f} | {rt['f1']:.4f} | {rt['macro_P']:.4f}"
                     f" | {ft['f1']:.4f} | {ft['macro_P']:.4f} |")
        rows.append({
            "p_floor": pf,
            "thresholds": thr.tolist(),
            "val_metrics": res["metrics"],
            "rapid_test_metrics": rt,
            "full_test_metrics": ft,
        })
        if ft["f1"] > best_full_f1:
            best_full_f1 = ft["f1"]
            best_row = rows[-1]

    lines.append("")
    lines.append("## Best by full_test F1")
    lines.append("```")
    lines.append(f"p_floor={best_row['p_floor']}  thresholds={best_row['thresholds']}")
    lines.append("rapid_test:")
    lines.append(fmt_metrics(best_row["rapid_test_metrics"], nc, "  "))
    lines.append("full_test:")
    lines.append(fmt_metrics(best_row["full_test_metrics"], nc, "  "))
    lines.append("```")
    lines.append("")
    lines.append(f"Δ vs baseline (full_test): F1 {best_row['full_test_metrics']['f1'] - base_ft['f1']:+.4f}, "
                 f"macroP {best_row['full_test_metrics']['macro_P'] - base_ft['macro_P']:+.4f}")

    Path("campaign/C_threshold_sweep.md").write_text("\n".join(lines))
    Path("campaign/C_threshold_sweep.json").write_text(json.dumps({
        "T": T, "baseline_rapid_test": base_rt, "baseline_full_test": base_ft,
        "rows": rows, "best_by_full_test_f1": best_row,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
