"""Phase D: joint sweep of temperature × per-class threshold.

Temperature changes how confident the ensemble is. Higher T flattens
probabilities, which interacts with thresholds (more abstention).
"""
import json
from pathlib import Path

import numpy as np

from campaign.lib import (
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
    test_P, test_y = [], None
    for entry in meta["ckpts"]:
        d = load_dump(full_dir / f"{entry['stem']}.npz")
        test_P.append(d["test_probs"])
        test_y = d["test_y"]
    return np.stack(test_P).mean(axis=0), test_y


def main():
    meta = load_meta("campaign/probs_rapid_1s")
    nc = meta["num_classes"]
    val_P, val_y, test_P, test_y, _ = stack_probs("campaign/probs_rapid_1s")
    val_ens = val_P.mean(axis=0)
    test_ens = test_P.mean(axis=0)
    full_ens, full_y = get_full(meta)
    print(f"shapes: val {val_y.shape}, rapid_test {test_y.shape}, full_test {full_y.shape}")

    Ts = [0.7, 1.0, 1.3, 1.5, 1.75, 2.0, 2.3, 2.7, 3.0, 3.5, 4.0]
    grid = list(np.linspace(0.0, 0.99, 50))
    p_floor = 0.65

    rows = []
    lines = ["# D. Joint sweep — temperature × per-class threshold", ""]
    lines.append(f"- p_floor = {p_floor}")
    lines.append(f"- threshold grid: 50 pts in [0, 0.99]")
    lines.append("")
    lines.append("| T | thresholds | val F1 | rapid_test F1 | rapid_test macroP | full_test F1 | full_test macroP | full_test cov |")
    lines.append("|---|---|---|---|---|---|---|---|")

    best_full_f1 = -1.0
    best = None
    for T in Ts:
        v = temperature_scale(val_ens, T)
        t = temperature_scale(test_ens, T)
        f = temperature_scale(full_ens, T)
        res = search_thresholds_f1_with_p_floor(v, val_y, nc, p_floor=p_floor, grid=grid)
        thr = np.array(res["thresholds"])
        rt = metrics_with_thresholds(t, test_y, thr, nc)
        ft = metrics_with_thresholds(f, full_y, thr, nc)
        lines.append(f"| {T:.2f} | {[round(x,2) for x in thr.tolist()]} "
                     f"| {res['metrics']['f1']:.4f} | {rt['f1']:.4f} | {rt['macro_P']:.4f}"
                     f" | {ft['f1']:.4f} | {ft['macro_P']:.4f} | {ft['coverage']:.3f} |")
        rows.append({"T": T, "thresholds": thr.tolist(),
                     "val_metrics": res["metrics"],
                     "rapid_test_metrics": rt, "full_test_metrics": ft})
        if ft["f1"] > best_full_f1:
            best_full_f1 = ft["f1"]
            best = rows[-1]

    lines.append("")
    lines.append("## Best by full_test F1")
    lines.append("```")
    lines.append(f"T={best['T']}  thresholds={best['thresholds']}")
    lines.append("rapid_test:")
    lines.append(fmt_metrics(best["rapid_test_metrics"], nc, "  "))
    lines.append("full_test:")
    lines.append(fmt_metrics(best["full_test_metrics"], nc, "  "))
    lines.append("```")
    Path("campaign/D_temp_threshold.md").write_text("\n".join(lines))
    Path("campaign/D_temp_threshold.json").write_text(json.dumps({
        "p_floor": p_floor, "rows": rows, "best": best,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
