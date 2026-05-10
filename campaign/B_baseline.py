"""Phase B: establish 5-seed ensemble baseline on rapid_1s.

Reads dumped probs, computes per-model + ensemble metrics on val and test,
fits temperature on val, applies to test, writes report.
"""
import json
from pathlib import Path

import numpy as np

from campaign.lib import (
    fit_temperature,
    fmt_metrics,
    load_meta,
    metrics,
    stack_probs,
    temperature_scale,
)


def main(probs_dir: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    meta = load_meta(probs_dir)
    nc = meta["num_classes"]
    cls = meta["classes"]

    val_P, val_y, test_P, test_y, stems = stack_probs(probs_dir)
    M = len(stems)
    print(f"loaded {M} models, val_y shape {val_y.shape}, test_y shape {test_y.shape}")

    lines = ["# B. Baseline on rapid_1s — 5-seed ensemble", ""]
    lines.append(f"- probs_dir: `{probs_dir}`")
    lines.append(f"- num_classes: {nc}")
    lines.append(f"- classes: {cls}")
    lines.append(f"- val n: {len(val_y)}, test n: {len(test_y)}")
    lines.append("")

    # Per-model metrics on val and test (no calibration)
    lines.append("## Per-model (raw, no temperature)")
    lines.append("")
    lines.append("| stem | val/F1 | val/macroP | val/MCC | test/F1 | test/macroP | test/MCC |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, stem in enumerate(stems):
        vm = metrics(val_P[i], val_y, nc)
        tm = metrics(test_P[i], test_y, nc)
        lines.append(f"| {stem} | {vm['f1']:.4f} | {vm['macro_P']:.4f} | {vm['mcc']:.4f}"
                     f" | {tm['f1']:.4f} | {tm['macro_P']:.4f} | {tm['mcc']:.4f} |")
    lines.append("")

    # Ensemble (mean prob)
    val_ens = val_P.mean(axis=0)
    test_ens = test_P.mean(axis=0)
    val_em = metrics(val_ens, val_y, nc)
    test_em = metrics(test_ens, test_y, nc)
    lines.append("## Ensemble (mean prob, no temperature)")
    lines.append("")
    lines.append("```")
    lines.append("VAL:  " + fmt_metrics(val_em, nc, "  "))
    lines.append("TEST: " + fmt_metrics(test_em, nc, "  "))
    lines.append("```")
    lines.append("")

    # Fit temperature on val ensemble
    T = fit_temperature(val_ens, val_y)
    val_ens_T = temperature_scale(val_ens, T)
    test_ens_T = temperature_scale(test_ens, T)
    val_emT = metrics(val_ens_T, val_y, nc)
    test_emT = metrics(test_ens_T, test_y, nc)
    lines.append(f"## Ensemble + temperature (T = {T:.3f}, fitted on val)")
    lines.append("")
    lines.append("```")
    lines.append("VAL:  " + fmt_metrics(val_emT, nc, "  "))
    lines.append("TEST: " + fmt_metrics(test_emT, nc, "  "))
    lines.append("```")
    lines.append("")

    # Save baseline metrics
    out = {
        "stems": stems,
        "T": T,
        "ensemble_val_raw": val_em,
        "ensemble_test_raw": test_em,
        "ensemble_val_T": val_emT,
        "ensemble_test_T": test_emT,
    }
    Path(out_dir, "B_ensemble_baseline.json").write_text(json.dumps(out, indent=2))
    Path(out_dir, "B_ensemble_baseline.md").write_text("\n".join(lines))
    print("\n".join(lines[-15:]))
    print(f"\nwrote {out_dir}/B_ensemble_baseline.{{md,json}}")


if __name__ == "__main__":
    main("campaign/probs_rapid_1s", "campaign")
