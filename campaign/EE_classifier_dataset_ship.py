"""Phase EE: save best Classifier_Dataset stacker to production location.

Best stacker for this dataset: LR(C=0.1, class_weight='balanced') +
per-class threshold sweep on top.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from campaign.lib import (
    fmt_metrics,
    load_dump,
    load_meta,
    metrics,
    metrics_with_thresholds,
    search_thresholds_f1_with_p_floor,
    stack_probs,
)


def main():
    probs_dir = "campaign/probs_classifier_dataset"
    meta = load_meta(probs_dir)
    nc = meta["num_classes"]
    classes = meta["classes"]
    val_P, val_y, test_P, test_y, stems = stack_probs(probs_dir)
    eps = 1e-8
    Xv = np.log(val_P + eps).transpose(1, 0, 2).reshape(len(val_y), -1)
    Xt = np.log(test_P + eps).transpose(1, 0, 2).reshape(len(test_y), -1)

    # Fit best variant
    clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs",
                              class_weight="balanced").fit(Xv, val_y)
    v_p = clf.predict_proba(Xv)
    t_p = clf.predict_proba(Xt)

    grid = list(np.linspace(0.0, 0.99, 50))
    res = search_thresholds_f1_with_p_floor(v_p, val_y, nc, p_floor=0.5, grid=grid)
    thr = np.array(res["thresholds"])
    raw_test_m = metrics(t_p, test_y, nc)
    cal_test_m = metrics_with_thresholds(t_p, test_y, thr, nc)

    out_dir = Path("lightning_logs/dirichlet_stacker_classifier_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "stacker.joblib")
    np.save(out_dir / "thresholds.npy", thr)
    np.save(out_dir / "final_test_probs.npy", t_p)
    with open(out_dir / "stacker_meta.json", "w") as f:
        json.dump({
            "classifier": "LR(C=0.1, class_weight=balanced)",
            "fit_on": f"Classifier_Dataset/val (n={len(val_y)})",
            "thresholds": thr.tolist(),
            "p_floor_used": 0.5,
            "ckpts": meta["ckpts"],
            "classes": classes,
            "raw_test_metrics": raw_test_m,
            "calibrated_test_metrics": cal_test_m,
        }, f, indent=2)

    lines = ["# Classifier_Dataset production stacker", ""]
    lines.append("## Setup")
    lines.append(f"- 5 HydroHydra Phase-I seed ckpts → log-prob features (5×4=20 dims)")
    lines.append(f"- Stacker: LR(C=0.1, class_weight='balanced')")
    lines.append(f"- Per-class thresholds: {[round(x,3) for x in thr.tolist()]}")
    lines.append(f"- Fit val: n={len(val_y)} (Cargo {np.bincount(val_y)[0]}, Passenger {np.bincount(val_y)[1]}, "
                 f"Tanker {np.bincount(val_y)[2]}, Tug {np.bincount(val_y)[3]})")
    lines.append(f"- Test: n={len(test_y)}")
    lines.append("")
    lines.append("## Test metrics")
    lines.append("```")
    lines.append("raw (no thresholds):")
    lines.append(fmt_metrics(raw_test_m, nc, "  "))
    lines.append("with thresholds:")
    lines.append(fmt_metrics(cal_test_m, nc, "  "))
    lines.append("```")
    lines.append("")
    lines.append("## Comparison vs baseline")
    base_f = val_P.mean(axis=0)
    base_t = test_P.mean(axis=0)
    base_test_m = metrics(test_P.mean(axis=0), test_y, nc)
    lines.append(f"- N=5 arith baseline: F1={base_test_m['f1']:.4f}, macroP={base_test_m['macro_P']:.4f}")
    lines.append(f"- Production stacker (raw): F1={raw_test_m['f1']:.4f}, macroP={raw_test_m['macro_P']:.4f}")
    lines.append(f"- Production stacker (cal): F1={cal_test_m['f1']:.4f}, macroP={cal_test_m['macro_P']:.4f}")
    lines.append(f"- **Δ F1 (cal vs baseline) = {cal_test_m['f1']-base_test_m['f1']:+.4f}**")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- `{out_dir}/stacker.joblib` — fitted LR")
    lines.append(f"- `{out_dir}/thresholds.npy` — per-class thresholds (pre-applied)")
    lines.append(f"- `{out_dir}/stacker_meta.json` — full metrics + provenance")
    lines.append(f"- `{out_dir}/final_test_probs.npy` — raw (pre-threshold) softmax probs")

    Path("campaign/EE_classifier_dataset_ship.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
