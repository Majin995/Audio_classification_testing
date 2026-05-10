"""Phase CC: full eval on Classifier_Dataset (1s clips, 50k val, 14k test).

Uses dumped probs from campaign/probs_classifier_dataset/ to:
  1. Apply the production stacker (lightning_logs/dirichlet_stacker_full/stacker.joblib)
     fit on Split1s_eval — measures CROSS-DATASET generalization.
  2. Refit a fresh MLP(64) K=10 stacker on Classifier_Dataset/val,
     eval on Classifier_Dataset/test — measures DATASET-SPECIFIC best.
  3. Also refit LR(C=0.1) Dirichlet for comparison.

Reports both side-by-side. Saves the refit stacker to
lightning_logs/dirichlet_stacker_classifier_dataset/.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from campaign.lib import (
    fmt_metrics,
    load_dump,
    load_meta,
    metrics,
    stack_probs,
)


def main():
    probs_dir = "campaign/probs_classifier_dataset"
    meta = load_meta(probs_dir)
    classes = meta["classes"]
    nc = meta["num_classes"]

    val_P, val_y, test_P, test_y, stems = stack_probs(probs_dir)
    eps = 1e-8
    print(f"loaded {len(stems)} models, val_y={val_y.shape}, test_y={test_y.shape}")

    Xv = np.log(val_P + eps).transpose(1, 0, 2).reshape(len(val_y), -1)
    Xt = np.log(test_P + eps).transpose(1, 0, 2).reshape(len(test_y), -1)

    # 0. Per-model raw + N=5 arith baseline
    per_model = []
    for i, s in enumerate(stems):
        per_model.append({"stem": s,
                           "val": metrics(val_P[i], val_y, nc),
                           "test": metrics(test_P[i], test_y, nc)})
    base_v = val_P.mean(axis=0)
    base_t = test_P.mean(axis=0)
    base_val_m = metrics(base_v, val_y, nc)
    base_test_m = metrics(base_t, test_y, nc)

    # 1. APPLY pre-fitted production stacker
    pre = joblib.load("lightning_logs/dirichlet_stacker_full/stacker.joblib")
    if isinstance(pre, list):
        applied_t = np.mean([m.predict_proba(Xt) for m in pre], axis=0)
    else:
        applied_t = pre.predict_proba(Xt)
    applied_test_m = metrics(applied_t, test_y, nc)

    # 2. REFIT MLP(64) K=10 on Classifier_Dataset val
    K = 10
    mlp_v = mlp_t = None
    mlps = []
    for s in range(K):
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000,
                             alpha=1e-2, random_state=s).fit(Xv, val_y)
        mlps.append(clf)
        v_p, t_p = clf.predict_proba(Xv), clf.predict_proba(Xt)
        mlp_v = v_p if mlp_v is None else mlp_v + v_p
        mlp_t = t_p if mlp_t is None else mlp_t + t_p
    mlp_v /= K; mlp_t /= K
    refit_val_m = metrics(mlp_v, val_y, nc)
    refit_test_m = metrics(mlp_t, test_y, nc)

    # 3. REFIT LR(C=0.1) Dirichlet
    lr = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(Xv, val_y)
    lr_t = lr.predict_proba(Xt)
    lr_test_m = metrics(lr_t, test_y, nc)

    # Save refit stacker artifact
    out_dir = Path("lightning_logs/dirichlet_stacker_classifier_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(mlps, out_dir / "stacker.joblib")
    np.save(out_dir / "final_test_probs.npy", mlp_t)
    with open(out_dir / "stacker_meta.json", "w") as f:
        json.dump({
            "classifier": "mlp64_k10",
            "fit_on": "Classifier_Dataset/val (n=" + str(len(val_y)) + ")",
            "ckpts": meta["ckpts"],
            "classes": classes,
            "test_metrics": refit_test_m,
            "val_metrics": refit_val_m,
        }, f, indent=2)

    # Build report
    lines = ["# CC. Full eval on Classifier_Dataset", ""]
    lines.append(f"- data_dir: `/var/mnt/5A009BF8009BD8F9/Data/Classifier_Dataset`")
    lines.append(f"- val n: {len(val_y)}, test n: {len(test_y)}")
    lines.append(f"- 5 ckpts (HydroHydra Phase-I seeds)")
    lines.append("")
    lines.append("## Per-model raw metrics")
    lines.append("")
    lines.append("| model | val/F1 | val/macroP | test/F1 | test/macroP |")
    lines.append("|---|---|---|---|---|")
    for r in per_model:
        lines.append(f"| {r['stem']} | {r['val']['f1']:.4f}"
                     f" | {r['val']['macro_P']:.4f} | {r['test']['f1']:.4f}"
                     f" | {r['test']['macro_P']:.4f} |")
    lines.append("")
    lines.append("## Headline comparison")
    lines.append("")
    lines.append("| stacker | fit on | test F1 | test macroP | test recall | test MCC | test AUROC |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| (none — N=5 arith) | — | {base_test_m['f1']:.4f}"
                 f" | {base_test_m['macro_P']:.4f} | {base_test_m['recall']:.4f}"
                 f" | {base_test_m['mcc']:.4f} | {base_test_m['auroc']:.4f} |")
    lines.append(f"| MLP(64) K=10 PRE-FIT (Split1s_eval) | (loaded) | {applied_test_m['f1']:.4f}"
                 f" | {applied_test_m['macro_P']:.4f} | {applied_test_m['recall']:.4f}"
                 f" | {applied_test_m['mcc']:.4f} | {applied_test_m['auroc']:.4f} |")
    lines.append(f"| LR(C=0.1) Dirichlet REFIT | Classifier_Dataset/val | {lr_test_m['f1']:.4f}"
                 f" | {lr_test_m['macro_P']:.4f} | {lr_test_m['recall']:.4f}"
                 f" | {lr_test_m['mcc']:.4f} | {lr_test_m['auroc']:.4f} |")
    lines.append(f"| **MLP(64) K=10 REFIT** | Classifier_Dataset/val | **{refit_test_m['f1']:.4f}**"
                 f" | **{refit_test_m['macro_P']:.4f}** | **{refit_test_m['recall']:.4f}**"
                 f" | **{refit_test_m['mcc']:.4f}** | **{refit_test_m['auroc']:.4f}** |")
    lines.append("")
    lines.append("## Per-class detail — refit MLP(64) K=10 (production-grade)")
    lines.append("```")
    lines.append("test:")
    lines.append(fmt_metrics(refit_test_m, nc, "  "))
    lines.append("```")
    lines.append("")
    lines.append("## Per-class detail — pre-fit (cross-dataset apply)")
    lines.append("```")
    lines.append("test:")
    lines.append(fmt_metrics(applied_test_m, nc, "  "))
    lines.append("```")
    lines.append("")
    lines.append("## Lift summary")
    df_apply = applied_test_m["f1"] - base_test_m["f1"]
    df_refit = refit_test_m["f1"] - base_test_m["f1"]
    df_lr = lr_test_m["f1"] - base_test_m["f1"]
    lines.append(f"- pre-fit (apply only) Δ F1 = {df_apply:+.4f}")
    lines.append(f"- LR refit Δ F1 = {df_lr:+.4f}")
    lines.append(f"- **MLP refit Δ F1 = {df_refit:+.4f}**")

    Path("campaign/CC_classifier_dataset_eval.md").write_text("\n".join(lines))
    Path("campaign/CC_classifier_dataset_eval.json").write_text(json.dumps({
        "per_model": per_model,
        "n=5_arith": {"val": base_val_m, "test": base_test_m},
        "applied_pre_fit": applied_test_m,
        "lr_refit": lr_test_m,
        "mlp_refit": refit_test_m,
    }, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
