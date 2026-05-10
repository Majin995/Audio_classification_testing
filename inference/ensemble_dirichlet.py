"""Stacker-based ensemble inference for HydroHydra.

Sibling of `inference/ensemble_eval.py`. Replaces arithmetic-mean ensembling
+ temperature scaling with a Dirichlet-style stacker fit on val log-probs.

Pipeline:
    1. Run each ckpt over val + test, collect softmax probs.
    2. Build 20-D log-prob features per sample (5 ckpts × 4 classes).
    3. Fit one of:
         - LogisticRegression(C=0.1) on log-probs  ("dirichlet_lr")
         - K-seed MLP(64) ensemble                  ("mlp64_k10")
       on val, evaluate on test.
    4. Save fitted classifier(s) as `stacker.joblib`,
       per-model val/test metrics + final test metrics as `stacker_eval.md`,
       and the final softmax probs `final_test_probs.npy` for downstream
       consumers.

Discovered 2026-05-10 in `campaign/` to lift HydroHydra full-Split1s test
F1 from 0.7432 (arithmetic mean baseline) → 0.8820 (mlp64_k10) with
macroP held at 0.88.

Usage:
    python -m inference.ensemble_dirichlet \
        --ckpts ckpt1 ckpt2 ckpt3 ckpt4 ckpt5 \
        --data_dir <DATA_DIR> \
        --out_dir lightning_logs/dirichlet_stacker \
        --classifier mlp64_k10
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_hydra import HydroHydra


@torch.no_grad()
def _collect_probs(model, dataloader, device, num_classes: int):
    model.eval()
    P, Y = [], []
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        if out.size(-1) > num_classes:
            out = out[:, :num_classes]
        P.append(F.softmax(out.float(), dim=-1).cpu().numpy())
        Y.append(y.detach().cpu().numpy())
    return np.concatenate(P), np.concatenate(Y)


def _features(P_stack: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Build (N, M*C) log-prob features from (M, N, C) per-model probs."""
    return np.log(P_stack + eps).transpose(1, 0, 2).reshape(P_stack.shape[1], -1)


def fit_stacker(name: str, X: np.ndarray, y: np.ndarray):
    """Return either a single sklearn classifier or a list (for K-seed MLP)."""
    if name == "dirichlet_lr":
        return LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(X, y)
    if name == "mlp64_k10":
        return [MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, alpha=1e-2,
                              random_state=s).fit(X, y) for s in range(10)]
    if name == "mlp64x32_k5":
        return [MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, alpha=1e-2,
                              random_state=s).fit(X, y) for s in range(5)]
    raise ValueError(f"unknown classifier {name}")


def predict_stacker(stacker, X: np.ndarray) -> np.ndarray:
    if isinstance(stacker, list):
        return np.mean([clf.predict_proba(X) for clf in stacker], axis=0)
    return stacker.predict_proba(X)


def report_metrics(probs: np.ndarray, y: np.ndarray, num_classes: int):
    pred = probs.argmax(axis=1)
    out = {
        "acc":     float(accuracy_score(y, pred)),
        "f1":      float(f1_score(y, pred, average="macro", zero_division=0)),
        "recall":  float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_P": float(precision_score(y, pred, average="macro", zero_division=0)),
        "micro_P": float(precision_score(y, pred, average="micro", zero_division=0)),
        "mcc":     float(matthews_corrcoef(y, pred)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y, probs, multi_class="ovr"))
    except Exception:
        out["auroc"] = float("nan")
    cm = confusion_matrix(y, pred, labels=list(range(num_classes)))
    out["confusion"] = cm.tolist()
    for c in range(num_classes):
        m_p = pred == c
        out[f"P_class_{c}"] = float((y[m_p] == c).mean()) if m_p.sum() else float("nan")
        m_t = y == c
        out[f"R_class_{c}"] = float((pred[m_t] == c).mean()) if m_t.sum() else 0.0
        prec, rec = out[f"P_class_{c}"], out[f"R_class_{c}"]
        out[f"F1_class_{c}"] = (2 * prec * rec / (prec + rec)
                                if not np.isnan(prec) and (prec + rec) > 0 else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="Dirichlet-stacker ensemble for HydroHydra")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_threads", type=int, default=8)
    ap.add_argument("--sample_rate", type=int, default=5_120)
    ap.add_argument("--fixed_len", type=int, default=5_120)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--classifier", choices=["dirichlet_lr", "mlp64_k10", "mlp64x32_k5"],
                    default="mlp64_k10",
                    help="stacker family. dirichlet_lr = simplest/fastest;"
                         " mlp64_k10 = best mean F1.")
    ap.add_argument("--stacker_path", default=None,
                    help="Optional path to a pre-fitted stacker.joblib. "
                         "When set, the stacker is NOT refit on val — instead "
                         "loaded from disk and applied to test only. Use this "
                         "for production inference on a new dataset.")
    args = ap.parse_args()

    if not args.data_dir:
        raise SystemExit("Set --data_dir or DATA_DIR")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = DALIAudioDataModule(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_threads=args.num_threads, target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
    )
    dm.setup()
    nc = dm.num_classes
    classes = [dm.idx_to_class[i] for i in range(nc)]
    print(f"[stacker] data_dir={args.data_dir} num_classes={nc} classes={classes}")

    val_P, test_P = [], []
    val_y = test_y = None
    per_model_metrics = []
    for ck in args.ckpts:
        print(f"[stacker] loading {ck}")
        m = HydroHydra.load_from_checkpoint(ck, map_location=device, strict=False).to(device).eval()
        # Try val (skip if missing — supports --apply mode without val)
        vp = vy = None
        try:
            dm.setup()
            vp, vy = _collect_probs(m, dm.val_dataloader(), device, nc)
        except FileNotFoundError:
            if args.stacker_path is None:
                raise
        dm.setup()
        tp, ty = _collect_probs(m, dm.test_dataloader(), device, nc)
        if vp is not None:
            val_P.append(vp); val_y = vy
        test_P.append(tp); test_y = ty
        entry = {"ckpt": ck, "test": report_metrics(tp, ty, nc)}
        if vp is not None:
            entry["val"] = report_metrics(vp, vy, nc)
        per_model_metrics.append(entry)
        del m
        torch.cuda.empty_cache()

    test_P = np.stack(test_P)
    Xt = _features(test_P)

    if args.stacker_path is not None:
        print(f"[stacker] loading pre-fitted stacker from {args.stacker_path}")
        stacker = joblib.load(args.stacker_path)
        val_m = None
    else:
        val_P = np.stack(val_P)
        Xv = _features(val_P)
        print(f"[stacker] fitting {args.classifier} on val features {Xv.shape}")
        stacker = fit_stacker(args.classifier, Xv, val_y)
        val_pred_p = predict_stacker(stacker, Xv)
        val_m = report_metrics(val_pred_p, val_y, nc)
        print(f"[stacker] val/F1 = {val_m['f1']:.4f}  val/macroP = {val_m['macro_P']:.4f}")

    test_pred_p = predict_stacker(stacker, Xt)
    test_m = report_metrics(test_pred_p, test_y, nc)
    print(f"[stacker] test/F1 = {test_m['f1']:.4f}  test/macroP = {test_m['macro_P']:.4f}")

    # Persist (skip stacker.joblib write when applying a pre-loaded stacker)
    if args.stacker_path is None:
        joblib.dump(stacker, out_dir / "stacker.joblib")
    np.save(out_dir / "final_test_probs.npy", test_pred_p)
    meta_out = {
        "classifier": args.classifier,
        "ckpts": args.ckpts,
        "classes": classes,
        "test_metrics": test_m,
        "per_model": per_model_metrics,
    }
    if val_m is not None:
        meta_out["val_metrics"] = val_m
    if args.stacker_path:
        meta_out["loaded_stacker"] = args.stacker_path
    with open(out_dir / "stacker_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    # Markdown summary
    lines = [
        "# Dirichlet-stacker ensemble report",
        "",
        f"- classifier: `{args.classifier}`",
        f"- data_dir: `{args.data_dir}`",
        f"- ckpts ({len(args.ckpts)}):",
    ]
    for ck in args.ckpts:
        lines.append(f"  - `{ck}`")
    lines.append("")
    lines.append("## Per-model metrics")
    lines.append("| model | val/F1 | val/macroP | test/F1 | test/macroP |")
    lines.append("|---|---|---|---|---|")
    for r in per_model_metrics:
        lines.append(f"| {Path(r['ckpt']).name} | {r['val']['f1']:.4f}"
                     f" | {r['val']['macro_P']:.4f} | {r['test']['f1']:.4f}"
                     f" | {r['test']['macro_P']:.4f} |")
    lines.append("")
    lines.append("## Stacker test metrics")
    lines.append(f"- F1:      {test_m['f1']:.4f}")
    lines.append(f"- macro_P: {test_m['macro_P']:.4f}")
    lines.append(f"- recall:  {test_m['recall']:.4f}")
    lines.append(f"- MCC:     {test_m['mcc']:.4f}")
    lines.append(f"- AUROC:   {test_m['auroc']:.4f}")
    for c in range(nc):
        lines.append(f"- {classes[c]}: P={test_m[f'P_class_{c}']:.3f} "
                     f"R={test_m[f'R_class_{c}']:.3f} "
                     f"F1={test_m[f'F1_class_{c}']:.3f}")
    lines.append("")
    lines.append("## Confusion matrix")
    lines.append("(rows = true, cols = predicted)")
    lines.append("```")
    cm = test_m["confusion"]
    header = "       " + " ".join(f"{c:>10}" for c in classes)
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"{classes[i]:>6}: " + " ".join(f"{v:>10}" for v in row))
    lines.append("```")
    (out_dir / "stacker_eval.md").write_text("\n".join(lines))
    written = ["stacker_eval.md", "stacker_meta.json", "final_test_probs.npy"]
    if args.stacker_path is None:
        written.append("stacker.joblib")
    print(f"[stacker] wrote {out_dir}/ → {', '.join(written)}")


if __name__ == "__main__":
    main()
