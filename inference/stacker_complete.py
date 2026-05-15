"""Single-model Dirichlet stacker for HydroComplete.

Companion to ``inference/ensemble_dirichlet.py`` (which is hard-coded to
HydroHydra and assumes ≥1 ckpt). For HydroComplete with a single trained
checkpoint, the "stacker" reduces to a per-class affine recalibration of
log-probs — equivalent to vector temperature + per-class bias. The
memory's Dirichlet-stacker campaign noted that vector temperature alone
gives a small (+1.0 pp F1) lift on a geom-2 base; this script measures
the analogous lift for a single HydroComplete ckpt.

Pipeline (matches the campaign exactly except 1 model instead of 5):
  1. Load HydroComplete ckpt; collect softmax probs over val and test.
  2. Build (N, num_classes) log-prob features.
  3. Fit LogisticRegression(C=0.1) on val (the "Dirichlet LR" stacker).
  4. Optionally fit MLP(64) k=10 too (matches the production stacker).
  5. Report stacker test F1 / macroP / per-class P/R/F1 + raw-argmax
     baseline (no stacker) for comparison.

Note: per memory, "stacker fit on training data HURTS — never fit
calibration on data the underlying model was trained on." The val split
must be data the model has not been trained on. That is the case here
(the standard train/val/test split is honored by DALIAudioDataModule).

Usage:
    python -m inference.stacker_complete \\
        --ckpt lightning_logs/hydro_complete_s42/.../complete-044-p0.7159.ckpt \\
        --data_dir /path/to/Classifier_Dataset \\
        --out_dir lightning_logs/hydro_complete_s42/stacker
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
from models.hydro_complete import HydroComplete


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


def _features(P: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Single-model log-prob features. P: (N, C) → (N, C)."""
    return np.log(P + eps)


def fit_stacker(name: str, X: np.ndarray, y: np.ndarray):
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
    ap = argparse.ArgumentParser(description="Single-model Dirichlet stacker for HydroComplete")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_threads", type=int, default=8)
    ap.add_argument("--sample_rate", type=int, default=5_120)
    ap.add_argument("--fixed_len", type=int, default=5_120)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--classifiers", nargs="+",
                    default=["dirichlet_lr", "mlp64_k10"],
                    help="Stacker families to fit + evaluate (each gets its own row).")
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

    print(f"[stacker] loading {args.ckpt}")
    m = HydroComplete.load_from_checkpoint(
        args.ckpt, map_location=device, strict=False,
    ).to(device).eval()

    dm.setup()
    val_P, val_y = _collect_probs(m, dm.val_dataloader(), device, nc)
    dm.setup()
    test_P, test_y = _collect_probs(m, dm.test_dataloader(), device, nc)
    del m
    torch.cuda.empty_cache()
    print(f"[stacker] val:  N={len(val_y):>6}  argmax F1 baseline = "
          f"{f1_score(val_y, val_P.argmax(1), average='macro', zero_division=0):.4f}")
    print(f"[stacker] test: N={len(test_y):>6}  argmax F1 baseline = "
          f"{f1_score(test_y, test_P.argmax(1), average='macro', zero_division=0):.4f}")

    # Apply temperature from temperature.pt if available — gives a fair
    # baseline (post-cal raw argmax) to compare the stacker against.
    ckpt_dir = Path(args.ckpt).parent
    T = None
    if (ckpt_dir / "temperature.pt").exists():
        T = float(torch.load(ckpt_dir / "temperature.pt", weights_only=False)["temperature"])
        print(f"[stacker] applying temperature T={T:.3f} to val/test logits-equivalent probs")
        # Post-cal probs: re-temperature the softmax distribution.
        # Equivalent to softmax(log_p / T) up to renorm.
        def re_T(P, T):
            logp = np.log(P + 1e-8) / T
            logp -= logp.max(axis=1, keepdims=True)
            ep = np.exp(logp)
            return ep / ep.sum(axis=1, keepdims=True)
        val_P_T  = re_T(val_P, T)
        test_P_T = re_T(test_P, T)
    else:
        val_P_T, test_P_T = val_P, test_P

    Xv = _features(val_P_T)
    Xt = _features(test_P_T)

    # Baselines.
    raw_test_m   = report_metrics(test_P,   test_y, nc)
    cal_test_m   = report_metrics(test_P_T, test_y, nc)

    # Fit each stacker family on val log-probs, predict on test.
    stacker_results = {}
    for name in args.classifiers:
        print(f"[stacker] fitting {name} on val features {Xv.shape}")
        stacker = fit_stacker(name, Xv, val_y)
        val_pred_p  = predict_stacker(stacker, Xv)
        test_pred_p = predict_stacker(stacker, Xt)
        val_m  = report_metrics(val_pred_p,  val_y,  nc)
        test_m = report_metrics(test_pred_p, test_y, nc)
        print(f"[stacker]   val/F1 = {val_m['f1']:.4f}  val/macroP = {val_m['macro_P']:.4f}")
        print(f"[stacker]   test/F1 = {test_m['f1']:.4f}  test/macroP = {test_m['macro_P']:.4f}")
        joblib.dump(stacker, out_dir / f"stacker_{name}.joblib")
        np.save(out_dir / f"final_test_probs_{name}.npy", test_pred_p)
        stacker_results[name] = {"val": val_m, "test": test_m}

    # Persist
    meta = {
        "ckpt": args.ckpt,
        "data_dir": args.data_dir,
        "classes": classes,
        "temperature": T,
        "raw_test_metrics":  raw_test_m,
        "post_cal_test_metrics": cal_test_m,
        "stackers": stacker_results,
    }
    with open(out_dir / "stacker_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Markdown summary
    lines = [
        "# HydroComplete single-model Dirichlet stacker",
        "",
        f"- ckpt: `{args.ckpt}`",
        f"- data_dir: `{args.data_dir}`",
        f"- temperature: {T if T is not None else 'n/a'}",
        f"- val N: {len(val_y)},  test N: {len(test_y)}",
        "",
        "## Headline test metrics",
        "| variant | F1 | macro_P | micro_P | recall | MCC | AUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    def row(name, m):
        return (f"| {name} | {m['f1']:.4f} | {m['macro_P']:.4f} | "
                f"{m['micro_P']:.4f} | {m['recall']:.4f} | "
                f"{m['mcc']:.4f} | {m['auroc']:.4f} |")
    lines.append(row("raw argmax (no T)", raw_test_m))
    lines.append(row("post-cal argmax (T)", cal_test_m))
    for name, r in stacker_results.items():
        lines.append(row(f"stacker {name}", r["test"]))

    lines.append("")
    lines.append("## Per-class precision / recall / F1 (test)")
    lines.append("| variant | " + " | ".join(f"P[{c}]" for c in classes) + " | " +
                 " | ".join(f"R[{c}]" for c in classes) + " | " +
                 " | ".join(f"F1[{c}]" for c in classes) + " |")
    lines.append("|" + "---|" * (1 + 3 * nc))
    def per_row(name, m):
        ps = " | ".join(f"{m[f'P_class_{c}']:.3f}" for c in range(nc))
        rs = " | ".join(f"{m[f'R_class_{c}']:.3f}" for c in range(nc))
        fs = " | ".join(f"{m[f'F1_class_{c}']:.3f}" for c in range(nc))
        return f"| {name} | {ps} | {rs} | {fs} |"
    lines.append(per_row("raw", raw_test_m))
    lines.append(per_row("post-cal", cal_test_m))
    for name, r in stacker_results.items():
        lines.append(per_row(name, r["test"]))

    lines.append("")
    lines.append("## Confusion matrix (best stacker)")
    best_name = max(stacker_results,
                    key=lambda n: stacker_results[n]["test"]["f1"])
    cm = stacker_results[best_name]["test"]["confusion"]
    lines.append(f"(rows = true, cols = predicted; stacker = `{best_name}`)")
    lines.append("```")
    lines.append("       " + " ".join(f"{c:>10}" for c in classes))
    for i, r in enumerate(cm):
        lines.append(f"{classes[i]:>6}: " + " ".join(f"{v:>10}" for v in r))
    lines.append("```")
    (out_dir / "stacker_eval.md").write_text("\n".join(lines))
    print(f"[stacker] wrote {out_dir}/stacker_eval.md, stacker_meta.json, "
          f"stacker_<name>.joblib, final_test_probs_<name>.npy")


if __name__ == "__main__":
    main()
