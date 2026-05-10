"""Multi-ckpt logit-level ensemble evaluation for HydroHydra.

Phase I multi-seed verification (2026-05-07) found seed-to-seed test/μP
variance of ±0.10 on Lexar's small Test split, while val/μP is stable to
±0.002. Greedy weight-soup fails because the per-seed minima are in
different basins (averaging weights crashes val to ~0.10).

Logit-level ensembling sidesteps the basin problem: each model emits an
independent prediction; we average softmax probabilities. Standard
variance-reduction trick when weight averaging is infeasible.

Usage:
    python -m inference.ensemble_eval \
        --ckpts ckpt1 ckpt2 ckpt3 \
        --data_dir <DATA_DIR> \
        --out_dir lightning_logs/phaseI_multiseed_ensemble
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_hydra import HydroHydra
from inference.calibrate import (
    search_thresholds_macroP_coverage,
    search_thresholds_recall_floor,
)
from inference.selective_pr import selective_pr_curve, write_markdown


@torch.no_grad()
def _collect_probs(model, dataloader, device, num_classes: int):
    """Return per-sample softmax probabilities and labels for one model."""
    model.eval()
    probs_all, targets_all = [], []
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        if out.size(-1) > num_classes:
            out = out[:, :num_classes]
        p = F.softmax(out.float(), dim=-1)
        probs_all.append(p.cpu())
        targets_all.append(y.cpu())
    return torch.cat(probs_all), torch.cat(targets_all)


def _fit_temperature(logits, targets, lr=1e-2, max_iter=200) -> float:
    log_T = nn.Parameter(torch.zeros(1))
    optim = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        optim.zero_grad()
        T = log_T.exp().clamp(min=1e-2, max=100.0)
        loss = F.cross_entropy(logits / T, targets)
        loss.backward()
        return loss

    optim.step(closure)
    return float(log_T.exp().clamp(min=1e-2, max=100.0).item())


def _test_metrics(probs: np.ndarray, targets: np.ndarray, num_classes: int):
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, matthews_corrcoef,
        accuracy_score, roc_auc_score,
    )
    pred = probs.argmax(axis=1)
    out = {
        "acc":      float(accuracy_score(targets, pred)),
        "f1":       float(f1_score(targets, pred, average="macro", zero_division=0)),
        "recall":   float(recall_score(targets, pred, average="macro", zero_division=0)),
        "micro_P":  float(precision_score(targets, pred, average="micro", zero_division=0)),
        "macro_P":  float(precision_score(targets, pred, average="macro", zero_division=0)),
        "mcc":      float(matthews_corrcoef(targets, pred)),
    }
    try:
        out["auroc"] = float(roc_auc_score(targets, probs, multi_class="ovr"))
    except Exception:
        out["auroc"] = float("nan")
    for c in range(num_classes):
        m = pred == c
        n = int(m.sum())
        if n == 0:
            out[f"P_class_{c}"] = float("nan")
        else:
            out[f"P_class_{c}"] = float((targets[m] == c).mean())
    return out


def main():
    p = argparse.ArgumentParser(description="Logit-ensemble eval for HydroHydra")
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--sample_rate", type=int, default=5_120)
    p.add_argument("--fixed_len",   type=int, default=5_120)
    p.add_argument("--target_coverage", type=float, default=0.85)
    p.add_argument("--recall_floor",    type=float, default=0.6)
    p.add_argument("--selective_coverages", type=str,
                   default="0.70,0.75,0.80,0.85,0.90,0.95")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    if not args.data_dir:
        raise ValueError("Set --data_dir or DATA_DIR")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
    )
    dm.setup()
    num_classes = dm.num_classes

    print(f"[ensemble] loading {len(args.ckpts)} checkpoints …")
    models = []
    for ck in args.ckpts:
        m = HydroHydra.load_from_checkpoint(ck, map_location=device, strict=False)
        m = m.to(device).eval()
        models.append((Path(ck).name, m))

    # Per-model probs on val and test
    val_probs_by_model = []
    val_targets = None
    print("[ensemble] running each model over val + test …")
    for name, m in models:
        vp, vy = _collect_probs(m, dm.val_dataloader(), device, num_classes)
        val_probs_by_model.append(vp)
        val_targets = vy
        single_val_mp = _test_metrics(vp.numpy(), vy.numpy(), num_classes)["macro_P"]
        print(f"  {name}: solo val/μP = {single_val_mp:.4f}")

    test_probs_by_model = []
    test_targets = None
    for name, m in models:
        tp, ty = _collect_probs(m, dm.test_dataloader(), device, num_classes)
        test_probs_by_model.append(tp)
        test_targets = ty
        single_test_mp = _test_metrics(tp.numpy(), ty.numpy(), num_classes)["macro_P"]
        print(f"  {name}: solo test/μP = {single_test_mp:.4f}")

    # Average probs across models
    val_probs_ens = torch.stack(val_probs_by_model, dim=0).mean(dim=0)  # (N,C)
    test_probs_ens = torch.stack(test_probs_by_model, dim=0).mean(dim=0)

    # Eval ensemble
    val_metrics = _test_metrics(val_probs_ens.numpy(),
                                val_targets.numpy(), num_classes)
    test_metrics = _test_metrics(test_probs_ens.numpy(),
                                 test_targets.numpy(), num_classes)
    print(f"[ensemble] val/μP = {val_metrics['macro_P']:.4f}")
    print(f"[ensemble] test/μP = {test_metrics['macro_P']:.4f}")

    # Optional: temperature scaling on the *ensemble* val probs
    # Convert probs back to log-probs (= log p) for temperature fitting.
    val_logits_ens = val_probs_ens.clamp(min=1e-8).log()
    T = _fit_temperature(val_logits_ens, val_targets)
    print(f"[ensemble] temperature (post-ensemble) = {T:.3f}")
    torch.save({"T": T}, out_dir / "temperature.pt")

    test_logits_ens = test_probs_ens.clamp(min=1e-8).log()
    test_probs_T = F.softmax(test_logits_ens / T, dim=-1).numpy()
    test_metrics_T = _test_metrics(test_probs_T, test_targets.numpy(), num_classes)
    print(f"[ensemble] test/μP (post-T) = {test_metrics_T['macro_P']:.4f}")

    # Selective-PR curve on val (post-T)
    val_probs_T = F.softmax(val_logits_ens / T, dim=-1).numpy()
    coverages = [float(c) for c in args.selective_coverages.split(",")]
    pr_rows = selective_pr_curve(
        val_probs_T, val_targets.numpy(), num_classes, coverages,
    )
    class_names = [dm.idx_to_class[i] for i in range(num_classes)]
    write_markdown(out_dir / "selective_pr.md", pr_rows, num_classes, class_names)

    # Macro-P @ target coverage thresholds
    th = search_thresholds_macroP_coverage(
        val_probs_T, val_targets.numpy(), num_classes,
        target_coverage=args.target_coverage,
    )
    rf = search_thresholds_recall_floor(
        val_probs_T, val_targets.numpy(), num_classes,
        recall_floor=args.recall_floor,
    )
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump({"macroP_coverage": th, "recall_floor": rf}, f, indent=2)

    # Markdown summary
    summary = []
    summary.append("# Multi-seed logit-level ensemble report\n")
    summary.append(f"## Inputs ({len(args.ckpts)} ckpts)\n")
    for ck in args.ckpts:
        summary.append(f"- `{ck}`")
    summary.append("\n## Per-model + ensemble metrics\n")
    summary.append("| model | val/μP | test/μP |")
    summary.append("|---|---|---|")
    for (name, _), vp, tp in zip(models, val_probs_by_model, test_probs_by_model):
        v = _test_metrics(vp.numpy(), val_targets.numpy(), num_classes)["macro_P"]
        t = _test_metrics(tp.numpy(), test_targets.numpy(), num_classes)["macro_P"]
        summary.append(f"| {name} | {v:.4f} | {t:.4f} |")
    summary.append(f"| **ensemble (mean prob)** | **{val_metrics['macro_P']:.4f}** | **{test_metrics['macro_P']:.4f}** |")
    summary.append(f"| **ensemble + post-T** | (val same) | **{test_metrics_T['macro_P']:.4f}** |")
    summary.append("")
    summary.append("## Ensemble test metrics (post-T)\n")
    summary.append(f"- macro-precision: {test_metrics_T['macro_P']:.4f}")
    summary.append(f"- accuracy:        {test_metrics_T['acc']:.4f}")
    summary.append(f"- F1:              {test_metrics_T['f1']:.4f}")
    summary.append(f"- recall:          {test_metrics_T['recall']:.4f}")
    summary.append(f"- MCC:             {test_metrics_T['mcc']:.4f}")
    summary.append(f"- AUROC:           {test_metrics_T['auroc']:.4f}")
    for c in range(num_classes):
        summary.append(f"- P[class_{c}]:      {test_metrics_T[f'P_class_{c}']:.4f}")
    summary.append("")
    summary.append(f"## Calibration\n- temperature: {T:.3f}")
    summary.append(f"- MP@cov{args.target_coverage}: {th['macro_precision']:.4f}")
    summary.append(f"- recall_floor feasible: {rf['feasible']}")
    (out_dir / "ensemble.md").write_text("\n".join(summary))

    print(f"[ensemble] wrote → {out_dir}/ensemble.md")
    print(f"[ensemble] wrote → {out_dir}/temperature.pt")
    print(f"[ensemble] wrote → {out_dir}/thresholds.json")
    print(f"[ensemble] wrote → {out_dir}/selective_pr.md")


if __name__ == "__main__":
    main()
