"""Standalone post-hoc calibration + test eval on a single Lightning ckpt.

Mirrors the post-cal block in ``training/train_precise_v2.py:main()`` so we
can re-run calibration on a chosen checkpoint without retraining. Writes
``temperature.pt``, ``thresholds.json``, ``selective_pr.md``, and
``test_metrics.md`` next to the ckpt.

Usage:
    python -m inference.postcal_one --ckpt <path>.ckpt \
        --data_dir <DATA_DIR> \
        [--rms_normalize --target_rms 1.0 --hpf_hz 20.0 ...]

The data_dir + preprocessing flags must match how the model was *trained*
or the temperature/thresholds will be miscalibrated. Inference-time
augmentation flags (RIR/pitch/etc.) are off by default — only the
preprocessing pipeline matters.
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
from models.hydro_precise_v2 import HydroPreciseV2
from models.hydro_hydra import HydroHydra
from inference.calibrate import (
    search_thresholds_macroP_coverage,
    search_thresholds_recall_floor,
)
from inference.selective_pr import selective_pr_curve, write_markdown


_MODEL_DISPATCH = {
    "precise": HydroPreciseV2,
    "hydra":   HydroHydra,
}


@torch.no_grad()
def _collect(model, dataloader, device, num_classes: int,
             tta_k: int = 1, tta_noise_snr: float = 25.0,
             tta_gain_range: float = 0.1):
    """Collect class-only logits.

    The HydroHydra MLP+Gamblers head emits ``num_classes+1`` logits; the last
    is the abstention column and is sliced off here so calibration is
    class-only.

    Phase I α.4 — Test-Time Augmentation:
    when ``tta_k > 1`` we run K augmented forward passes per batch
    (Gaussian noise at the supplied SNR + small random gain) and average
    softmax probabilities, then take the log to recover an "averaged
    logit" that downstream temperature scaling can consume cleanly.
    K=1 is the legacy single-pass behaviour.
    """
    import math
    model.eval()
    logits_all, targets_all = [], []
    snr_lin = 10 ** (tta_noise_snr / 20.0)
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if tta_k <= 1:
            out = model(x)
            if out.size(-1) > num_classes:
                out = out[:, :num_classes]
        else:
            probs_sum = None
            for _ in range(tta_k):
                # Random gain in [1-r, 1+r]; Gaussian noise scaled to SNR.
                gain = 1.0 + (torch.rand(x.size(0), 1, device=device) * 2 - 1) * tta_gain_range
                rms = x.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
                n = torch.randn_like(x) * (rms / snr_lin)
                x_aug = x * gain + n
                out = model(x_aug)
                if out.size(-1) > num_classes:
                    out = out[:, :num_classes]
                p = F.softmax(out.float(), dim=-1)
                probs_sum = p if probs_sum is None else probs_sum + p
            mean_p = (probs_sum / tta_k).clamp(min=1e-8)
            out = mean_p.log()  # log-prob ≈ logit + const, OK for T fitting
        logits_all.append(out.cpu())
        targets_all.append(y.cpu())
    return torch.cat(logits_all), torch.cat(targets_all)


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
        precision_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score,
    )
    pred = probs.argmax(axis=1)
    out = {
        "acc":      float(accuracy_score(targets, pred)),
        "f1":       float(f1_score(targets, pred, average="macro", zero_division=0)),
        "micro_P":  float(precision_score(targets, pred, average="micro", zero_division=0)),
        "macro_P":  float(precision_score(targets, pred, average="macro", zero_division=0)),
        "mcc":      float(matthews_corrcoef(targets, pred)),
    }
    try:
        out["auroc"] = float(roc_auc_score(targets, probs, multi_class="ovr"))
    except Exception:
        out["auroc"] = float("nan")
    # Per-class precision
    for c in range(num_classes):
        m = pred == c
        if m.sum() == 0:
            out[f"P_class_{c}"] = float("nan")
        else:
            out[f"P_class_{c}"] = float((targets[m] == c).mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", default="precise", choices=sorted(_MODEL_DISPATCH))
    p.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--sample_rate", type=int, default=5_120)
    p.add_argument("--fixed_len",   type=int, default=5_120)
    # Preprocessing flags (must match training)
    p.add_argument("--rms_normalize", action="store_true")
    p.add_argument("--target_rms",   type=float, default=0.1)
    p.add_argument("--hpf_hz",       type=float, default=0.0)
    p.add_argument("--hpf_order",    type=int,   default=4)
    p.add_argument("--denoise",      default="off")
    # Post-cal flags
    p.add_argument("--target_coverage", type=float, default=0.85)
    p.add_argument("--recall_floor",    type=float, default=0.6)
    p.add_argument("--selective_coverages", type=str,
                   default="0.70,0.75,0.80,0.85,0.90,0.95")
    p.add_argument("--out_dir", default="",
                   help="Where to write artifacts (defaults to ckpt's directory).")
    # Phase I α.4 — TTA at inference
    p.add_argument("--tta_k",         type=int, default=1,
                   help="K augmented forward passes per batch; K=1 disables TTA.")
    p.add_argument("--tta_noise_snr", type=float, default=25.0)
    p.add_argument("--tta_gain_range", type=float, default=0.1)
    args = p.parse_args()

    if not args.data_dir:
        raise ValueError("Set --data_dir or DATA_DIR")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size, num_threads=args.num_threads,
        target_sr=args.sample_rate, fixed_len=args.fixed_len,
        oversample_train=False, denoise_method=args.denoise,
        rms_normalize=args.rms_normalize, target_rms=args.target_rms,
        hpf_hz=args.hpf_hz, hpf_order=args.hpf_order,
    )
    data.setup()
    nc = data.num_classes
    class_names = [data.idx_to_class[i] for i in range(nc)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = _MODEL_DISPATCH[args.model]
    model = cls.load_from_checkpoint(
        args.ckpt, map_location=device, strict=False,
    ).to(device).eval()

    print(f"[postcal] model = {args.model}  ckpt = {args.ckpt}")
    print(f"[postcal] writing to {out_dir}")

    # ── Val: collect logits, fit T, sweep thresholds + selective PR ──
    print("[postcal] collecting val logits …")
    v_logits, v_targets = _collect(model, data.val_dataloader(), device, nc,
                                   tta_k=args.tta_k,
                                   tta_noise_snr=args.tta_noise_snr,
                                   tta_gain_range=args.tta_gain_range)
    T = _fit_temperature(v_logits, v_targets)
    v_probs = F.softmax(v_logits / T, dim=-1).numpy()
    v_y     = v_targets.numpy()
    print(f"  temperature = {T:.4f}")

    res_macro = search_thresholds_macroP_coverage(
        v_probs, v_y, num_classes=nc, target_coverage=args.target_coverage,
    )
    print(f"  macroP@cov{args.target_coverage}: macro_P={res_macro['macro_precision']:.4f} "
          f"thresholds={res_macro['thresholds']}")

    res_floor = search_thresholds_recall_floor(
        v_probs, v_y, num_classes=nc, recall_floor=args.recall_floor,
    )
    tag = "OK" if res_floor["feasible"] else "INFEASIBLE"
    print(f"  recall_floor={args.recall_floor}: {tag}  micro_P={res_floor.get('micro_precision', 0):.4f}")

    coverages = [float(x) for x in args.selective_coverages.split(",") if x.strip()]
    sel_rows = selective_pr_curve(v_probs, v_y, num_classes=nc, coverages=coverages)

    # ── Test: collect predictions, dump headline metrics ──
    print("[postcal] collecting test logits …")
    t_logits, t_targets = _collect(model, data.test_dataloader(), device, nc,
                                   tta_k=args.tta_k,
                                   tta_noise_snr=args.tta_noise_snr,
                                   tta_gain_range=args.tta_gain_range)
    t_probs = F.softmax(t_logits / T, dim=-1).numpy()
    t_y     = t_targets.numpy()
    test_m  = _test_metrics(t_probs, t_y, nc)
    print(f"  test/macro_P = {test_m['macro_P']:.4f}  "
          f"test/micro_P = {test_m['micro_P']:.4f}  "
          f"f1 = {test_m['f1']:.4f}  mcc = {test_m['mcc']:.4f}  auroc = {test_m['auroc']:.4f}")

    # ── Write artifacts ──
    torch.save({"temperature": T}, out_dir / "temperature.pt")
    thr_payload = {
        "macroP_coverage": res_macro,
        "recall_floor":    res_floor,
    }
    (out_dir / "thresholds.json").write_text(json.dumps(thr_payload, indent=2))
    write_markdown(out_dir / "selective_pr.md", sel_rows,
                   num_classes=nc, class_names=class_names)

    md = ["# Test-set headline metrics", "",
          f"ckpt: `{Path(args.ckpt).name}`", f"temperature: {T:.4f}", ""]
    md.append("| metric | value |")
    md.append("|---|---:|")
    for k in ("micro_P", "macro_P", "f1", "acc", "mcc", "auroc"):
        md.append(f"| test/{k} | {test_m[k]:.4f} |")
    md.append("")
    md.append("| class | per-class precision |")
    md.append("|---|---:|")
    for c in range(nc):
        md.append(f"| {class_names[c]} | {test_m[f'P_class_{c}']:.4f} |")
    (out_dir / "test_metrics.md").write_text("\n".join(md) + "\n")

    print(f"\n  saved → {out_dir / 'temperature.pt'}")
    print(f"  saved → {out_dir / 'thresholds.json'}")
    print(f"  saved → {out_dir / 'selective_pr.md'}")
    print(f"  saved → {out_dir / 'test_metrics.md'}")


if __name__ == "__main__":
    main()
