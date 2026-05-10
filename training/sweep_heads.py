"""Phase A — head-only sweep on cached HydroPreciseV2 embeddings.

Two subcommands:

  ``extract_features`` — load a HydroPreciseV2 checkpoint, run train/val/test
    splits through ``_features(x)`` once, and cache the results. Augmentation
    is OFF (model is in ``eval()`` mode).

  ``train`` — for each (head_type × feature_norm) config in the grid, train a
    fresh head on the cached embeddings (tiny PyTorch loop, no Lightning) and
    record val metrics. Writes a CSV.

Both subcommands tag cache files by the SHA-8 of the source checkpoint, so a
mismatched ckpt + cache combination raises rather than silently using a stale
cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score,
    MulticlassMatthewsCorrCoef, MulticlassPrecision,
)

from data.audio_lightning_loader import DALIAudioDataModule
from models.heads import HEAD_NAMES, build_head, PrototypeHead
from models.hydro_precise_v2 import HydroPreciseV2
from models.hydro_conformer import FocalLoss


# ── Cache helpers ───────────────────────────────────────────────────────────

def _ckpt_sha(ckpt_path: str) -> str:
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _cache_dir(out_dir: Path, sha: str) -> Path:
    d = out_dir / sha
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Extract features ────────────────────────────────────────────────────────

@torch.no_grad()
def _extract_split(model: HydroPreciseV2, loader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, labs = [], []
    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        f = model._features(x).float().cpu()
        feats.append(f)
        labs.append(y.cpu())
    return torch.cat(feats), torch.cat(labs)


def cmd_extract(args):
    ckpt = args.ckpt
    sha = _ckpt_sha(ckpt)
    out = _cache_dir(Path(args.out_dir), sha)
    print(f"[extract] ckpt sha={sha}  out={out}")

    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        oversample_train=False,                       # cache raw distribution
    )
    data.setup()
    print(f"[extract] {data.num_classes} classes, "
          f"class_weights={[round(w, 3) for w in data.class_weights]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HydroPreciseV2.load_from_checkpoint(ckpt, map_location=device, strict=False)
    model = model.to(device).eval()

    splits = {
        "train": data.train_dataloader(),
        "val":   data.val_dataloader(),
        "test":  data.test_dataloader(),
    }
    metadata = {
        "ckpt_sha":      sha,
        "ckpt_path":     ckpt,
        "num_classes":   data.num_classes,
        "class_weights": data.class_weights,
        "class_to_idx":  data.class_to_idx,
    }
    torch.save(metadata, out / "metadata.pt")

    for name, loader in splits.items():
        t0 = time.time()
        feats, labs = _extract_split(model, loader, device)
        torch.save({"feats": feats, "labels": labs, "ckpt_sha": sha},
                   out / f"{name}.pt")
        print(f"[extract] {name}: feats={tuple(feats.shape)} "
              f"labels={tuple(labs.shape)}  ({time.time()-t0:.1f}s)")
    print(f"[extract] done. Cache at {out}")


# ── Loss to match the original training conditions ─────────────────────────

class _LogitAdjustedFocal(nn.Module):
    """Focal loss + logit adjustment + label smoothing — mirrors the original
    training recipe so a head's Phase A score is comparable to the joint-trained
    baseline."""

    def __init__(self, num_classes: int, class_weights, focal_gamma: float,
                 label_smoothing: float, logit_adjust_tau: float):
        super().__init__()
        self.focal = FocalLoss(class_weights=class_weights, gamma=focal_gamma,
                               label_smoothing=label_smoothing)
        self.logit_adjust_tau = float(logit_adjust_tau)
        if class_weights is not None and self.logit_adjust_tau > 0:
            cw = torch.tensor(class_weights, dtype=torch.float32)
            prior = 1.0 / cw
            prior = prior / prior.sum()
            self.register_buffer("log_prior", torch.log(prior + 1e-12))
        else:
            self.register_buffer("log_prior", torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Logit adjustment is applied at *evaluation* in the parent model, so
        # the loss is computed on raw logits — same as production.
        return self.focal(logits, y)


# ── Tiny train-eval loop on cached embeddings ──────────────────────────────

def _make_metrics(num_classes: int, device: str) -> Dict[str, nn.Module]:
    return {
        "val_acc":    MulticlassAccuracy(num_classes=num_classes, average="macro").to(device),
        "val_f1":     MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
        "val_micro":  MulticlassPrecision(num_classes=num_classes, average="micro").to(device),
        "val_macro":  MulticlassPrecision(num_classes=num_classes, average="macro").to(device),
        "val_mcc":    MulticlassMatthewsCorrCoef(num_classes=num_classes).to(device),
        "val_auroc":  MulticlassAUROC(num_classes=num_classes).to(device),
    }


def _reset_metrics(ms): [m.reset() for m in ms.values()]


def _compute_metrics(ms, val_logits, val_y, val_probs) -> Dict[str, float]:
    ms["val_acc"  ](val_logits, val_y)
    ms["val_f1"   ](val_logits, val_y)
    ms["val_micro"](val_logits, val_y)
    ms["val_macro"](val_logits, val_y)
    ms["val_mcc"  ](val_logits, val_y)
    ms["val_auroc"](val_probs,  val_y)
    return {k: float(m.compute()) for k, m in ms.items()}


def _train_head(
    head: nn.Module, feature_norm: str,
    train_feats: torch.Tensor, train_y: torch.Tensor,
    val_feats:   torch.Tensor, val_y:   torch.Tensor,
    *, num_classes: int, class_weights, epochs: int, batch_size: int,
    lr: float, weight_decay: float, focal_gamma: float, label_smoothing: float,
    logit_adjust_tau: float, head_uses_labels: bool, device: str,
) -> Tuple[Dict[str, float], int, float]:
    """Train ``head`` on cached features, return best val metrics, n_params, seconds."""
    head = head.to(device)
    # Optional LayerNorm + L2 on the embedding before the head (a fresh module
    # so the head sees the same input shape as the joint-trained variant).
    norm: nn.Module = nn.Identity()
    if feature_norm == "layernorm_l2":
        norm = nn.LayerNorm(train_feats.shape[1]).to(device)

    crit = _LogitAdjustedFocal(num_classes=num_classes, class_weights=class_weights,
                               focal_gamma=focal_gamma, label_smoothing=label_smoothing,
                               logit_adjust_tau=logit_adjust_tau).to(device)
    params = list(head.parameters()) + list(norm.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_train = train_feats.size(0)
    train_feats_d = train_feats.to(device)
    train_y_d     = train_y.to(device)
    val_feats_d   = val_feats.to(device)
    val_y_d       = val_y.to(device)

    metrics = _make_metrics(num_classes, device)
    best = {"val_micro": -1.0}
    best_full = {}

    t0 = time.time()
    for ep in range(epochs):
        head.train()
        if not isinstance(norm, nn.Identity):
            norm.train()
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            x = train_feats_d[idx]
            y = train_y_d[idx]
            x = norm(x)
            if feature_norm == "layernorm_l2":
                x = F.normalize(x, dim=-1)
            logits = head(x, y) if head_uses_labels else head(x, None)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        head.eval()
        if not isinstance(norm, nn.Identity):
            norm.eval()
        with torch.no_grad():
            xv = norm(val_feats_d)
            if feature_norm == "layernorm_l2":
                xv = F.normalize(xv, dim=-1)
            # Eval: ArcFace also gets labels=None to disable margin.
            logits_v = head(xv, None)
            probs_v = F.softmax(logits_v, dim=-1)

        _reset_metrics(metrics)
        out = _compute_metrics(metrics, logits_v, val_y_d, probs_v)
        if out["val_micro"] > best["val_micro"]:
            best = out
            best_full = out

    n_params = sum(p.numel() for p in params)
    elapsed = time.time() - t0
    return best_full, n_params, elapsed


# ── Sweep ───────────────────────────────────────────────────────────────────

def _grid() -> List[Tuple[str, str]]:
    """Return list of (head_type, feature_norm) configs to evaluate."""
    out = [("mlp", "none")]                                     # baseline
    for h in ("cosine", "prototype", "arcface", "mlp_wide"):
        for fn in ("none", "layernorm_l2"):
            out.append((h, fn))
    return out


def cmd_train(args):
    ckpt = args.ckpt
    sha = _ckpt_sha(ckpt)
    cache = Path(args.cache_dir) / sha
    if not cache.exists():
        raise FileNotFoundError(
            f"No cache for ckpt sha {sha} at {cache}. "
            f"Run `extract_features` first."
        )
    meta = torch.load(cache / "metadata.pt", weights_only=False)
    if meta["ckpt_sha"] != sha:
        raise RuntimeError(f"Cache metadata sha {meta['ckpt_sha']} != ckpt sha {sha}")
    num_classes = meta["num_classes"]
    class_weights = meta["class_weights"]

    train_blob = torch.load(cache / "train.pt", weights_only=False)
    val_blob   = torch.load(cache / "val.pt",   weights_only=False)
    train_feats, train_y = train_blob["feats"], train_blob["labels"]
    val_feats,   val_y   = val_blob["feats"],   val_blob["labels"]
    in_dim = train_feats.shape[1]
    print(f"[train] cache sha={sha}  in_dim={in_dim}  "
          f"train={tuple(train_feats.shape)}  val={tuple(val_feats.shape)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    grid = _grid()
    for head_type, feature_norm in grid:
        head = build_head(
            name=head_type,
            in_dim=in_dim,
            num_classes=num_classes,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            arcface_margin=args.arcface_margin,
            arcface_scale=args.arcface_scale,
            cosine_scale_init=args.cosine_scale_init,
        )
        if head_type == "prototype":
            with torch.no_grad():
                cents = torch.stack([
                    train_feats[train_y == c].mean(dim=0) for c in range(num_classes)
                ])
                head.init_from_centroids(cents)

        best, n_params, elapsed = _train_head(
            head, feature_norm,
            train_feats, train_y, val_feats, val_y,
            num_classes=num_classes, class_weights=class_weights,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
            logit_adjust_tau=args.logit_adjust_tau,
            head_uses_labels=(head_type == "arcface"),
            device=device,
        )
        row = {
            "head":            head_type,
            "feature_norm":    feature_norm,
            "val_micro_p":     round(best["val_micro"], 4),
            "val_macro_p":     round(best["val_macro"], 4),
            "val_f1":          round(best["val_f1"],    4),
            "val_acc":         round(best["val_acc"],   4),
            "val_mcc":         round(best["val_mcc"],   4),
            "val_auroc":       round(best["val_auroc"], 4),
            "params":          n_params,
            "train_seconds":   round(elapsed, 1),
            "ckpt_sha":        sha,
        }
        rows.append(row)
        print(f"[train] {head_type:10s} {feature_norm:13s}  "
              f"μP={row['val_micro_p']:.4f} MP={row['val_macro_p']:.4f} "
              f"f1={row['val_f1']:.4f} mcc={row['val_mcc']:.4f}  "
              f"params={n_params:>7d}  ({elapsed:.1f}s)")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[train] wrote {out_csv}")

    # Print top-3 by val_micro_p
    rows.sort(key=lambda r: r["val_micro_p"], reverse=True)
    print("\nTop 3 by val/micro_precision:")
    for r in rows[:3]:
        print(f"  {r['head']:10s} {r['feature_norm']:13s}  μP={r['val_micro_p']:.4f}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract_features")
    pe.add_argument("--ckpt", required=True)
    pe.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    pe.add_argument("--batch_size", type=int, default=64)
    pe.add_argument("--num_threads", type=int, default=8)
    pe.add_argument("--sample_rate", type=int, default=5_120)
    pe.add_argument("--fixed_len", type=int, default=5_120)
    pe.add_argument("--out_dir", default="lightning_logs/sweep_heads/cache")
    pe.set_defaults(fn=cmd_extract)

    pt = sub.add_parser("train")
    pt.add_argument("--ckpt", required=True)
    pt.add_argument("--cache_dir", default="lightning_logs/sweep_heads/cache")
    pt.add_argument("--out", default="lightning_logs/sweep_heads/results.csv")
    pt.add_argument("--epochs", type=int, default=30)
    pt.add_argument("--batch_size", type=int, default=512)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--weight_decay", type=float, default=1e-4)
    pt.add_argument("--fusion_dim", type=int, default=128,
                    help="Hidden dim for MLP heads (matches winner ckpt fusion_dim).")
    pt.add_argument("--dropout", type=float, default=0.10)
    pt.add_argument("--focal_gamma", type=float, default=2.0)
    pt.add_argument("--label_smoothing", type=float, default=0.05)
    pt.add_argument("--logit_adjust_tau", type=float, default=0.5)
    pt.add_argument("--arcface_margin", type=float, default=0.2)
    pt.add_argument("--arcface_scale", type=float, default=30.0)
    pt.add_argument("--cosine_scale_init", type=float, default=10.0)
    pt.set_defaults(fn=cmd_train)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
