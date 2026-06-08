"""④ AST-10s-B — MLMC (multi-label) training variant.

Same AST fine-tuning recipe as ``campaign/train_ast2.py`` (AudioSet-pretrained
ViT, diff-LR AdamW, cosine, SpecAugment, balanced sampler) with one structural
change: the 4-way **softmax + CE** head becomes 4 **independent sigmoid heads +
BCEWithLogitsLoss**, so the model predicts each class independently and emits a
one-hot / multi-hot row.

Key differences from the single-label trainer
---------------------------------------------
- targets are multi-hot float vectors (single-label data → one positive bit);
- ``BCEWithLogitsLoss(pos_weight=...)`` (focal-BCE option) replaces CE/focal-CE;
  ``pos_weight`` = inverse-frequency, the multi-label analogue of class weights;
- mixup mixes the multi-hot targets directly (no two-term CE);
- epoch selection on **clip-level multi-label macro-F1** at 0.5 (low variance —
  same rationale as the original's clip-level selection);
- after training, per-class thresholds are tuned on val sigmoid scores and a
  one-hot val prediction + thresholds are saved for downstream source dumping.

Honest contract: Test split never read here (use a dump script for test, as in
the original pipeline). Outputs: ``ast_best.pt``, ``val_onehot.npy``,
``thresholds.json``, ``result.json``.

Run (10s build):
  python campaign/mlmc/train_ast2_mlmc.py \
      --data_root /var/.../Combined_IARA_Deepship_10s \
      --clip_len 160000 --src_sr 16000 --loss focal --class_weight --specaug \
      --mixup 0 --lr_head 3e-4 --seed 2024 --out lightning_logs/mlmc/ast10b
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# repo root + campaign on path for ast_common / mlmc helpers
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[1]))            # campaign/
from ast_common import collect_clips, wav_to_fbank, load_wav, CLASSES  # noqa: E402
from data.mlmc_windowed_loader import (              # noqa: E402
    tune_thresholds_per_class, probs_to_onehot, multilabel_report,
)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class MultiHotClipDS(Dataset):
    """fbank + multi-hot float target (single-label data → one-hot)."""

    def __init__(self, paths, labels, clip_len, src_sr, n_cls):
        self.paths, self.labels = paths, labels
        self.clip_len, self.src_sr, self.n_cls = clip_len, src_sr, n_cls

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        fb = wav_to_fbank(load_wav(self.paths[i], self.clip_len), self.src_sr)
        y = torch.zeros(self.n_cls)
        y[int(self.labels[i])] = 1.0
        return fb, y


def bce_loss(logits, target, args, pos_weight):
    """BCE (optionally focal-BCE) with per-class pos_weight."""
    if args.loss == "focal":
        # focal binary cross-entropy (per-class), then pos_weight reweight.
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=pos_weight)
        pt = target * p + (1 - target) * (1 - p)
        return ((1 - pt) ** args.focal_gamma * ce).mean()
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight)


def spec_augment(fb, n_freq=2, n_time=2, fw=16, tw=80):
    B, T, M = fb.shape
    for _ in range(n_freq):
        f = random.randint(0, fw); f0 = random.randint(0, max(0, M - f))
        fb[:, :, f0:f0 + f] = 0.0
    for _ in range(n_time):
        t = random.randint(0, tw); t0 = random.randint(0, max(0, T - t))
        fb[:, t0:t0 + t, :] = 0.0
    return fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--clip_len", type=int, required=True)
    ap.add_argument("--src_sr", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--train_cap", type=int, default=60)
    ap.add_argument("--val_cap", type=int, default=40)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--class_weight", action="store_true",
                    help="inverse-frequency pos_weight in BCE")
    ap.add_argument("--specaug", action="store_true")
    ap.add_argument("--single_label", action="store_true",
                    help="strict argmax one-hot for val predictions (single-"
                         "label data); epoch selection also uses argmax.")
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--model_name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    set_seed(args.seed)
    dev = torch.device("cuda")
    nC = len(CLASSES)
    root = Path(args.data_root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[args] {vars(args)}", flush=True)

    tr_p, tr_y, _ = collect_clips(root, "Train", args.train_cap, args.seed)
    va_p, va_y, va_s = collect_clips(root, "Val", args.val_cap, args.seed)
    print(f"[data] train clips={len(tr_p)} val clips={len(va_p)} "
          f"train counts={np.bincount(tr_y).tolist()}", flush=True)

    cls_cnt = np.bincount(tr_y, minlength=nC).astype(np.float64)
    pos_weight = None
    if args.class_weight:
        neg = cls_cnt.sum() - cls_cnt
        pw = neg / np.maximum(cls_cnt, 1)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=dev)
        print(f"[loss] pos_weight={pos_weight.tolist()}", flush=True)
    sample_w = (1.0 / np.maximum(cls_cnt, 1))[tr_y]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                    num_samples=len(tr_y), replacement=True)
    tr_dl = DataLoader(MultiHotClipDS(tr_p, tr_y, args.clip_len, args.src_sr, nC),
                       batch_size=args.batch, sampler=sampler,
                       num_workers=args.workers, pin_memory=True, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
    va_dl = DataLoader(MultiHotClipDS(va_p, va_y, args.clip_len, args.src_sr, nC),
                       batch_size=args.batch * 2, shuffle=False,
                       num_workers=args.workers, pin_memory=True,
                       persistent_workers=True, prefetch_factor=4)

    from transformers import ASTForAudioClassification
    model = ASTForAudioClassification.from_pretrained(
        args.model_name, num_labels=nC, ignore_mismatched_sizes=True,
        problem_type="multi_label_classification").to(dev)

    head_ids = {id(p) for n, p in model.named_parameters() if "classifier" in n}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr_backbone},
                             {"params": head, "lr": args.lr_head}],
                            weight_decay=args.wd)
    steps = len(tr_dl) * args.epochs
    warm = int(steps * args.warmup)

    def lr_scale(s):
        if s < warm:
            return s / max(1, warm)
        prog = (s - warm) / max(1, steps - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))

    scaler = torch.amp.GradScaler("cuda")
    gstep, best_f1, best_ep = 0, -1.0, -1
    va_multihot = np.zeros((len(va_y), nC), dtype=np.float32)
    va_multihot[np.arange(len(va_y)), va_y] = 1.0

    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for fb, y in tr_dl:
            fb = fb.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            if args.specaug:
                fb = spec_augment(fb)
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm = torch.randperm(fb.size(0), device=dev)
                fb = lam * fb + (1 - lam) * fb[perm]
                y = lam * y + (1 - lam) * y[perm]      # mix multi-hot targets
            sc = lr_scale(gstep)
            for g, base in zip(opt.param_groups, (args.lr_backbone, args.lr_head)):
                g["lr"] = base * sc
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_values=fb).logits
                loss = bce_loss(logits, y, args, pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            run += loss.item(); gstep += 1

        # val: collect sigmoid scores, select epoch on clip-level macro-F1@0.5
        model.eval()
        sc_list = []
        with torch.no_grad():
            for fb, _ in va_dl:
                fb = fb.to(dev, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    sc_list.append(torch.sigmoid(
                        model(input_values=fb).logits).float().cpu())
        va_scores = torch.cat(sc_list).numpy()
        pred05 = probs_to_onehot(va_scores, np.full(nC, 0.5), force_one=True,
                                 single_label=args.single_label)
        m05 = _macro_f1(va_multihot, pred05)
        print(f"[ep {ep:02d}] loss={run/len(tr_dl):.4f} "
              f"val_clip_macroF1@0.5={m05:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if m05 > best_f1:
            best_f1, best_ep = m05, ep
            torch.save({"state_dict": model.state_dict(), "args": vars(args),
                        "epoch": ep, "val_macroF1": m05}, out / "ast_best.pt")
            np.save(out / "val_scores.npy", va_scores)
            print(f"  -> saved best (val_clip_macroF1={m05:.4f})", flush=True)

    # final: tune per-class thresholds on val scores of the best epoch, save
    va_scores = np.load(out / "val_scores.npy")
    thr = tune_thresholds_per_class(va_scores, va_multihot)
    val_onehot = probs_to_onehot(va_scores, thr, force_one=True,
                                 single_label=args.single_label)
    print("\n--- VAL (best epoch, tuned thresholds) ---")
    m = multilabel_report("VAL", va_multihot, val_onehot, CLASSES)
    np.save(out / "val_onehot.npy", val_onehot)
    (out / "thresholds.json").write_text(json.dumps(
        {"per_class": dict(zip(CLASSES, thr.tolist()))}, indent=2))
    (out / "result.json").write_text(json.dumps(
        {"best_val_macroF1@0.5": best_f1, "best_epoch": best_ep,
         "val_tuned": m, "args": vars(args)}, indent=2))
    print(f"[done] best val macroF1@0.5={best_f1:.4f} @ ep {best_ep} → {out}")


def _macro_f1(y, p):
    y = y.astype(bool); p = p.astype(bool)
    fs = []
    for c in range(y.shape[1]):
        tp = int((p[:, c] & y[:, c]).sum()); fp = int((p[:, c] & ~y[:, c]).sum())
        fn = int((~p[:, c] & y[:, c]).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(fs))


if __name__ == "__main__":
    main()
