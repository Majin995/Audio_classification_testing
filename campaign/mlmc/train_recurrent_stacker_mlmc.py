"""⑤ HydroRecurrentStacker — MLMC (multi-label) training variant.

Same architecture and frozen 6-ckpt per-clip ensemble features as
``campaign/train_recurrent_stacker.py``, but trained as a **multi-label**
source classifier: the deep-gambler softmax head is dropped (``gambler=False``
→ 4 independent class logits), the loss is ``BCEWithLogitsLoss`` with
inverse-frequency ``pos_weight``, and per-source predictions are emitted as
one-hot / multi-hot via per-class thresholds tuned on val.

Differences from the single-label stacker trainer
--------------------------------------------------
- ``HydroRecurrentStacker(..., gambler=False)`` so ``head`` outputs exactly
  ``num_classes`` logits (no abstain logit — abstention is a single-label
  selective-prediction notion; multi-label abstention = an all-zero row);
- targets are multi-hot; loss = BCE on the final-step logits + ``deep_aux_w`` ·
  BCE over all GRU steps (deep supervision, as before but with BCE);
- model selection on val multi-label macro-F1 (thresholds tuned on val each
  eval); test is scored by ``eval_*`` downstream, not here.

Honest contract: train sources for SGD, val for selection + thresholds; the
frozen ensemble ckpts are read from the cached NPZ. Outputs:
``best.pt``, ``thresholds.json``, ``history.json``, ``result.json``.

Run:
  python campaign/mlmc/train_recurrent_stacker_mlmc.py --steps 8000 \
      --K_train 30 --K_eval 60 --scales 1,3,10 --lr 2e-3 \
      --out_dir lightning_logs/mlmc/recurrent_stacker
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F

from models.hydro_recurrent_stacker import HydroRecurrentStacker
from campaign.train_recurrent import (
    CLASSES, SourceSampler, read_batch_threaded, preprocess, scan_split,
)
from campaign.train_recurrent_stacker import EnsembleCache
from data.mlmc_windowed_loader import (
    tune_thresholds_per_class, probs_to_onehot, multilabel_report,
)


def _multihot(labels, n_cls):
    M = np.zeros((len(labels), n_cls), dtype=np.float32)
    M[np.arange(len(labels)), labels] = 1.0
    return M


@torch.no_grad()
def eval_scores(model, sources_by_class, executor, device, cache, K_eval,
                target_rms=0.1, hpf_hz=20.0, bs=4):
    """Return (src_scores[S,4] sigmoid, src_multihot[S,4])."""
    model.eval()
    items = []
    for ci, c in enumerate(CLASSES):
        for sid, paths in sources_by_class.get(c, {}).items():
            items.append((sid, ci, paths))
    y = _multihot(np.array([it[1] for it in items], dtype=np.int64), len(CLASSES))
    S = len(items)
    scores = np.zeros((S, len(CLASSES)), dtype=np.float32)
    for start in range(0, S, bs):
        chunk = items[start:start + bs]
        clip_lists, masks = [], []
        for _, _, paths in chunk:
            ps = sorted(paths)
            if len(ps) >= K_eval:
                stride = max(1, len(ps) // K_eval)
                clip_lists.append(ps[::stride][:K_eval]); masks.append([True] * K_eval)
            else:
                pad = K_eval - len(ps)
                clip_lists.append(ps + [ps[0]] * pad)
                masks.append([True] * len(ps) + [False] * pad)
        flat = [p for lst in clip_lists for p in lst]
        wav = torch.from_numpy(read_batch_threaded(flat, executor)).to(device)
        wav = preprocess(wav, target_rms, hpf_hz).view(len(chunk), K_eval, -1)
        ens_np = cache.lookup(flat).reshape(len(chunk), K_eval, cache.M, cache.C)
        ens = torch.from_numpy(ens_np).to(device)
        mask = torch.tensor(masks, dtype=torch.bool, device=device)
        logits = model(wav, ens, mask=mask)            # (chunk, 4)
        scores[start:start + len(chunk)] = torch.sigmoid(logits).float().cpu().numpy()
    return scores, y


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/var/mnt/5A009BF8009BD8F9/Data/Combined_IARA_Deepship_1s")
    p.add_argument("--ens_npz", default="campaign/probs_combined_recurrent_stacker.npz")
    p.add_argument("--out_dir", default="lightning_logs/mlmc/recurrent_stacker")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--per_class", type=int, default=4)
    p.add_argument("--K_train", type=int, default=30)
    p.add_argument("--K_eval", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--n_bands", type=int, default=24)
    p.add_argument("--embed_dim", type=int, default=48)
    p.add_argument("--gru_hidden", type=int, default=64)
    p.add_argument("--gru_layers", type=int, default=1)
    p.add_argument("--head_hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--band_dropout", type=float, default=0.10)
    p.add_argument("--tcn_dilations", default="1,4,16,64")
    p.add_argument("--scales", default="1,3,10")
    p.add_argument("--deep_aux_w", type=float, default=0.2)
    p.add_argument("--class_weight_pow", type=float, default=0.5)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--target_rms", type=float, default=0.1)
    p.add_argument("--hpf_hz", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_workers", type=int, default=12)
    p.add_argument("--exclude_iara_glider", action="store_true")
    p.add_argument("--single_label", action="store_true",
                   help="strict argmax one-hot for val predictions (single-"
                        "label data); ignores tuned per-class thresholds.")
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    nC = len(CLASSES)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    train_src = scan_split(args.data_dir, "Train", exclude_iara_glider=args.exclude_iara_glider)
    val_src = scan_split(args.data_dir, "Val", exclude_iara_glider=args.exclude_iara_glider)
    for lbl, d in (("train", train_src), ("val", val_src)):
        print(f"{lbl:>5s} sources: { {k: len(d[k]) for k in CLASSES} }")

    cache = EnsembleCache(Path(args.ens_npz), Path(args.data_dir))
    sampler = SourceSampler(train_src, args.per_class, args.K_train,
                            random.Random(args.seed))

    tcn = tuple(int(d) for d in args.tcn_dilations.split(","))
    scales = tuple(int(s) for s in args.scales.split(","))
    model = HydroRecurrentStacker(
        num_classes=nC, n_bands=args.n_bands, embed_dim=args.embed_dim,
        ens_n_ckpts=cache.M, ens_n_classes=cache.C,
        gru_hidden=args.gru_hidden, gru_layers=args.gru_layers,
        head_hidden=args.head_hidden, dropout=args.dropout,
        band_dropout=args.band_dropout, tcn_dilations=tcn,
        scales=scales, gambler=False,        # ← multi-label: no abstain logit
    ).to(device)
    print(f"HydroRecurrentStacker params = {model.n_params():,}  scales={scales}")

    # pos_weight = (#neg/#pos) over sources, softened by class_weight_pow.
    counts = np.array([len(train_src[c]) for c in CLASSES], dtype=np.float64)
    neg = counts.sum() - counts
    pw = (neg / np.maximum(counts, 1)) ** args.class_weight_pow
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
    print(f"pos_weight: {dict(zip(CLASSES, [f'{w:.2f}' for w in pw]))}")

    def bce(logits, target):
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    executor = ThreadPoolExecutor(max_workers=args.n_workers)
    best_val, best_step, patience_left = -1.0, -1, args.patience
    history = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        labels, clip_paths = sampler.draw()
        flat = [p for lst in clip_paths for p in lst]
        wav = torch.from_numpy(read_batch_threaded(flat, executor)).to(device, non_blocking=True)
        wav = preprocess(wav, args.target_rms, args.hpf_hz)
        N, K = len(labels), args.K_train
        wav = wav.view(N, K, -1)
        ens_np = cache.lookup(flat).reshape(N, K, cache.M, cache.C)
        ens = torch.from_numpy(ens_np).to(device, non_blocking=True)
        target = torch.from_numpy(_multihot(labels, nC)).to(device)

        logits_seq = model(wav, ens, mask=None, return_seq=True)   # (N,K_total,4)
        loss = bce(logits_seq[:, K - 1], target)
        if args.deep_aux_w > 0:
            Kt = logits_seq.size(1)
            tgt_rep = target.unsqueeze(1).expand(-1, Kt, -1).reshape(-1, nC)
            loss = loss + args.deep_aux_w * bce(logits_seq.reshape(-1, nC), tgt_rep)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()

        if step % 50 == 0:
            print(f"  step {step:5d}/{args.steps}  loss={loss.item():.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            scores, y = eval_scores(model, val_src, executor, device, cache,
                                    K_eval=args.K_eval, target_rms=args.target_rms,
                                    hpf_hz=args.hpf_hz)
            thr = tune_thresholds_per_class(scores, y)
            pred = probs_to_onehot(scores, thr, force_one=True,
                                   single_label=args.single_label)
            m = multilabel_report(f"val@{step}", y, pred, CLASSES)
            mf = m["macro_f1"]
            history.append({"step": step, **m, "thr": thr.tolist()})
            if mf > best_val:
                best_val, best_step, patience_left = mf, step, args.patience
                torch.save({"state_dict": model.state_dict(), "args": vars(args),
                            "step": step, "val_macro_f1": mf,
                            "thresholds": thr.tolist(),
                            "ckpt_names": cache.ckpt_names}, out_dir / "best.pt")
                (out_dir / "thresholds.json").write_text(json.dumps(
                    {"per_class": dict(zip(CLASSES, thr.tolist()))}, indent=2))
                print(f"    ↑ best (step {step}, val macroF1={mf:.4f}) saved", flush=True)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("  early stop", flush=True); break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "result.json").write_text(json.dumps(
        {"best_val_macro_f1": best_val, "best_step": best_step,
         "args": vars(args), "n_params": model.n_params()}, indent=2))
    print(f"\nDONE. best val macroF1 = {best_val:.4f} at step {best_step}")


if __name__ == "__main__":
    main()
