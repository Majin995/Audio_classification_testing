"""
Active learning pipeline for HydroPrecise.

Builds an initial stratified labeled subset of the train pool, trains
HydroPrecise (verify_B preset from the v1 grid by default), then iteratively
queries the most informative unlabeled examples and adds them to the labeled
set. Each round writes Lightning logs under
``lightning_logs/<run_name>/round_NN/`` and the final per-round metrics land in
``<run_name>/summary.csv`` / ``summary.json`` at the run root.

Strategies supported:
  - entropy     : top-k by softmax entropy
  - margin      : top-k by smallest (p1 - p2) margin
  - least_conf  : top-k by smallest max-prob
  - bald        : MC-dropout BALD (T forward passes)
  - random      : uniform-random baseline

Usage (verify_B preset, default):
    python training/train_active.py \
        --data_dir "/run/media/damo/Lexar M2/Data/Classifier_Dataset" \
        --run_name al_entropy_v1 \
        --strategy entropy --init_size 1000 --query_size 1000 --n_rounds 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from data.audio_lightning_loader import DALIAudioDataModule
from data._subset import AUDIO_EXTS, scan_train_pool, stratified_init
from models.hydro_precise import HydroPrecise


# ═══════════════════════════════════════════════════════════════════════
#  CLI — verify_B preset defaults (lmf_margin=0.30, lmf_gamma=2.0,
#  label_smoothing=0.05, gambler_weight=0.0, lr=3e-4, bs=64,
#  max_epochs=60, warmup=8, patience=15, target_coverage=0.85, bf16-mixed)
# ═══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description="Active learning for HydroPrecise")

    # Data
    p.add_argument("--data_dir",       default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--num_threads",    type=int, default=8)
    p.add_argument("--no_oversample",  action="store_true")
    p.add_argument("--denoise",        default="off",
                   choices=["off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"])

    # Audio
    p.add_argument("--sample_rate",    type=int, default=5_120)
    p.add_argument("--fixed_len",      type=int, default=5_120)

    # Model widths (verify_B = HydroPrecise defaults)
    p.add_argument("--gabor_n_filters", type=int, default=64)
    p.add_argument("--gabor_kernel",    type=int, default=257)
    p.add_argument("--gabor_ch",        type=int, default=128)
    p.add_argument("--cqt_n_bins",      type=int, default=84)
    p.add_argument("--cqt_bpo",         type=int, default=12)
    p.add_argument("--cqt_hop",         type=int, default=64)
    p.add_argument("--cqt_ch",          type=int, default=128)
    p.add_argument("--demon_hop",       type=int, default=64)
    p.add_argument("--demon_ch",        type=int, default=64)
    p.add_argument("--demon_n_fft",     type=int, default=2048)
    p.add_argument("--demon_mod_f_min", type=float, default=0.0)
    p.add_argument("--demon_mod_f_max", type=float, default=50.0)
    p.add_argument("--fusion_T",        type=int, default=64)
    p.add_argument("--fusion_dim",      type=int, default=192)
    p.add_argument("--n_heads",         type=int, default=2)
    p.add_argument("--dropout",         type=float, default=0.15)

    # Loss — verify_B
    p.add_argument("--loss",            default="lmf", choices=["lmf", "focal"])
    p.add_argument("--lmf_gamma",       type=float, default=2.0)
    p.add_argument("--lmf_margin",      type=float, default=0.30)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--gambler_o",       type=float, default=0.3)
    p.add_argument("--gambler_weight",  type=float, default=0.0)

    # Augmentation
    p.add_argument("--noise_prob",     type=float, default=0.5)
    p.add_argument("--noise_snr_min",  type=float, default=15.0)
    p.add_argument("--noise_snr_max",  type=float, default=30.0)
    p.add_argument("--gain_prob",      type=float, default=0.5)
    p.add_argument("--gain_range",     type=float, default=0.3)

    # Training — verify_B
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-2)
    p.add_argument("--max_epochs",     type=int,   default=60)
    p.add_argument("--warmup_epochs",  type=int,   default=8)
    p.add_argument("--patience",       type=int,   default=15)
    p.add_argument("--precision",      default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--seed",           type=int,   default=2026)
    p.add_argument("--run_name",       default="hydro_precise_al")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)

    # Active-learning controls
    p.add_argument("--init_size",   type=int, default=1000,
                   help="Total initial labeled count (stratified across classes).")
    p.add_argument("--init_per_class", type=int, default=0,
                   help="If >0, take exactly this many files per class for init "
                        "(overrides --init_size).")
    p.add_argument("--query_size", type=int, default=1000,
                   help="Files to query per round (added to labeled).")
    p.add_argument("--n_rounds",   type=int, default=5,
                   help="Number of AL rounds (round 0 = train on init only; "
                        "n_rounds queries follow).")
    p.add_argument("--strategy",   default="entropy",
                   choices=["entropy", "margin", "least_conf", "bald", "random"])
    p.add_argument("--bald_T",     type=int, default=10,
                   help="MC-dropout passes for BALD.")
    p.add_argument("--score_batch_size", type=int, default=128,
                   help="Batch size for pool scoring (no grad — can be larger).")
    p.add_argument("--warm_start", action="store_true",
                   help="Reuse last round's checkpoint as init weights.")
    p.add_argument("--stratified_query", action="store_true",
                   help="Distribute query budget proportionally across classes "
                        "(uses model's argmax as proxy class).")
    p.add_argument("--target_coverage", type=float, default=0.85)
    p.add_argument("--final_calibration", action="store_true",
                   help="Run temperature scaling + threshold sweep on the "
                        "final round only.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Pool / labeled bookkeeping (scan_train_pool + stratified_init are
#  imported from data/_subset.py so they're shared with train_precise_v2.)
# ═══════════════════════════════════════════════════════════════════════


def remaining_pool(pool: Dict[str, List[str]],
                   labeled: Dict[str, List[str]]
                  ) -> List[tuple]:
    """Return list of (path, true_class) for the unlabeled pool."""
    labeled_set = {p for fs in labeled.values() for p in fs}
    out = []
    for c in sorted(pool.keys()):
        for p in pool[c]:
            if p not in labeled_set:
                out.append((p, c))
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Acquisition scoring
# ═══════════════════════════════════════════════════════════════════════

def _enable_dropout_only(model: nn.Module) -> None:
    """eval() everything, then re-enable Dropout-family modules."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def _forward_softmax(model: HydroPrecise, audio: torch.Tensor) -> torch.Tensor:
    logits = model(audio)[:, :model.num_classes]
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def score_pool(model: HydroPrecise, datamodule: DALIAudioDataModule,
               pool_files: List[str], strategy: str, bald_T: int,
               score_batch_size: int, device: torch.device,
              ) -> tuple:
    """
    Score every file in ``pool_files`` under the chosen strategy.
    Returns (scores: np.ndarray of shape (N,), argmax_class: np.ndarray (N,)).
    Higher score == higher acquisition priority.
    """
    n = len(pool_files)
    scores  = np.full(n, -np.inf, dtype=np.float64)
    pred_cls = np.full(n, -1, dtype=np.int64)

    loader = datamodule.make_score_loader(pool_files, batch_size=score_batch_size)

    if strategy == "bald":
        _enable_dropout_only(model)
        for audio, idx in loader:
            audio = audio.to(device, non_blocking=True)
            probs_T = []
            for _ in range(bald_T):
                probs_T.append(_forward_softmax(model, audio))    # (B, C)
            probs_T = torch.stack(probs_T, dim=0)                 # (T, B, C)
            mean_p = probs_T.mean(dim=0)                          # (B, C)
            entropy_mean = -(mean_p * (mean_p.clamp_min(1e-12)).log()).sum(-1)
            mean_entropy = -(probs_T * probs_T.clamp_min(1e-12).log()).sum(-1).mean(0)
            bald = entropy_mean - mean_entropy                     # (B,)
            idx_np = idx.cpu().numpy()
            scores[idx_np]   = bald.cpu().numpy()
            pred_cls[idx_np] = mean_p.argmax(-1).cpu().numpy()
        return scores, pred_cls

    # Deterministic strategies
    model.eval()
    for audio, idx in loader:
        audio = audio.to(device, non_blocking=True)
        probs = _forward_softmax(model, audio)                    # (B, C)
        if strategy == "entropy":
            s = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
        elif strategy == "margin":
            top2, _ = probs.topk(2, dim=-1)
            s = -(top2[:, 0] - top2[:, 1])                         # smaller margin = higher score
        elif strategy == "least_conf":
            s = 1.0 - probs.max(-1).values
        elif strategy == "random":
            s = torch.rand(probs.shape[0], device=probs.device)
        else:
            raise ValueError(f"unknown strategy: {strategy}")
        idx_np = idx.cpu().numpy()
        scores[idx_np]   = s.cpu().numpy()
        pred_cls[idx_np] = probs.argmax(-1).cpu().numpy()
    return scores, pred_cls


def select_query(scores: np.ndarray, pred_cls: np.ndarray, k: int,
                 stratified: bool, num_classes: int) -> np.ndarray:
    """Return indices of the top-k pool entries by score."""
    k = min(k, (scores > -np.inf).sum())
    if k <= 0:
        return np.array([], dtype=np.int64)
    if not stratified:
        # argpartition for top-k, then sort that slice descending
        cand = np.argpartition(-scores, k - 1)[:k]
        cand = cand[np.argsort(-scores[cand])]
        return cand
    # Stratified: split budget by predicted class
    base, rem = divmod(k, num_classes)
    quotas = [base + (1 if i < rem else 0) for i in range(num_classes)]
    picks: list = []
    for c, q in enumerate(quotas):
        cls_idx = np.where(pred_cls == c)[0]
        if cls_idx.size == 0 or q == 0:
            continue
        ord_ = cls_idx[np.argsort(-scores[cls_idx])]
        picks.append(ord_[:q])
    if picks:
        out = np.concatenate(picks)
    else:
        out = np.array([], dtype=np.int64)
    # Top up if any quota was short on a class
    if out.size < k:
        already = set(out.tolist())
        rest = np.array([i for i in np.argsort(-scores) if i not in already],
                        dtype=np.int64)
        out = np.concatenate([out, rest[:k - out.size]])
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Train one round
# ═══════════════════════════════════════════════════════════════════════

def build_model(args, num_classes: int, class_weights: list) -> HydroPrecise:
    return HydroPrecise(
        num_classes=num_classes,
        class_weights=class_weights,
        sample_rate=args.sample_rate,
        gabor_n_filters=args.gabor_n_filters, gabor_kernel=args.gabor_kernel, gabor_ch=args.gabor_ch,
        cqt_n_bins=args.cqt_n_bins, cqt_bpo=args.cqt_bpo, cqt_hop=args.cqt_hop, cqt_ch=args.cqt_ch,
        demon_hop=args.demon_hop, demon_ch=args.demon_ch,
        demon_n_fft=args.demon_n_fft,
        demon_mod_f_min=args.demon_mod_f_min,
        demon_mod_f_max=args.demon_mod_f_max,
        fusion_T=args.fusion_T, fusion_dim=args.fusion_dim,
        n_heads=args.n_heads, dropout=args.dropout,
        loss=args.loss, lmf_gamma=args.lmf_gamma, lmf_margin=args.lmf_margin,
        label_smoothing=args.label_smoothing,
        gambler_o=args.gambler_o, gambler_weight=args.gambler_weight,
        noise_prob=args.noise_prob, noise_snr_min=args.noise_snr_min,
        noise_snr_max=args.noise_snr_max,
        gain_prob=args.gain_prob, gain_range=args.gain_range,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
    )


def train_round(args, datamodule: DALIAudioDataModule, round_idx: int,
                init_state: dict | None
               ) -> tuple:
    """
    Train HydroPrecise on the current labeled subset. Returns
    (best_ckpt_path, val_micro_precision, val_metrics_dict, model_in_memory).
    """
    pl.seed_everything(args.seed + round_idx, workers=True)
    torch.set_float32_matmul_precision("high")

    model = build_model(args, datamodule.num_classes, datamodule.class_weights)
    if init_state is not None:
        model.load_state_dict(init_state, strict=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[round {round_idx}] HydroPrecise — {n_params/1e6:.2f}M params")

    round_name = f"{args.run_name}/round_{round_idx:02d}"
    ckpt_cb = ModelCheckpoint(
        monitor="val/micro_precision", mode="max", save_top_k=1,
        filename="precise-{epoch:03d}-p{val/micro_precision:.4f}",
        auto_insert_metric_name=False, verbose=False,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val/micro_precision", patience=args.patience,
                      mode="max", verbose=False),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision if torch.cuda.is_available() else 32,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=[
            CSVLogger("lightning_logs", name=round_name),
            TensorBoardLogger("lightning_logs", name=round_name),
        ],
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        deterministic=False,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    trainer.fit(model, datamodule)
    val_metrics = {k: float(v) for k, v in trainer.callback_metrics.items()
                   if k.startswith("val/")}

    best_path  = ckpt_cb.best_model_path
    best_score = ckpt_cb.best_model_score
    val_mp = float(best_score) if best_score is not None else float("nan")
    print(f"[round {round_idx}] best val/micro_precision = {val_mp:.4f}")
    print(f"[round {round_idx}] ckpt: {best_path}")

    # Reload best weights for scoring
    if best_path and Path(best_path).exists():
        ckpt_model = HydroPrecise.load_from_checkpoint(best_path, strict=False)
    else:
        ckpt_model = model
    return best_path, val_mp, val_metrics, ckpt_model


# ═══════════════════════════════════════════════════════════════════════
#  Calibration helpers (mirrors train_precise.py)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_logits(model, dataloader, device):
    model.eval()
    logits_all, targets_all = [], []
    for x, y in dataloader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits_all.append(model(x)[:, :model.num_classes].cpu())
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


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    if not args.data_dir:
        raise ValueError("Set --data_dir or DATA_DIR.")

    rng = random.Random(args.seed)

    # 1. Scan full train pool from disk
    data_dir = Path(args.data_dir).resolve()
    pool = scan_train_pool(data_dir)
    classes = sorted(pool.keys())
    num_classes = len(classes)
    print(f"Pool: {sum(len(v) for v in pool.values())} files across "
          f"{num_classes} classes: { {c: len(pool[c]) for c in classes} }")

    # 2. Stratified initial labeled subset
    labeled = stratified_init(pool, args.init_size, args.init_per_class, rng)
    n_init = sum(len(v) for v in labeled.values())
    print(f"Initial labeled: {n_init} files "
          f"{ {c: len(labeled[c]) for c in classes} }")

    # 3. Single DALIAudioDataModule reused across rounds
    dm = DALIAudioDataModule(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        oversample_train=not args.no_oversample,
        denoise_method=args.denoise,
        train_files_override=labeled,
    )
    dm.setup()

    summary_dir = Path("lightning_logs") / args.run_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv  = summary_dir / "summary.csv"
    summary_json = summary_dir / "summary.json"
    selections_jsonl = summary_dir / "selections.jsonl"

    rows: list = []
    init_state = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Round loop. Round 0 = train on init only; subsequent rounds add a
    # query before training.
    for r in range(args.n_rounds + 1):
        # Round r > 0: query top-k from current pool using last round's model
        if r > 0:
            unlabeled = remaining_pool(pool, labeled)
            if not unlabeled:
                print(f"[round {r}] pool exhausted, stopping.")
                break
            pool_files = [p for p, _ in unlabeled]
            pool_truth = [c for _, c in unlabeled]
            print(f"[round {r}] scoring {len(pool_files)} pool files "
                  f"with strategy={args.strategy}")

            scoring_model = last_model.to(device)
            scores, pred_cls = score_pool(
                scoring_model, dm, pool_files,
                strategy=args.strategy, bald_T=args.bald_T,
                score_batch_size=args.score_batch_size, device=device,
            )
            picked = select_query(scores, pred_cls, args.query_size,
                                  stratified=args.stratified_query,
                                  num_classes=num_classes)
            picked_files = [pool_files[i] for i in picked]
            picked_truth = [pool_truth[i] for i in picked]

            for path, truth in zip(picked_files, picked_truth):
                labeled.setdefault(truth, []).append(path)

            # Log this round's selection (compact: counts + first 50 paths)
            with selections_jsonl.open("a") as f:
                f.write(json.dumps({
                    "round":           r,
                    "strategy":        args.strategy,
                    "queried":         int(picked.size),
                    "queried_by_truth": {c: int(picked_truth.count(c))
                                          for c in classes},
                    "queried_by_pred":  {int(c): int((pred_cls[picked] == c).sum())
                                          for c in range(num_classes)},
                    "first_paths": picked_files[:50],
                }) + "\n")

            dm.set_train_files_override(labeled)
            torch.cuda.empty_cache()

        n_lab = sum(len(v) for v in labeled.values())
        per_class = {c: len(labeled.get(c, [])) for c in classes}
        print(f"[round {r}] labeled: {n_lab} files {per_class}")

        best_path, val_mp, val_metrics, ckpt_model = train_round(
            args, dm, round_idx=r,
            init_state=init_state if args.warm_start else None,
        )
        last_model = ckpt_model

        if args.warm_start and best_path and Path(best_path).exists():
            init_state = {k: v.clone() for k, v in ckpt_model.state_dict().items()}

        # Optional: temperature scaling on the final round
        T = None
        if args.final_calibration and r == args.n_rounds and best_path:
            print(f"[round {r}] running temperature scaling")
            ckpt_model.to(device).eval()
            logits, targets = _collect_logits(ckpt_model, dm.val_dataloader(), device)
            T = _fit_temperature(logits, targets)
            torch.save({"temperature": T}, Path(best_path).parent / "temperature.pt")

        row = {
            "round":          r,
            "strategy":       args.strategy,
            "n_labeled":      n_lab,
            "per_class":      per_class,
            "val_micro_precision": val_mp,
            "val_macro_precision": val_metrics.get("val/macro_precision", float("nan")),
            "val_acc":            val_metrics.get("val/acc", float("nan")),
            "val_f1":             val_metrics.get("val/f1", float("nan")),
            "val_mcc":            val_metrics.get("val/mcc", float("nan")),
            "val_auroc":          val_metrics.get("val/auroc", float("nan")),
            "best_ckpt":     best_path,
            "temperature":   T,
        }
        rows.append(row)

        # Append summary CSV after every round (resumable, observable mid-run)
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["round", "strategy", "n_labeled",
                            "val_micro_precision", "val_macro_precision",
                            "val_acc", "val_f1", "val_mcc", "val_auroc",
                            "best_ckpt", "temperature"],
            )
            writer.writeheader()
            for rr in rows:
                rrr = {k: rr[k] for k in writer.fieldnames}
                writer.writerow(rrr)
        with summary_json.open("w") as f:
            json.dump({
                "run_name":   args.run_name,
                "strategy":   args.strategy,
                "init_size":  n_init,
                "query_size": args.query_size,
                "n_rounds":   args.n_rounds,
                "rounds":     rows,
            }, f, indent=2, default=str)

    print("\n=== AL run complete ===")
    print(f"Summary CSV : {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
