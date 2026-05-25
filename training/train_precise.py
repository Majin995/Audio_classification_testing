"""
Training script for HydroPrecise — high-precision UATR classifier.

Usage:
    export DATA_DIR=/path/to/Split1s
    python training/train_precise.py [options]

Monitors ``val/micro_precision`` for checkpointing.  After training, performs
post-hoc temperature scaling on the validation set and sweeps per-class
softmax thresholds to maximise macro-precision at coverage >= ``--target_coverage``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from data.audio_lightning_loader import DALIAudioDataModule
from data.loader_factory          import LOADER_CHOICES, build_loader
from models.hydro_precise        import HydroPrecise


# ═══════════════════════════════════════════════════════════════════════
#  Post-hoc calibration + per-class threshold search
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_logits(model: HydroPrecise, dataloader, device: torch.device):
    model.eval()
    logits_all, targets_all = [], []
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)[:, :model.num_classes]
        logits_all.append(logits.cpu())
        targets_all.append(y.cpu())
    return torch.cat(logits_all), torch.cat(targets_all)


def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor,
                     lr: float = 1e-2, max_iter: int = 200) -> float:
    """Fit a single positive scalar T minimising NLL(softmax(logits / T), targets)."""
    log_T = nn.Parameter(torch.zeros(1))        # T = exp(log_T) > 0
    optim = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        optim.zero_grad()
        T = log_T.exp().clamp(min=1e-2, max=100.0)
        loss = F.cross_entropy(logits / T, targets)
        loss.backward()
        return loss

    optim.step(closure)
    return float(log_T.exp().clamp(min=1e-2, max=100.0).item())


def _search_thresholds(probs: torch.Tensor, targets: torch.Tensor,
                       num_classes: int, target_coverage: float = 0.85) -> dict:
    """
    Greedy per-class threshold search maximising macro-precision with the
    constraint that predicted coverage >= target_coverage.

    A sample 'covered' = ``max(probs) >= threshold_of_argmax_class``.
    """
    probs_np   = probs.numpy()
    targets_np = targets.numpy()
    argmax     = probs_np.argmax(axis=1)
    p_max      = probs_np.max(axis=1)

    grid = np.linspace(0.0, 0.95, 20)

    def score(thr_vec: np.ndarray):
        """Return (macro_precision, coverage) for a given per-class threshold vector."""
        keep = p_max >= thr_vec[argmax]
        if keep.sum() == 0:
            return 0.0, 0.0
        preds = argmax[keep]
        gts   = targets_np[keep]
        precs = []
        for c in range(num_classes):
            p_mask = preds == c
            if p_mask.sum() == 0:
                continue
            precs.append((gts[p_mask] == c).mean())
        if not precs:
            return 0.0, float(keep.mean())
        return float(np.mean(precs)), float(keep.mean())

    # Start all at 0, greedily increase the threshold of whichever class
    # currently has the lowest precision, stop when coverage would drop below target.
    thr = np.zeros(num_classes)
    best_prec, best_cov = score(thr)
    best_thr = thr.copy()

    improved = True
    while improved:
        improved = False
        for c in range(num_classes):
            for g in grid:
                if g <= thr[c]:
                    continue
                cand = thr.copy(); cand[c] = g
                prec, cov = score(cand)
                if cov < target_coverage:
                    continue
                if prec > best_prec + 1e-6:
                    best_prec, best_cov, best_thr = prec, cov, cand
                    thr = cand
                    improved = True
        # one sweep per outer iteration
    return {
        "thresholds":     best_thr.tolist(),
        "macro_precision": best_prec,
        "coverage":        best_cov,
        "target_coverage": target_coverage,
    }


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description="Train HydroPrecise (high-precision UATR)")

    # Data
    p.add_argument("--data_dir",        default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--num_threads",     type=int, default=8)
    p.add_argument("--no_oversample",   action="store_true")
    p.add_argument("--denoise",         default="off",
                   choices=["off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"])
    # Loader selection — DALI vs threaded backend, splitting vs non-splitting.
    p.add_argument("--loader",          default="dali", choices=list(LOADER_CHOICES))
    p.add_argument("--window_sec",      type=float, default=None)
    p.add_argument("--hop_sec",         type=float, default=None)
    p.add_argument("--num_workers",     type=int, default=8)

    # Audio
    p.add_argument("--sample_rate",     type=int, default=5_120)
    p.add_argument("--fixed_len",       type=int, default=5_120)

    # Model widths
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

    # Loss
    p.add_argument("--loss",            default="lmf", choices=["lmf", "focal"])
    p.add_argument("--lmf_gamma",       type=float, default=2.0)
    p.add_argument("--lmf_margin",      type=float, default=0.5)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--gambler_o",       type=float, default=0.3)
    p.add_argument("--gambler_weight",  type=float, default=0.1)

    # Augmentation
    p.add_argument("--noise_prob",      type=float, default=0.5)
    p.add_argument("--noise_snr_min",   type=float, default=15.0)
    p.add_argument("--noise_snr_max",   type=float, default=30.0)
    p.add_argument("--gain_prob",       type=float, default=0.5)
    p.add_argument("--gain_range",      type=float, default=0.3)

    # Training
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--max_epochs",      type=int,   default=100)
    p.add_argument("--warmup_epochs",   type=int,   default=10)
    p.add_argument("--patience",        type=int,   default=20)
    p.add_argument("--precision",       default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--run_name",        default="hydro_precise")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)

    # Post-hoc
    p.add_argument("--target_coverage", type=float, default=0.85,
                   help="Coverage lower bound for per-class threshold search.")
    p.add_argument("--skip_calibration", action="store_true")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    if args is None:
        args = get_args()

    if not args.data_dir:
        raise ValueError("Set --data_dir or export DATA_DIR=/path/to/Split1s")

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data = build_loader(
        args.loader,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_workers=args.num_workers,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        oversample_train=not args.no_oversample,
        denoise_method=args.denoise,
    )
    data.setup()

    model = HydroPrecise(
        num_classes=data.num_classes,
        class_weights=data.class_weights,
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHydroPrecise — {n_params / 1e6:.2f}M trainable parameters")
    print(f"Loss: {args.loss}  (margin={args.lmf_margin}, γ={args.lmf_gamma}, gambler_w={args.gambler_weight})")

    ckpt_cb = ModelCheckpoint(
        monitor="val/micro_precision", mode="max", save_top_k=1,
        filename="precise-{epoch:03d}-p{val/micro_precision:.4f}",
        auto_insert_metric_name=False, verbose=True,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val/micro_precision", patience=args.patience,
                      mode="max", verbose=True),
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
            CSVLogger("lightning_logs", name=args.run_name),
            TensorBoardLogger("lightning_logs", name=args.run_name),
        ],
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        deterministic=False,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")

    best_path  = ckpt_cb.best_model_path
    best_score = ckpt_cb.best_model_score
    print(f"\nBest checkpoint:       {best_path}")
    print(f"Best val/micro_prec:   {best_score:.4f}" if best_score is not None else "Best val/micro_prec: n/a")

    # ── Post-hoc calibration + threshold sweep ───────────────────────────
    if not args.skip_calibration and best_path:
        print("\n── Post-hoc calibration + threshold sweep ──")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cal_model = HydroPrecise.load_from_checkpoint(
            best_path, map_location=device, strict=False,
        )
        cal_model.to(device).eval()

        logits, targets = _collect_logits(cal_model, data.val_dataloader(), device)
        T = _fit_temperature(logits, targets)
        probs = F.softmax(logits / T, dim=-1)
        print(f"  temperature    = {T:.3f}")

        res = _search_thresholds(probs, targets, num_classes=data.num_classes,
                                 target_coverage=args.target_coverage)
        print(f"  temp-scaled val micro-precision (no gating) = "
              f"{float(((probs.argmax(-1) == targets).float()).mean()):.4f}")
        print(f"  thresholds     = {['%.2f' % t for t in res['thresholds']]}")
        print(f"  macro_precision = {res['macro_precision']:.4f}  "
              f"coverage = {res['coverage']:.4f}  (target >= {res['target_coverage']})")

        out_dir = Path(best_path).parent
        torch.save({"temperature": T}, out_dir / "temperature.pt")
        with open(out_dir / "thresholds.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"  saved → {out_dir / 'temperature.pt'}")
        print(f"  saved → {out_dir / 'thresholds.json'}")

    return float(best_score) if best_score is not None else float("nan"), best_path


if __name__ == "__main__":
    main()
