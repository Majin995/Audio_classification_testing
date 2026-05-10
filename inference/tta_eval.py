"""Test-time augmentation evaluation for HydroPreciseV2.

For each K in ``--ks``, average softmax across K stochastic forward passes.
``_WaveformAug.forward`` is invoked manually with ``force_train=True`` while
the model itself stays in ``eval()`` mode — so dropout, BN running stats,
and the teacher path remain untouched. The only stochasticity is the
waveform-level augmentation (corpus noise, RIR, pitch, gain, Gaussian noise)
plus per-branch SpecAugment, which we re-enable explicitly.

Reports val/μP and test/μP for each K. Compare to K=1 baseline.
"""
from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score,
)

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_precise_v2 import HydroPreciseV2
from models.hydro_catfish import SpecAugment1D
from models.hydro_precise_v2 import _WaveformAug


# ── Augmentation gating ────────────────────────────────────────────────────

@contextmanager
def _enable_specaug(model: nn.Module):
    """Flip only the per-branch SpecAugment1D modules into train() so they
    actually mask. Everything else (dropout, BN, head) stays in eval()."""
    aug_modules = [m for m in model.modules() if isinstance(m, SpecAugment1D)]
    prev = [(m, m.training) for m in aug_modules]
    for m in aug_modules:
        m.train()
    try:
        yield
    finally:
        for m, t in prev:
            m.train(t)


# ── TTA forward ────────────────────────────────────────────────────────────

def _tta_logits(model: HydroPreciseV2, x: torch.Tensor, augment: bool) -> torch.Tensor:
    """Run a single forward, optionally with stochastic waveform aug applied.

    Critical: the model stays in ``eval()`` throughout. We invoke
    ``model.wave_aug(x, force_train=True)`` to bypass its ``self.training``
    gate without flipping the rest of the model's mode (which would re-enable
    dropout, BN updates, etc.). Per-branch SpecAugment is re-enabled via the
    `_enable_specaug` context manager because branches consult `self.training`
    of the SpecAugment1D module directly.
    """
    if not augment:
        return model(x)
    x = model.wave_aug(x, force_train=True)
    with _enable_specaug(model):
        return model(x)


@torch.no_grad()
def _tta_predict_split(model: HydroPreciseV2, loader, device: str, K: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Average softmax over K augmented forwards. Returns (probs, labels).

    DALI yields samples in the same order across passes when shuffle=False
    (val/test loaders are not shuffled). We accumulate softmax sums per-batch
    and divide by K at the end.
    """
    sum_probs, labs = None, []
    for k in range(K):
        per_batch = []
        labs_k = []
        offset = 0
        for batch in loader:
            x, y = batch
            x = x.to(device, non_blocking=True)
            logits = _tta_logits(model, x, augment=(K > 1))
            p = F.softmax(logits, dim=-1).cpu().numpy()
            if k == 0:
                per_batch.append(p)
                labs_k.append(y.cpu().numpy())
            else:
                assert sum_probs is not None
                sum_probs[offset:offset + p.shape[0]] += p
                offset += p.shape[0]
        if k == 0:
            sum_probs = np.concatenate(per_batch)
            labs = np.concatenate(labs_k)
    assert sum_probs is not None
    return sum_probs / K, labs


# ── Main ───────────────────────────────────────────────────────────────────

def _metrics(y, probs, nc):
    pred = probs.argmax(axis=1)
    return {
        "acc":      accuracy_score(y, pred),
        "f1":       f1_score(y, pred, average="macro", zero_division=0),
        "micro_p":  precision_score(y, pred, average="micro", zero_division=0),
        "macro_p":  precision_score(y, pred, average="macro", zero_division=0),
        "mcc":      matthews_corrcoef(y, pred),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--sample_rate", type=int, default=5_120)
    p.add_argument("--fixed_len", type=int, default=5_120)
    p.add_argument("--ks", default="1,4,8",
                   help="Comma-separated TTA pass counts to evaluate.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="lightning_logs/tta/results.md")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    Ks: List[int] = [int(k) for k in args.ks.split(",")]
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    data = DALIAudioDataModule(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_threads=args.num_threads, target_sr=args.sample_rate,
        fixed_len=args.fixed_len, oversample_train=False,
    )
    data.setup()
    nc = data.num_classes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HydroPreciseV2.load_from_checkpoint(args.ckpt, map_location=device, strict=False)
    model = model.to(device).eval()

    md = [f"# TTA — `{Path(args.ckpt).name}`", ""]
    md.append("| K | split | μP | MP | f1 | acc | mcc |")
    md.append("|---|---|---|---|---|---|---|")

    for K in Ks:
        print(f"\n[tta] K={K} val ...")
        pv, yv = _tta_predict_split(model, data.val_dataloader(),  device, K)
        print(f"[tta] K={K} test ...")
        pt, yt = _tta_predict_split(model, data.test_dataloader(), device, K)
        mv = _metrics(yv, pv, nc); mt = _metrics(yt, pt, nc)
        md.append(f"| {K} | val  | {mv['micro_p']:.4f} | {mv['macro_p']:.4f} | "
                  f"{mv['f1']:.4f} | {mv['acc']:.4f} | {mv['mcc']:.4f} |")
        md.append(f"| {K} | test | {mt['micro_p']:.4f} | {mt['macro_p']:.4f} | "
                  f"{mt['f1']:.4f} | {mt['acc']:.4f} | {mt['mcc']:.4f} |")
        print(f"[tta] K={K}  val/μP={mv['micro_p']:.4f}  test/μP={mt['micro_p']:.4f}")

    text = "\n".join(md) + "\n"
    out_path.write_text(text)
    print("\n" + text)
    print(f"[tta] wrote {out_path}")


if __name__ == "__main__":
    main()
