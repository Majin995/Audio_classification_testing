"""Per-class val-vs-test breakdown for a HydroPreciseV2 ckpt.

For each split (val, test):
  * predict probabilities via the ckpt's head (no KNN, no blend — pure model)
  * compute per-class precision / recall / support / mean SNR

The mean SNR per class comes from ``processing/denoise/snr.estimate_snr_noise_floor``
applied to each clip's raw waveform. SNR loop uses a multiprocessing pool.

Output: a side-by-side markdown table at ``--out`` showing, per class, where
performance changes val→test and whether SNR concentration explains the drop.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import precision_score, recall_score

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_precise_v2 import HydroPreciseV2
from processing.denoise.snr import estimate_snr_noise_floor


# ── Per-clip SNR (multiprocessed) ───────────────────────────────────────────

def _snr_one(path: str) -> Tuple[str, float]:
    try:
        wav, sr = torchaudio.load(path)
        x = wav.mean(dim=0).numpy().astype(np.float32)      # mono
        # Normalise to [-1, 1]
        peak = max(abs(x.max()), abs(x.min()), 1e-9)
        x = x / peak
        return path, estimate_snr_noise_floor(x, fs=sr)
    except Exception:
        return path, float("nan")


def _per_class_snr(files_by_class: Dict[str, List[str]],
                   workers: int) -> Dict[str, float]:
    """Mean SNR (dB) per class across all files in the split."""
    paths, classes = [], []
    for cls, fs in files_by_class.items():
        paths.extend(fs)
        classes.extend([cls] * len(fs))
    with mp.Pool(workers) as pool:
        results = pool.map(_snr_one, paths)
    by_cls: Dict[str, List[float]] = {c: [] for c in files_by_class}
    for (_, snr), cls in zip(results, classes):
        if not np.isnan(snr):
            by_cls[cls].append(snr)
    return {c: float(np.mean(v)) if v else float("nan") for c, v in by_cls.items()}


# ── Per-class predictions ───────────────────────────────────────────────────

@torch.no_grad()
def _predict_split(model, loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    preds, labs = [], []
    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        labs.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(labs)


def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       num_classes: int) -> Dict[str, np.ndarray]:
    p = precision_score(y_true, y_pred, average=None,
                        labels=list(range(num_classes)), zero_division=0)
    r = recall_score(y_true, y_pred, average=None,
                     labels=list(range(num_classes)), zero_division=0)
    n = np.array([(y_true == c).sum() for c in range(num_classes)])
    return {"precision": p, "recall": r, "support": n}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--sample_rate", type=int, default=5_120)
    p.add_argument("--fixed_len", type=int, default=5_120)
    p.add_argument("--snr_workers", type=int, default=8)
    p.add_argument("--out", default="lightning_logs/diagnostic/val_vs_test.md")
    args = p.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    data = DALIAudioDataModule(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_threads=args.num_threads, target_sr=args.sample_rate,
        fixed_len=args.fixed_len, oversample_train=False,
    )
    data.setup()
    nc = data.num_classes
    classes = sorted(data.class_to_idx, key=lambda c: data.class_to_idx[c])
    print(f"[breakdown] {nc} classes  class_to_idx={data.class_to_idx}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HydroPreciseV2.load_from_checkpoint(args.ckpt, map_location=device, strict=False)
    model = model.to(device).eval()

    print("[breakdown] predicting val ...")
    pv, yv = _predict_split(model, data.val_dataloader(),  device)
    print("[breakdown] predicting test ...")
    pt, yt = _predict_split(model, data.test_dataloader(), device)

    val_m  = _per_class_metrics(yv, pv, nc)
    test_m = _per_class_metrics(yt, pt, nc)

    print("[breakdown] computing per-class mean SNR (val) ...")
    val_files  = data._scan_split("val")
    val_snr    = _per_class_snr(val_files,  args.snr_workers)
    print("[breakdown] computing per-class mean SNR (test) ...")
    test_files = data._scan_split("test")
    test_snr   = _per_class_snr(test_files, args.snr_workers)

    # Markdown
    md = [f"# Val vs Test breakdown — `{Path(args.ckpt).name}`", ""]
    md.append("| class | val_P | val_R | val_N | val_SNR(dB) | test_P | test_R | test_N | test_SNR(dB) | ΔP | ΔR | ΔSNR |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ci, c in enumerate(classes):
        vP = val_m["precision"][ci];  tP = test_m["precision"][ci]
        vR = val_m["recall"][ci];     tR = test_m["recall"][ci]
        vN = int(val_m["support"][ci]); tN = int(test_m["support"][ci])
        vSnr = val_snr.get(c, float("nan"));  tSnr = test_snr.get(c, float("nan"))
        md.append(
            f"| {c} | {vP:.4f} | {vR:.4f} | {vN} | {vSnr:.2f} | "
            f"{tP:.4f} | {tR:.4f} | {tN} | {tSnr:.2f} | "
            f"{tP - vP:+.4f} | {tR - vR:+.4f} | {tSnr - vSnr:+.2f} |"
        )
    md.append("")
    text = "\n".join(md)
    out_path.write_text(text)
    print("\n" + text)
    print(f"\n[breakdown] wrote {out_path}")


if __name__ == "__main__":
    main()
