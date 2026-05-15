"""
Honest ensemble of multiple HydroSpark seeds.

Loads N saved ``best.pt`` checkpoints, runs the SAME preprocessing pipeline
as training (RMS-normalize + HPF), averages softmax probs across seeds, then
reports per-clip and per-source test metrics (real-only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio.functional as AF

from models.hydro_spark import HydroSpark


CLASSES = ["Cargo", "Passenger", "Tanker", "Tug"]


# ───────────────────────── Honest metrics (copy from train_spark) ───────────

def _confusion(y_true, y_pred, K):
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _macro_prf1(cm):
    K = cm.shape[0]
    p, r, f = np.zeros(K), np.zeros(K), np.zeros(K)
    for k in range(K):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp
        p[k] = tp / max(tp + fp, 1)
        r[k] = tp / max(tp + fn, 1)
        f[k] = 2 * p[k] * r[k] / max(p[k] + r[k], 1e-9)
    return p, r, f, p.mean(), r.mean(), f.mean()


def _source(p):
    parts = Path(p).stem.split("_")
    return f"{parts[0]}_{parts[1]}_{parts[2]}" if len(parts) >= 4 else Path(p).stem


def _hpf_torch(x, sr, cutoff, order=4):
    if cutoff <= 0:
        return x
    T = x.shape[-1]
    freqs = torch.fft.rfftfreq(T, 1.0 / sr).to(x.device)
    ratio = freqs / max(cutoff, 1e-9)
    mag = ratio.pow(2 * order)
    mask = (mag / (1.0 + mag)).sqrt().to(x.dtype)
    Xf = torch.fft.rfft(x.float(), dim=-1)
    return torch.fft.irfft(Xf * mask, n=T, dim=-1).to(x.dtype)


# ───────────────────────── Per-checkpoint loader ────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> HydroSpark:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ck["args"]
    tcn_dilations = tuple(int(d) for d in args["tcn_dilations"].split(","))
    model = HydroSpark(
        num_classes=4, n_bands=args["n_bands"],
        sinc_kernel=args["sinc_kernel"], env_decim=args["env_decim"],
        tcn_kernel=args["tcn_kernel"], tcn_dilations=tcn_dilations,
        expansion=args["expansion"], head_hidden=args["head_hidden"],
        dropout=args.get("dropout", 0.1),
        band_dropout=args.get("band_dropout", 0.0),
        freeze_sinc=args.get("freeze_sinc", False),
        use_log_energy=not args.get("no_log_energy", False),
        gambler=True,
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, args


# ───────────────────────── File enumeration (real-only test) ────────────────

def enumerate_test_real(data_dir: str):
    files, labels = [], []
    for ci, cls in enumerate(CLASSES):
        d = Path(data_dir) / "test" / cls
        if not d.is_dir():
            continue
        for p in sorted(d.glob(f"{cls}_real_*.wav")):
            files.append(str(p))
            labels.append(ci)
    return files, np.array(labels, dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/var/mnt/5A009BF8009BD8F9/Data/Classifier_Dataset")
    p.add_argument("--out_dir", default="lightning_logs/spark_ensemble")
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    files, labels = enumerate_test_real(args.data_dir)
    print(f"Test files: {len(files)} (real-only)")

    # Load all models
    models, hparams = [], None
    for c in args.ckpts:
        m, h = load_model(c, device)
        models.append(m)
        if hparams is None:
            hparams = h
        print(f"  loaded {c} (params={m.n_params():,})")

    # Read audio in batches; apply same preprocessing as training; accumulate probs
    target_sr = 5120; fixed_len = 5120
    K = 4
    sum_probs = np.zeros((len(files), K), dtype=np.float64)
    target_rms = hparams.get("target_rms", 0.1)
    rms_normalize = hparams.get("rms_normalize", True)
    hpf_hz = hparams.get("hpf_hz", 20.0)
    bs = args.batch_size

    for i in range(0, len(files), bs):
        batch = files[i:i + bs]
        wavs = []
        for fp in batch:
            a, sr = sf.read(fp, dtype="float32", always_2d=False)
            if a.ndim > 1: a = a.mean(axis=1)
            if sr != target_sr:
                a = AF.resample(torch.from_numpy(a), sr, target_sr).numpy()
            if len(a) < fixed_len: a = np.pad(a, (0, fixed_len - len(a)))
            else: a = a[:fixed_len]
            wavs.append(a)
        wav = torch.from_numpy(np.stack(wavs).astype(np.float32)).to(device)
        if rms_normalize:
            wav = (wav - wav.mean(dim=1, keepdim=True)) / (
                wav.std(dim=1, keepdim=True) + 1e-9
            ) * target_rms
        if hpf_hz > 0:
            wav = _hpf_torch(wav, target_sr, hpf_hz, 4)
        batch_probs = None
        with torch.no_grad():
            for m in models:
                logits = m(wav)
                p_ = F.softmax(logits[:, :K], dim=-1).float().cpu().numpy()
                batch_probs = p_ if batch_probs is None else batch_probs + p_
        batch_probs /= len(models)
        sum_probs[i:i + len(batch)] = batch_probs
        if i % (bs * 20) == 0:
            print(f"  inference {i + len(batch):>6d} / {len(files)}")

    # Per-clip eval
    per_clip = sum_probs.argmax(-1)
    cm = _confusion(labels, per_clip, K)
    _, _, fcls, mP, mR, mF = _macro_prf1(cm)
    print("\n" + "═" * 72)
    print(f"PER-CLIP ENSEMBLE  (n_clips={len(files)}, n_models={len(models)})")
    print(f"  macroP={mP:.4f}  macroR={mR:.4f}  macroF1={mF:.4f}")
    print(f"  cm={cm.tolist()}")

    # Per-source eval — log-mean of mean-softmax
    src_probs = defaultdict(list); src_label = {}
    for i, fp in enumerate(files):
        s = _source(fp)
        src_probs[s].append(sum_probs[i])
        src_label[s] = labels[i]
    src_list = sorted(src_probs.keys())
    agg = np.zeros((len(src_list), K))
    for j, s in enumerate(src_list):
        P = np.stack(src_probs[s])
        agg[j] = np.exp(np.log(np.clip(P, 1e-8, 1.0)).mean(axis=0))
    per_src = agg.argmax(-1)
    src_labels = np.array([src_label[s] for s in src_list])
    cm_s = _confusion(src_labels, per_src, K)
    _, _, fcls_s, mP_s, mR_s, mF_s = _macro_prf1(cm_s)
    print("\nPER-SOURCE ENSEMBLE")
    print(f"  n_sources={len(src_list)}")
    print(f"  macroP={mP_s:.4f}  macroR={mR_s:.4f}  macroF1={mF_s:.4f}")
    print(f"  per-class F1: {dict(zip(CLASSES, fcls_s.round(4)))}")
    print(f"  cm={cm_s.tolist()}")
    print("═" * 72)

    out = {
        "ckpts": args.ckpts, "n_models": len(models),
        "n_clips": int(len(files)), "n_sources": int(len(src_list)),
        "per_clip": {"cm": cm.tolist(), "macroP": float(mP),
                     "macroR": float(mR), "macroF1": float(mF)},
        "per_source": {"cm": cm_s.tolist(), "macroP": float(mP_s),
                       "macroR": float(mR_s), "macroF1": float(mF_s),
                       "per_class_F1": fcls_s.tolist()},
        "params_per_model": models[0].n_params(),
    }
    with open(out_dir / "ensemble.json", "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(out_dir / "probs.npz",
                        files=np.array(files), labels=labels,
                        probs=sum_probs)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
