"""Average softmax probs across multiple HydroSpark seeds on Deepship_1s test."""

from __future__ import annotations
import argparse, json, sys, os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from models.hydro_spark import HydroSpark

CLASSES = ["Cargo", "Passenger", "Tanker", "Tug"]


def _source(p):
    parts = Path(p).stem.split("_")
    if len(parts) >= 4: return f"{parts[0]}_{parts[1]}_{parts[2]}"
    if len(parts) == 3: return f"{parts[0]}_{parts[1]}"
    return Path(p).stem


def _confusion(y, p, K):
    cm = np.zeros((K, K), dtype=np.int64)
    for a, b in zip(y, p): cm[a, b] += 1
    return cm


def _macro(cm):
    K = cm.shape[0]
    p, r, f = np.zeros(K), np.zeros(K), np.zeros(K)
    for k in range(K):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp
        p[k] = tp / max(tp + fp, 1); r[k] = tp / max(tp + fn, 1)
        f[k] = 2 * p[k] * r[k] / max(p[k] + r[k], 1e-9)
    return p, r, f, p.mean(), r.mean(), f.mean()


def _hpf(x, sr, cutoff, order=4):
    if cutoff <= 0: return x
    T = x.shape[-1]
    freqs = torch.fft.rfftfreq(T, 1.0 / sr).to(x.device)
    ratio = freqs / max(cutoff, 1e-9)
    mag = ratio.pow(2 * order)
    mask = (mag / (1.0 + mag)).sqrt().to(x.dtype)
    Xf = torch.fft.rfft(x.float(), dim=-1)
    return torch.fft.irfft(Xf * mask, n=T, dim=-1).to(x.dtype)


def _read(p, target_sr=5120, fixed_len=5120):
    a, sr = sf.read(p, dtype="float32", always_2d=False)
    if a.ndim > 1: a = a.mean(axis=1)
    if sr != target_sr:
        a = AF.resample(torch.from_numpy(a), sr, target_sr).numpy()
    if len(a) < fixed_len: a = np.pad(a, (0, fixed_len - len(a)))
    else: a = a[:fixed_len]
    return a


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    dils = tuple(int(d) for d in a["tcn_dilations"].split(","))
    m = HydroSpark(
        num_classes=4, n_bands=a["n_bands"], sinc_kernel=a["sinc_kernel"],
        env_decim=a["env_decim"], tcn_kernel=a["tcn_kernel"],
        tcn_dilations=dils, expansion=a["expansion"],
        head_hidden=a["head_hidden"], dropout=a.get("dropout", 0.1),
        band_dropout=a.get("band_dropout", 0.0),
        freeze_sinc=a.get("freeze_sinc", False),
        use_log_energy=not a.get("no_log_energy", False),
        use_delta=not a.get("no_delta", False),
        use_coherence=not a.get("no_coherence", False),
        gambler=True,
    ).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m, a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--data_dir", default="/var/mnt/5A009BF8009BD8F9/Data/Deepship_1s")
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--out_dir", default="lightning_logs/ensemble_dship")
    args = p.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    files, labels = [], []
    for ci, c in enumerate(CLASSES):
        for fp in sorted((Path(args.data_dir) / args.split / c).glob("*.wav")):
            files.append(str(fp)); labels.append(ci)
    labels = np.array(labels, dtype=np.int64)
    print(f"{args.split}: {len(files)} clips, {len(args.ckpts)} models")

    sum_probs = np.zeros((len(files), 4), dtype=np.float64)
    a0 = None
    for ckpt in args.ckpts:
        m, a = load_model(ckpt, device)
        if a0 is None: a0 = a
        print(f"  loaded {ckpt} (params={m.n_params():,})")
        target_rms = a.get("target_rms", 0.1); hpf_hz = a.get("hpf_hz", 20.0)
        rms = a.get("rms_normalize", True)
        executor = ThreadPoolExecutor(max_workers=12)
        bs = args.batch_size
        probs = np.zeros((len(files), 4), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(files), bs):
                batch = files[i:i+bs]
                audio = np.stack(list(executor.map(_read, batch))).astype(np.float32)
                wav = torch.from_numpy(audio).to(device)
                if rms:
                    wav = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-9) * target_rms
                if hpf_hz > 0:
                    wav = _hpf(wav, 5120, hpf_hz, 4)
                logits = m(wav)
                probs[i:i+len(batch)] = F.softmax(logits[:, :4], dim=-1).float().cpu().numpy()
        sum_probs += probs
        executor.shutdown(wait=True)
        del m
        torch.cuda.empty_cache()
    sum_probs /= len(args.ckpts)

    yp = sum_probs.argmax(-1)
    cm = _confusion(labels, yp, 4)
    _, _, fcls, mP, mR, mF = _macro(cm)
    print()
    print("═" * 72)
    print(f"PER-CLIP   n={len(files)}: macroP={mP:.4f} macroR={mR:.4f} macroF1={mF:.4f}")

    src_probs, src_label = defaultdict(list), {}
    for i, fp in enumerate(files):
        s = _source(fp); src_probs[s].append(sum_probs[i]); src_label[s] = labels[i]
    src_list = sorted(src_probs.keys())
    agg = np.zeros((len(src_list), 4))
    for j, s in enumerate(src_list):
        P = np.stack(src_probs[s])
        agg[j] = np.exp(np.log(np.clip(P, 1e-8, 1.0)).mean(axis=0))
    yp_s = agg.argmax(-1)
    y_s = np.array([src_label[s] for s in src_list])
    cm_s = _confusion(y_s, yp_s, 4)
    _, _, fcls_s, mP_s, mR_s, mF_s = _macro(cm_s)
    print(f"PER-SOURCE n={len(src_list)}: macroP={mP_s:.4f} macroR={mR_s:.4f} macroF1={mF_s:.4f}")
    print(f"  per-class F1: {dict(zip(CLASSES, fcls_s.round(4)))}")
    print(f"  cm = {cm_s.tolist()}")
    print("═" * 72)

    with open(out_dir / "ensemble.json", "w") as f:
        json.dump({
            "ckpts": args.ckpts, "n_models": len(args.ckpts),
            "split": args.split,
            "per_clip": {"cm": cm.tolist(), "macroF1": float(mF), "macroP": float(mP), "macroR": float(mR)},
            "per_source": {"cm": cm_s.tolist(), "macroF1": float(mF_s), "macroP": float(mP_s), "macroR": float(mR_s), "per_class_F1": fcls_s.tolist()},
        }, f, indent=2)
    np.savez_compressed(out_dir / "probs.npz", files=np.array(files), labels=labels, probs=sum_probs)


if __name__ == "__main__":
    main()
