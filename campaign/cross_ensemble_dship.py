"""Ensemble HydroSpark (per-clip log-mean) with HydroSetSpark (per-source).

For each test source:
  • Run HydroSpark per-clip on all clips → log-mean → softmax → P_hs
  • Run HydroSetSpark on (up to K_eval) clips of the source → softmax → P_ss
  • Final source prob = w * P_hs + (1-w) * P_ss
  • argmax for class prediction

Honest: weight ``w`` is selected on VAL macro-F1 (not test), then test is touched once.
"""

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
from models.hydro_set_spark import HydroSetSpark

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


def load_spark(ckpt, device):
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


def load_setspark(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    a = ck["args"]
    dils = tuple(int(d) for d in a["tcn_dilations"].split(","))
    m = HydroSetSpark(
        num_classes=4, n_bands=a["n_bands"],
        tcn_dilations=dils, expansion=a["expansion"],
        embed_dim=a["embed_dim"], n_heads=a["n_heads"],
        head_hidden=a["head_hidden"], dropout=a["dropout"],
        band_dropout=a.get("band_dropout", 0.0),
        use_delta=not a.get("no_delta", False),
        use_coherence=not a.get("no_coherence", False),
        gambler=True,
    ).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m, a


def per_clip_probs(model, files, device, args, batch_size=256, n_workers=12):
    """Return P_hs (n_files, 4) for HydroSpark per-clip predictions."""
    target_rms = args.get("target_rms", 0.1); hpf_hz = args.get("hpf_hz", 20.0)
    rms = args.get("rms_normalize", True)
    executor = ThreadPoolExecutor(max_workers=n_workers)
    probs = np.zeros((len(files), 4), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            audio = np.stack(list(executor.map(_read, batch))).astype(np.float32)
            wav = torch.from_numpy(audio).to(device)
            if rms:
                wav = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-9) * target_rms
            if hpf_hz > 0:
                wav = _hpf(wav, 5120, hpf_hz, 4)
            logits = model(wav)
            probs[i:i+len(batch)] = F.softmax(logits[:, :4], dim=-1).float().cpu().numpy()
    executor.shutdown(wait=True)
    return probs


def per_source_setspark(model, sources_dict, device, args,
                        K_eval=32, batch_sources=8, n_workers=12):
    target_rms = args.get("target_rms", 0.1); hpf_hz = args.get("hpf_hz", 20.0)
    items = []
    for ci, c in enumerate(CLASSES):
        for sid, paths in sources_dict.get(c, {}).items():
            items.append((f"{c}_{sid}", ci, paths))
    src_ids = [it[0] for it in items]
    src_lab = np.array([it[1] for it in items], dtype=np.int64)
    out_probs = np.zeros((len(items), 4), dtype=np.float32)
    executor = ThreadPoolExecutor(max_workers=n_workers)
    for bs in range(0, len(items), batch_sources):
        chunk = items[bs:bs+batch_sources]
        clips_lists, masks = [], []
        for _, _, paths in chunk:
            if len(paths) >= K_eval:
                clips_lists.append(sorted(paths)[:K_eval])
                masks.append([True]*K_eval)
            else:
                pad = K_eval - len(paths)
                clips_lists.append(sorted(paths) + [paths[0]]*pad)
                masks.append([True]*len(paths) + [False]*pad)
        flat = [p for lst in clips_lists for p in lst]
        audio = np.stack(list(executor.map(_read, flat))).astype(np.float32)
        wav = torch.from_numpy(audio).to(device)
        wav = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-9) * target_rms
        if hpf_hz > 0:
            wav = _hpf(wav, 5120, hpf_hz, 4)
        wav = wav.view(len(chunk), K_eval, -1)
        mask = torch.tensor(masks, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = model(wav, mask=mask)
        p = F.softmax(logits[:, :4], dim=-1).float().cpu().numpy()
        out_probs[bs:bs+len(chunk)] = p
    executor.shutdown(wait=True)
    return src_ids, src_lab, out_probs


def enumerate_split(data_dir, split):
    sources, files, labels = defaultdict(lambda: defaultdict(list)), [], []
    for ci, c in enumerate(CLASSES):
        for fp in sorted((Path(data_dir) / split / c).glob("*.wav")):
            sources[c][_source(str(fp))[len(c)+1:]].append(str(fp))
            files.append(str(fp)); labels.append(ci)
    return dict(sources), files, np.array(labels)


def hs_per_source(files, labels, probs):
    """Per-source log-mean of probs for HydroSpark predictions."""
    src_probs, src_label = defaultdict(list), {}
    for i, fp in enumerate(files):
        s = _source(fp); src_probs[s].append(probs[i]); src_label[s] = labels[i]
    src_list = sorted(src_probs.keys())
    out = np.zeros((len(src_list), 4))
    for j, s in enumerate(src_list):
        P = np.stack(src_probs[s])
        out[j] = np.exp(np.log(np.clip(P, 1e-8, 1.0)).mean(axis=0))
    return src_list, np.array([src_label[s] for s in src_list]), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/var/mnt/5A009BF8009BD8F9/Data/Deepship_1s")
    ap.add_argument("--spark_ckpts", nargs="+", required=True)
    ap.add_argument("--setspark_ckpts", nargs="+", required=True)
    ap.add_argument("--out_dir", default="lightning_logs/cross_ens")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--K_eval", type=int, default=32)
    ap.add_argument("--w_grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="Comma-separated mixing weights of P_hs (per-clip ensemble).")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    val_src, val_files, val_y = enumerate_split(args.data_dir, "val")
    test_src, test_files, test_y = enumerate_split(args.data_dir, "test")
    print(f"VAL: {len(val_files)} clips / {sum(len(s) for s in val_src.values())} sources")
    print(f"TEST: {len(test_files)} clips / {sum(len(s) for s in test_src.values())} sources")

    # ── HydroSpark per-clip (averaged across seeds) ──
    hs_args = None
    val_hs_sum = np.zeros((len(val_files), 4))
    test_hs_sum = np.zeros((len(test_files), 4))
    for c in args.spark_ckpts:
        m, a = load_spark(c, device)
        print(f"  HydroSpark loaded {c}")
        val_hs_sum += per_clip_probs(m, val_files, device, a, batch_size=args.batch_size)
        test_hs_sum += per_clip_probs(m, test_files, device, a, batch_size=args.batch_size)
        hs_args = a
        del m; torch.cuda.empty_cache()
    val_hs_sum /= len(args.spark_ckpts); test_hs_sum /= len(args.spark_ckpts)

    val_hs_src, val_hs_y, val_hs_psrc = hs_per_source(val_files, val_y, val_hs_sum)
    test_hs_src, test_hs_y, test_hs_psrc = hs_per_source(test_files, test_y, test_hs_sum)

    # ── HydroSetSpark per-source (averaged across seeds) ──
    val_ss_sum_dict, test_ss_sum_dict = None, None
    val_ss_src_keys, test_ss_src_keys = None, None
    val_ss_y, test_ss_y = None, None
    for c in args.setspark_ckpts:
        m, a = load_setspark(c, device)
        print(f"  HydroSetSpark loaded {c}")
        v_ids, v_y, v_p = per_source_setspark(m, val_src, device, a, K_eval=args.K_eval)
        t_ids, t_y, t_p = per_source_setspark(m, test_src, device, a, K_eval=args.K_eval)
        if val_ss_sum_dict is None:
            val_ss_sum_dict = {s: v_p[i] for i, s in enumerate(v_ids)}
            test_ss_sum_dict = {s: t_p[i] for i, s in enumerate(t_ids)}
            val_ss_src_keys = v_ids; test_ss_src_keys = t_ids
            val_ss_y = v_y; test_ss_y = t_y
        else:
            for i, s in enumerate(v_ids):
                val_ss_sum_dict[s] = val_ss_sum_dict[s] + v_p[i]
            for i, s in enumerate(t_ids):
                test_ss_sum_dict[s] = test_ss_sum_dict[s] + t_p[i]
        del m; torch.cuda.empty_cache()
    n_ss = len(args.setspark_ckpts)
    val_ss_p = np.stack([val_ss_sum_dict[s] / n_ss for s in val_ss_src_keys])
    test_ss_p = np.stack([test_ss_sum_dict[s] / n_ss for s in test_ss_src_keys])
    val_ss_src = val_ss_src_keys; test_ss_src = test_ss_src_keys

    # Align by source id
    def align(src_a, prob_a, src_b, prob_b, labels_a):
        idx = {s: i for i, s in enumerate(src_b)}
        kept = [(s, i, idx[s]) for i, s in enumerate(src_a) if s in idx]
        srcs = [k[0] for k in kept]
        pa = np.stack([prob_a[k[1]] for k in kept])
        pb = np.stack([prob_b[k[2]] for k in kept])
        labs = np.array([labels_a[k[1]] for k in kept])
        return srcs, labs, pa, pb

    val_srcs, val_labs, val_pa, val_pb = align(val_hs_src, val_hs_psrc, val_ss_src, val_ss_p, val_hs_y)
    test_srcs, test_labs, test_pa, test_pb = align(test_hs_src, test_hs_psrc, test_ss_src, test_ss_p, test_hs_y)

    # ── Sweep mix weights on VAL, pick best, report TEST ──
    weights = [float(x) for x in args.w_grid.split(",")]
    val_curve = []
    best_w, best_val_f1 = None, -1.0
    for w in weights:
        p = w * val_pa + (1 - w) * val_pb
        cm = _confusion(val_labs, p.argmax(-1), 4)
        _, _, _, mP, mR, mF = _macro(cm)
        val_curve.append({"w": w, "macroP": float(mP), "macroR": float(mR), "macroF1": float(mF)})
        if mF > best_val_f1:
            best_val_f1 = mF; best_w = w
    print(f"\nVAL mixing sweep: best w (HydroSpark weight) = {best_w:.2f}, val macroF1 = {best_val_f1:.4f}")
    for r in val_curve:
        print(f"  w={r['w']:.2f}  val macroP={r['macroP']:.4f}  macroR={r['macroR']:.4f}  macroF1={r['macroF1']:.4f}")

    # Report test at best_w
    p_test = best_w * test_pa + (1 - best_w) * test_pb
    yp = p_test.argmax(-1)
    cm = _confusion(test_labs, yp, 4)
    p_, r_, f_, mP, mR, mF = _macro(cm)
    print()
    print("═" * 72)
    print(f"TEST  (cross-ensemble, val-selected w={best_w:.2f}, n_sources={len(test_srcs)})")
    print(f"  macroP = {mP:.4f}")
    print(f"  macroR = {mR:.4f}")
    print(f"  macroF1= {mF:.4f}")
    print(f"  per-class F1: {dict(zip(CLASSES, f_.round(4)))}")
    print(f"  cm = {cm.tolist()}")
    print("═" * 72)

    with open(out / "result.json", "w") as f:
        json.dump({
            "spark_ckpts": args.spark_ckpts, "setspark_ckpt": args.setspark_ckpt,
            "val_curve": val_curve, "best_w": best_w,
            "best_val_f1": float(best_val_f1),
            "test_per_source": {
                "n": int(len(test_srcs)), "cm": cm.tolist(),
                "macroP": float(mP), "macroR": float(mR), "macroF1": float(mF),
                "per_class_F1": f_.tolist(),
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
