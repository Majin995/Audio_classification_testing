"""
Train HydroSetSpark — source-level training and honest val/test eval.

Each training step samples N sources (class-balanced) and K clips per source,
reads them with soundfile, preprocesses (RMS norm + HPF) and runs them
through the model in a single forward. Loss = source-level cross-entropy.

Honest contract
───────────────
  • Train: real-only train sources.
  • Val:   real-only val sources, all clips per source aggregated by the
           attention pool itself (no log-mean post-processing — the model's
           own pool IS the aggregator). Used for model selection.
  • Test:  touched ONCE at the end on real-only test sources.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from models.hydro_set_spark import HydroSetSpark


CLASSES = ["Cargo", "Passenger", "Tanker", "Tug"]


# ─────────────────────────── Source enumeration ─────────────────────────────

def scan_real_split(data_dir: str, split: str) -> dict[str, dict[str, list[str]]]:
    """Returns {class: {source_id: [abs_paths]}} for the given split.

    Two filename conventions handled:
      • Classifier_Dataset: <class>_<kind>_<sourceID>_<chunkIdx>.wav
        (here we filter to kind=='real')
      • Deepship_1s:        <class>_<sourceID>_<chunkIdx>.wav
        (no kind marker — all are real)
    """
    out: dict[str, dict[str, list[str]]] = {c: {} for c in CLASSES}
    for c in CLASSES:
        d = Path(data_dir) / split / c
        if not d.is_dir():
            continue
        for p in sorted(d.glob(f"{c}_*.wav")):
            parts = p.stem.split("_")
            if len(parts) >= 4:
                if parts[1] != "real":
                    continue
                src_id = parts[2]
            elif len(parts) == 3:
                src_id = parts[1]
            else:
                continue
            out[c].setdefault(src_id, []).append(str(p))
    return out


# ─────────────────────────── Audio I/O ─────────────────────────────────────

def _read_clip(path, target_sr=5120, fixed_len=5120):
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1: a = a.mean(axis=1)
    if sr != target_sr:
        a = AF.resample(torch.from_numpy(a), sr, target_sr).numpy()
    if len(a) < fixed_len:
        a = np.pad(a, (0, fixed_len - len(a)))
    else:
        a = a[:fixed_len]
    return a


def read_batch_threaded(paths, executor, target_sr=5120, fixed_len=5120):
    fn = lambda p: _read_clip(p, target_sr, fixed_len)
    return np.stack(list(executor.map(fn, paths))).astype(np.float32)


def hpf_torch(x, sr, cutoff, order=4):
    if cutoff <= 0: return x
    T = x.shape[-1]
    freqs = torch.fft.rfftfreq(T, 1.0 / sr).to(x.device)
    ratio = freqs / max(cutoff, 1e-9)
    mag = ratio.pow(2 * order)
    mask = (mag / (1.0 + mag)).sqrt().to(x.dtype)
    Xf = torch.fft.rfft(x.float(), dim=-1)
    return torch.fft.irfft(Xf * mask, n=T, dim=-1).to(x.dtype)


def preprocess(audio: torch.Tensor, target_rms: float, hpf_hz: float,
               sr: int = 5120) -> torch.Tensor:
    # audio: (..., T). RMS-normalize per clip → HPF.
    x = audio
    x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-9) * target_rms
    x = hpf_torch(x, sr, hpf_hz, 4)
    return x


# ─────────────────────────── Source batch sampler ───────────────────────────

class SourceSampler:
    """Class-balanced sampler that yields (sources, labels, clip_paths) per batch.

    Each draw: pick ``per_class`` sources from each class (with replacement
    if a class has fewer than ``per_class`` sources), and ``K`` random clip
    paths from each source.
    """

    def __init__(self, sources_by_class: dict, per_class: int, K: int,
                 rng: random.Random):
        self.sources_by_class = {
            c: [(sid, paths) for sid, paths in d.items()]
            for c, d in sources_by_class.items()
        }
        self.per_class = int(per_class)
        self.K = int(K)
        self.rng = rng
        self.classes = [c for c in CLASSES if self.sources_by_class.get(c)]

    def __iter__(self):
        while True:
            yield self.draw()

    def draw(self):
        srcs, labels, clip_paths = [], [], []
        for ci, c in enumerate(self.classes):
            pool = self.sources_by_class[c]
            replace = len(pool) < self.per_class
            picks = (self.rng.choices(pool, k=self.per_class) if replace
                     else self.rng.sample(pool, self.per_class))
            for sid, paths in picks:
                if len(paths) >= self.K:
                    clips = self.rng.sample(paths, self.K)
                else:
                    clips = self.rng.choices(paths, k=self.K)
                srcs.append(f"{c}_{sid}")
                labels.append(ci)
                clip_paths.append(clips)
        return srcs, np.array(labels, dtype=np.int64), clip_paths


# ─────────────────────────── Metrics ────────────────────────────────────────

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


# ─────────────────────────── Eval (val / test) ──────────────────────────────

@torch.no_grad()
def eval_split(model, sources_by_class, executor, device,
               K_eval: int, target_rms: float, hpf_hz: float,
               batch_size_sources: int = 8):
    """Evaluate one source per group, using up to K_eval clips per source.

    For sources with > K_eval clips, we sample K_eval ones at random (we use
    a deterministic shuffle by sorting paths). For sources with < K_eval, we
    pad with mask=False — the attention pool ignores those positions.
    """
    model.eval()
    K = 4  # num classes
    items = []
    for ci, c in enumerate(CLASSES):
        for sid, paths in sources_by_class.get(c, {}).items():
            items.append((f"{c}_{sid}", ci, paths))

    y_true = np.array([it[1] for it in items], dtype=np.int64)
    y_pred = np.zeros(len(items), dtype=np.int64)
    probs = np.zeros((len(items), K), dtype=np.float32)

    for batch_start in range(0, len(items), batch_size_sources):
        chunk = items[batch_start:batch_start + batch_size_sources]
        clip_lists = []
        masks = []
        for sid, ci, paths in chunk:
            if len(paths) >= K_eval:
                # deterministic: take first K_eval in sorted order
                clip_lists.append(sorted(paths)[:K_eval])
                masks.append([True] * K_eval)
            else:
                pad_n = K_eval - len(paths)
                clip_lists.append(sorted(paths) + [paths[0]] * pad_n)
                masks.append([True] * len(paths) + [False] * pad_n)
        flat_paths = [p for lst in clip_lists for p in lst]
        audio = read_batch_threaded(flat_paths, executor)
        wav = torch.from_numpy(audio).to(device)
        wav = preprocess(wav, target_rms, hpf_hz)
        wav = wav.view(len(chunk), K_eval, -1)
        mask = torch.tensor(masks, dtype=torch.bool, device=device)
        logits = model(wav, mask=mask)
        p = F.softmax(logits[:, :K], dim=-1).float().cpu().numpy()
        probs[batch_start:batch_start + len(chunk)] = p
        y_pred[batch_start:batch_start + len(chunk)] = p.argmax(-1)
    return y_true, y_pred, probs


# ─────────────────────────── Args & main ────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/var/mnt/5A009BF8009BD8F9/Data/Classifier_Dataset")
    p.add_argument("--out_dir", default="lightning_logs/setspark")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--per_class", type=int, default=4, help="sources per class per batch")
    p.add_argument("--K_train", type=int, default=12, help="clips per source at train")
    p.add_argument("--K_eval", type=int, default=32, help="clips per source at eval")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--n_bands", type=int, default=24)
    p.add_argument("--embed_dim", type=int, default=48)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--head_hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--band_dropout", type=float, default=0.10)
    p.add_argument("--tcn_dilations", default="1,4,16,64")
    p.add_argument("--expansion", type=int, default=1)
    p.add_argument("--no_delta", action="store_true")
    p.add_argument("--no_coherence", action="store_true")
    p.add_argument("--smoothing", type=float, default=0.05)
    p.add_argument("--mixup", type=float, default=0.0,
                   help="Source-level Mixup α (0=off).")
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--patience", type=int, default=10,
                   help="patience in val_every-step units")
    p.add_argument("--target_rms", type=float, default=0.1)
    p.add_argument("--hpf_hz", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_workers", type=int, default=12)
    p.add_argument("--select_metric", default="macroF1",
                   choices=["macroF1", "macroP"])
    p.add_argument("--encoder_init", default="",
                   help="Path to a HydroSpark best.pt to initialize the encoder "
                        "(loads sinc, env, pcen, tcn weights — projection is "
                        "re-initialized from scratch).")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze the encoder during source-level fine-tuning.")
    p.add_argument("--focal_gamma", type=float, default=0.0)
    p.add_argument("--class_weight_pow", type=float, default=0.0)
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # Scan sources
    train_src = scan_real_split(args.data_dir, "train")
    val_src   = scan_real_split(args.data_dir, "val")
    test_src  = scan_real_split(args.data_dir, "test")
    for split, d in (("train", train_src), ("val", val_src), ("test", test_src)):
        counts = {c: len(d[c]) for c in CLASSES}
        clips = {c: sum(len(p) for p in d[c].values()) for c in CLASSES}
        print(f"{split:>5s} sources: {counts}  clips: {clips}")

    rng = random.Random(args.seed)
    sampler = SourceSampler(train_src, per_class=args.per_class,
                            K=args.K_train, rng=rng)

    tcn_dils = tuple(int(d) for d in args.tcn_dilations.split(","))
    model = HydroSetSpark(
        num_classes=4, n_bands=args.n_bands,
        tcn_dilations=tcn_dils, expansion=args.expansion,
        embed_dim=args.embed_dim, n_heads=args.n_heads,
        head_hidden=args.head_hidden, dropout=args.dropout,
        band_dropout=args.band_dropout,
        use_delta=not args.no_delta, use_coherence=not args.no_coherence,
        gambler=True,
    ).to(device)
    print(f"HydroSetSpark params = {model.n_params():,}")

    # Optional encoder pre-init from a per-clip HydroSpark checkpoint
    if args.encoder_init:
        ck = torch.load(args.encoder_init, map_location="cpu", weights_only=False)
        sd_src = ck["state_dict"]
        loaded = 0; skipped = 0
        target_sd = model.state_dict()
        for k, v in sd_src.items():
            # HydroSpark uses 'sinc', 'env', 'pcen', 'tcn', 'head' top-level.
            # SetSpark's encoder uses 'encoder.sinc', 'encoder.env', etc.
            new_k = f"encoder.{k}"
            if new_k in target_sd and target_sd[new_k].shape == v.shape:
                target_sd[new_k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(target_sd, strict=False)
        print(f"Pre-loaded {loaded} tensors from {args.encoder_init} "
              f"(skipped {skipped}).")
        if args.freeze_encoder:
            for n, p in model.named_parameters():
                if n.startswith("encoder.") and not n.startswith("encoder.proj"):
                    p.requires_grad_(False)
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Encoder frozen. Trainable params now: {n_train:,}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps,
        pct_start=0.1, anneal_strategy="cos",
    )

    executor = ThreadPoolExecutor(max_workers=args.n_workers)
    best_val, best_state, stale = -1.0, None, 0
    log = []

    sampler_iter = iter(sampler)
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        _, labels, clip_paths = next(sampler_iter)
        flat = [p for lst in clip_paths for p in lst]
        audio = read_batch_threaded(flat, executor)
        wav = torch.from_numpy(audio).to(device)
        wav = preprocess(wav, args.target_rms, args.hpf_hz)
        N = len(labels); K_t = args.K_train
        wav = wav.view(N, K_t, -1)
        y = torch.from_numpy(labels).to(device)

        # Build class weights once
        if step == 0 and args.class_weight_pow > 0:
            ct = np.array([len(train_src[c]) for c in CLASSES], dtype=np.float64)
            w = (ct.max() / ct) ** args.class_weight_pow
            w = w / w.mean()
            class_weight = torch.tensor(w, dtype=torch.float32, device=device)
            print(f"class_weight (source counts^{args.class_weight_pow}): "
                  f"{dict(zip(CLASSES, w.round(3)))}")
        elif step == 0:
            class_weight = None

        # Source-level Mixup
        if args.mixup > 0:
            lam = float(np.random.beta(args.mixup, args.mixup))
            idx = torch.randperm(N, device=device)
            # mix clip sets: take lam fraction from A and 1-lam from B (clip-wise)
            n_a = max(1, int(round(K_t * lam)))
            sel = torch.zeros(K_t, dtype=torch.long, device=device)
            sel[:n_a] = 0; sel[n_a:] = 1
            wav_b = wav[idx]
            mixed = torch.where(sel.view(1, K_t, 1).bool(), wav, wav_b)
            logits = model(mixed)
            loss = (n_a / K_t) * F.cross_entropy(logits[:, :4], y, label_smoothing=args.smoothing) \
                   + (1 - n_a / K_t) * F.cross_entropy(logits[:, :4], y[idx], label_smoothing=args.smoothing)
        else:
            logits = model(wav)
            log_p = F.log_softmax(logits[:, :4], dim=-1)
            ls = args.smoothing
            if ls > 0:
                n_cls = 4
                td = torch.full_like(log_p, ls / (n_cls - 1))
                td.scatter_(1, y.unsqueeze(1), 1.0 - ls)
                per_sample = -(td * log_p).sum(dim=-1)
            else:
                per_sample = F.nll_loss(log_p, y, reduction="none")
            if args.focal_gamma > 0:
                pp = log_p.exp().gather(1, y.unsqueeze(1)).squeeze(1)
                per_sample = per_sample * (1 - pp).clamp(0, 1).pow(args.focal_gamma)
            if class_weight is not None:
                per_sample = per_sample * class_weight[y]
            loss = per_sample.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        try: sched.step()
        except Exception: pass

        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            y_t, y_p, _ = eval_split(model, val_src, executor, device,
                                     K_eval=args.K_eval,
                                     target_rms=args.target_rms,
                                     hpf_hz=args.hpf_hz)
            cm = _confusion(y_t, y_p, 4)
            _, _, _, mP, mR, mF = _macro_prf1(cm)
            elapsed = time.time() - t0
            print(f"step {step+1:>5d} | {elapsed:6.1f}s | loss={loss.item():.4f} | "
                  f"VAL macroP={mP:.4f} macroR={mR:.4f} macroF1={mF:.4f}")
            log.append({"step": step + 1, "loss": float(loss.item()),
                        "val_macroP": float(mP), "val_macroR": float(mR),
                        "val_macroF1": float(mF), "val_cm": cm.tolist()})
            sel = mP if args.select_metric == "macroP" else mF
            if sel > best_val + 1e-4:
                best_val = sel
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
                print(f"   ↳ new best val {args.select_metric}: {best_val:.4f}")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"   ↳ early stop after {args.patience} validations without improvement.")
                    break

    # ─── HONEST TEST (touched ONCE) ───
    if best_state is not None:
        model.load_state_dict(best_state)
    y_t, y_p, probs = eval_split(model, test_src, executor, device,
                                 K_eval=args.K_eval,
                                 target_rms=args.target_rms,
                                 hpf_hz=args.hpf_hz)
    cm_test = _confusion(y_t, y_p, 4)
    p, r, fcls, mP, mR, mF = _macro_prf1(cm_test)
    print()
    print("═" * 72)
    print(f"HONEST PER-SOURCE TEST (real-only, n={len(y_t)})")
    print(f"  params={model.n_params():,}")
    print(f"  macroP = {mP:.4f}")
    print(f"  macroR = {mR:.4f}")
    print(f"  macroF1= {mF:.4f}")
    print(f"  per-class F1: {dict(zip(CLASSES, fcls.round(4)))}")
    print(f"  cm = {cm_test.tolist()}")
    print("═" * 72)

    # ─── Save ───
    if best_state is not None:
        torch.save({"state_dict": best_state, "args": vars(args)},
                   out_dir / "best.pt")
    with open(out_dir / "result.json", "w") as f:
        json.dump({
            "params": int(model.n_params()),
            "args": vars(args),
            "best_val": float(best_val),
            "test_per_source": {
                "n_sources": int(len(y_t)),
                "cm": cm_test.tolist(),
                "macroP": float(mP), "macroR": float(mR), "macroF1": float(mF),
                "per_class_F1": fcls.tolist(),
            },
            "log": log,
        }, f, indent=2)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
