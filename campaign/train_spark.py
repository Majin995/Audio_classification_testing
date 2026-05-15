"""
Train HydroSpark and report HONEST val + test metrics.

Honesty contract
────────────────
  • Train on the *train* split only.
  • Use the *val* split for early-stopping and final-epoch selection.
  • Touch the *test* split EXACTLY ONCE, at the very end, using the val-selected
    checkpoint state.

For the rapid dataset (1 s clips, 4 classes), eval is per-clip. For
Classifier_Dataset, per-source aggregation (log-mean) is enabled via the
``--per_source`` flag — sources are derived from the filename pattern
``<class>_<kind>_<sourceID>_<chunkIdx>.wav``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_spark import HydroSpark


# ───────────────────────────── Args ─────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="lightning_logs/hydro_spark")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--n_bands", type=int, default=24)
    p.add_argument("--sinc_kernel", type=int, default=257)
    p.add_argument("--env_decim", type=int, default=32)
    p.add_argument("--tcn_kernel", type=int, default=7)
    p.add_argument("--tcn_dilations", default="1,4,16",
                   help="Comma-separated dilations for the TCN blocks.")
    p.add_argument("--expansion", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--head_hidden", type=int, default=48)
    p.add_argument("--smoothing", type=float, default=0.05)
    p.add_argument("--gambler", type=float, default=0.1,
                   help="weight on the gambler abstention auxiliary loss")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per_source", action="store_true",
                   help="Aggregate per-source by log-mean of softmax probs.")
    p.add_argument("--rms_normalize", action="store_true", default=True)
    p.add_argument("--no_rms", dest="rms_normalize", action="store_false")
    p.add_argument("--target_rms", type=float, default=0.1)
    p.add_argument("--hpf_hz", type=float, default=20.0)
    p.add_argument("--real_only", action="store_true",
                   help="Restrict train/val/test to files matching *_real_*.wav. "
                        "Use this on Classifier_Dataset to enforce the no-synth "
                        "constraint.")
    p.add_argument("--mixup", type=float, default=0.2,
                   help="Mixup α — 0 disables.")
    p.add_argument("--band_dropout", type=float, default=0.10,
                   help="Probability that any one sinc band is zeroed in train.")
    p.add_argument("--select_metric", default="macroF1",
                   choices=["macroF1", "macroP"],
                   help="Validation metric used for checkpoint selection.")
    p.add_argument("--freeze_sinc", action="store_true",
                   help="Freeze the parametric SincBank at its mel-spaced init.")
    p.add_argument("--no_log_energy", action="store_true",
                   help="Disable the per-band log-energy head input.")
    p.add_argument("--no_delta", action="store_true",
                   help="Disable the delta-energy head input.")
    p.add_argument("--no_coherence", action="store_true",
                   help="Disable the cross-band coherence head input.")
    p.add_argument("--focal_gamma", type=float, default=0.0,
                   help="Focal-loss γ. 0 = standard CE.")
    p.add_argument("--class_weight_pow", type=float, default=0.0,
                   help="Class-weight = (max_count/count)^pow. 0 = uniform.")
    p.add_argument("--denoise", default="off",
                   choices=["off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"],
                   help="Audio pre-processing denoising (CPU-side, after DALI).")
    return p.parse_args()


# ───────────────────────────── Honest metrics ───────────────────────────────

def _confusion(y_true, y_pred, K):
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _macro_prf1(cm):
    K = cm.shape[0]
    p, r, f = np.zeros(K), np.zeros(K), np.zeros(K)
    for k in range(K):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        p[k] = tp / max(tp + fp, 1)
        r[k] = tp / max(tp + fn, 1)
        f[k] = (2 * p[k] * r[k]) / max(p[k] + r[k], 1e-9)
    return p, r, f, p.mean(), r.mean(), f.mean()


def _source_from_filename(p: str) -> str:
    """Return a source identifier from a clip filename.

    Two conventions handled:
      • Classifier_Dataset (4 parts): <class>_<kind>_<sourceID>_<chunkIdx>.wav
        → source = <class>_<kind>_<sourceID>
      • Deepship_1s (3 parts):       <class>_<sourceID>_<chunkIdx>.wav
        → source = <class>_<sourceID>
    """
    name = Path(p).stem
    parts = name.split("_")
    if len(parts) >= 4:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    if len(parts) == 3:
        return f"{parts[0]}_{parts[1]}"
    return name


# ───────────────────────────── Loss ─────────────────────────────────────────

class _GamblerCE(torch.nn.Module):
    """CE + label smoothing + optional class-weighted focal CE + Gambler aux.

    logits: (B, K+1) where the last logit is the abstain logit.

    ``class_weight``: (K,) torch tensor or None.
    ``focal_gamma``: 0 disables focal weighting.
    """

    def __init__(self, K: int, smoothing: float = 0.05, gambler_w: float = 0.1,
                 abstain_o: float = 2.2,
                 class_weight: "torch.Tensor | None" = None,
                 focal_gamma: float = 0.0):
        super().__init__()
        self.K = int(K)
        self.smoothing = float(smoothing)
        self.gambler_w = float(gambler_w)
        self.abstain_o = float(abstain_o)
        self.focal_gamma = float(focal_gamma)
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.float())
        else:
            self.class_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        K = self.K
        ce_logits = logits[:, :K]
        log_p = F.log_softmax(ce_logits, dim=-1)
        p = log_p.exp()
        if self.smoothing > 0:
            n = ce_logits.size(1)
            true_dist = torch.full_like(log_p, self.smoothing / (n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            per_sample = -(true_dist * log_p).sum(dim=-1)              # (B,)
        else:
            per_sample = F.nll_loss(log_p, target, reduction="none")
        # Focal modulation
        if self.focal_gamma > 0:
            py = p.gather(1, target.unsqueeze(1)).squeeze(1)
            per_sample = per_sample * (1 - py).clamp(0, 1).pow(self.focal_gamma)
        # Class-weight
        if self.class_weight is not None:
            w = self.class_weight[target]
            per_sample = per_sample * w
        ce = per_sample.mean()

        if self.gambler_w == 0 or logits.size(1) == K:
            return ce
        p_full = F.softmax(logits, dim=-1)
        py = p_full.gather(1, target.unsqueeze(1)).squeeze(1)
        pabs = p_full[:, K]
        gambler = -torch.log(py + pabs / self.abstain_o + 1e-8).mean()
        return ce + self.gambler_w * gambler


# ───────────────────────────── Train loop ───────────────────────────────────

def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    dm = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        oversample_train=True,
        rms_normalize=args.rms_normalize,
        target_rms=args.target_rms,
        hpf_hz=args.hpf_hz,
        hpf_order=4,
        denoise_method=args.denoise,
    )
    dm.setup()

    if args.real_only:
        # Filter every split to ``*_real_*.wav`` files only — synth samples
        # are disqualified per the active goal.
        def _real_filter(files_by_class):
            return {c: [p for p in fs if "_real_" in Path(p).name]
                    for c, fs in files_by_class.items()}
        real_train = _real_filter(dm._apply_merge(dm._scan_split("train")))
        dm.set_train_files_override(real_train)
        # Patch _scan_split so val/test loaders also become real-only.
        _orig_scan = dm._scan_split
        def _real_scan(split):
            return _real_filter(_orig_scan(split))
        dm._scan_split = _real_scan
        # Print new counts
        print("REAL-ONLY filter applied. Counts:")
        for split in ("train", "val", "test"):
            sub = _real_scan(split)
            counts = {c: len(v) for c, v in sub.items()}
            print(f"  {split}: {counts}")
    K = dm.num_classes
    idx_to_class = dm.idx_to_class
    classes = [idx_to_class[i] for i in range(K)]
    print(f"Classes: {classes}")

    device = torch.device("cuda")
    tcn_dilations = tuple(int(d) for d in args.tcn_dilations.split(","))
    model = HydroSpark(
        num_classes=K, n_bands=args.n_bands, sinc_kernel=args.sinc_kernel,
        env_decim=args.env_decim, tcn_kernel=args.tcn_kernel,
        tcn_dilations=tcn_dilations, expansion=args.expansion,
        head_hidden=args.head_hidden, dropout=args.dropout,
        band_dropout=args.band_dropout,
        freeze_sinc=args.freeze_sinc,
        use_log_energy=not args.no_log_energy,
        use_delta=not args.no_delta,
        use_coherence=not args.no_coherence,
        gambler=True,
    ).to(device)
    print(f"params = {model.n_params():,}")

    # Class weights from training counts (real-only filter applied).
    cls_counts = []
    train_files_now = (dm.train_files_override
                       if dm.train_files_override is not None
                       else dm._apply_merge(dm._scan_split("train")))
    for c in classes:
        cls_counts.append(max(1, len(train_files_now.get(c, []))))
    cls_counts = np.array(cls_counts, dtype=np.float64)
    if args.class_weight_pow > 0:
        w = (cls_counts.max() / cls_counts) ** args.class_weight_pow
        w = w / w.mean()
        print(f"class_weight = {dict(zip(classes, w.round(3)))}")
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        class_weight = None

    criterion = _GamblerCE(K, smoothing=args.smoothing, gambler_w=args.gambler,
                           class_weight=class_weight,
                           focal_gamma=args.focal_gamma)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.wd)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    # OneCycle schedule properly sized to the actual number of steps.
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos",
    )

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improve = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time(); n_seen = 0; loss_sum = 0.0; correct = 0
        for audio, label in train_loader:
            audio = audio.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # Waveform Mixup (regularizer — same family as classic Mixup, but
            # on the raw waveform). Mixing in the time domain is well-defined
            # for additive sounds.
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                idx = torch.randperm(audio.size(0), device=device)
                audio = lam * audio + (1 - lam) * audio[idx]
                label_b = label[idx]
            else:
                lam = 1.0; label_b = label

            logits = model(audio)
            if args.mixup > 0 and lam < 1.0:
                loss = lam * criterion(logits, label) + (1 - lam) * criterion(logits, label_b)
            else:
                loss = criterion(logits, label)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            try: sched.step()
            except Exception: pass
            n_seen += audio.size(0)
            loss_sum += loss.item() * audio.size(0)
            correct += (logits[:, :K].argmax(-1) == label).sum().item()

        train_loss = loss_sum / max(n_seen, 1)
        train_acc = correct / max(n_seen, 1)

        # Val
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for audio, label in val_loader:
                audio = audio.to(device, non_blocking=True)
                logits = model(audio)
                val_logits.append(logits[:, :K].float().cpu().numpy())
                val_labels.append(label.cpu().numpy())
        val_logits = np.concatenate(val_logits)
        val_labels = np.concatenate(val_labels)
        val_pred = val_logits.argmax(-1)
        cm = _confusion(val_labels, val_pred, K)
        _, _, _, mP, mR, mF = _macro_prf1(cm)

        dt = time.time() - t0
        print(
            f"epoch {epoch:3d} | {dt:5.1f}s | train loss={train_loss:.4f} "
            f"acc={train_acc:.3f} | VAL macroP={mP:.4f} macroR={mR:.4f} "
            f"macroF1={mF:.4f}"
        )

        sel = mP if args.select_metric == "macroP" else mF
        if sel > best_val_f1 + 1e-4:
            best_val_f1 = sel
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_without_improve = 0
            print(f"   ↳ new best val {args.select_metric}: {best_val_f1:.4f}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"   ↳ early stop at epoch {epoch} "
                      f"(no val {args.select_metric} improvement for "
                      f"{args.patience} epochs)")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ─── Honest test eval (touched once) ───
    # IMPORTANT: dm.make_score_loader() uses the audio_pipeline DEFAULTS —
    # it does NOT respect dm.rms_normalize, dm.target_rms, or dm.hpf_hz. So
    # the audio it yields is preprocessed DIFFERENTLY from the train/val
    # loaders. To keep the test eval honest, we apply the same waveform
    # preprocessing manually here (soundfile read → resample if needed →
    # pad/crop → RMS-normalize → HPF) so the test waveform distribution
    # matches what the model saw at train time.
    import soundfile as sf
    import torchaudio.functional as AF

    test_files_by_class = dm._apply_merge(dm._scan_split("test"))
    test_files, test_labels_list = [], []
    for cls in classes:
        for path in test_files_by_class.get(cls, []):
            test_files.append(path)
            test_labels_list.append(dm.class_to_idx[cls])
    test_labels = np.array(test_labels_list, dtype=np.int64)

    target_sr = dm.target_sr
    fixed_len = dm.fixed_len

    def _read_batch(paths):
        wavs = []
        for p in paths:
            a, sr = sf.read(p, dtype="float32", always_2d=False)
            if a.ndim > 1:
                a = a.mean(axis=1)
            if sr != target_sr:
                ta = torch.from_numpy(a)
                ta = AF.resample(ta, sr, target_sr)
                a = ta.numpy()
            if len(a) < fixed_len:
                a = np.pad(a, (0, fixed_len - len(a)))
            else:
                a = a[:fixed_len]
            wavs.append(a)
        return np.stack(wavs).astype(np.float32)

    def _hpf_torch(x: torch.Tensor, sr: int, cutoff: float, order: int = 4):
        if cutoff <= 0:
            return x
        T = x.shape[-1]
        freqs = torch.fft.rfftfreq(T, 1.0 / sr).to(x.device)
        ratio = freqs / max(cutoff, 1e-9)
        mag = ratio.pow(2 * order)
        mask = (mag / (1.0 + mag)).sqrt().to(x.dtype)
        Xf = torch.fft.rfft(x.float(), dim=-1)
        return torch.fft.irfft(Xf * mask, n=T, dim=-1).to(x.dtype)

    test_probs = np.zeros((len(test_files), K), dtype=np.float32)
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, len(test_files), bs):
            batch = test_files[i:i + bs]
            wav_np = _read_batch(batch)
            wav = torch.from_numpy(wav_np).to(device)
            # Match DALI's preprocessing exactly.
            if args.rms_normalize:
                wav = (wav - wav.mean(dim=1, keepdim=True)) / (
                    wav.std(dim=1, keepdim=True) + 1e-9
                ) * args.target_rms
            if args.hpf_hz > 0:
                wav = _hpf_torch(wav, target_sr, args.hpf_hz, 4)
            if args.denoise != "off":
                from processing.denoise.transform import DenoiseTransform
                if not hasattr(args, "_denoise_t"):
                    args._denoise_t = DenoiseTransform(
                        method=args.denoise, sample_rate=target_sr
                    )
                wav = args._denoise_t(wav)
            logits = model(wav)
            p = F.softmax(logits[:, :K], dim=-1).float().cpu().numpy()
            test_probs[i:i + len(batch)] = p
            if i % (bs * 20) == 0:
                print(f"   test inference {i + len(batch):>6d} / {len(test_files)}")

    # Per-clip metrics
    per_clip_pred = test_probs.argmax(-1)
    cm = _confusion(test_labels, per_clip_pred, K)
    _, _, fcls, mP, mR, mF = _macro_prf1(cm)
    print()
    print("═" * 72)
    print(f"HONEST PER-CLIP TEST  (n={len(test_files)}):")
    print(f"  macroP = {mP:.4f}")
    print(f"  macroR = {mR:.4f}")
    print(f"  macroF1= {mF:.4f}")
    print(f"  cm = {cm.tolist()}")
    print("═" * 72)

    out = {
        "params": int(model.n_params()),
        "args": vars(args),
        "classes": classes,
        "best_val_f1": float(best_val_f1),
        "test_per_clip": {
            "n": int(len(test_files)),
            "cm": cm.tolist(),
            "macroP": float(mP),
            "macroR": float(mR),
            "macroF1": float(mF),
            "per_class_F1": fcls.tolist(),
        },
    }

    # Per-source metrics
    if args.per_source:
        source_probs = defaultdict(list)
        source_label = {}
        for i, path in enumerate(test_files):
            src = _source_from_filename(path)
            source_probs[src].append(test_probs[i])
            source_label[src] = test_labels[i]
        src_list = sorted(source_probs.keys())
        # log-mean of probs (clipped)
        agg = np.zeros((len(src_list), K), dtype=np.float64)
        for i, s in enumerate(src_list):
            P = np.stack(source_probs[s])
            agg[i] = np.exp(np.log(np.clip(P, 1e-8, 1.0)).mean(axis=0))
        per_src_pred = agg.argmax(-1)
        per_src_label = np.array([source_label[s] for s in src_list])
        cm_s = _confusion(per_src_label, per_src_pred, K)
        _, _, fcls_s, mP_s, mR_s, mF_s = _macro_prf1(cm_s)
        print(f"HONEST PER-SOURCE TEST  (n_sources={len(src_list)}):")
        print(f"  macroP = {mP_s:.4f}")
        print(f"  macroR = {mR_s:.4f}")
        print(f"  macroF1= {mF_s:.4f}")
        print(f"  cm = {cm_s.tolist()}")
        print("═" * 72)
        out["test_per_source"] = {
            "n_sources": int(len(src_list)),
            "cm": cm_s.tolist(),
            "macroP": float(mP_s),
            "macroR": float(mR_s),
            "macroF1": float(mF_s),
            "per_class_F1": fcls_s.tolist(),
        }

    # Save
    if best_state is not None:
        torch.save({"state_dict": best_state, "args": vars(args),
                    "classes": classes},
                   out_dir / "best.pt")
    with open(out_dir / "result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
