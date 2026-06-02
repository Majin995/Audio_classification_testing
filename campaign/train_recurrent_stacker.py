"""Train HydroRecurrentStacker — recurrent + multi-scale with FROZEN 6-ckpt
ensemble per-clip features attached.

Per-clip features come from the cached NPZ
`campaign/probs_combined_recurrent_stacker.npz` produced by
dump_ensemble_combined.py. Lookup is by relative clip path; we precompute a
path → row-index dict once at startup.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from models.hydro_recurrent_stacker import HydroRecurrentStacker
from campaign.train_recurrent import (
    CLASSES, SourceSampler, read_batch_threaded, preprocess, scan_split,
    GamblerCE, macro_prf1,
)


# ─── Ensemble cache ────────────────────────────────────────────────────────

class EnsembleCache:
    """Random-access per-clip ensemble probs, keyed by ABS path."""

    def __init__(self, npz_path: Path, data_dir: Path):
        z = np.load(npz_path, allow_pickle=True)
        rel = z["paths"].tolist()
        self.probs = z["probs"]          # (M, N, 4)
        self.ckpt_names = list(z["ckpt_names"].tolist())
        # Build abs-path → row index
        self.idx = {str(data_dir / p): i for i, p in enumerate(rel)}
        # Shape convenience
        self.M = self.probs.shape[0]
        self.C = self.probs.shape[2]
        print(f"EnsembleCache: {self.probs.shape}  ckpts={self.ckpt_names}")

    def lookup(self, abs_paths: list[str]) -> np.ndarray:
        """Returns (len(paths), M, C) per-clip probs."""
        rows = np.empty((len(abs_paths), self.M, self.C), dtype=np.float32)
        for j, p in enumerate(abs_paths):
            i = self.idx[p]
            rows[j] = self.probs[:, i, :]
        return rows


# ─── Eval ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_split(model, sources_by_class, executor, device, cache: EnsembleCache,
               K_eval, target_rms=0.1, hpf_hz=20.0, batch_size_sources=4):
    model.eval()
    items = []
    for ci, c in enumerate(CLASSES):
        for sid, paths in sources_by_class.get(c, {}).items():
            src_type = 'iara' if sid.startswith('iara-') else 'deepship'
            items.append((sid, ci, paths, src_type))
    y_true = np.array([it[1] for it in items], dtype=np.int64)
    src_type = np.array([it[3] for it in items])
    S = len(items)
    final_probs = np.zeros((S, 4), dtype=np.float32)
    final_abstain = np.zeros(S, dtype=np.float32)

    for start in range(0, S, batch_size_sources):
        chunk = items[start:start + batch_size_sources]
        clip_lists, masks = [], []
        for _, _, paths, _ in chunk:
            ps = sorted(paths)
            if len(ps) >= K_eval:
                stride = max(1, len(ps) // K_eval)
                taken = ps[::stride][:K_eval]
                clip_lists.append(taken); masks.append([True] * K_eval)
            else:
                pad = K_eval - len(ps)
                clip_lists.append(ps + [ps[0]] * pad)
                masks.append([True] * len(ps) + [False] * pad)
        flat = [p for lst in clip_lists for p in lst]
        wav = torch.from_numpy(read_batch_threaded(flat, executor)).to(device)
        wav = preprocess(wav, target_rms, hpf_hz)
        wav = wav.view(len(chunk), K_eval, -1)
        ens_np = cache.lookup(flat).reshape(len(chunk), K_eval, cache.M, cache.C)
        ens = torch.from_numpy(ens_np).to(device)
        mask = torch.tensor(masks, dtype=torch.bool, device=device)
        logits = model(wav, ens, mask=mask)
        p_full = F.softmax(logits, dim=-1)
        p_cls = F.softmax(logits[..., :4], dim=-1).float().cpu().numpy()
        final_probs[start:start + len(chunk)] = p_cls
        final_abstain[start:start + len(chunk)] = p_full[..., 4].float().cpu().numpy()
    pred = final_probs.argmax(-1)
    return y_true, pred, final_probs, final_abstain, src_type


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/var/mnt/5A009BF8009BD8F9/Data/Combined_IARA_Deepship_1s")
    p.add_argument("--ens_npz", default="campaign/probs_combined_recurrent_stacker.npz")
    p.add_argument("--out_dir", default="lightning_logs/hydro_recurrent_stacker_combined")
    p.add_argument("--loader", choices=("dali", "dali_split", "threaded", "threaded_split"),
                   default="threaded",
                   help="Audio backend. Stacker uses internal threaded I/O; "
                        "'dali*' choices are accepted for uniformity but treated as 'threaded'.")
    p.add_argument("--class_depth", type=int, default=1,
                   help="Model depth knob. Scales gru_layers and the tcn_dilations stack.")
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
    p.add_argument("--smoothing", type=float, default=0.05)
    p.add_argument("--gambler_w", type=float, default=0.1)
    p.add_argument("--abstain_o", type=float, default=2.2)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--deep_aux_w", type=float, default=0.2)
    p.add_argument("--class_weight_pow", type=float, default=0.5)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--target_rms", type=float, default=0.1)
    p.add_argument("--hpf_hz", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_workers", type=int, default=12)
    p.add_argument("--exclude_iara_glider", action="store_true",
                   help="Drop IARA partitions F+G (Wave Glider) from train/val.")
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    train_src = scan_split(args.data_dir, "Train", exclude_iara_glider=args.exclude_iara_glider)
    val_src = scan_split(args.data_dir, "Val", exclude_iara_glider=args.exclude_iara_glider)
    if args.exclude_iara_glider:
        print("** IARA Glider partitions (F, G) EXCLUDED from train and val **")
    for lbl, d in (("train", train_src), ("val", val_src)):
        c = {k: len(d[k]) for k in CLASSES}
        n = {k: sum(len(p) for p in d[k].values()) for k in CLASSES}
        print(f"{lbl:>5s} sources: {c}  clips: {n}")

    cache = EnsembleCache(Path(args.ens_npz), Path(args.data_dir))

    sampler = SourceSampler(train_src, args.per_class, args.K_train,
                            random.Random(args.seed))

    tcn_dils = tuple(int(d) for d in args.tcn_dilations.split(","))
    scales = tuple(int(s) for s in args.scales.split(","))
    # --class_depth scales gru_layers and repeats tcn_dilations.
    cd = max(1, int(args.class_depth))
    eff_gru = max(args.gru_layers, cd)
    eff_tcn = tcn_dils * cd
    if args.loader.startswith("dali"):
        print(f"[note] --loader={args.loader}: stacker uses internal threaded I/O; flag recorded only.")
    model = HydroRecurrentStacker(
        num_classes=4, n_bands=args.n_bands, embed_dim=args.embed_dim,
        ens_n_ckpts=cache.M, ens_n_classes=cache.C,
        gru_hidden=args.gru_hidden, gru_layers=eff_gru,
        head_hidden=args.head_hidden, dropout=args.dropout,
        band_dropout=args.band_dropout, tcn_dilations=eff_tcn,
        scales=scales, gambler=True,
    ).to(device)
    print(f"HydroRecurrentStacker params = {model.n_params():,}  scales={scales}")

    counts = np.array([len(train_src[c]) for c in CLASSES], dtype=np.float64)
    cw = (counts.sum() / (counts * len(CLASSES))) ** args.class_weight_pow
    class_weight = torch.tensor(cw, dtype=torch.float32, device=device)
    print(f"Class weights: {dict(zip(CLASSES, [f'{w:.2f}' for w in cw]))}")

    criterion = GamblerCE(K=4, smoothing=args.smoothing,
                          gambler_w=args.gambler_w, abstain_o=args.abstain_o,
                          class_weight=class_weight, focal_gamma=args.focal_gamma)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    executor = ThreadPoolExecutor(max_workers=args.n_workers)
    best_val = -1.0; best_step = -1; patience_left = args.patience
    history = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        labels, clip_paths = sampler.draw()
        flat_paths = [p for lst in clip_paths for p in lst]
        audio = read_batch_threaded(flat_paths, executor)
        wav = torch.from_numpy(audio).to(device, non_blocking=True)
        wav = preprocess(wav, args.target_rms, args.hpf_hz)
        N = len(labels); K = args.K_train
        wav = wav.view(N, K, -1)
        ens_np = cache.lookup(flat_paths).reshape(N, K, cache.M, cache.C)
        ens = torch.from_numpy(ens_np).to(device, non_blocking=True)
        targets = torch.from_numpy(labels).to(device)

        logits_seq = model(wav, ens, mask=None, return_seq=True)
        final_logits = logits_seq[:, K - 1]
        loss_final = criterion(final_logits, targets)
        if args.deep_aux_w > 0:
            K_total = logits_seq.size(1)
            tgt_rep = targets.unsqueeze(1).expand(-1, K_total).reshape(-1)
            seq_flat = logits_seq.reshape(N * K_total, -1)
            loss_aux = criterion(seq_flat, tgt_rep)
            loss = loss_final + args.deep_aux_w * loss_aux
        else:
            loss = loss_final

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()

        if step % 50 == 0:
            print(f"  step {step:5d}/{args.steps}  loss={loss.item():.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)",
                  flush=True)

        if step % args.val_every == 0 or step == args.steps:
            y, yp, _, abst, _ = eval_split(model, val_src, executor, device, cache,
                                           K_eval=args.K_eval,
                                           target_rms=args.target_rms,
                                           hpf_hz=args.hpf_hz)
            P, R, F1, mp, mr, mf, _ = macro_prf1(y, yp, 4)
            print(f"  [val @ step {step}] macroF1={mf:.4f} macroP={mp:.4f} "
                  f"macroR={mr:.4f}  abstain⟨{abst.mean():.3f}⟩", flush=True)
            history.append({"step": step, "macro_f1": float(mf),
                            "macro_p": float(mp), "macro_r": float(mr),
                            "mean_abstain": float(abst.mean())})
            if mf > best_val:
                best_val = float(mf); best_step = step; patience_left = args.patience
                torch.save({
                    "state_dict": model.state_dict(),
                    "args": vars(args), "step": step,
                    "val_macro_f1": mf, "class_weight": cw.tolist(),
                    "ckpt_names": cache.ckpt_names,
                }, out_dir / "best.pt")
                print(f"    ↑ best (step {step}, val_F1={mf:.4f}) saved", flush=True)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"  early stop", flush=True)
                    break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "result.json").write_text(json.dumps({
        "best_val_macro_f1": best_val, "best_step": best_step,
        "args": vars(args), "n_params": model.n_params(),
    }, indent=2))
    print(f"\nDONE. best val macroF1 = {best_val:.4f} at step {best_step}")


if __name__ == "__main__":
    main()
