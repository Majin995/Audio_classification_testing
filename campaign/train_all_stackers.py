"""Sequential stacker training for an arbitrary new dataset.

Trains a registered set of non-recurrent stacker models one after another on
the same dataset and the same per-clip ensemble cache, writing each model's
artifacts to ``<out_root>/<model_name>/``.

Optional ``--finetune`` warm-starts every model from a prior best.pt at
``<finetune_from>/<model_name>/best.pt``, at a reduced LR (5e-4 default).
GraphProto's k-means prototype init is skipped automatically when warm-
starting (prototypes already in the loaded state-dict).

Uses the standalone ``data.threaded_audio_loader`` (no NVIDIA DALI) so the
script runs even when the DALI env is broken or unavailable.

Registry today: HydroGraphProto only (no recurrent variants by design).
Add a new entry to ``MODEL_REGISTRY`` to enroll another model — each entry
supplies build_model(args, ens), and the training loop below covers loss,
eval, ckpt I/O for any model with the ``(cls_logits, abstain, cluster_scores,
s)`` forward signature.
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

from data.threaded_audio_loader import (
    CLASSES, scan_split, read_batch_threaded, preprocess,
)
from models.hydro_graph_proto import HydroGraphProto, sinkhorn_knopp


# ─── Per-clip ensemble cache ───────────────────────────────────────────────

class EnsembleCache:
    def __init__(self, npz_path: Path, data_dir: Path):
        z = np.load(npz_path, allow_pickle=True)
        rel = z["paths"].tolist()
        self.probs = z["probs"]
        self.ckpt_names = list(z["ckpt_names"].tolist())
        self.idx = {str(data_dir / p): i for i, p in enumerate(rel)}
        self.M = self.probs.shape[0]
        self.C = self.probs.shape[2]
        print(f"  EnsembleCache: {self.probs.shape}  ckpts={self.ckpt_names}")

    def lookup(self, abs_paths):
        rows = np.empty((len(abs_paths), self.M, self.C), dtype=np.float32)
        for j, p in enumerate(abs_paths):
            rows[j] = self.probs[:, self.idx[p], :]
        return rows


# ─── Sampler ───────────────────────────────────────────────────────────────

class SourceSampler:
    def __init__(self, sources_by_class, per_class, K, rng):
        self.sbc = {c: list(d.items()) for c, d in sources_by_class.items()}
        self.per_class = int(per_class); self.K = int(K); self.rng = rng
        self.classes = [c for c in CLASSES if self.sbc.get(c)]

    def draw(self):
        labels, clip_paths = [], []
        for ci, c in enumerate(self.classes):
            pool = self.sbc[c]
            replace = len(pool) < self.per_class
            picks = (self.rng.choices(pool, k=self.per_class) if replace
                     else self.rng.sample(pool, self.per_class))
            for sid, paths in picks:
                clips = (self.rng.sample(paths, self.K) if len(paths) >= self.K
                         else self.rng.choices(paths, k=self.K))
                self.rng.shuffle(clips)
                labels.append(ci)
                clip_paths.append(clips)
        return np.array(labels, dtype=np.int64), clip_paths


class DomainBalancedSampler(SourceSampler):
    """50/50 IARA vs DeepShip per class when both exist."""

    def __init__(self, sources_by_class, per_class, K, rng):
        super().__init__(sources_by_class, per_class, K, rng)
        self.iara = {c: [(s, p) for s, p in sources_by_class[c].items()
                          if s.startswith('iara-')] for c in self.classes}
        self.dship = {c: [(s, p) for s, p in sources_by_class[c].items()
                           if not s.startswith('iara-')] for c in self.classes}

    def _pick(self, pool, n):
        if not pool: return []
        replace = len(pool) < n
        return (self.rng.choices(pool, k=n) if replace
                else self.rng.sample(pool, n))

    def draw(self):
        labels, clip_paths = [], []
        n_each = max(1, self.per_class // 2)
        for ci, c in enumerate(self.classes):
            ip, dp = self.iara[c], self.dship[c]
            n_i = n_each if ip else 0
            n_d = self.per_class - n_i if dp else 0
            if n_i == 0: n_d = self.per_class
            if n_d == 0: n_i = self.per_class
            for sid, paths in self._pick(ip, n_i) + self._pick(dp, n_d):
                clips = (self.rng.sample(paths, self.K) if len(paths) >= self.K
                         else self.rng.choices(paths, k=self.K))
                self.rng.shuffle(clips)
                labels.append(ci)
                clip_paths.append(clips)
        return np.array(labels, dtype=np.int64), clip_paths


# ─── Metric ────────────────────────────────────────────────────────────────

def macro_prf1(y_true, y_pred, K):
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    P, R, F1 = np.zeros(K), np.zeros(K), np.zeros(K)
    for k in range(K):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k].sum() - tp
        P[k] = tp / (tp + fp) if tp + fp else 0.0
        R[k] = tp / (tp + fn) if tp + fn else 0.0
        F1[k] = 2 * P[k] * R[k] / (P[k] + R[k]) if P[k] + R[k] else 0.0
    return P, R, F1, P.mean(), R.mean(), F1.mean(), cm


# ─── Losses ────────────────────────────────────────────────────────────────

def proto_ce(cls_logits, targets, class_weight=None, smoothing=0.05):
    return F.cross_entropy(cls_logits, targets, weight=class_weight,
                           label_smoothing=smoothing)


def swav_loss(cluster_scores, epsilon=0.05, sinkhorn_iters=3, temp=0.1):
    with torch.no_grad():
        Q = sinkhorn_knopp(cluster_scores.detach() / epsilon,
                           n_iters=sinkhorn_iters, epsilon=1.0)
    return -(Q * F.log_softmax(cluster_scores / temp, dim=-1)).sum(-1).mean()


def gambler_loss(abstain_logits, cls_logits, targets, abstain_o=2.2):
    p_abst = torch.sigmoid(abstain_logits)
    p_cls = F.softmax(cls_logits, dim=-1)
    p_correct = p_cls.gather(1, targets.unsqueeze(1)).squeeze(1)
    reward = p_correct * (1.0 - p_abst) + p_abst / abstain_o
    return -torch.log(reward.clamp_min(1e-8)).mean()


# ─── Eval ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_split(model, sources_by_class, executor, device, cache,
               K_eval, target_rms=0.1, hpf_hz=20.0, batch_size_sources=4):
    model.eval()
    items = [(sid, ci, paths) for ci, c in enumerate(CLASSES)
             for sid, paths in sources_by_class.get(c, {}).items()]
    if not items:
        return np.array([]), np.array([]), np.zeros((0, 4)), np.zeros(0)
    y_true = np.array([it[1] for it in items], dtype=np.int64)
    S = len(items)
    final_probs = np.zeros((S, 4), dtype=np.float32)
    final_abstain = np.zeros(S, dtype=np.float32)

    for start in range(0, S, batch_size_sources):
        chunk = items[start:start + batch_size_sources]
        clip_lists, masks = [], []
        for _, _, paths in chunk:
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
        cls_logits, abstain, _, _ = model(wav, ens, mask=mask)
        final_probs[start:start + len(chunk)] = F.softmax(cls_logits, -1).cpu().numpy()
        final_abstain[start:start + len(chunk)] = torch.sigmoid(abstain).cpu().numpy()
    return y_true, final_probs.argmax(-1), final_probs, final_abstain


# ─── k-means prototype init (skipped on warm start) ────────────────────────

@torch.no_grad()
def kmeans_init_prototypes(model, train_src, executor, device, cache,
                           K_eval=30, n_per_class=48, kmeans_iters=20,
                           target_rms=0.1, hpf_hz=20.0):
    model.eval()
    M = model.M
    proto_dim = model.prototypes.shape[1]
    new_protos = torch.zeros_like(model.prototypes)
    for ci, c in enumerate(CLASSES):
        sids = list(train_src.get(c, {}).keys()); random.shuffle(sids)
        sids = sids[:n_per_class]
        vecs = []
        for sid in sids:
            paths = sorted(train_src[c][sid])
            if len(paths) >= K_eval:
                stride = max(1, len(paths) // K_eval)
                taken = paths[::stride][:K_eval]
            else:
                taken = paths + [paths[0]] * (K_eval - len(paths))
            wav = torch.from_numpy(read_batch_threaded(taken, executor)).to(device)
            wav = preprocess(wav, target_rms, hpf_hz).view(1, K_eval, -1)
            ens_np = cache.lookup(taken).reshape(1, K_eval, cache.M, cache.C)
            ens = torch.from_numpy(ens_np).to(device)
            _, _, _, s = model(wav, ens)
            vecs.append(F.normalize(s, dim=-1).squeeze(0))
        if not vecs:
            continue
        X = torch.stack(vecs)
        idx = torch.randperm(X.size(0), device=X.device)[:M]
        if X.size(0) < M:
            extra = torch.randn(M - X.size(0), proto_dim, device=device) * 0.02
            centroids = torch.cat([X[idx], extra], dim=0)
        else:
            centroids = X[idx].clone()
        for _ in range(kmeans_iters):
            d = torch.cdist(X, centroids); assign = d.argmin(dim=1)
            for m in range(M):
                pick = X[assign == m]
                if pick.size(0) > 0: centroids[m] = pick.mean(dim=0)
        new_protos[ci * M:(ci + 1) * M] = centroids
    model.prototypes.data.copy_(new_protos)


# ─── Model registry ────────────────────────────────────────────────────────

def _build_hydro_graph_proto(args, cache):
    return HydroGraphProto(
        num_classes=4, n_bands=args.n_bands, embed_dim=args.embed_dim,
        tcn_dilations=tuple(int(d) for d in args.tcn_dilations.split(",")),
        ens_n_ckpts=cache.M, ens_n_classes=cache.C,
        gnn_dim=args.gnn_dim, gnn_depth=args.gnn_depth,
        gnn_heads=args.gnn_heads, kNN=args.kNN,
        temporal_ring=args.temporal_ring,
        perceiver_queries=args.perceiver_queries,
        perceiver_depth=args.perceiver_depth,
        n_prototypes_per_class=args.proto_per_class,
        proto_temp=args.proto_temp, proto_dim=args.proto_dim,
        dropout=args.dropout, band_dropout=args.band_dropout,
    )


# name -> (factory, needs_kmeans_init)
MODEL_REGISTRY = {
    "hydro_graph_proto": (_build_hydro_graph_proto, True),
}


# ─── Single-model training loop ────────────────────────────────────────────

def train_one(name, factory, needs_kmeans, args, train_src, val_src, cache,
              executor, device, out_dir, finetune_from=None):
    print(f"\n{'='*78}\n=== {name}  (out={out_dir})\n{'='*78}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = factory(args, cache).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params = {n_params:,}")

    warm_started = False
    if finetune_from is not None:
        ckpt_path = Path(finetune_from) / name / "best.pt"
        if not ckpt_path.is_file():
            print(f"  finetune ckpt MISSING ({ckpt_path}) — training from scratch")
        else:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
            warm_started = True
            print(f"  warm-started from {ckpt_path} "
                  f"(prior val_F1={ck.get('val_macro_f1', float('nan')):.4f}, "
                  f"missing={len(missing)} unexpected={len(unexpected)})")

    counts = np.array([len(train_src[c]) for c in CLASSES], dtype=np.float64)
    cw = (counts.sum() / (counts.clip(min=1) * len(CLASSES))) ** args.class_weight_pow
    class_weight = torch.tensor(cw, dtype=torch.float32, device=device)
    print(f"  class weights: {dict(zip(CLASSES, [f'{w:.2f}' for w in cw]))}")

    if args.domain_balance:
        sampler = DomainBalancedSampler(train_src, args.per_class, args.K_train,
                                        random.Random(args.seed))
        print("  ** domain-balanced sampling enabled **")
    else:
        sampler = SourceSampler(train_src, args.per_class, args.K_train,
                                random.Random(args.seed))

    lr = args.finetune_lr if warm_started else args.lr
    print(f"  lr = {lr:.1e}  (finetune={warm_started})")
    if needs_kmeans and not warm_started:
        print("  k-means prototype init...")
        kmeans_init_prototypes(model, train_src, executor, device, cache,
                               K_eval=args.K_train,
                               target_rms=args.target_rms, hpf_hz=args.hpf_hz)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    best_val, best_step, patience_left = -1.0, -1, args.patience
    history = []; t0 = time.time()

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

        cls_logits, abstain, cluster_scores, _ = model(wav, ens, mask=None)
        l_cls = proto_ce(cls_logits, targets, class_weight=class_weight,
                          smoothing=args.smoothing)
        l_swav = (swav_loss(cluster_scores, epsilon=args.swav_epsilon,
                            temp=args.swav_temp)
                  if args.swav_w > 0 else cls_logits.new_zeros(()))
        l_gamb = (gambler_loss(abstain, cls_logits, targets, args.abstain_o)
                  if args.gambler_w > 0 else cls_logits.new_zeros(()))
        loss = l_cls + args.swav_w * l_swav + args.gambler_w * l_gamb

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()

        if step % 50 == 0:
            print(f"    step {step:5d}/{args.steps}  loss={loss.item():.4f} "
                  f"(cls={l_cls.item():.3f} swav={l_swav.item():.3f} "
                  f"gamb={l_gamb.item():.3f})  lr={sched.get_last_lr()[0]:.2e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            y, yp, _, abst = eval_split(model, val_src, executor, device, cache,
                                         K_eval=args.K_eval,
                                         target_rms=args.target_rms,
                                         hpf_hz=args.hpf_hz)
            _, _, _, mp, mr, mf, _ = macro_prf1(y, yp, 4)
            print(f"    [val @ {step}] macroF1={mf:.4f} macroP={mp:.4f} "
                  f"macroR={mr:.4f}  abstain⟨{abst.mean():.3f}⟩", flush=True)
            history.append({"step": step, "macro_f1": float(mf),
                            "macro_p": float(mp), "macro_r": float(mr),
                            "mean_abstain": float(abst.mean())})
            if mf > best_val:
                best_val = float(mf); best_step = step; patience_left = args.patience
                torch.save({
                    "state_dict": model.state_dict(), "args": vars(args),
                    "step": step, "val_macro_f1": mf, "ckpt_names": cache.ckpt_names,
                    "warm_started_from": str(finetune_from) if warm_started else None,
                }, out_dir / "best.pt")
                print(f"      ↑ best saved", flush=True)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"    early stop", flush=True); break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "result.json").write_text(json.dumps({
        "model": name, "best_val_macro_f1": best_val, "best_step": best_step,
        "warm_started": warm_started, "n_params": n_params,
        "args": vars(args),
    }, indent=2))
    print(f"  DONE [{name}] best val_F1={best_val:.4f} at step {best_step}")
    return {"model": name, "best_val_macro_f1": best_val, "best_step": best_step,
            "n_params": n_params, "warm_started": warm_started}


# ─── Main ──────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="dataset root containing Train/Val/Test/<class>/*.wav")
    p.add_argument("--ens_npz", required=True,
                   help="per-clip ensemble cache NPZ for this dataset")
    p.add_argument("--out_root", required=True,
                   help="output root; each model writes to <out_root>/<model_name>/")
    p.add_argument("--models", default="hydro_graph_proto",
                   help="comma-separated list of model names from MODEL_REGISTRY")
    # Finetune control — same flag toggles for all models in the list.
    p.add_argument("--finetune", action="store_true",
                   help="warm-start each model from <finetune_from>/<model_name>/best.pt")
    p.add_argument("--finetune_from", default=None,
                   help="root dir holding prior <model_name>/best.pt artifacts")
    p.add_argument("--finetune_lr", type=float, default=5e-4)
    # Standard knobs
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--per_class", type=int, default=4)
    p.add_argument("--K_train", type=int, default=30)
    p.add_argument("--K_eval", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=5e-2)
    p.add_argument("--n_bands", type=int, default=24)
    p.add_argument("--embed_dim", type=int, default=48)
    p.add_argument("--gnn_dim", type=int, default=48)
    p.add_argument("--gnn_depth", type=int, default=1)
    p.add_argument("--gnn_heads", type=int, default=4)
    p.add_argument("--kNN", type=int, default=4)
    p.add_argument("--temporal_ring", type=int, default=2)
    p.add_argument("--perceiver_queries", type=int, default=4)
    p.add_argument("--perceiver_depth", type=int, default=1)
    p.add_argument("--proto_per_class", type=int, default=8)
    p.add_argument("--proto_temp", type=float, default=10.0)
    p.add_argument("--proto_dim", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--band_dropout", type=float, default=0.20)
    p.add_argument("--tcn_dilations", default="1,4,16,64")
    p.add_argument("--smoothing", type=float, default=0.05)
    p.add_argument("--swav_w", type=float, default=0.0)
    p.add_argument("--swav_temp", type=float, default=0.1)
    p.add_argument("--swav_epsilon", type=float, default=0.05)
    p.add_argument("--gambler_w", type=float, default=0.1)
    p.add_argument("--abstain_o", type=float, default=2.2)
    p.add_argument("--class_weight_pow", type=float, default=0.5)
    p.add_argument("--val_every", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--target_rms", type=float, default=0.1)
    p.add_argument("--hpf_hz", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_workers", type=int, default=12)
    p.add_argument("--domain_balance", action="store_true", default=True)
    p.add_argument("--no_domain_balance", dest="domain_balance", action="store_false")
    return p.parse_args()


def main():
    args = get_args()
    if args.finetune and not args.finetune_from:
        raise SystemExit("--finetune requires --finetune_from <prior_out_root>")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dataset: {args.data_dir}")
    train_src = scan_split(args.data_dir, "Train")
    val_src = scan_split(args.data_dir, "Val")
    for lbl, d in (("train", train_src), ("val", val_src)):
        c = {k: len(d[k]) for k in CLASSES}
        n = {k: sum(len(p) for p in d[k].values()) for k in CLASSES}
        print(f"  {lbl:>5s} sources={c}  clips={n}")

    cache = EnsembleCache(Path(args.ens_npz), Path(args.data_dir))
    executor = ThreadPoolExecutor(max_workers=args.n_workers)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_names:
        if m not in MODEL_REGISTRY:
            raise SystemExit(f"unknown model '{m}'. Known: {list(MODEL_REGISTRY)}")

    finetune_from = args.finetune_from if args.finetune else None
    summary = []
    for name in model_names:
        factory, needs_kmeans = MODEL_REGISTRY[name]
        out_dir = out_root / name
        res = train_one(name, factory, needs_kmeans, args, train_src, val_src,
                        cache, executor, device, out_dir,
                        finetune_from=finetune_from)
        summary.append(res)

    (out_root / "all_models_summary.json").write_text(json.dumps({
        "models": summary, "args": vars(args),
        "data_dir": args.data_dir, "ens_npz": args.ens_npz,
    }, indent=2))
    print(f"\n=== ALL DONE ===")
    for r in summary:
        print(f"  {r['model']:>20s}  val_F1={r['best_val_macro_f1']:.4f}  "
              f"step={r['best_step']}  params={r['n_params']:,}  "
              f"warm_started={r['warm_started']}")


if __name__ == "__main__":
    main()
