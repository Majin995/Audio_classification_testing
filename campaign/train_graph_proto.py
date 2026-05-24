"""Train HydroGraphProto — 1D frontend → GNN → Perceiver → Sinkhorn-Knopp
prototype head, with frozen 6-ckpt cargo_confirm per-clip ensemble features
concatenated to the node tokens.

Loss = ProtoNet-style CE (soft-min over M prototypes per class) +
       swav_w * SwAV swapped-assignment auxiliary (Sinkhorn balancing) +
       gambler_w * Deep-Gamblers abstain auxiliary.

Honest contract preserved: the 6 ensemble ckpts are FROZEN (read from
`campaign/probs_combined_recurrent_stacker.npz`); only the encoder, GNN,
Perceiver, prototypes, and abstain head train. Val OOF only for selection.
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

from models.hydro_graph_proto import HydroGraphProto, sinkhorn_knopp
from campaign.train_recurrent import (
    CLASSES, SourceSampler, read_batch_threaded, preprocess, scan_split,
    macro_prf1,
)
from campaign.train_recurrent_stacker import EnsembleCache


class DomainBalancedSampler:
    """Per-class draws that force ~50/50 IARA vs DeepShip per batch.

    Source IDs prefixed 'iara-' are IARA, else DeepShip. Falls back to the
    available pool when one domain is empty for a class.
    """

    def __init__(self, sources_by_class, per_class, K, rng):
        self.per_class = int(per_class); self.K = int(K); self.rng = rng
        self.classes = [c for c in CLASSES if sources_by_class.get(c)]
        self.iara = {c: [(s, p) for s, p in sources_by_class[c].items()
                          if s.startswith('iara-')] for c in self.classes}
        self.dship = {c: [(s, p) for s, p in sources_by_class[c].items()
                           if not s.startswith('iara-')] for c in self.classes}

    def _pick(self, pool, n):
        if not pool:
            return []
        replace = len(pool) < n
        return (self.rng.choices(pool, k=n) if replace
                else self.rng.sample(pool, n))

    def draw(self):
        labels, clip_paths = [], []
        n_each = max(1, self.per_class // 2)
        for ci, c in enumerate(self.classes):
            iara_pool = self.iara[c]; dship_pool = self.dship[c]
            n_i = n_each if iara_pool else 0
            n_d = self.per_class - n_i if dship_pool else 0
            if n_i == 0:
                n_d = self.per_class
            if n_d == 0:
                n_i = self.per_class
            picks = self._pick(iara_pool, n_i) + self._pick(dship_pool, n_d)
            for sid, paths in picks:
                if len(paths) >= self.K:
                    clips = self.rng.sample(paths, self.K)
                else:
                    clips = self.rng.choices(paths, k=self.K)
                self.rng.shuffle(clips)
                labels.append(ci)
                clip_paths.append(clips)
        return np.array(labels, dtype=np.int64), clip_paths


# ─── Loss helpers ──────────────────────────────────────────────────────────

def proto_ce(cls_logits, targets, class_weight=None, smoothing=0.05):
    """Standard CE on the class logits (already logsumexp over M prototypes)."""
    return F.cross_entropy(cls_logits, targets, weight=class_weight,
                           label_smoothing=smoothing)


def swav_loss(cluster_scores, epsilon=0.05, sinkhorn_iters=3, temp=0.1):
    """SwAV-style swapped assignment on a single batch.

    cluster_scores: (N, P) cosine-sim logits (un-temperature-scaled).
    Computes Sinkhorn assignment Q on detached scores, then CE between
    log-softmax(scores/temp) and Q.
    """
    with torch.no_grad():
        Q = sinkhorn_knopp(cluster_scores.detach() / epsilon,
                           n_iters=sinkhorn_iters, epsilon=1.0)
    log_p = F.log_softmax(cluster_scores / temp, dim=-1)
    return -(Q * log_p).sum(dim=-1).mean()


def gambler_loss(abstain_logits, cls_logits, targets, abstain_o=2.2):
    """Deep-Gamblers reward: encourage abstention when cls is wrong.

    Treat sigmoid(abstain) as p_abstain ∈ (0,1). The deep-gamblers objective
    pays log( p_correct * (1 - p_abstain) + p_abstain / abstain_o ).
    """
    p_abst = torch.sigmoid(abstain_logits)
    p_cls = F.softmax(cls_logits, dim=-1)
    p_correct = p_cls.gather(1, targets.unsqueeze(1)).squeeze(1)
    reward = p_correct * (1.0 - p_abst) + p_abst / abstain_o
    return -torch.log(reward.clamp_min(1e-8)).mean()


# ─── Prototype init via k-means on val embeddings (one-time) ────────────────

@torch.no_grad()
def kmeans_init_prototypes(model, train_src, executor, device, cache,
                           K_eval=30, n_per_class=64, kmeans_iters=20,
                           target_rms=0.1, hpf_hz=20.0):
    """Encode up to ``n_per_class`` sources per class, run k-means(M) per
    class on the resulting source vectors, and write the centroids into
    ``model.prototypes``. Replaces random init with a sensible warm start."""
    model.eval()
    M = model.M
    proto_dim = model.prototypes.shape[1]
    new_protos = torch.zeros_like(model.prototypes)

    for ci, c in enumerate(CLASSES):
        sids = list(train_src.get(c, {}).keys())
        random.shuffle(sids)
        sids = sids[:n_per_class]
        if len(sids) < M:
            print(f"  init: class {c} has only {len(sids)} sources, padding random")
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
        X = torch.stack(vecs)                                       # (S, Dp)
        # k-means
        idx = torch.randperm(X.size(0), device=X.device)[:M]
        if X.size(0) < M:
            extra = torch.randn(M - X.size(0), proto_dim, device=device) * 0.02
            centroids = torch.cat([X[idx], extra], dim=0)
        else:
            centroids = X[idx].clone()
        for _ in range(kmeans_iters):
            d = torch.cdist(X, centroids)                            # (S, M)
            assign = d.argmin(dim=1)                                  # (S,)
            for m in range(M):
                pick = X[assign == m]
                if pick.size(0) > 0:
                    centroids[m] = pick.mean(dim=0)
        new_protos[ci * M:(ci + 1) * M] = centroids
    model.prototypes.data.copy_(new_protos)
    print(f"  prototype init: k-means done (M={M} per class)")


# ─── Eval ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_split(model, sources_by_class, executor, device, cache,
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
        cls_logits, abstain, _, _ = model(wav, ens, mask=mask)
        p = F.softmax(cls_logits, dim=-1).float().cpu().numpy()
        a = torch.sigmoid(abstain).float().cpu().numpy()
        final_probs[start:start + len(chunk)] = p
        final_abstain[start:start + len(chunk)] = a
    pred = final_probs.argmax(-1)
    return y_true, pred, final_probs, final_abstain, src_type


# ─── Main ──────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/var/mnt/5A009BF8009BD8F9/Data/Combined_IARA_Deepship_1s")
    p.add_argument("--ens_npz", default="campaign/probs_combined_recurrent_stacker.npz")
    p.add_argument("--out_dir", default="lightning_logs/hydro_graph_proto_combined")
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
    p.add_argument("--domain_balance", action="store_true", default=True)
    p.add_argument("--no_domain_balance", dest="domain_balance", action="store_false")
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
    p.add_argument("--kmeans_init", action="store_true", default=True)
    p.add_argument("--no_kmeans_init", dest="kmeans_init", action="store_false")
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    train_src = scan_split(args.data_dir, "Train")
    val_src = scan_split(args.data_dir, "Val")
    for lbl, d in (("train", train_src), ("val", val_src)):
        c = {k: len(d[k]) for k in CLASSES}
        n = {k: sum(len(p) for p in d[k].values()) for k in CLASSES}
        print(f"{lbl:>5s} sources: {c}  clips: {n}")

    cache = EnsembleCache(Path(args.ens_npz), Path(args.data_dir))

    if args.domain_balance:
        sampler = DomainBalancedSampler(train_src, args.per_class, args.K_train,
                                        random.Random(args.seed))
        print("** Domain-balanced sampling: ~50/50 IARA/DeepShip per class **")
    else:
        sampler = SourceSampler(train_src, args.per_class, args.K_train,
                                random.Random(args.seed))

    tcn_dils = tuple(int(d) for d in args.tcn_dilations.split(","))
    model = HydroGraphProto(
        num_classes=4, n_bands=args.n_bands, embed_dim=args.embed_dim,
        tcn_dilations=tcn_dils,
        ens_n_ckpts=cache.M, ens_n_classes=cache.C,
        gnn_dim=args.gnn_dim, gnn_depth=args.gnn_depth, gnn_heads=args.gnn_heads,
        kNN=args.kNN, temporal_ring=args.temporal_ring,
        perceiver_queries=args.perceiver_queries,
        perceiver_depth=args.perceiver_depth,
        n_prototypes_per_class=args.proto_per_class,
        proto_temp=args.proto_temp, proto_dim=args.proto_dim,
        dropout=args.dropout, band_dropout=args.band_dropout,
    ).to(device)
    print(f"HydroGraphProto params = {model.n_params():,}")

    counts = np.array([len(train_src[c]) for c in CLASSES], dtype=np.float64)
    cw = (counts.sum() / (counts * len(CLASSES))) ** args.class_weight_pow
    class_weight = torch.tensor(cw, dtype=torch.float32, device=device)
    print(f"Class weights: {dict(zip(CLASSES, [f'{w:.2f}' for w in cw]))}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    executor = ThreadPoolExecutor(max_workers=args.n_workers)

    if args.kmeans_init:
        print("Initializing prototypes via k-means on train embeddings...")
        kmeans_init_prototypes(model, train_src, executor, device, cache,
                               K_eval=args.K_train, n_per_class=48,
                               target_rms=args.target_rms, hpf_hz=args.hpf_hz)

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

        cls_logits, abstain, cluster_scores, _ = model(wav, ens, mask=None)
        l_cls = proto_ce(cls_logits, targets, class_weight=class_weight,
                          smoothing=args.smoothing)
        l_swav = swav_loss(cluster_scores, epsilon=args.swav_epsilon,
                           temp=args.swav_temp) if args.swav_w > 0 else cls_logits.new_zeros(())
        l_gamb = gambler_loss(abstain, cls_logits, targets, args.abstain_o) \
                 if args.gambler_w > 0 else cls_logits.new_zeros(())
        loss = l_cls + args.swav_w * l_swav + args.gambler_w * l_gamb

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()

        if step % 50 == 0:
            print(f"  step {step:5d}/{args.steps}  loss={loss.item():.4f} "
                  f"(cls={l_cls.item():.3f} swav={l_swav.item():.3f} "
                  f"gamb={l_gamb.item():.3f})  lr={sched.get_last_lr()[0]:.2e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

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
