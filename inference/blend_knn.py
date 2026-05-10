"""Phase C — compare three inference paths on a HydroPrecise(V2) checkpoint:

  * **head-only**          : ``softmax(head_logits / T)`` with the saved
                             temperature, if any.
  * **head + KNN blend**   : ``alpha * head_p + (1-alpha) * knn_p``, with alpha
                             chosen on val/micro_precision over a sweep grid.
  * **KNN-only**           : sklearn ``KNeighborsClassifier(5, cosine,
                             weights="distance")`` fit on train embeddings.

When ``--ckpt`` receives a comma-separated list, head probabilities are
arithmetic-mean-ensembled across the listed ckpts (each with its own saved
temperature). The KNN classifier is fit on the first ckpt's train embeddings
only — KNN already dominates the blend, so multi-fit doesn't help in practice.

Use ``--model_version v1`` to score a HydroPrecise (V1) ckpt; the V1 abstention
column is dropped before softmax.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score,
)

from data.audio_lightning_loader import DALIAudioDataModule


# ── Model loader (handles V1 / V2) ──────────────────────────────────────────

def _load_model(ckpt: str, version: str, device: str):
    if version == "v2":
        from models.hydro_precise_v2 import HydroPreciseV2
        m = HydroPreciseV2.load_from_checkpoint(ckpt, map_location=device, strict=False)
    elif version == "v1":
        from models.hydro_precise import HydroPrecise
        m = HydroPrecise.load_from_checkpoint(ckpt, map_location=device, strict=False)
    else:
        raise ValueError(f"--model_version: {version!r}")
    return m.to(device).eval()


# ── Embedding extraction (match Phase A semantics) ─────────────────────────

@torch.no_grad()
def _extract(model, loader, device: str, version: str
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (feats, head_logits, labels) for the whole loader.

    For V1, the head outputs ``num_classes + 1`` logits (last column is the
    abstention logit). We drop it here so the downstream softmax compares on
    the same C-class distribution as V2.
    """
    feats, logits, labs = [], [], []
    nc = getattr(model, "num_classes", None)
    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        f = model._features(x)
        if version == "v2":
            l = model.head(f, None)
            if not model.training and getattr(model, "logit_adjust_tau", 0.0) > 0:
                l = l - model.logit_adjust_tau * model.log_prior
        else:                                            # v1
            l = model.head(f)
            l = l[:, :nc] if nc is not None else l[:, :-1]
        feats.append(f.float().cpu().numpy())
        logits.append(l.float().cpu().numpy())
        labs.append(y.cpu().numpy())
    return np.concatenate(feats), np.concatenate(logits), np.concatenate(labs)


# ── Saved temperature (post-hoc calibration) ────────────────────────────────

def _load_temperature(ckpt: str) -> float:
    T_path = Path(ckpt).parent / "temperature.pt"
    if not T_path.exists():
        return 1.0
    loaded = torch.load(T_path, weights_only=False)
    return float(loaded["temperature"]) if isinstance(loaded, dict) else float(loaded)


# ── Metrics helper ─────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> dict:
    pred = probs.argmax(axis=1)
    out = {
        "acc":       accuracy_score(y_true, pred),
        "f1_macro":  f1_score(y_true, pred, average="macro", zero_division=0),
        "micro_p":   precision_score(y_true, pred, average="micro", zero_division=0),
        "macro_p":   precision_score(y_true, pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, pred, average="macro", zero_division=0),
        "mcc":       matthews_corrcoef(y_true, pred),
    }
    try:
        out["auroc"] = roc_auc_score(y_true, probs, multi_class="ovr",
                                     labels=list(range(num_classes)))
    except Exception:
        out["auroc"] = float("nan")
    return out


def _row(mode: str, alpha: str, m: dict) -> str:
    return (f"| {mode:18s} | {alpha:>6s} | {m['micro_p']:.4f} | {m['macro_p']:.4f} | "
            f"{m['f1_macro']:.4f} | {m['acc']:.4f} | {m['mcc']:.4f} | {m['auroc']:.4f} |")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Single checkpoint, or comma-separated list for ensemble.")
    p.add_argument("--model_version", default="v2", choices=["v1", "v2"],
                   help="Which model class to load. V1 drops the abstention column.")
    p.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--sample_rate", type=int, default=5_120)
    p.add_argument("--fixed_len", type=int, default=5_120)
    p.add_argument("--knn_k", type=int, default=5)
    p.add_argument("--alphas", default="0.0,0.1,0.3,0.5,0.7,0.9,1.0",
                   help="Comma-separated blend coefficients to sweep on val.")
    p.add_argument("--out", default="lightning_logs/blend_knn/results.md")
    args = p.parse_args()

    ckpts: List[str] = [c.strip() for c in args.ckpt.split(",") if c.strip()]
    alphas = [float(a) for a in args.alphas.split(",")]
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[blend] {len(ckpts)} ckpt(s); model_version={args.model_version}")
    for c in ckpts:
        print(f"        - {c}")

    # Data — shared across all ckpts (same loader semantics)
    data = DALIAudioDataModule(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_threads=args.num_threads, target_sr=args.sample_rate,
        fixed_len=args.fixed_len, oversample_train=False,
    )
    data.setup()
    num_classes = data.num_classes
    print(f"[blend] {num_classes} classes  class_to_idx={data.class_to_idx}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    head_p_va_sum = None
    head_p_te_sum = None
    f_tr_first = y_tr_first = f_va_first = y_va_first = f_te_first = y_te_first = None

    for i, ckpt in enumerate(ckpts):
        T = _load_temperature(ckpt)
        print(f"\n[blend] === ckpt {i+1}/{len(ckpts)} ===  T={T:.4f}")
        model = _load_model(ckpt, args.model_version, device)
        print("[blend] extracting train ...")
        f_tr, _,    y_tr = _extract(model, data.train_dataloader(), device, args.model_version)
        print("[blend] extracting val ...")
        f_va, l_va, y_va = _extract(model, data.val_dataloader(),   device, args.model_version)
        print("[blend] extracting test ...")
        f_te, l_te, y_te = _extract(model, data.test_dataloader(),  device, args.model_version)

        # Accumulate head softmax
        head_p_va = torch.softmax(torch.tensor(l_va) / T, dim=-1).numpy()
        head_p_te = torch.softmax(torch.tensor(l_te) / T, dim=-1).numpy()
        head_p_va_sum = head_p_va if head_p_va_sum is None else (head_p_va_sum + head_p_va)
        head_p_te_sum = head_p_te if head_p_te_sum is None else (head_p_te_sum + head_p_te)

        # First ckpt is the KNN source (already-good embedding from any one
        # member is enough; multi-fit gains are negligible vs cost).
        if i == 0:
            f_tr_first, y_tr_first = f_tr, y_tr
            f_va_first, y_va_first = f_va, y_va
            f_te_first, y_te_first = f_te, y_te

        # Free the model — we may load several
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    head_p_va = head_p_va_sum / len(ckpts)
    head_p_te = head_p_te_sum / len(ckpts)
    y_va = y_va_first; y_te = y_te_first

    # KNN on the first ckpt's embeddings
    print(f"\n[blend] fitting KNN(k={args.knn_k}, cosine, weights=distance) "
          f"on first ckpt's train embeddings ({len(f_tr_first)} samples) ...")
    knn = KNeighborsClassifier(
        n_neighbors=args.knn_k, metric="cosine", weights="distance", n_jobs=-1,
    )
    knn.fit(f_tr_first, y_tr_first)
    print("[blend] knn.predict_proba(val) ...")
    knn_p_va = knn.predict_proba(f_va_first)
    print("[blend] knn.predict_proba(test) ...")
    knn_p_te = knn.predict_proba(f_te_first)

    # Sweep alpha on val
    print("\n[blend] sweeping alpha on val:")
    print(f"{'alpha':>6s}  μP      MP      f1      acc     mcc     auroc")
    best_alpha, best_micro = 1.0, -1.0
    for a in alphas:
        p_blend = a * head_p_va + (1.0 - a) * knn_p_va
        m = _metrics(y_va, p_blend, num_classes)
        print(f"{a:6.2f}  {m['micro_p']:.4f}  {m['macro_p']:.4f}  "
              f"{m['f1_macro']:.4f}  {m['acc']:.4f}  {m['mcc']:.4f}  {m['auroc']:.4f}")
        if m["micro_p"] > best_micro:
            best_micro, best_alpha = m["micro_p"], a
    print(f"\n[blend] best alpha = {best_alpha:.2f}  (val/micro_precision={best_micro:.4f})")

    # Test metrics
    head_only_te = head_p_te
    knn_only_te  = knn_p_te
    blend_te     = best_alpha * head_p_te + (1.0 - best_alpha) * knn_p_te

    rows = []
    rows.append(_row("head-only",  "1.00", _metrics(y_te, head_only_te, num_classes)))
    rows.append(_row(f"blend (alpha*)", f"{best_alpha:.2f}",
                     _metrics(y_te, blend_te, num_classes)))
    rows.append(_row("knn-only",   "0.00", _metrics(y_te, knn_only_te, num_classes)))

    title = (f"Phase C — {len(ckpts)}-ckpt {args.model_version} "
             f"on `{Path(ckpts[0]).name}`" + (f" + {len(ckpts)-1} more" if len(ckpts) > 1 else ""))
    md = [f"# {title}", ""]
    md.append(f"- model_version = {args.model_version}")
    md.append(f"- ensemble size = {len(ckpts)}")
    md.append(f"- best alpha (val) = {best_alpha:.2f}  (val/micro_precision={best_micro:.4f})")
    md.append(f"- KNN: k={args.knn_k}, cosine, distance-weighted, fit on first ckpt's train ({len(f_tr_first)} samples)")
    for c in ckpts:
        md.append(f"  - ckpt: `{c}`  (T={_load_temperature(c):.4f})")
    md.append("")
    md.append("| mode | alpha | test/μP | test/MP | test/f1 | test/acc | test/mcc | test/auroc |")
    md.append("|---|---|---|---|---|---|---|---|")
    md.extend(rows)
    md.append("")
    text = "\n".join(md)
    out_path.write_text(text)
    print("\n" + text)
    print(f"\n[blend] wrote {out_path}")


if __name__ == "__main__":
    main()
