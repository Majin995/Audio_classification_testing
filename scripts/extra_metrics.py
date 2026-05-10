"""Extended metric suite for HydroHydra checkpoints — beyond F1 / Pr / Rec.

Adds the metrics the Phase H summary did NOT cover:
- Per-class precision / recall / F1 / FPR
- macro/micro AUROC, AUPR
- Cohen's κ (linear)
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
- Multiclass Brier score
- Class-balanced accuracy
- Confusion matrix (val + test)
- Optional temperature-scaled variants of all calibration metrics

Reads the model's logits via DALI val + test loaders, applies the saved
``temperature.pt`` if present, and writes ``extra_metrics.md`` and
``extra_metrics.json`` next to the checkpoint.

Usage:
    python -m scripts.extra_metrics \
        --ckpts ck1.ckpt ck2.ckpt ... \
        --data_dir $DATA_DIR \
        --batch_size 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_hydra import HydroHydra


@torch.no_grad()
def _collect_logits(model, loader, device, num_classes: int):
    model.eval()
    logits_all, targets_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        if out.size(-1) > num_classes:
            out = out[:, :num_classes]
        logits_all.append(out.cpu().float())
        targets_all.append(y.cpu())
    return torch.cat(logits_all), torch.cat(targets_all)


def _ece_mce(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15):
    """Expected and Maximum Calibration Error.

    For each prediction the predicted class confidence is its top-1
    probability; correctness is whether argmax matches the target.
    Bins are equal-width over [0, 1].
    """
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == targets).astype(np.float32)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(targets)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confs > lo) & (confs <= hi if i == n_bins - 1 else confs <= hi)
        if i == 0:
            mask = (confs >= lo) & (confs <= hi)
        m_n = mask.sum()
        if m_n == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc = correct[mask].mean()
        gap = abs(avg_conf - avg_acc)
        ece += (m_n / N) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def _brier_multiclass(probs: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    """Multiclass Brier score: mean over samples of Σ_c (p_c − y_c)²."""
    N = len(targets)
    onehot = np.zeros((N, num_classes), dtype=np.float32)
    onehot[np.arange(N), targets] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def _per_class(probs: np.ndarray, targets: np.ndarray, num_classes: int):
    """Per-class precision, recall, F1, FPR. Returns list of dicts."""
    preds = probs.argmax(axis=1)
    out = []
    for c in range(num_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        tn = int(((preds != c) & (targets != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        support = int((targets == c).sum())
        out.append(dict(
            class_idx=c, support=support,
            precision=float(p), recall=float(r), f1=float(f1), fpr=float(fpr),
            tp=tp, fp=fp, fn=fn, tn=tn,
        ))
    return out


def _confmat(probs: np.ndarray, targets: np.ndarray, num_classes: int):
    preds = probs.argmax(axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


def _macro_auroc_aupr(probs: np.ndarray, targets: np.ndarray, num_classes: int):
    """Macro-averaged AUROC and AUPR (one-vs-rest)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(targets)), targets] = 1.0
    try:
        auroc_macro = float(roc_auc_score(onehot, probs, average="macro"))
        auroc_micro = float(roc_auc_score(onehot, probs, average="micro"))
        aupr_macro  = float(average_precision_score(onehot, probs, average="macro"))
        aupr_micro  = float(average_precision_score(onehot, probs, average="micro"))
    except Exception:
        auroc_macro = auroc_micro = aupr_macro = aupr_micro = float("nan")
    return auroc_macro, auroc_micro, aupr_macro, aupr_micro


def _cohen_kappa(probs: np.ndarray, targets: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(targets, probs.argmax(axis=1)))


def _mcc(probs: np.ndarray, targets: np.ndarray) -> float:
    from sklearn.metrics import matthews_corrcoef
    return float(matthews_corrcoef(targets, probs.argmax(axis=1)))


def _balanced_acc(probs: np.ndarray, targets: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(targets, probs.argmax(axis=1)))


def _all_metrics(logits: torch.Tensor, targets: torch.Tensor,
                 num_classes: int, T: float = 1.0):
    probs = F.softmax(logits / T, dim=-1).numpy()
    targets = targets.numpy()
    macro_auroc, micro_auroc, macro_aupr, micro_aupr = _macro_auroc_aupr(
        probs, targets, num_classes,
    )
    ece, mce = _ece_mce(probs, targets, n_bins=15)
    brier = _brier_multiclass(probs, targets, num_classes)
    return dict(
        T=float(T),
        macro_auroc=macro_auroc, micro_auroc=micro_auroc,
        macro_aupr=macro_aupr,   micro_aupr=micro_aupr,
        ece=ece, mce=mce, brier=brier,
        cohen_kappa=_cohen_kappa(probs, targets),
        mcc=_mcc(probs, targets),
        balanced_acc=_balanced_acc(probs, targets),
        per_class=_per_class(probs, targets, num_classes),
        confmat=_confmat(probs, targets, num_classes).tolist(),
    )


def _format_md(name: str, val_m: dict, test_m: dict,
               class_names: list[str]) -> str:
    lines = [f"# Extended metrics — {name}", ""]
    if val_m.get("T") and val_m.get("T") != 1.0:
        lines.append(f"Temperature applied: T = {val_m['T']:.4f}")
        lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("| metric | val | test |")
    lines.append("|---|---|---|")
    for k in ("macro_auroc", "micro_auroc", "macro_aupr", "micro_aupr",
              "cohen_kappa", "mcc", "balanced_acc",
              "ece", "mce", "brier"):
        lines.append(f"| {k} | {val_m[k]:.4f} | {test_m[k]:.4f} |")
    lines.append("")

    for split, m in (("val", val_m), ("test", test_m)):
        lines.append(f"## Per-class — {split}")
        lines.append("| class | support | precision | recall | F1 | FPR |")
        lines.append("|---|---|---|---|---|---|")
        for pc in m["per_class"]:
            cn = class_names[pc["class_idx"]] if pc["class_idx"] < len(class_names) else f"c{pc['class_idx']}"
            lines.append(
                f"| {cn} | {pc['support']} | {pc['precision']:.4f} | "
                f"{pc['recall']:.4f} | {pc['f1']:.4f} | {pc['fpr']:.4f} |"
            )
        lines.append("")
        lines.append(f"### Confusion matrix — {split} (rows=true, cols=pred)")
        lines.append("| | " + " | ".join(class_names) + " |")
        lines.append("|" + "---|" * (len(class_names) + 1))
        cm = m["confmat"]
        for i, row in enumerate(cm):
            lines.append(f"| **{class_names[i]}** | " + " | ".join(str(x) for x in row) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_threads", type=int, default=4)
    p.add_argument("--sample_rate", type=int, default=5_120)
    p.add_argument("--fixed_len",   type=int, default=5_120)
    p.add_argument("--use_temperature", action="store_true",
                   help="If set, look for temperature.pt next to ckpt and apply it.")
    args = p.parse_args()

    if not args.data_dir:
        raise SystemExit("Set --data_dir or DATA_DIR")

    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size, num_threads=args.num_threads,
        target_sr=args.sample_rate, fixed_len=args.fixed_len,
        oversample_train=False,
    )
    data.setup()
    nc = data.num_classes
    class_names = [data.idx_to_class[i] for i in range(nc)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ck in args.ckpts:
        ck_path = Path(ck)
        out_dir = ck_path.parent
        print(f"\n[extra-metrics] {ck}")

        T = 1.0
        if args.use_temperature:
            tp = ck_path.parent / "temperature.pt"
            if tp.exists():
                T = float(torch.load(tp, weights_only=False).get("temperature", 1.0))
                print(f"  loaded temperature = {T:.4f}")

        m = HydroHydra.load_from_checkpoint(ck, map_location=device, strict=False).to(device).eval()
        v_logits, v_targets = _collect_logits(m, data.val_dataloader(), device, nc)
        t_logits, t_targets = _collect_logits(m, data.test_dataloader(), device, nc)
        del m
        torch.cuda.empty_cache()

        val_m  = _all_metrics(v_logits, v_targets, nc, T=T)
        test_m = _all_metrics(t_logits, t_targets, nc, T=T)

        name = ck_path.parent.parent.parent.name + "/" + ck_path.name
        md_path  = out_dir / "extra_metrics.md"
        json_path = out_dir / "extra_metrics.json"
        md_path.write_text(_format_md(name, val_m, test_m, class_names) + "\n")
        json_path.write_text(json.dumps(
            {"name": name, "T": T, "val": val_m, "test": test_m},
            indent=2,
        ))
        print(f"  saved → {md_path}")
        print(f"  test: macro_auroc={test_m['macro_auroc']:.4f}  "
              f"macro_aupr={test_m['macro_aupr']:.4f}  "
              f"κ={test_m['cohen_kappa']:.4f}  ECE={test_m['ece']:.4f}  "
              f"Brier={test_m['brier']:.4f}")


if __name__ == "__main__":
    main()
