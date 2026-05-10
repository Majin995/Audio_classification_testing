"""
Geometric-mean softmax ensemble for HydroHydra / HydroPrecise checkpoints.

For each supplied checkpoint:
  1. Load the LightningModule (inferred from ckpt hparams, or from --model_name).
  2. Compute logits on val + test.
  3. If ``temperature.pt`` exists alongside the ckpt, use that T for calibration.
     Otherwise fit a fresh T on the val logits.
  4. Convert to softmax probabilities.

Across all ckpts, take the geometric mean of per-class probabilities
(exp(mean(log p)), re-normalised).  Run the per-class threshold search on the
ensembled val probabilities at ``--target_coverage``.  Report gated macro-
precision on test.  Save ``ensemble_results.json``.

Usage
-----
    python scripts/ensemble_predict.py \
        --ckpt /path/to/hydra1/best.ckpt \
        --ckpt /path/to/hydra2/best.ckpt \
        --data_dir /abs/path/to/Split1s \
        --target_coverage 0.85 \
        --out lightning_logs/ensemble_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from data.audio_lightning_loader import DALIAudioDataModule
from training.train_precise      import _collect_logits, _fit_temperature, _search_thresholds


_MODEL_NAME_TO_CLS = {
    "hydra":   "models.hydro_hydra:HydroHydra",
    "precise": "models.hydro_precise:HydroPrecise",
}


def _load_cls(path: str):
    mod_name, cls_name = path.split(":")
    mod = __import__(mod_name, fromlist=[cls_name])
    return getattr(mod, cls_name)


def _infer_model_name(ckpt_path: str) -> str:
    """Inspect ckpt 'hyper_parameters' dict to guess which model class to load."""
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = d.get("hyper_parameters", {})
    # HydroHydra has use_gabor / use_scattering; HydroPrecise has cqt_n_bins
    if "use_scattering" in hparams or "use_gabor" in hparams:
        return "hydra"
    if "cqt_n_bins" in hparams:
        return "precise"
    raise ValueError(
        f"Cannot infer model class for {ckpt_path}. "
        "Pass --model_name alongside --ckpt."
    )


@torch.no_grad()
def _logits_for_loader(model, loader, device):
    model.eval()
    logits_all, targets_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        # drop abstention if present
        n = getattr(model, "num_classes", out.shape[-1])
        logits_all.append(out[:, :n].cpu())
        targets_all.append(y.cpu())
    return torch.cat(logits_all), torch.cat(targets_all)


def _geometric_mean(probs_list, eps: float = 1e-12) -> torch.Tensor:
    """Geometric mean of a list of (B, C) probability tensors; re-normalised."""
    log_p = torch.stack([p.clamp(min=eps).log() for p in probs_list])     # (N, B, C)
    p = log_p.mean(dim=0).exp()                                            # (B, C)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=eps)
    return p


def _confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


def get_args():
    p = argparse.ArgumentParser(description="Geo-mean softmax ensemble")
    p.add_argument("--ckpt", action="append", required=True,
                   help="One checkpoint path; repeat --ckpt for each.")
    p.add_argument("--model_name", action="append", default=[],
                   help="Optional model name per --ckpt (same order). "
                        "Valid: hydra, precise. Inferred from hparams if omitted.")
    p.add_argument("--data_dir", default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--sample_rate", type=int, default=5_120)
    p.add_argument("--fixed_len",  type=int, default=5_120)
    p.add_argument("--target_coverage", type=float, default=0.85)
    p.add_argument("--out", default="lightning_logs/ensemble")
    return p.parse_args()


def main():
    args = get_args()
    if not args.data_dir:
        raise ValueError("Set --data_dir or export DATA_DIR")

    names = list(args.model_name) if args.model_name else []
    while len(names) < len(args.ckpt):
        names.append(_infer_model_name(args.ckpt[len(names)]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        oversample_train=True,
        denoise_method="off",
    )
    data.setup()
    num_classes = data.num_classes

    val_probs_list, test_probs_list = [], []
    val_targets, test_targets = None, None
    per_model_temps = []

    for ckpt_path, model_name in zip(args.ckpt, names):
        if model_name not in _MODEL_NAME_TO_CLS:
            raise ValueError(
                f"Unknown model_name '{model_name}'. "
                f"Known: {list(_MODEL_NAME_TO_CLS)}"
            )
        cls = _load_cls(_MODEL_NAME_TO_CLS[model_name])
        print(f"\n── loading {model_name} from {ckpt_path}")
        m = cls.load_from_checkpoint(ckpt_path, map_location=device, strict=False).to(device).eval()

        # Temperature: reuse saved one if present
        temp_path = Path(ckpt_path).parent / "temperature.pt"
        if temp_path.exists():
            T = float(torch.load(temp_path, map_location="cpu", weights_only=False)["temperature"])
            print(f"   reuse temperature.pt → T = {T:.3f}")
        else:
            val_logits, val_targets_ = _logits_for_loader(m, data.val_dataloader(), device)
            T = _fit_temperature(val_logits, val_targets_)
            print(f"   fitted new T = {T:.3f}")
            if val_targets is None:
                val_targets = val_targets_
        per_model_temps.append({"ckpt": ckpt_path, "model_name": model_name, "temperature": T})

        val_logits,  val_targets_  = _logits_for_loader(m, data.val_dataloader(),  device)
        test_logits, test_targets_ = _logits_for_loader(m, data.test_dataloader(), device)
        val_probs_list.append(F.softmax(val_logits  / T, dim=-1))
        test_probs_list.append(F.softmax(test_logits / T, dim=-1))

        if val_targets  is None: val_targets  = val_targets_
        if test_targets is None: test_targets = test_targets_

    # ── Geo-mean ensemble ────────────────────────────────────────────────
    val_probs  = _geometric_mean(val_probs_list)
    test_probs = _geometric_mean(test_probs_list)

    # ── Threshold sweep on val ensemble → apply to test ──────────────────
    val_res = _search_thresholds(
        val_probs, val_targets, num_classes=num_classes,
        target_coverage=args.target_coverage,
    )
    print(f"\n── Ensemble result ({len(args.ckpt)} models) ──")
    print(f"  thresholds (from val)  = {['%.2f' % t for t in val_res['thresholds']]}")
    print(f"  val macro_precision    = {val_res['macro_precision']:.4f}  "
          f"coverage = {val_res['coverage']:.4f}")

    # Apply selected thresholds on the test set
    thr = np.asarray(val_res["thresholds"])
    test_argmax = test_probs.numpy().argmax(axis=1)
    test_pmax   = test_probs.numpy().max(axis=1)
    test_targets_np = test_targets.numpy()
    keep = test_pmax >= thr[test_argmax]

    if keep.sum() == 0:
        test_prec = 0.0
        test_cov  = 0.0
        test_cm   = _confusion_matrix(np.zeros(0), np.zeros(0), num_classes)
    else:
        preds_kept  = test_argmax[keep]
        gts_kept    = test_targets_np[keep]
        per_class   = []
        for c in range(num_classes):
            pm = preds_kept == c
            if pm.sum() == 0:
                continue
            per_class.append((gts_kept[pm] == c).mean())
        test_prec = float(np.mean(per_class)) if per_class else 0.0
        test_cov  = float(keep.mean())
        test_cm   = _confusion_matrix(preds_kept, gts_kept, num_classes)

    print(f"  test macro_precision   = {test_prec:.4f}  coverage = {test_cov:.4f}")
    print(f"  test confusion matrix (gated):\n{test_cm}")

    results = {
        "n_models": len(args.ckpt),
        "per_model": per_model_temps,
        "target_coverage": args.target_coverage,
        "val": {
            "thresholds":      val_res["thresholds"],
            "macro_precision": val_res["macro_precision"],
            "coverage":        val_res["coverage"],
        },
        "test": {
            "macro_precision":   test_prec,
            "coverage":          test_cov,
            "confusion_matrix":  test_cm.tolist(),
        },
    }
    out_file = out_dir / "ensemble_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out_file}")


if __name__ == "__main__":
    main()
