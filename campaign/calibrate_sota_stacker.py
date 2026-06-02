"""Extra calibration on the SOTA HydroRecurrentStacker.

The existing pipeline (eval_recurrent_stacker.py) applies MSP-isotonic at
selection time → 0.715 test@cov0.85. This adds two further calibrators on
top of the raw final_probs, fit on val, evaluated on test:

  1. Per-class isotonic regression: fp_v[:, c] → empirical P(y=c|...)
  2. Single global temperature T on logits (Platt-style, scalar)

Reports both raw and calibrated test macroF1 (full, no abstain) plus the
combined per-class iso + selective@cov0.85 number.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar

_R = Path(__file__).resolve().parents[1]
if str(_R) not in sys.path: sys.path.insert(0, str(_R))

from campaign.eval_recurrent_stacker import (  # noqa: E402
    scan_split, predict_split, metric_block, CLASSES,
)
from models.hydro_recurrent_stacker import HydroRecurrentStacker
from campaign.train_recurrent_stacker import EnsembleCache
from concurrent.futures import ThreadPoolExecutor
import torch


def _temperature_scale(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(T):
        z = logits / max(T, 1e-3)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()
    r = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
    return float(r.x)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/var/mnt/5A009BF8009BD8F9/Data/Combined_IARA_Deepship_1s")
    ap.add_argument("--ens_npz", default="campaign/probs_combined_recurrent_stacker.npz")
    ap.add_argument("--ckpt", default="lightning_logs/hydro_recurrent_stacker_combined/best.pt")
    ap.add_argument("--K_eval", type=int, default=60)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = EnsembleCache(Path(args.ens_npz), Path(args.data_dir))

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    hp = ck.get("hparams") or {}
    scales = ck.get("scales", hp.get("scales", [1, 3, 10]))
    model = HydroRecurrentStacker(
        num_classes=cache.C,
        ens_n_ckpts=cache.M,
        ens_n_classes=cache.C,
        scales=scales, gambler=True,
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    print(f"Loaded stacker val_F1={ck.get('val_macro_f1'):.4f}")

    exe = ThreadPoolExecutor(max_workers=12)
    print("[cal] scanning splits + running val/test")
    val_src  = scan_split(args.data_dir, "Val")
    test_src = scan_split(args.data_dir, "Test")
    yv, fp_v, fa_v, _, _ = predict_split(model, val_src,  exe, device, cache, args.K_eval)
    yt, fp_t, fa_t, _, _ = predict_split(model, test_src, exe, device, cache, args.K_eval)
    K = fp_v.shape[1]

    # Raw test baseline
    m_raw = metric_block(yt, fp_t.argmax(-1))
    print(f"\n[raw]            test macroF1 = {m_raw['macro_f1']:.4f}")

    # 1) Per-class isotonic on val
    iso_v = np.zeros_like(fp_v); iso_t = np.zeros_like(fp_t)
    for c in range(K):
        y_bin = (yv == c).astype(np.int32)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(fp_v[:, c], y_bin)
        iso_v[:, c] = ir.transform(fp_v[:, c])
        iso_t[:, c] = ir.transform(fp_t[:, c])
    iso_v /= iso_v.sum(1, keepdims=True).clip(min=1e-9)
    iso_t /= iso_t.sum(1, keepdims=True).clip(min=1e-9)
    m_iso = metric_block(yt, iso_t.argmax(-1))
    print(f"[per-class iso]  test macroF1 = {m_iso['macro_f1']:.4f}")

    # 2) Temperature scaling on val (treating fp as p, log -> logits proxy)
    logits_v = np.log(np.clip(fp_v, 1e-9, 1.0))
    logits_t = np.log(np.clip(fp_t, 1e-9, 1.0))
    T = _temperature_scale(logits_v, yv)
    z_t = logits_t / T
    z_t -= z_t.max(1, keepdims=True)
    p_t = np.exp(z_t); p_t /= p_t.sum(1, keepdims=True)
    m_tmp = metric_block(yt, p_t.argmax(-1))
    print(f"[T-scaled T={T:.3f}]  test macroF1 = {m_tmp['macro_f1']:.4f}")

    # 3) Selective@cov0.85 with per-class iso replacing fp
    msp_v = iso_v.max(1)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(msp_v, (iso_v.argmax(-1) == yv).astype(np.int32))
    p_corr_v = cal.predict(msp_v)
    cut = float(np.quantile(p_corr_v, 1 - 0.85))
    msp_t = iso_t.max(1); p_corr_t = cal.predict(msp_t)
    keep = p_corr_t >= cut
    m_sel = metric_block(yt[keep], iso_t[keep].argmax(-1))
    print(f"[iso + MSP-iso sel cov0.85]  cov={keep.mean():.3f}  "
          f"test macroF1 = {m_sel['macro_f1']:.4f}")

    # Per-class breakdown best variant
    best = max([("raw", m_raw), ("iso", m_iso), ("tmp", m_tmp), ("iso+sel", m_sel)],
               key=lambda x: x[1]['macro_f1'])
    print(f"\n→ best: {best[0]}  macroF1={best[1]['macro_f1']:.4f}")
    for i, c in enumerate(CLASSES):
        print(f"  {c:>10s}  F1={best[1]['per_class_f1'][i]:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
