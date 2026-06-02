"""Alternate calibration methods on the SOTA stacker — pick the best.

Fits each method on Val, reports Test macroF1 (full, no abstain) plus the
best selective@cov0.85 variant on top.

Methods:
  raw                                  — no calibration baseline
  per_class_iso                        — per-class isotonic regression
  temperature                          — single scalar T on log-probs
  vector_scaling                       — per-class T + bias (Platt vector)
  matrix_scaling                       — full K×K linear on log-probs
  beta_per_class                       — per-class Beta calibration
  per_source_type_iso                  — separate per-class iso for iara / deepship
  threshold_tune                       — per-class decision threshold on val
  best + MSP-isotonic selective@0.85   — sel layer on top of the winner
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize, minimize_scalar
from concurrent.futures import ThreadPoolExecutor

_R = Path(__file__).resolve().parents[1]
if str(_R) not in sys.path: sys.path.insert(0, str(_R))

from campaign.eval_recurrent_stacker import (  # noqa: E402
    scan_split, predict_split, metric_block, CLASSES,
)
from campaign.train_recurrent_stacker import EnsembleCache
from models.hydro_recurrent_stacker import HydroRecurrentStacker


def _softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def _norm(p): return p / p.sum(1, keepdims=True).clip(min=1e-9)


def _temperature(logits, y):
    def nll(T):
        p = _softmax(logits / max(T, 1e-3))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()
    return float(minimize_scalar(nll, bounds=(0.5, 5), method="bounded").x)


def _vector_scaling(logits, y, K):
    # logits/Wc + bc
    theta0 = np.r_[np.ones(K), np.zeros(K)]
    def nll(theta):
        W, b = theta[:K], theta[K:]
        z = logits / np.maximum(W, 1e-3) + b
        p = _softmax(z)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()
    r = minimize(nll, theta0, method="L-BFGS-B")
    W, b = r.x[:K], r.x[K:]
    return lambda z: _softmax(z / np.maximum(W, 1e-3) + b)


def _matrix_scaling(logits, y, K):
    # multi-class logistic regression on logits
    lr = LogisticRegression(max_iter=2000, multi_class="multinomial", C=1.0)
    lr.fit(logits, y)
    return lambda z: lr.predict_proba(z)


def _beta(p_v, y_v):
    # Beta calibration per class: fit logistic on (log p, log(1-p))
    eps = 1e-6
    p = np.clip(p_v, eps, 1 - eps)
    X = np.column_stack([np.log(p), -np.log(1 - p)])
    lr = LogisticRegression(max_iter=500, C=10.0)
    lr.fit(X, y_v)
    def f(p_t):
        p_t = np.clip(p_t, eps, 1 - eps)
        Xt = np.column_stack([np.log(p_t), -np.log(1 - p_t)])
        return lr.predict_proba(Xt)[:, 1]
    return f


def _threshold_tune(p_v, y_v, K):
    # Per-class additive offset in logit-space, fit by maximizing val macroF1.
    log_p = np.log(np.clip(p_v, 1e-9, 1))
    def f1_at(off):
        z = log_p + off[None, :]
        pred = z.argmax(1)
        f1s = []
        for c in range(K):
            tp = ((pred == c) & (y_v == c)).sum()
            fp = ((pred == c) & (y_v != c)).sum()
            fn = ((pred != c) & (y_v == c)).sum()
            d = 2*tp + fp + fn
            f1s.append(2*tp / d if d > 0 else 0.0)
        return -np.mean(f1s)
    r = minimize(f1_at, np.zeros(K), method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 2000})
    return r.x


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
    scales = ck.get("scales", [1, 3, 10])
    model = HydroRecurrentStacker(num_classes=cache.C, ens_n_ckpts=cache.M,
                                  ens_n_classes=cache.C, scales=scales,
                                  gambler=True).to(device)
    model.load_state_dict(ck["state_dict"])

    exe = ThreadPoolExecutor(max_workers=12)
    print("[cmp] predicting val + test")
    val_src  = scan_split(args.data_dir, "Val")
    test_src = scan_split(args.data_dir, "Test")
    yv, fp_v, fa_v, stv, _ = predict_split(model, val_src,  exe, device, cache, args.K_eval)
    yt, fp_t, fa_t, stt, _ = predict_split(model, test_src, exe, device, cache, args.K_eval)
    K = fp_v.shape[1]
    log_v = np.log(np.clip(fp_v, 1e-9, 1.0))
    log_t = np.log(np.clip(fp_t, 1e-9, 1.0))

    results = {}
    def record(name, p_t):
        pred = p_t.argmax(1)
        m = metric_block(yt, pred)
        results[name] = m['macro_f1']
        print(f"  {name:30s} test macroF1 = {m['macro_f1']:.4f}")
        return p_t

    print("\n=== Method comparison ===")
    record("raw", fp_t)

    # per-class iso
    iso_v = np.zeros_like(fp_v); iso_t = np.zeros_like(fp_t)
    for c in range(K):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
        ir.fit(fp_v[:, c], (yv == c).astype(int))
        iso_v[:, c] = ir.transform(fp_v[:, c]); iso_t[:, c] = ir.transform(fp_t[:, c])
    iso_v_n, iso_t_n = _norm(iso_v), _norm(iso_t)
    record("per_class_iso", iso_t_n)

    # temperature
    T = _temperature(log_v, yv)
    p_T = _softmax(log_t / T)
    print(f"    (T={T:.3f})")
    record("temperature", p_T)

    # vector scaling
    vs = _vector_scaling(log_v, yv, K)
    record("vector_scaling", vs(log_t))

    # matrix scaling
    try:
        ms = _matrix_scaling(log_v, yv, K)
        record("matrix_scaling", ms(log_t))
    except Exception as e:
        print(f"  matrix_scaling failed: {e}")

    # beta per-class
    beta_v = np.zeros_like(fp_v); beta_t = np.zeros_like(fp_t)
    for c in range(K):
        f = _beta(fp_v[:, c], (yv == c).astype(int))
        beta_v[:, c] = f(fp_v[:, c]); beta_t[:, c] = f(fp_t[:, c])
    record("beta_per_class", _norm(beta_t))

    # per-source-type iso
    pst_v = np.zeros_like(fp_v); pst_t = np.zeros_like(fp_t)
    for s in np.unique(stv):
        mv = (stv == s); mt = (stt == s)
        if mt.sum() == 0: continue
        for c in range(K):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
            ir.fit(fp_v[mv, c], (yv[mv] == c).astype(int))
            pst_v[mv, c] = ir.transform(fp_v[mv, c])
            pst_t[mt, c] = ir.transform(fp_t[mt, c])
    record("per_source_type_iso", _norm(pst_t))

    # threshold tune (logit-offset)
    off = _threshold_tune(fp_v, yv, K)
    p_thr_t = _softmax(log_t + off[None, :])
    record(f"threshold_tune (off={np.round(off,2).tolist()})", p_thr_t)

    # Selective@cov0.85 on top of the winner
    winner = max(results, key=results.get)
    print(f"\nwinner (raw): {winner}  macroF1={results[winner]:.4f}")
    # rebuild winner's probs
    winners_map = {
        "raw": fp_t, "per_class_iso": iso_t_n, "temperature": p_T,
        "vector_scaling": vs(log_t), "beta_per_class": _norm(beta_t),
        "per_source_type_iso": _norm(pst_t),
    }
    if "matrix_scaling" in results: winners_map["matrix_scaling"] = ms(log_t)
    winners_map[[k for k in results if k.startswith("threshold")][0]] = p_thr_t
    wp_t = winners_map[winner]
    if winner == "per_class_iso":
        wp_v = iso_v_n
    elif winner == "temperature":
        wp_v = _softmax(log_v / T)
    elif winner == "vector_scaling":
        wp_v = vs(log_v)
    elif winner == "matrix_scaling":
        wp_v = ms(log_v)
    elif winner == "beta_per_class":
        wp_v = _norm(beta_v)
    elif winner == "per_source_type_iso":
        wp_v = _norm(pst_v)
    elif winner.startswith("threshold"):
        wp_v = _softmax(log_v + off[None, :])
    else:
        wp_v = fp_v

    # MSP-isotonic on top
    msp_v = wp_v.max(1)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(msp_v, (wp_v.argmax(1) == yv).astype(int))
    p_corr_v = cal.predict(msp_v)
    cut = float(np.quantile(p_corr_v, 1 - 0.85))
    p_corr_t = cal.predict(wp_t.max(1))
    keep = p_corr_t >= cut
    m_sel = metric_block(yt[keep], wp_t[keep].argmax(1))
    print(f"  {winner} + MSP-iso sel@0.85  cov={keep.mean():.3f}  "
          f"test macroF1 = {m_sel['macro_f1']:.4f}")
    print("  per-class F1:", {CLASSES[i]: round(m_sel['per_class_f1'][i],3) for i in range(K)})


if __name__ == "__main__":
    raise SystemExit(main())
