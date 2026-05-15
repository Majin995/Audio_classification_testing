"""Conservative stacker — minimal selection-DoF to avoid val-OOF overfitting.

What changed vs clean_redo baseline:
  - Per-ckpt temperature calibration with EXPANDED range T ∈ [0.5, 10.0]
    (clean_redo had no calibration; rich_v1 capped at 3.0 and all 5 ckpts saturated).
  - All 5 ckpts always (NO subset selection — that overfit in rich_v1).
  - Single feature: log(mean P_clip) per ckpt-class — same as baseline.
  - Single Tug booster only (τ=0.95 fixed a-priori, same as baseline).
  - Pre-committed HGB grid; pick on val 5-fold OOF.

Expected gain vs clean_redo: from calibration alone if probs were overconfident.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, matthews_corrcoef,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

meta = json.load(open('campaign/probs_classifier_dataset/_meta.json'))
data_dir = Path(meta['data_dir'])
classes = meta['classes']
NUM_CLASSES = 4
TUG = classes.index('Tug')
CLIP_RE = re.compile(r'_(\d{6})\.wav$', re.IGNORECASE)


def list_files(split):
    out = []
    for cls in sorted((data_dir / split).iterdir()):
        if not cls.is_dir():
            continue
        ci = classes.index(cls.name)
        for fn in sorted(p.name for p in cls.iterdir() if p.suffix.lower() == '.wav'):
            m = CLIP_RE.search(fn)
            src = fn[:m.start()] if m else fn
            out.append((fn, ci, src))
    return out


def groupby(rows, n):
    rows = rows[:n]
    G = []
    cur, idx, cy = None, [], None
    for i, (_, ci, src) in enumerate(rows):
        if src != cur:
            if cur is not None:
                G.append((np.array(idx), cy, cur))
            cur, idx, cy = src, [i], ci
        else:
            idx.append(i)
    if cur is not None:
        G.append((np.array(idx), cy, cur))
    return G


def fit_temperature(probs_val, y_val, T_grid):
    eps = 1e-8
    log_p = np.log(np.clip(probs_val, eps, 1.0))
    best_T, best_nll = 1.0, float('inf')
    for T in T_grid:
        z = log_p / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=1, keepdims=True)
        nll = -np.log(p[np.arange(len(y_val)), y_val] + eps).mean()
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return float(best_T)


def apply_T(probs, T, eps=1e-8):
    log_p = np.log(np.clip(probs, eps, 1.0))
    z = log_p / T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def src_aggregate_logmean(P_stack, G, eps=1e-8):
    M, _, C = P_stack.shape
    X = np.zeros((len(G), M * C), dtype=np.float32)
    y = np.zeros(len(G), dtype=np.int64)
    for gi, (idx, gy, _) in enumerate(G):
        ch = P_stack[:, idx]  # (M, k, C)
        agg = np.log(ch.mean(axis=1) + eps)
        X[gi] = agg.flatten()
        y[gi] = gy
    return X, y


def run_oof(X, y, cfg, tau, k_folds=5, seed=0):
    lr_, depth, iters, l2 = cfg
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    P_oof = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    p_tug_oof = np.zeros(len(y), dtype=np.float32)
    for tr, te in skf.split(np.zeros_like(y), y):
        m = HistGradientBoostingClassifier(
            max_iter=iters, learning_rate=lr_, max_depth=depth,
            l2_regularization=l2, random_state=0).fit(X[tr], y[tr])
        P_oof[te] = m.predict_proba(X[te])
        b = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced').fit(
            X[tr], (y[tr] == TUG).astype(int))
        p_tug_oof[te] = b.predict_proba(X[te])[:, 1]
    pred = P_oof.argmax(1)
    pred = np.where(p_tug_oof > tau, TUG, pred)
    return dict(
        f1=f1_score(y, pred, average='macro', zero_division=0),
        macroP=precision_score(y, pred, average='macro', zero_division=0),
        recall=recall_score(y, pred, average='macro', zero_division=0),
        mcc=matthews_corrcoef(y, pred) if len(set(y.tolist())) > 1 else 0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='lightning_logs/rich_stacker_v3')
    ap.add_argument('--no_calib', action='store_true')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_probs_raw = {ck['stem']: (np.load(ck['npz'])['val_probs'],
                                  np.load(ck['npz'])['test_probs'],
                                  np.load(ck['npz'])['val_y'])
                     for ck in meta['ckpts']}
    ckpt_stems = [ck['stem'] for ck in meta['ckpts']]
    rv = list_files('val')
    rt = list_files('test')
    n_val = all_probs_raw[ckpt_stems[0]][0].shape[0]
    n_tst = all_probs_raw[ckpt_stems[0]][1].shape[0]
    GV = groupby(rv, n_val)
    GT = groupby(rt, n_tst)
    print(f'val: {n_val} clips → {len(GV)} sources;  test: {n_tst} clips → {len(GT)} sources', flush=True)

    # Phase 1: temperature calibration with EXPANDED range
    if args.no_calib:
        print('\n=== Phase 1 SKIPPED (T=1 fixed)', flush=True)
        cal_probs = {s: (p[0], p[1]) for s, p in all_probs_raw.items()}
        temps = {s: 1.0 for s in ckpt_stems}
    else:
        print('\n=== Phase 1: per-ckpt temperature on val (range 0.5..10.0)', flush=True)
        T_grid = np.concatenate([np.linspace(0.5, 3.0, 26), np.linspace(3.1, 10.0, 70)])
        cal_probs = {}
        temps = {}
        for s in ckpt_stems:
            Pv, Pt, yv = all_probs_raw[s]
            T = fit_temperature(Pv, yv, T_grid)
            temps[s] = T
            cal_probs[s] = (apply_T(Pv, T), apply_T(Pt, T))
            print(f'  {s}: T={T:.3f}', flush=True)

    # ALL 5 ckpts — no subset selection
    sub = tuple(ckpt_stems)
    V = np.stack([cal_probs[s][0] for s in sub])
    Xv, ysv = src_aggregate_logmean(V, GV)
    print(f'\nfeature shape: {Xv.shape}', flush=True)

    # Phase 2: HGB grid on val OOF
    hgb_cfgs = [
        (0.10, 3, 500, 1.0),
        (0.10, 4, 500, 1.0),
        (0.15, 4, 500, 1.0),
        (0.20, 3, 1000, 1.0),
        (0.20, 4, 800, 1.0),
        (0.30, 4, 500, 2.0),
    ]
    tau = 0.95
    print('\n=== Phase 2: pick HGB cfg on val OOF (Tug τ=0.95 fixed)', flush=True)
    t0 = time.time()
    best = (-1.0, None)
    for cfg in hgb_cfgs:
        m = run_oof(Xv, ysv, cfg, tau)
        print(f'  [{time.time()-t0:5.1f}s] cfg={cfg}  val_OOF_F1={m["f1"]:.4f}  '
              f'macroP={m["macroP"]:.4f}  recall={m["recall"]:.4f}', flush=True)
        if m['f1'] > best[0]:
            best = (m['f1'], cfg)
    f1_best, cfg_best = best
    print(f'\nWINNER cfg: {cfg_best}  val_OOF_F1={f1_best:.4f}', flush=True)

    # Phase 3: refit, evaluate test once
    print('\n=== Phase 3: refit, evaluate test once', flush=True)
    T_stack = np.stack([cal_probs[s][1] for s in sub])
    Xt, yst = src_aggregate_logmean(T_stack, GT)

    lr_, depth, iters, l2 = cfg_best
    main_clf = HistGradientBoostingClassifier(
        max_iter=iters, learning_rate=lr_, max_depth=depth,
        l2_regularization=l2, random_state=0).fit(Xv, ysv)
    booster = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced').fit(
        Xv, (ysv == TUG).astype(int))

    P_test = main_clf.predict_proba(Xt)
    p_tug_test = booster.predict_proba(Xt)[:, 1]
    pred_test = P_test.argmax(1)
    pred_test = np.where(p_tug_test > tau, TUG, pred_test)

    test_f1 = f1_score(yst, pred_test, average='macro', zero_division=0)
    test_mp = precision_score(yst, pred_test, average='macro', zero_division=0)
    test_rec = recall_score(yst, pred_test, average='macro', zero_division=0)
    test_mcc = matthews_corrcoef(yst, pred_test)
    cm = confusion_matrix(yst, pred_test, labels=list(range(4)))

    print(f'\n### TEST (touched once)', flush=True)
    print(f'  F1     = {test_f1:.4f}', flush=True)
    print(f'  macroP = {test_mp:.4f}', flush=True)
    print(f'  recall = {test_rec:.4f}', flush=True)
    print(f'  MCC    = {test_mcc:.4f}', flush=True)
    print(f'  CM     = {cm.tolist()}', flush=True)
    for i, c in enumerate(classes):
        p_ = float((yst[pred_test == i] == i).mean()) if (pred_test == i).sum() else 0.0
        r_ = float((pred_test[yst == i] == i).mean()) if (yst == i).sum() else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ > 0 else 0.0
        print(f'  {c:<10s} P={p_:.3f} R={r_:.3f} F1={f_:.3f}', flush=True)

    joblib.dump({
        'main_stacker':   main_clf,
        'tug_booster':    booster,
        'tug_tau':        float(tau),
        'temperatures':   temps,
        'ckpt_subset':    list(sub),
        'cfg_main':       cfg_best,
        'data_dir':       str(data_dir),
        'val_oof_f1':     float(f1_best),
        'test_f1':        float(test_f1),
        'test_macroP':    float(test_mp),
        'test_recall':    float(test_rec),
    }, out_dir / 'rich_stacker_v3.joblib')
    print(f'\nSaved: {out_dir}/rich_stacker_v3.joblib', flush=True)


if __name__ == '__main__':
    main()
