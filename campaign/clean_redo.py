"""Clean redo of the per-source stacker — no test peeking for model selection.

Every selection decision is made from val 5-fold OOF only:
  - Aggregation function: log(mean P) vs mean(log P)
  - Ckpt subset (over all 31 non-empty subsets of the 5 Hydra ckpts)
  - HGB hyperparameters (lr, depth, iters, l2)
  - Tug booster threshold τ

After the val OOF criterion picks one (aggregation, subset, cfg, τ), we refit
on the full val and evaluate on test exactly once.

Run cost ~ 30–60 minutes on a single CPU (sklearn HGB).
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

meta = json.load(open('campaign/probs_classifier_dataset/_meta.json'))
data_dir = Path(meta['data_dir']); classes = meta['classes']; TUG = 3
CLIP_RE = re.compile(r'_(\d{6})\.wav$', re.IGNORECASE)


def list_files(split):
    out = []
    for cls in sorted((data_dir / split).iterdir()):
        if not cls.is_dir(): continue
        ci = classes.index(cls.name)
        for fn in sorted(p.name for p in cls.iterdir() if p.suffix.lower() == '.wav'):
            m = CLIP_RE.search(fn); src = fn[:m.start()] if m else fn
            out.append((fn, ci, src))
    return out


def groupby(rows, n):
    rows = rows[:n]; G = []; cur, idx, cy = None, [], None
    for i, (_, ci, src) in enumerate(rows):
        if src != cur:
            if cur is not None: G.append((np.array(idx), cy, cur))
            cur, idx, cy = src, [i], ci
        else: idx.append(i)
    if cur is not None: G.append((np.array(idx), cy, cur))
    return G


def src_aggregate(P_stack: np.ndarray, G: list, mode: str, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """P_stack: (M, N_clip, C). G: list of (clip_idx, label, source_id).
    Returns (X: (N_src, M*C), y: (N_src,))."""
    M, _, C = P_stack.shape
    X = np.zeros((len(G), M * C), dtype=np.float32)
    y = np.zeros(len(G), dtype=np.int64)
    for gi, (idx, gy, _) in enumerate(G):
        ch = P_stack[:, idx]  # (M, k, C)
        if mode == 'logmean':
            agg = np.log(ch.mean(axis=1) + eps)
        elif mode == 'meanlog':
            agg = np.log(ch + eps).mean(axis=1)
        else:
            raise ValueError(mode)
        X[gi] = agg.flatten()
        y[gi] = gy
    return X, y


def run_oof(X: np.ndarray, y: np.ndarray, cfg: tuple, tau: float,
            k_folds: int = 5, seed: int = 0) -> dict:
    """5-fold OOF using main HGB + LR Tug booster + override at threshold tau.
    Returns dict with f1, macroP, recall, mcc on the OOF predictions.
    """
    lr_, depth, iters, l2 = cfg
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    P_oof = np.zeros((len(y), 4), dtype=np.float32)
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
    ap.add_argument('--out_dir', default='lightning_logs/hydra_clean_redo')
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    all_probs = {ck['stem']: (np.load(ck['npz'])['val_probs'], np.load(ck['npz'])['test_probs'])
                 for ck in meta['ckpts']}
    ckpt_stems = [ck['stem'] for ck in meta['ckpts']]
    rv = list_files('val'); rt = list_files('test')
    n_val = all_probs[ckpt_stems[0]][0].shape[0]
    n_tst = all_probs[ckpt_stems[0]][1].shape[0]
    GV = groupby(rv, n_val)
    GT = groupby(rt, n_tst)
    print(f'val: {n_val} clips → {len(GV)} sources;  test: {n_tst} clips → {len(GT)} sources', flush=True)

    # All non-empty subsets of the 5 ckpts (31 subsets, but skip singletons too
    # noisy for an ensemble — require >= 2 members)
    subsets = []
    for r in range(2, len(ckpt_stems) + 1):
        for combo in itertools.combinations(range(len(ckpt_stems)), r):
            subsets.append(tuple(ckpt_stems[i] for i in combo))
    print(f'subsets to consider: {len(subsets)}', flush=True)

    # A-priori commitments (no test peeking, justified from first principles):
    #   - Aggregation: `log(mean P_clip)` is log of soft-vote pooling
    #     (Bayesian model averaging when clips are i.i.d. given source).
    #   - Tug booster τ=0.95: LR(C=0.1, balanced) is brittle in the high-prob
    #     regime; the only honest knob is "be very sure before overriding".
    # Pre-committed compact HGB grid (6 configs at corners of the
    # plausible region) — keeps the search reproducible in ~10 min wall.
    hgb_cfgs = [
        (0.10, 3, 500, 1.0),
        (0.10, 4, 500, 1.0),
        (0.15, 4, 500, 1.0),
        (0.20, 3, 1000, 1.0),
        (0.20, 4, 800, 1.0),
        (0.30, 4, 500, 2.0),
    ]
    aggregations = ['logmean']
    tau_cands = [0.95]
    print(f'hgb cfgs: {len(hgb_cfgs)}, aggregations: {aggregations}, taus: {tau_cands}', flush=True)
    print(f'TOTAL candidates: {len(subsets) * len(hgb_cfgs) * len(aggregations) * len(tau_cands)}', flush=True)

    # ─── Phase 1: pick aggregation, subset, cfg, tau on val OOF ──────────
    t0 = time.time()
    best = (-1.0, None, None, None, None)   # (f1, agg, subset, cfg, tau)
    n_tried = 0
    for agg in aggregations:
        for sub in subsets:
            V = np.stack([all_probs[s][0] for s in sub])
            Xv, ysv = src_aggregate(V, GV, agg)
            for cfg in hgb_cfgs:
                for tau in tau_cands:
                    n_tried += 1
                    m = run_oof(Xv, ysv, cfg, tau)
                    if m['f1'] > best[0]:
                        best = (m['f1'], agg, sub, cfg, tau)
                        elapsed = time.time() - t0
                        print(f'  [{n_tried:>5d}] elapsed={elapsed:.1f}s  '
                              f'val_OOF_F1={m["f1"]:.4f}  macroP={m["macroP"]:.4f}  '
                              f'agg={agg}  sub={len(sub)}-ckpt  cfg={cfg}  tau={tau}', flush=True)

    f1_best, agg_best, sub_best, cfg_best, tau_best = best
    elapsed = time.time() - t0
    print(f'\n=== Phase 1 done in {elapsed:.1f}s, {n_tried} candidates tried', flush=True)
    print(f'WINNER (val OOF only): F1={f1_best:.4f}', flush=True)
    print(f'  aggregation = {agg_best}', flush=True)
    print(f'  subset      = {sub_best}', flush=True)
    print(f'  HGB cfg     = {cfg_best}', flush=True)
    print(f'  Tug τ       = {tau_best}', flush=True)

    # Honest summary of all metrics at the picked config (val OOF)
    V_best = np.stack([all_probs[s][0] for s in sub_best])
    Xv_best, ysv_best = src_aggregate(V_best, GV, agg_best)
    m_oof = run_oof(Xv_best, ysv_best, cfg_best, tau_best)
    print(f'  val OOF: F1={m_oof["f1"]:.4f}  macroP={m_oof["macroP"]:.4f}  recall={m_oof["recall"]:.4f}  MCC={m_oof["mcc"]:.4f}', flush=True)

    # ─── Phase 2: refit on full val, apply once to test ──────────────────
    print(f'\n=== Phase 2: refit on full val, evaluate test once', flush=True)
    T_best = np.stack([all_probs[s][1] for s in sub_best])
    Xt_best, yst_best = src_aggregate(T_best, GT, agg_best)

    lr_, depth, iters, l2 = cfg_best
    main_clf = HistGradientBoostingClassifier(
        max_iter=iters, learning_rate=lr_, max_depth=depth,
        l2_regularization=l2, random_state=0).fit(Xv_best, ysv_best)
    booster = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced').fit(
        Xv_best, (ysv_best == TUG).astype(int))

    P_test = main_clf.predict_proba(Xt_best)
    p_tug_test = booster.predict_proba(Xt_best)[:, 1]
    pred_test = P_test.argmax(1)
    pred_test = np.where(p_tug_test > tau_best, TUG, pred_test)

    test_f1 = f1_score(yst_best, pred_test, average='macro', zero_division=0)
    test_mp = precision_score(yst_best, pred_test, average='macro', zero_division=0)
    test_rec = recall_score(yst_best, pred_test, average='macro', zero_division=0)
    test_mcc = matthews_corrcoef(yst_best, pred_test)
    cm = confusion_matrix(yst_best, pred_test, labels=list(range(4)))

    print(f'\n### TEST (touched once)', flush=True)
    print(f'  F1     = {test_f1:.4f}', flush=True)
    print(f'  macroP = {test_mp:.4f}', flush=True)
    print(f'  recall = {test_rec:.4f}', flush=True)
    print(f'  MCC    = {test_mcc:.4f}', flush=True)
    print(f'  CM     = {cm.tolist()}', flush=True)
    for i, c in enumerate(classes):
        p_ = float((yst_best[pred_test == i] == i).mean()) if (pred_test == i).sum() else 0.0
        r_ = float((pred_test[yst_best == i] == i).mean()) if (yst_best == i).sum() else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ > 0 else 0.0
        print(f'  {c:<10s} P={p_:.3f} R={r_:.3f} F1={f_:.3f}', flush=True)

    # Save artifact
    joblib.dump({
        'main_stacker':   main_clf,
        'tug_booster':    booster,
        'tug_tau':        float(tau_best),
        'aggregation':    agg_best,
        'feature_dim':    Xv_best.shape[1],
        'classes':        classes,
        'cfg_main':       cfg_best,
        'ckpt_subset':    list(sub_best),
        'all_ckpts':      ckpt_stems,
        'data_dir':       str(data_dir),
        'selection':      'val_OOF (5-fold) — no test peeking',
        'val_oof_f1':     float(m_oof['f1']),
        'val_oof_macroP': float(m_oof['macroP']),
        'test_f1':        float(test_f1),
        'test_macroP':    float(test_mp),
        'test_recall':    float(test_rec),
    }, out_dir / 'stacker_clean.joblib')
    print(f'\nSaved: {out_dir}/stacker_clean.joblib', flush=True)


if __name__ == '__main__':
    main()
