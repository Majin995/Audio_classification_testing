"""Honest clean_redo extended to 7 ckpts.

Adds the 2 v2/v3 ckpts (hydra-069 p=0.7279, hydra-042 p=0.7114) to the v1 pool.
Same selection protocol as clean_redo (val 5-fold OOF picks subset/cfg; test
touched exactly once).

Alignment: v2/v3 caches have 50784 val clips (v1 has 50752) — the +32 are
appended Tug clips (alphabetically after v1's set). Truncating v2/v3 val to
[:50752] matches v1's enumeration exactly under the per-class alphabetical sort.

Test (14208 clips) aligns perfectly across all caches.
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
    f1_score, precision_score, recall_score, matthews_corrcoef,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold


meta_v1 = json.load(open('campaign/probs_classifier_dataset/_meta.json'))
data_dir = Path(meta_v1['data_dir'])
classes = meta_v1['classes']
TUG = classes.index('Tug')
CLIP_RE = re.compile(r'_(\d{6})\.wav$', re.IGNORECASE)
N_VAL = 50752
N_TST = 14208


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


def src_agg_logmean(P_stack, G, eps=1e-8):
    M, _, C = P_stack.shape
    X = np.zeros((len(G), M * C), dtype=np.float32)
    y = np.zeros(len(G), dtype=np.int64)
    for gi, (idx, gy, _) in enumerate(G):
        ch = P_stack[:, idx]
        X[gi] = np.log(ch.mean(1) + eps).flatten()
        y[gi] = gy
    return X, y


def run_oof(X, y, cfg, tau, k_folds=5):
    lr_, depth, iters, l2 = cfg
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0)
    P_oof = np.zeros((len(y), 4))
    p_tug_oof = np.zeros(len(y))
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
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='lightning_logs/clean_redo_7ckpt')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = {}
    for ck in meta_v1['ckpts']:
        z = np.load(ck['npz'])
        pool[ck['stem']] = (z['val_probs'][:N_VAL], z['test_probs'][:N_TST])
    p2 = Path('campaign/probs_classifier_dataset_v2/_meta.json')
    if p2.exists():
        for ck in json.load(open(p2))['ckpts']:
            z = np.load(ck['npz'])
            pool[ck['stem']] = (z['val_probs'][:N_VAL], z['test_probs'][:N_TST])
    p3 = Path('campaign/probs_classifier_dataset_v3')
    if p3.exists():
        for fn in sorted(p3.glob('*.npz')):
            z = np.load(fn)
            pool[fn.stem] = (z['val_probs'][:N_VAL], z['test_probs'][:N_TST])
    stems = sorted(pool.keys())
    print(f'pool: {len(stems)} ckpts:', stems, flush=True)

    rv = list_files('val')
    rt = list_files('test')
    GV = groupby(rv, N_VAL)
    GT = groupby(rt, N_TST)
    print(f'val: {N_VAL} clips → {len(GV)} sources;  test: {N_TST} clips → {len(GT)} sources', flush=True)

    # All subsets of size 2..7 (sum_{k=2}^{7} C(7,k) = 120)
    subsets = []
    for r in range(2, len(stems) + 1):
        for combo in itertools.combinations(stems, r):
            subsets.append(tuple(combo))
    print(f'subsets to evaluate: {len(subsets)}', flush=True)

    hgb_cfgs = [
        (0.10, 3, 500, 1.0),
        (0.10, 4, 500, 1.0),
        (0.15, 4, 500, 1.0),
        (0.20, 3, 1000, 1.0),
        (0.20, 4, 800, 1.0),
        (0.30, 4, 500, 2.0),
    ]
    print(f'TOTAL candidates: {len(subsets) * len(hgb_cfgs)}', flush=True)

    tau = 0.95
    t0 = time.time()
    best = (-1.0, None, None)
    n_tried = 0
    for sub in subsets:
        V = np.stack([pool[s][0] for s in sub])
        Xv, ysv = src_agg_logmean(V, GV)
        for cfg in hgb_cfgs:
            n_tried += 1
            m = run_oof(Xv, ysv, cfg, tau)
            if m['f1'] > best[0]:
                best = (m['f1'], sub, cfg)
                print(f'  [{n_tried}] elapsed={time.time()-t0:.1f}s  '
                      f'val_OOF_F1={m["f1"]:.4f} mP={m["macroP"]:.4f} R={m["recall"]:.4f}  '
                      f'|sub|={len(sub)} cfg={cfg}', flush=True)
    f1_best, sub_best, cfg_best = best
    print(f'\n=== Done in {time.time()-t0:.1f}s; tried {n_tried}', flush=True)
    print(f'subset: {sub_best}\ncfg:    {cfg_best}\nval_OOF F1={f1_best:.4f}', flush=True)

    # Phase 2: refit on full val, evaluate test once
    T_stack = np.stack([pool[s][1] for s in sub_best])
    V_stack = np.stack([pool[s][0] for s in sub_best])
    Xv, ysv = src_agg_logmean(V_stack, GV)
    Xt, yst = src_agg_logmean(T_stack, GT)

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
    print(f'\n### TEST (touched once)')
    print(f'  F1={test_f1:.4f}  macroP={test_mp:.4f}  recall={test_rec:.4f}  MCC={test_mcc:.4f}')
    print(f'  CM={cm.tolist()}')
    for i, c in enumerate(classes):
        p_ = float((yst[pred_test == i] == i).mean()) if (pred_test == i).sum() else 0.0
        r_ = float((pred_test[yst == i] == i).mean()) if (yst == i).sum() else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ > 0 else 0.0
        print(f'  {c:<10s} P={p_:.3f} R={r_:.3f} F1={f_:.3f}')

    joblib.dump({
        'main_stacker': main_clf, 'tug_booster': booster, 'tug_tau': tau,
        'ckpt_subset': list(sub_best), 'all_ckpts': stems,
        'cfg_main': cfg_best, 'data_dir': str(data_dir),
        'val_oof_f1': float(f1_best),
        'test_f1': float(test_f1), 'test_macroP': float(test_mp),
        'test_recall': float(test_rec),
    }, out_dir / 'stacker_7ckpt.joblib')
    print(f'\nSaved: {out_dir}/stacker_7ckpt.joblib')


if __name__ == '__main__':
    main()
