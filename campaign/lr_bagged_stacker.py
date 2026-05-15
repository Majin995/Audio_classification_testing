"""LR bagged stacker — fewer parameters per model, average over seeds.

Architecture: small multinomial LR with class_weight='balanced' on the
source-level log-mean features (20-dim for 5 ckpts, 28-dim for 7 ckpts).
Single tunable: regularization C, picked on val 5-fold OOF macro-F1.
Bag across 10 random seeds for variance reduction.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
import numpy as np
import joblib
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


def run_oof_lr_bag(X, y, C, n_bags=10, k_folds=5):
    P_oof = np.zeros((len(y), 4), dtype=np.float32)
    p_tug_oof = np.zeros(len(y), dtype=np.float32)
    for bag in range(n_bags):
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=bag)
        for tr, te in skf.split(np.zeros_like(y), y):
            m = LogisticRegression(
                C=C, max_iter=5000,
                random_state=bag,
            ).fit(X[tr], y[tr])
            P_oof[te] += m.predict_proba(X[te])
            b = LogisticRegression(
                C=0.1, max_iter=5000, class_weight='balanced',
                random_state=bag,
            ).fit(X[tr], (y[tr] == TUG).astype(int))
            p_tug_oof[te] += b.predict_proba(X[te])[:, 1]
    P_oof /= n_bags
    p_tug_oof /= n_bags
    return P_oof, p_tug_oof


def metrics(y, pred):
    return dict(
        f1=f1_score(y, pred, average='macro', zero_division=0),
        macroP=precision_score(y, pred, average='macro', zero_division=0),
        recall=recall_score(y, pred, average='macro', zero_division=0),
        mcc=matthews_corrcoef(y, pred) if len(set(y)) > 1 else 0.0,
    )


def report(name, y, pred):
    m = metrics(y, pred)
    cm = confusion_matrix(y, pred, labels=list(range(4)))
    print(f'[{name}] F1={m["f1"]:.4f} mP={m["macroP"]:.4f} R={m["recall"]:.4f} MCC={m["mcc"]:.4f}')
    print(f'   CM={cm.tolist()}')
    for i, c in enumerate(classes):
        p_ = float((y[pred == i] == i).mean()) if (pred == i).sum() else 0.0
        r_ = float((pred[y == i] == i).mean()) if (y == i).sum() else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ > 0 else 0.0
        print(f'   {c}: P={p_:.3f} R={r_:.3f} F1={f_:.3f}')
    return m


def main():
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

    rv = list_files('val')
    rt = list_files('test')
    GV = groupby(rv, N_VAL)
    GT = groupby(rt, N_TST)
    print(f'val: {N_VAL} clips → {len(GV)} sources;  test: {N_TST} clips → {len(GT)} sources')

    # Pre-committed subsets based on val_p
    val_p = {'hydra-010-p0.6943': 0.6943, 'hydra-012-p0.6929': 0.6929,
             'hydra-026-p0.6908': 0.6908, 'hydra-031-p0.7042': 0.7042,
             'hydra-032-p0.7010': 0.7010, 'hydra-042-p0.7114': 0.7114,
             'hydra-069-p0.7279': 0.7279}
    stems = sorted(pool.keys())
    ranked = sorted(stems, key=lambda s: -val_p[s])

    cands = [
        ('baseline_subset', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042']),
        ('baseline_plus_069_042', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042', 'hydra-069-p0.7279', 'hydra-042-p0.7114']),
        ('top4', ranked[:4]),
        ('all7', ranked),
    ]
    C_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

    print('\n=== Phase 1: subset × C search on val 5-fold OOF macro-F1 (LR x10 bag) ===')
    t0 = time.time()
    best = (-1.0, None, None, None, None)  # f1, subset_name, subset, C, oof_pred
    for name, subset in cands:
        V = np.stack([pool[s][0] for s in subset])
        Xv, ysv = src_agg_logmean(V, GV)
        for C in C_grid:
            P_oof, p_tug_oof = run_oof_lr_bag(Xv, ysv, C)
            pred = P_oof.argmax(1)
            pred_tugged = np.where(p_tug_oof > 0.95, TUG, pred)
            m1 = metrics(ysv, pred)
            m2 = metrics(ysv, pred_tugged)
            print(f'  [{time.time()-t0:5.1f}s] {name:24s} C={C:6.3f} '
                  f'noBoost F1={m1["f1"]:.4f} mP={m1["macroP"]:.4f} | '
                  f'+tug F1={m2["f1"]:.4f} mP={m2["macroP"]:.4f}')
            if m2['f1'] > best[0]:
                best = (m2['f1'], name, subset, C, pred_tugged)

    f1_best, name_best, sub_best, C_best, _ = best
    print(f'\nWINNER subset={name_best} C={C_best}  val_OOF F1={f1_best:.4f}')

    # Phase 2: refit on full val, evaluate test once
    print('\n=== Phase 2: refit on full val, evaluate test ===')
    V_best = np.stack([pool[s][0] for s in sub_best])
    T_best = np.stack([pool[s][1] for s in sub_best])
    Xv, ysv = src_agg_logmean(V_best, GV)
    Xt, yst = src_agg_logmean(T_best, GT)

    n_bags = 10
    P_test = np.zeros((len(yst), 4), dtype=np.float32)
    p_tug_test = np.zeros(len(yst), dtype=np.float32)
    for bag in range(n_bags):
        m = LogisticRegression(C=C_best, max_iter=5000,
                               random_state=bag).fit(Xv, ysv)
        P_test += m.predict_proba(Xt)
        b = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced',
                               random_state=bag).fit(Xv, (ysv == TUG).astype(int))
        p_tug_test += b.predict_proba(Xt)[:, 1]
    P_test /= n_bags
    p_tug_test /= n_bags
    pred_test = P_test.argmax(1)
    pred_test = np.where(p_tug_test > 0.95, TUG, pred_test)
    m = report('TEST', yst, pred_test)

    out_dir = Path('lightning_logs/lr_bagged_stacker')
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'method': 'LR bagged (n_bags=10) + Tug LR booster',
        'ckpt_subset': sub_best, 'subset_name': name_best,
        'C_best': C_best,
        'val_oof_f1': float(f1_best),
        'test_f1': float(m['f1']), 'test_macroP': float(m['macroP']),
        'test_recall': float(m['recall']), 'test_mcc': float(m['mcc']),
    }, out_dir / 'lr_bagged.joblib')
    print(f'\nSaved → {out_dir}/lr_bagged.joblib')


if __name__ == '__main__':
    main()
