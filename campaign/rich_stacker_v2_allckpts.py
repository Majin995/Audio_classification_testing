"""Rich per-source stacker v2 — uses ALL 7 cached ckpts (5 v1 + 2 v2/v3).

Architectural additions vs v1:
  * Adds hydra-069 (val_OOF p=0.7279) and hydra-042 (p=0.7114) — Phase H/G' ckpts
    that introduce architectural diversity (DEMON-MoE and different stream-mix).
  * Soft probability blending between main HGB and per-class booster.
  * Same val OOF selection protocol, test touched exactly once.

Alignment: v2/v3 caches have 50784 val clips (vs v1's 50752; +32 in Tug class).
All caches share the same test enumeration (14208 clips). For val we truncate v2/v3
to the first 50752 entries — class boundaries 0..45368 are identical across all,
and Tug clips 45368..50752 are the first 5384 of v2/v3's 5416 alphabetically-sorted
Tug clips, which match v1's enumeration.
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

# Build ckpt pool: 5 v1 + 2 extras
meta_v1 = json.load(open('campaign/probs_classifier_dataset/_meta.json'))
data_dir = Path(meta_v1['data_dir'])
classes = meta_v1['classes']
NUM_CLASSES = 4
TUG = classes.index('Tug')
CARGO = classes.index('Cargo')
TANKER = classes.index('Tanker')
CLIP_RE = re.compile(r'_(\d{6})\.wav$', re.IGNORECASE)


def load_all_ckpts():
    pool = {}
    for ck in meta_v1['ckpts']:
        z = np.load(ck['npz'])
        pool[ck['stem']] = (z['val_probs'], z['test_probs'], z['val_y'], z['test_y'])
    # v2
    p2 = Path('campaign/probs_classifier_dataset_v2/_meta.json')
    if p2.exists():
        for ck in json.load(open(p2))['ckpts']:
            z = np.load(ck['npz'])
            pool[ck['stem']] = (z['val_probs'], z['test_probs'], z['val_y'], z['test_y'])
    # v3 (lacks meta.json, scan dir)
    p3 = Path('campaign/probs_classifier_dataset_v3')
    if p3.exists():
        for fn in sorted(p3.glob('*.npz')):
            z = np.load(fn)
            stem = fn.stem
            pool[stem] = (z['val_probs'], z['test_probs'], z['val_y'], z['test_y'])
    return pool


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


def fit_temperature(probs_val: np.ndarray, y_val: np.ndarray) -> float:
    eps = 1e-8
    log_p = np.log(np.clip(probs_val, eps, 1.0))
    best_T, best_nll = 1.0, float('inf')
    for T in np.linspace(0.5, 3.0, 26):
        z = log_p / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=1, keepdims=True)
        nll = -np.log(p[np.arange(len(y_val)), y_val] + eps).mean()
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return float(best_T)


def apply_temperature(probs: np.ndarray, T: float, eps: float = 1e-8) -> np.ndarray:
    log_p = np.log(np.clip(probs, eps, 1.0))
    z = log_p / T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def rich_features(P_stack: np.ndarray, G: list, modes: tuple, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    M, _, C = P_stack.shape
    parts = []
    n = len(G)
    for mode in modes:
        if mode in ('logmean', 'logmax', 'log_q90'):
            X = np.zeros((n, M * C), dtype=np.float32)
            for gi, (idx, _, _) in enumerate(G):
                ch = P_stack[:, idx]  # (M, k, C)
                if mode == 'logmean':
                    s = np.log(ch.mean(1) + eps)
                elif mode == 'logmax':
                    s = np.log(ch.max(1) + eps)
                else:  # log_q90
                    s = np.log(np.quantile(ch, 0.9, axis=1) + eps)
                X[gi] = s.reshape(M * C)
            parts.append(X)
        elif mode == 'var':
            X = np.zeros((n, M * C), dtype=np.float32)
            for gi, (idx, _, _) in enumerate(G):
                ch = P_stack[:, idx]
                X[gi] = ch.var(1).reshape(M * C)
            parts.append(X)
        elif mode == 'entropy':
            X = np.zeros((n, M), dtype=np.float32)
            for gi, (idx, _, _) in enumerate(G):
                ch = P_stack[:, idx]  # (M, k, C)
                ent = -(ch * np.log(np.clip(ch, eps, 1.0))).sum(-1).mean(-1)
                X[gi] = ent
            parts.append(X)
        else:
            raise ValueError(mode)
    y = np.array([g[1] for g in G], dtype=np.int64)
    return np.concatenate(parts, axis=1), y


def run_oof_soft(X: np.ndarray, y: np.ndarray, cfg: tuple,
                 boost_classes: tuple, boost_weights: dict,
                 k_folds: int = 5, seed: int = 0) -> tuple[dict, np.ndarray]:
    """OOF with SOFT booster blending: P_final = P_main + w_c * P_booster[c]
    (across class c in boost_classes), normalized at the end. Argmax for prediction.
    """
    lr_, depth, iters, l2 = cfg
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    P_oof = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    boost_oof = {c: np.zeros(len(y), dtype=np.float32) for c in boost_classes}
    for tr, te in skf.split(np.zeros_like(y), y):
        m = HistGradientBoostingClassifier(
            max_iter=iters, learning_rate=lr_, max_depth=depth,
            l2_regularization=l2, random_state=0).fit(X[tr], y[tr])
        P_oof[te] = m.predict_proba(X[te])
        for c in boost_classes:
            yc = (y[tr] == c).astype(int)
            if yc.sum() == 0 or yc.sum() == len(yc):
                continue
            b = LogisticRegression(
                C=0.1, max_iter=5000, class_weight='balanced').fit(X[tr], yc)
            boost_oof[c][te] = b.predict_proba(X[te])[:, 1]
    P_blend = P_oof.copy()
    for c in boost_classes:
        if c in boost_weights:
            P_blend[:, c] += boost_weights[c] * boost_oof[c]
    P_blend = P_blend / P_blend.sum(axis=1, keepdims=True).clip(min=1e-8)
    pred = P_blend.argmax(1)
    return dict(
        f1=f1_score(y, pred, average='macro', zero_division=0),
        macroP=precision_score(y, pred, average='macro', zero_division=0),
        recall=recall_score(y, pred, average='macro', zero_division=0),
        mcc=matthews_corrcoef(y, pred) if len(set(y.tolist())) > 1 else 0.0,
    ), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='lightning_logs/rich_stacker_v2')
    ap.add_argument('--mode', default='hard', choices=['hard', 'soft'])
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_all_ckpts()
    ckpt_stems = sorted(pool.keys())
    print(f'pool size: {len(ckpt_stems)} ckpts: {ckpt_stems}', flush=True)

    # Align to v1 lengths (n_val=50752, n_test=14208)
    n_val = 50752
    n_tst = 14208
    aligned = {}
    for s in ckpt_stems:
        Pv, Pt, yv, yt = pool[s]
        if Pv.shape[0] < n_val or Pt.shape[0] < n_tst:
            print(f'  SKIP {s}: shapes too small {Pv.shape}, {Pt.shape}', flush=True)
            continue
        aligned[s] = (Pv[:n_val], Pt[:n_tst], yv[:n_val], yt[:n_tst])
    ckpt_stems = sorted(aligned.keys())

    rv = list_files('val')
    rt = list_files('test')
    GV = groupby(rv, n_val)
    GT = groupby(rt, n_tst)
    print(f'val: {n_val} clips → {len(GV)} sources;  test: {n_tst} clips → {len(GT)} sources', flush=True)

    # Per-ckpt temperature calibration on val
    print('\n=== Phase 1: temperature calibration on val', flush=True)
    cal = {}
    temps = {}
    for s in ckpt_stems:
        Pv, Pt, yv, _ = aligned[s]
        T = fit_temperature(Pv, yv)
        temps[s] = T
        cal[s] = (apply_temperature(Pv, T), apply_temperature(Pt, T))
        print(f'  {s}: T={T:.3f}', flush=True)

    # Compact subsets: full pool always, plus pruned sub-sets that drop singletons
    # We'll try a fixed handful of strategic subsets (avoid combinatorial blowup):
    subsets = [
        tuple(ckpt_stems),                                          # all 7
        tuple(s for s in ckpt_stems if s.startswith('hydra-0')),    # all hydra
    ]
    # Also: top-3, top-4, top-5 by val accuracy (estimated from softmax probs)
    val_acc = {}
    for s in ckpt_stems:
        Pv, _, yv, _ = aligned[s]
        val_acc[s] = float((Pv.argmax(1) == yv).mean())
    sorted_by_acc = sorted(ckpt_stems, key=lambda x: -val_acc[x])
    for k in (3, 4, 5, 6):
        subsets.append(tuple(sorted_by_acc[:k]))
    # Deduplicate
    subsets = list({s: None for s in subsets}.keys())
    print(f'\nval_acc:', {s: f'{v:.4f}' for s, v in val_acc.items()}, flush=True)
    print(f'subsets to evaluate: {len(subsets)}', flush=True)

    hgb_cfgs = [
        (0.10, 3, 500, 1.0),
        (0.10, 4, 500, 1.0),
        (0.15, 4, 500, 1.0),
        (0.20, 4, 800, 1.0),
        (0.30, 4, 500, 2.0),
    ]
    feature_sets = [
        ('logmean',),
        ('logmean', 'logmax'),
        ('logmean', 'logmax', 'var'),
        ('logmean', 'logmax', 'log_q90'),
        ('logmean', 'logmax', 'var', 'entropy'),
    ]

    # ─── Phase 2: subset × cfg × feature search on val OOF ────────────────
    print('\n=== Phase 2: subset × feature × cfg search on val OOF', flush=True)
    t0 = time.time()
    best = (-1.0, None, None, None)   # (f1, sub, modes, cfg)
    n_tried = 0
    for modes in feature_sets:
        for sub in subsets:
            V = np.stack([cal[s][0] for s in sub])
            Xv, ysv = rich_features(V, GV, modes)
            for cfg in hgb_cfgs:
                n_tried += 1
                m, _ = run_oof_soft(Xv, ysv, cfg, boost_classes=(), boost_weights={})
                if m['f1'] > best[0]:
                    best = (m['f1'], sub, modes, cfg)
                    print(f'  [{n_tried}] elapsed={time.time()-t0:.1f}s  val_OOF_F1={m["f1"]:.4f}  '
                          f'macroP={m["macroP"]:.4f}  modes={modes} |sub|={len(sub)} cfg={cfg}', flush=True)
    f1_best, sub_best, modes_best, cfg_best = best
    print(f'\n=== Phase 2 done in {time.time()-t0:.1f}s; tried {n_tried}', flush=True)
    print(f'subset: {sub_best}\nmodes: {modes_best}\ncfg: {cfg_best}\nval_OOF F1={f1_best:.4f}', flush=True)

    # ─── Phase 3: pick booster weights on val OOF ─────────────────────────
    print(f'\n=== Phase 3: per-class booster blending ({args.mode}) on val OOF', flush=True)
    V_best = np.stack([cal[s][0] for s in sub_best])
    Xv_best, ysv_best = rich_features(V_best, GV, modes_best)

    if args.mode == 'soft':
        weight_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        boost_classes_list = [
            (),
            (TUG,),
            (TUG, CARGO),
            (TUG, TANKER),
            (TUG, CARGO, TANKER),
        ]
        best_b = (f1_best, (), {})
        t0 = time.time()
        n_tried = 0
        for bc in boost_classes_list:
            for ws in itertools.product(weight_grid, repeat=len(bc)):
                n_tried += 1
                bw = {c: ws[i] for i, c in enumerate(bc)}
                m, _ = run_oof_soft(Xv_best, ysv_best, cfg_best,
                                    boost_classes=bc, boost_weights=bw)
                if m['f1'] > best_b[0]:
                    best_b = (m['f1'], bc, dict(bw))
                    print(f'  [{n_tried}] elapsed={time.time()-t0:.1f}s  F1={m["f1"]:.4f} '
                          f'macroP={m["macroP"]:.4f} bc={bc} ws={bw}', flush=True)
        f1_b, bc_best, bw_best = best_b
        print(f'\n=== Phase 3 done in {time.time()-t0:.1f}s; tried {n_tried}', flush=True)
        print(f'boosters: {bc_best}  ws={bw_best}  val_OOF F1={f1_b:.4f}', flush=True)
    else:
        # Hard-override mode
        tau_grid = [0.55, 0.65, 0.75, 0.85, 0.95]
        boost_classes_list = [
            (),
            (TUG,),
            (TUG, CARGO),
            (TUG, TANKER),
            (TUG, CARGO, TANKER),
        ]
        best_b = (f1_best, (), {})
        t0 = time.time()
        n_tried = 0
        for bc in boost_classes_list:
            for taus in itertools.product(tau_grid, repeat=len(bc)):
                n_tried += 1
                bt = {c: taus[i] for i, c in enumerate(bc)}
                # Use hard override (set boost weight extremely high above τ via threshold)
                # Reuse simpler hard-override semantics:
                lr_, depth, iters, l2 = cfg_best
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
                P_oof = np.zeros((len(ysv_best), 4))
                boost = {c: np.zeros(len(ysv_best)) for c in bc}
                for tr, te in skf.split(np.zeros_like(ysv_best), ysv_best):
                    m = HistGradientBoostingClassifier(
                        max_iter=iters, learning_rate=lr_, max_depth=depth,
                        l2_regularization=l2, random_state=0).fit(Xv_best[tr], ysv_best[tr])
                    P_oof[te] = m.predict_proba(Xv_best[te])
                    for c in bc:
                        yc = (ysv_best[tr] == c).astype(int)
                        if yc.sum() == 0 or yc.sum() == len(yc):
                            continue
                        bn = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced').fit(Xv_best[tr], yc)
                        boost[c][te] = bn.predict_proba(Xv_best[te])[:, 1]
                pred = P_oof.argmax(1)
                for c in (TUG, TANKER, CARGO):
                    if c in bc:
                        pred = np.where(boost[c] > bt[c], c, pred)
                f1 = f1_score(ysv_best, pred, average='macro', zero_division=0)
                mp = precision_score(ysv_best, pred, average='macro', zero_division=0)
                if f1 > best_b[0]:
                    best_b = (f1, bc, dict(bt))
                    print(f'  [{n_tried}] elapsed={time.time()-t0:.1f}s  F1={f1:.4f} mP={mp:.4f} bc={bc} taus={bt}', flush=True)
        f1_b, bc_best, bw_best = best_b
        print(f'\n=== Phase 3 done in {time.time()-t0:.1f}s; tried {n_tried}', flush=True)
        print(f'boosters: {bc_best}  taus={bw_best}  val_OOF F1={f1_b:.4f}', flush=True)

    # ─── Phase 4: refit, evaluate test once ───────────────────────────────
    print('\n=== Phase 4: refit, evaluate test once', flush=True)
    T_best = np.stack([cal[s][1] for s in sub_best])
    Xt_best, yst_best = rich_features(T_best, GT, modes_best)
    lr_, depth, iters, l2 = cfg_best
    main_clf = HistGradientBoostingClassifier(
        max_iter=iters, learning_rate=lr_, max_depth=depth,
        l2_regularization=l2, random_state=0).fit(Xv_best, ysv_best)
    P_test = main_clf.predict_proba(Xt_best)
    boosters = {}
    for c in bc_best:
        boosters[c] = LogisticRegression(
            C=0.1, max_iter=5000, class_weight='balanced'
        ).fit(Xv_best, (ysv_best == c).astype(int))

    if args.mode == 'soft':
        P_final = P_test.copy()
        for c in bc_best:
            P_final[:, c] += bw_best[c] * boosters[c].predict_proba(Xt_best)[:, 1]
        P_final = P_final / P_final.sum(1, keepdims=True).clip(min=1e-8)
        pred_test = P_final.argmax(1)
    else:
        pred_test = P_test.argmax(1)
        for c in (TUG, TANKER, CARGO):
            if c in bc_best:
                p_c = boosters[c].predict_proba(Xt_best)[:, 1]
                pred_test = np.where(p_c > bw_best[c], c, pred_test)

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

    joblib.dump({
        'main_stacker':   main_clf,
        'boosters':       boosters,
        'boost_taus_or_ws': bw_best,
        'boost_classes':  bc_best,
        'temperatures':   temps,
        'ckpt_subset':    list(sub_best),
        'modes':          modes_best,
        'cfg_main':       cfg_best,
        'mode':           args.mode,
        'data_dir':       str(data_dir),
        'val_oof_f1':     float(f1_b),
        'test_f1':        float(test_f1),
        'test_macroP':    float(test_mp),
        'test_recall':    float(test_rec),
    }, out_dir / 'rich_stacker_v2.joblib')
    print(f'\nSaved: {out_dir}/rich_stacker_v2.joblib', flush=True)


if __name__ == '__main__':
    main()
