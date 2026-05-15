"""Rich per-source stacker — architectural improvements only.

Pipeline (every selection decision made on val 5-fold OOF):
  1. Per-checkpoint temperature calibration on val OOF (NLL).
  2. Rich per-source aggregation features per ckpt:
       a) log(mean P_clip)        (existing baseline)
       b) log(max P_clip)         (peak-confidence)
       c) var(P_clip)             (within-source dispersion)
       d) max - min over clips    (range)
       e) mean per-clip entropy   (uncertainty)
  3. Main HGB stacker on (M_ckpt * 4 * 4) + (M_ckpt * 1) = 84-dim features.
  4. Per-class boosters (LR, class_weight='balanced') for {Cargo, Tanker, Tug}
     each with τ_c chosen on val OOF to maximise macro F1 (no test peek).
  5. Final refit on full val → evaluate test exactly once.

Selection budget is small and grid points are committed up-front.
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

meta = json.load(open('campaign/probs_classifier_dataset/_meta.json'))
data_dir = Path(meta['data_dir'])
classes = meta['classes']
NUM_CLASSES = 4
TUG = classes.index('Tug')
CARGO = classes.index('Cargo')
TANKER = classes.index('Tanker')
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


def fit_temperature(probs_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fit a single temperature T on val NLL: P' = softmax(log(P)/T).
    Returns best T from a 1-D grid (avoids gradient deps).
    """
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


def rich_aggregate(P_stack: np.ndarray, G: list, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """P_stack: (M_ckpt, N_clip, C). G: list of (clip_idx, label, source).
    Per source, per ckpt, per class compute:
      - log(mean P)
      - log(max P)
      - var P
      - max - min P
    Per source, per ckpt (scalar): mean per-clip entropy.
    Returns (X: (N_src, M*(4*C+1)), y: (N_src,))."""
    M, _, C = P_stack.shape
    feat_per_ckpt = 4 * C + 1
    X = np.zeros((len(G), M * feat_per_ckpt), dtype=np.float32)
    y = np.zeros(len(G), dtype=np.int64)
    for gi, (idx, gy, _) in enumerate(G):
        for mi in range(M):
            ch = P_stack[mi, idx]  # (k, C)
            mean_p = ch.mean(axis=0)
            max_p = ch.max(axis=0)
            min_p = ch.min(axis=0)
            var_p = ch.var(axis=0)
            ent_p = -(ch * np.log(np.clip(ch, eps, 1.0))).sum(axis=1).mean()
            base = mi * feat_per_ckpt
            X[gi, base + 0 * C:base + 1 * C] = np.log(mean_p + eps)
            X[gi, base + 1 * C:base + 2 * C] = np.log(max_p + eps)
            X[gi, base + 2 * C:base + 3 * C] = var_p
            X[gi, base + 3 * C:base + 4 * C] = max_p - min_p
            X[gi, base + 4 * C] = ent_p
        y[gi] = gy
    return X, y


def run_oof(X: np.ndarray, y: np.ndarray, cfg: tuple,
            boost_classes: tuple, boost_taus: dict,
            k_folds: int = 5, seed: int = 0) -> tuple[dict, np.ndarray]:
    """5-fold OOF using main HGB + per-class LR boosters with hard override.
    Returns (metrics_dict, oof_predictions)."""
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
            # If a class has zero positives in this fold, skip
            if yc.sum() == 0 or yc.sum() == len(yc):
                boost_oof[c][te] = 0.0
                continue
            b = LogisticRegression(
                C=0.1, max_iter=5000, class_weight='balanced'
            ).fit(X[tr], yc)
            boost_oof[c][te] = b.predict_proba(X[te])[:, 1]
    pred = P_oof.argmax(1)
    # Apply boosters in priority order: Tug > Tanker > Cargo (override → stronger)
    priority = [TUG, TANKER, CARGO]
    for c in priority:
        if c in boost_classes and c in boost_taus:
            pred = np.where(boost_oof[c] > boost_taus[c], c, pred)
    return dict(
        f1=f1_score(y, pred, average='macro', zero_division=0),
        macroP=precision_score(y, pred, average='macro', zero_division=0),
        recall=recall_score(y, pred, average='macro', zero_division=0),
        mcc=matthews_corrcoef(y, pred) if len(set(y.tolist())) > 1 else 0.0,
    ), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='lightning_logs/rich_stacker_v1')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_probs_raw = {}
    for ck in meta['ckpts']:
        z = np.load(ck['npz'])
        all_probs_raw[ck['stem']] = (z['val_probs'], z['test_probs'], z['val_y'], z['test_y'])
    ckpt_stems = [ck['stem'] for ck in meta['ckpts']]
    rv = list_files('val')
    rt = list_files('test')
    n_val = all_probs_raw[ckpt_stems[0]][0].shape[0]
    n_tst = all_probs_raw[ckpt_stems[0]][1].shape[0]
    GV = groupby(rv, n_val)
    GT = groupby(rt, n_tst)
    print(f'val: {n_val} clips → {len(GV)} sources;  test: {n_tst} clips → {len(GT)} sources', flush=True)

    # ── Phase 1: per-ckpt temperature calibration on val OOF ──────────────
    # Fit T once per ckpt using val clips (this is allowed: val is selection set)
    print('\n=== Phase 1: per-ckpt temperature calibration on val', flush=True)
    cal_probs = {}
    temps = {}
    for s in ckpt_stems:
        Pv, Pt, yv, _ = all_probs_raw[s]
        T = fit_temperature(Pv, yv)
        temps[s] = T
        cal_probs[s] = (apply_temperature(Pv, T), apply_temperature(Pt, T))
        print(f'  {s}: T={T:.3f}', flush=True)

    # All non-empty subsets of size ≥ 2
    subsets = []
    for r in range(2, len(ckpt_stems) + 1):
        for combo in itertools.combinations(range(len(ckpt_stems)), r):
            subsets.append(tuple(ckpt_stems[i] for i in combo))
    print(f'subsets to consider: {len(subsets)}', flush=True)

    hgb_cfgs = [
        (0.10, 3, 500, 1.0),
        (0.10, 4, 500, 1.0),
        (0.15, 4, 500, 1.0),
        (0.20, 3, 1000, 1.0),
        (0.20, 4, 800, 1.0),
        (0.30, 4, 500, 2.0),
    ]
    boost_classes_candidates = [
        (TUG,),                  # baseline
        (TUG, CARGO),            # add Cargo booster
        (TUG, TANKER),           # add Tanker booster
        (TUG, CARGO, TANKER),    # all three
    ]
    tau_grid = [0.55, 0.65, 0.75, 0.85, 0.95]

    print(f'hgb cfgs: {len(hgb_cfgs)} | booster sets: {len(boost_classes_candidates)} | tau grid: {tau_grid}', flush=True)

    # ─── Phase 2: pick subset+cfg on val OOF (single Tug booster τ=0.95) ──
    print('\n=== Phase 2: select subset + HGB cfg on val OOF (Tug τ=0.95)', flush=True)
    t0 = time.time()
    best = (-1.0, None, None)
    n_tried = 0
    for sub in subsets:
        V = np.stack([cal_probs[s][0] for s in sub])
        Xv, ysv = rich_aggregate(V, GV)
        for cfg in hgb_cfgs:
            n_tried += 1
            m, _ = run_oof(Xv, ysv, cfg,
                           boost_classes=(TUG,), boost_taus={TUG: 0.95})
            if m['f1'] > best[0]:
                best = (m['f1'], sub, cfg)
                print(f'  [{n_tried}] elapsed={time.time()-t0:.1f}s  val_OOF_F1={m["f1"]:.4f}  '
                      f'macroP={m["macroP"]:.4f}  sub={len(sub)}-ckpt  cfg={cfg}', flush=True)
    f1_best, sub_best, cfg_best = best
    print(f'\n=== Phase 2 done in {time.time()-t0:.1f}s; tried {n_tried}', flush=True)
    print(f'subset: {sub_best}', flush=True)
    print(f'cfg:    {cfg_best}', flush=True)
    print(f'val_OOF F1={f1_best:.4f}', flush=True)

    # ─── Phase 3: pick booster set + per-class τ on val OOF ───────────────
    print('\n=== Phase 3: select boosters + per-class τ on val OOF', flush=True)
    V_best = np.stack([cal_probs[s][0] for s in sub_best])
    Xv_best, ysv_best = rich_aggregate(V_best, GV)

    best_boost = (-1.0, None, None)
    t0 = time.time()
    n_tried = 0
    for bc in boost_classes_candidates:
        # Search τ per class in bc independently is large; we use a small structured grid.
        # For each candidate booster set, try the cross product of τ's restricted to a coarse grid.
        # To keep total runtime manageable, use the same τ grid for all members.
        tau_combos = list(itertools.product(tau_grid, repeat=len(bc)))
        for taus in tau_combos:
            n_tried += 1
            boost_taus = {c: taus[i] for i, c in enumerate(bc)}
            m, _ = run_oof(Xv_best, ysv_best, cfg_best,
                           boost_classes=bc, boost_taus=boost_taus)
            if m['f1'] > best_boost[0]:
                best_boost = (m['f1'], bc, dict(boost_taus))
                print(f'  [{n_tried}] elapsed={time.time()-t0:.1f}s  val_OOF F1={m["f1"]:.4f}  '
                      f'macroP={m["macroP"]:.4f}  bc={bc}  taus={boost_taus}', flush=True)
    f1_b, bc_best, taus_best = best_boost
    print(f'\n=== Phase 3 done in {time.time()-t0:.1f}s; tried {n_tried}', flush=True)
    print(f'boosters: {bc_best}  τs={taus_best}  val_OOF F1={f1_b:.4f}', flush=True)

    # ─── Phase 4: refit on full val, evaluate test once ───────────────────
    print('\n=== Phase 4: refit on full val, evaluate test once', flush=True)
    T_best = np.stack([cal_probs[s][1] for s in sub_best])
    Xt_best, yst_best = rich_aggregate(T_best, GT)

    lr_, depth, iters, l2 = cfg_best
    main_clf = HistGradientBoostingClassifier(
        max_iter=iters, learning_rate=lr_, max_depth=depth,
        l2_regularization=l2, random_state=0).fit(Xv_best, ysv_best)
    boosters = {}
    for c in bc_best:
        boosters[c] = LogisticRegression(
            C=0.1, max_iter=5000, class_weight='balanced'
        ).fit(Xv_best, (ysv_best == c).astype(int))

    P_test = main_clf.predict_proba(Xt_best)
    pred_test = P_test.argmax(1)
    priority = [TUG, TANKER, CARGO]
    for c in priority:
        if c in bc_best:
            p_c = boosters[c].predict_proba(Xt_best)[:, 1]
            pred_test = np.where(p_c > taus_best[c], c, pred_test)

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
        'boosters':       boosters,
        'boost_taus':     taus_best,
        'boost_classes':  bc_best,
        'temperatures':   temps,
        'ckpt_subset':    list(sub_best),
        'all_ckpts':      ckpt_stems,
        'cfg_main':       cfg_best,
        'feature_layout': '(M_ckpt * (4*C + 1)) per source: '
                          '[log_mean, log_max, var, max-min] × C, then ent_mean',
        'data_dir':       str(data_dir),
        'selection':      'val 5-fold OOF only; test touched once',
        'val_oof_f1':     float(f1_b),
        'test_f1':        float(test_f1),
        'test_macroP':    float(test_mp),
        'test_recall':    float(test_rec),
    }, out_dir / 'rich_stacker_v1.joblib')
    print(f'\nSaved: {out_dir}/rich_stacker_v1.joblib', flush=True)


if __name__ == '__main__':
    main()
