"""Cargo-confirmation ensemble — threshold picked on val, test touched once.

Architectural rule:
  - default prediction = arg-max of baseline_subset log-mean
  - if precise_cargo_prob > τ AND Cargo is in baseline's top-2 → override to Cargo

Single threshold τ chosen by maximizing val macro-F1 on the source-level metric.
"""
import json
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, matthews_corrcoef,
    confusion_matrix,
)

meta = json.load(open('campaign/probs_classifier_dataset/_meta.json'))
data_dir = Path(meta['data_dir'])
classes = meta['classes']
CARGO = classes.index('Cargo')
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


def src_logmean(P, G, eps=1e-8):
    out = np.zeros((len(G), P.shape[-1]), dtype=np.float32)
    y = np.zeros(len(G), dtype=np.int64)
    for gi, (idx, gy, _) in enumerate(G):
        out[gi] = np.log(P[idx].mean(0) + eps)
        y[gi] = gy
    return out, y


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


def cargo_confirm(base_lm, precise_lm, tau):
    base_pred = base_lm.argmax(1)
    precise_prob = np.exp(precise_lm)
    precise_prob /= precise_prob.sum(1, keepdims=True)
    p_cargo_precise = precise_prob[:, CARGO]
    base_top2 = np.argsort(-base_lm, axis=1)[:, :2]
    base_cargo_in_top2 = (base_top2[:, 0] == CARGO) | (base_top2[:, 1] == CARGO)
    mask = (p_cargo_precise > tau) & base_cargo_in_top2
    pred = np.where(mask, CARGO, base_pred)
    return pred


def main():
    # Load probs (test)
    pool = {}
    for ck in meta['ckpts']:
        z = np.load(ck['npz'])
        pool[ck['stem']] = (z['val_probs'][:N_VAL], z['test_probs'][:N_TST])
    for ck in json.load(open('campaign/probs_classifier_dataset_v2/_meta.json'))['ckpts']:
        z = np.load(ck['npz'])
        pool[ck['stem']] = (z['val_probs'][:N_VAL], z['test_probs'][:N_TST])
    for fn in sorted(Path('campaign/probs_classifier_dataset_v3').glob('*.npz')):
        z = np.load(fn)
        pool[fn.stem] = (z['val_probs'][:N_VAL], z['test_probs'][:N_TST])
    z = np.load('campaign/probs_classifier_dataset_precise/precise-013-aligned.npz')
    pool['precise-013-p0.7444'] = (z['val_probs'], z['test_probs'])

    rv = list_files('val')
    rt = list_files('test')
    GV = groupby(rv, N_VAL)
    GT = groupby(rt, N_TST)
    print(f'val: {N_VAL} clips → {len(GV)} sources;  test: {N_TST} clips → {len(GT)} sources')

    baseline = ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042']
    val_base = np.stack([src_logmean(pool[s][0], GV)[0] for s in baseline]).mean(0)
    test_base = np.stack([src_logmean(pool[s][1], GT)[0] for s in baseline]).mean(0)
    val_y = src_logmean(pool[baseline[0]][0], GV)[1]
    test_y = src_logmean(pool[baseline[0]][1], GT)[1]
    val_prec = src_logmean(pool['precise-013-p0.7444'][0], GV)[0]
    test_prec = src_logmean(pool['precise-013-p0.7444'][1], GT)[0]

    # Pre-committed grid
    tau_grid = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    print('\n=== Phase 1: val selection of τ (no test peek) ===')
    best = (-1.0, None)
    for tau in tau_grid:
        pred = cargo_confirm(val_base, val_prec, tau)
        m = metrics(val_y, pred)
        print(f'  tau={tau:.2f}  val F1={m["f1"]:.4f}  mP={m["macroP"]:.4f}  R={m["recall"]:.4f}')
        if m['f1'] > best[0]:
            best = (m['f1'], tau)
    f1_best, tau_best = best
    print(f'\nWINNER τ={tau_best}  val F1={f1_best:.4f}')

    # Phase 2: evaluate on test once
    print('\n=== Phase 2: evaluate on test once ===')
    pred_test = cargo_confirm(test_base, test_prec, tau_best)
    report(f'TEST (τ={tau_best})', test_y, pred_test)

    # Also report no-overlap baseline for comparison
    print('\n=== Reference: baseline_subset alone (no precise) ===')
    report('baseline_subset', test_y, test_base.argmax(1))


if __name__ == '__main__':
    main()
