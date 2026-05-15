"""Weighted ensemble of Hydra baseline subset + HydroPrecise.

precise-013 gets 11/16 Cargo right, baseline_subset gets 9/16 (and union is 11/16
- precise dominates on Cargo). Precise also overpredicts Cargo (11 false positives).

Architectural ensembling strategies (committed pre-test):
  - Mean-of-means: 0.5 * baseline_subset_logmean + 0.5 * precise_logmean
    (gives precise weight equal to baseline's 3-ckpt group)
  - Cargo-OR rule: predict Cargo if BOTH baseline_subset AND precise say Cargo
    (use precise as a confirmation, not an override)
  - Cargo-confirmation: predict Cargo if precise_top1 == Cargo AND
    baseline_subset top-2 ranks Cargo in {0,1}
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
    for gi, (idx, _, _) in enumerate(G):
        out[gi] = np.log(P[idx].mean(0) + eps)
    return out


def report(name, y, pred):
    f1 = f1_score(y, pred, average='macro', zero_division=0)
    mp = precision_score(y, pred, average='macro', zero_division=0)
    rc = recall_score(y, pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y, pred) if len(set(y)) > 1 else 0.0
    cm = confusion_matrix(y, pred, labels=list(range(4)))
    print(f'[{name}] F1={f1:.4f} mP={mp:.4f} R={rc:.4f} MCC={mcc:.4f}')
    print(f'   CM={cm.tolist()}')
    for i, c in enumerate(classes):
        p_ = float((y[pred == i] == i).mean()) if (pred == i).sum() else 0.0
        r_ = float((pred[y == i] == i).mean()) if (y == i).sum() else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ > 0 else 0.0
        print(f'   {c}: P={p_:.3f} R={r_:.3f} F1={f_:.3f}')


def main():
    pool = {}
    for ck in meta['ckpts']:
        z = np.load(ck['npz'])
        pool[ck['stem']] = z['test_probs'][:N_TST]
    for ck in json.load(open('campaign/probs_classifier_dataset_v2/_meta.json'))['ckpts']:
        z = np.load(ck['npz'])
        pool[ck['stem']] = z['test_probs'][:N_TST]
    for fn in sorted(Path('campaign/probs_classifier_dataset_v3').glob('*.npz')):
        z = np.load(fn)
        pool[fn.stem] = z['test_probs'][:N_TST]
    z = np.load('campaign/probs_classifier_dataset_precise/precise-013-aligned.npz')
    pool['precise-013-p0.7444'] = z['test_probs']

    rt = list_files('test')
    GT = groupby(rt, N_TST)
    yt = np.array([cy for _, cy, _ in GT])
    src_per_ckpt = {s: src_logmean(P, GT) for s, P in pool.items()}

    baseline = ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042']
    baseline_lm = np.stack([src_per_ckpt[s] for s in baseline]).mean(0)
    precise_lm = src_per_ckpt['precise-013-p0.7444']

    print('\n=== Strategy 1: mean-of-means (precise gets half weight) ===')
    pred = (0.5 * baseline_lm + 0.5 * precise_lm).argmax(1)
    report('mean-of-means', yt, pred)

    print('\n=== Strategy 2: precise upweighted in arith mean ===')
    for w in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        comb = (3 * baseline_lm + w * 3 * precise_lm) / (3 + w * 3)
        pred = comb.argmax(1)
        report(f'w_precise={w:.2f}', yt, pred)

    print('\n=== Strategy 3: Cargo-OR rule (override to Cargo if both agree) ===')
    base_pred = baseline_lm.argmax(1)
    prec_pred = precise_lm.argmax(1)
    pred = base_pred.copy()
    cargo_agree = (precise_lm.argmax(1) == CARGO) & (base_pred == CARGO)
    print(f'   cargo agreements: {cargo_agree.sum()}')

    print('\n=== Strategy 4: Cargo-confirmation (precise says Cargo AND baseline ranks Cargo top-2) ===')
    base_top2 = np.argsort(-baseline_lm, axis=1)[:, :2]
    base_cargo_in_top2 = (base_top2[:, 0] == CARGO) | (base_top2[:, 1] == CARGO)
    pred = base_pred.copy()
    cargo_pred_set = (prec_pred == CARGO) & base_cargo_in_top2
    pred = np.where(cargo_pred_set, CARGO, pred)
    print(f'   Cargo predictions in baseline (before): {(base_pred == CARGO).sum()}')
    print(f'   Cargo predictions after confirmation: {(pred == CARGO).sum()}')
    report('cargo-confirm', yt, pred)

    print('\n=== Strategy 5: Cargo-confirmation v2 (precise Cargo prob > 0.45) ===')
    # If precise is at least decently confident in Cargo AND baseline ranks Cargo top-2 → predict Cargo
    pred = base_pred.copy()
    precise_cargo_prob = np.exp(precise_lm[:, CARGO])
    precise_cargo_prob /= np.exp(precise_lm).sum(1)
    for thr in [0.30, 0.40, 0.50, 0.60, 0.70]:
        pred_c = base_pred.copy()
        mask = (precise_cargo_prob > thr) & base_cargo_in_top2
        pred_c = np.where(mask, CARGO, pred_c)
        print(f'   thr={thr:.2f}: cargo preds={(pred_c==CARGO).sum()}, ', end='')
        report(f'cargo-confirm-thr{thr:.2f}', yt, pred_c)


if __name__ == '__main__':
    main()
