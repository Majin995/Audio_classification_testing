"""Zero-fit ensemble baseline — no learned stacker, pre-committed aggregation.

For each ckpt: log(mean over clips of softmax probs) per source.
Then arithmetic mean across ckpts. argmax → prediction. No fit, no selection.

This is the structurally simplest honest baseline. If it underperforms the
clean_redo subset baseline, it tells us the subset selection had real signal.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, matthews_corrcoef,
    confusion_matrix,
)


meta_v1 = json.load(open('campaign/probs_classifier_dataset/_meta.json'))
data_dir = Path(meta_v1['data_dir'])
classes = meta_v1['classes']
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
        print(f'   {c:<10s} P={p_:.3f} R={r_:.3f} F1={f_:.3f}')
    return dict(f1=float(f1), macroP=float(mp), recall=float(rc), mcc=float(mcc))


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

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

    rt = list_files('test')
    GT = groupby(rt, N_TST)
    print(f'test: {N_TST} clips → {len(GT)} sources')
    print(f'pool size: {len(pool)}')

    # Group by file ordering, build per-ckpt source-level matrix on test
    src_y = None
    src_per_ckpt = {}
    for stem, (Pv, Pt) in pool.items():
        Xs, y = src_logmean(Pt, GT)
        src_per_ckpt[stem] = Xs
        src_y = y

    # Try several pre-committed ensembles, evaluate test for each. No fit, but we
    # also report a clean "no peeking" reading: the metric is just descriptive.
    stems = sorted(pool.keys())
    print('\n--- single-ckpt test (descriptive only):')
    for s in stems:
        pred = src_per_ckpt[s].argmax(1)
        report(f'single-{s}', src_y, pred)

    # Honest fixed ensembles (no test peeking — selection by val_OOF p)
    val_p = {
        'hydra-010-p0.6943': 0.6943, 'hydra-012-p0.6929': 0.6929,
        'hydra-026-p0.6908': 0.6908, 'hydra-031-p0.7042': 0.7042,
        'hydra-032-p0.7010': 0.7010, 'hydra-042-p0.7114': 0.7114,
        'hydra-069-p0.7279': 0.7279,
    }
    ranked = sorted(stems, key=lambda s: -val_p[s])
    print(f'\nranked by val_p: {ranked}')

    # Pre-committed eval candidates (these decisions made on val_p alone)
    candidates = [
        ('top1',      ranked[:1]),
        ('top2',      ranked[:2]),
        ('top3',      ranked[:3]),
        ('top4',      ranked[:4]),
        ('top5',      ranked[:5]),
        ('top6',      ranked[:6]),
        ('all7',      ranked),
        ('baseline_subset', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042']),
    ]

    print('\n--- ensemble candidates (logmean across ckpts then argmax):')
    for name, subset in candidates:
        Xs = np.stack([src_per_ckpt[s] for s in subset]).mean(0)
        pred = Xs.argmax(1)
        report(f'ens-{name}', src_y, pred)


if __name__ == '__main__':
    main()
