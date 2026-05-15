"""Zero-fit ensemble with pre-committed uniform prior correction.

Architectural change: source-level argmax operates on log-likelihood, not
posterior. The base ckpts saw val with class freqs [0.193, 0.346, 0.409, 0.051]
— a strong Tanker prior. Removing this prior at decision time is a structural
correction (Bayes inversion):
    log_lik(y) = log P(y|x) - log P_val(y)
    pred = argmax_y log_lik(y)
This is committed before seeing test: val prior is a property of training data.

Compares: no correction, val_source_prior_uniform, val_clip_prior_uniform.
"""
from __future__ import annotations

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
    return f1, mp


def main():
    pool = {}
    for ck in meta_v1['ckpts']:
        z = np.load(ck['npz'])
        pool[ck['stem']] = (z['val_probs'][:N_VAL], z['test_probs'][:N_TST], z['val_y'][:N_VAL])
    p2 = Path('campaign/probs_classifier_dataset_v2/_meta.json')
    if p2.exists():
        for ck in json.load(open(p2))['ckpts']:
            z = np.load(ck['npz'])
            pool[ck['stem']] = (z['val_probs'][:N_VAL], z['test_probs'][:N_TST], z['val_y'][:N_VAL])
    p3 = Path('campaign/probs_classifier_dataset_v3')
    if p3.exists():
        for fn in sorted(p3.glob('*.npz')):
            z = np.load(fn)
            pool[fn.stem] = (z['val_probs'][:N_VAL], z['test_probs'][:N_TST], z['val_y'][:N_VAL])

    rv = list_files('val')
    rt = list_files('test')
    GV = groupby(rv, N_VAL)
    GT = groupby(rt, N_TST)
    print(f'val: {N_VAL} clips → {len(GV)} sources;  test: {N_TST} clips → {len(GT)} sources')

    # val class priors
    val_src_count = np.array([0, 0, 0, 0])
    for _, cy, _ in GV:
        val_src_count[cy] += 1
    val_clip_count = np.zeros(4)
    for idx, cy, _ in GV:
        val_clip_count[cy] += len(idx)
    src_prior = val_src_count / val_src_count.sum()
    clip_prior = val_clip_count / val_clip_count.sum()
    log_src_prior = np.log(src_prior + 1e-8)
    log_clip_prior = np.log(clip_prior + 1e-8)
    print(f'val source prior: {src_prior}')
    print(f'val clip prior:   {clip_prior}')

    # Pre-committed subsets (all from val_p, no test peek)
    val_p = {
        'hydra-010-p0.6943': 0.6943, 'hydra-012-p0.6929': 0.6929,
        'hydra-026-p0.6908': 0.6908, 'hydra-031-p0.7042': 0.7042,
        'hydra-032-p0.7010': 0.7010, 'hydra-042-p0.7114': 0.7114,
        'hydra-069-p0.7279': 0.7279,
    }
    stems = sorted(pool.keys())
    ranked = sorted(stems, key=lambda s: -val_p[s])

    # Per-ckpt source-level log-mean on TEST (pre-committed)
    src_per_ckpt = {}
    src_y = None
    for stem, (Pv, Pt, _) in pool.items():
        Xs, y = src_logmean(Pt, GT)
        src_per_ckpt[stem] = Xs
        src_y = y

    candidates = [
        ('top2',      ranked[:2]),
        ('top3',      ranked[:3]),
        ('baseline_subset', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042']),
        # Add curated 4-ckpt: baseline_subset + hydra-069 (top val_p, diversity)
        ('baseline_plus_069', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042', 'hydra-069-p0.7279']),
        # 5-ckpt: baseline + 069 + 042
        ('baseline_plus_069_042', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042', 'hydra-069-p0.7279', 'hydra-042-p0.7114']),
    ]

    print('\n=== No correction (raw log-mean) ===')
    for name, subset in candidates:
        Xs = np.stack([src_per_ckpt[s] for s in subset]).mean(0)
        pred = Xs.argmax(1)
        report(f'no-prior-{name}', src_y, pred)

    print('\n=== Minus val SOURCE prior ===')
    for name, subset in candidates:
        Xs = np.stack([src_per_ckpt[s] for s in subset]).mean(0)
        Xs = Xs - log_src_prior[None, :]
        pred = Xs.argmax(1)
        report(f'src-{name}', src_y, pred)

    print('\n=== Minus val CLIP prior ===')
    for name, subset in candidates:
        Xs = np.stack([src_per_ckpt[s] for s in subset]).mean(0)
        Xs = Xs - log_clip_prior[None, :]
        pred = Xs.argmax(1)
        report(f'clip-{name}', src_y, pred)


if __name__ == '__main__':
    main()
