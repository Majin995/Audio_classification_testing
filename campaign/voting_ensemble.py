"""Hard-voting and weighted-voting ensemble across all 8 ckpts.

Each ckpt votes its argmax per source. Majority wins; tie-break by sum of
log-mean probs. NO tuning knobs.

Variants:
  - Equal vote (all 8)
  - Restricted to baseline_subset + precise
  - Weighted vote: precise counts as 2 votes
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
CLIP_RE = re.compile(r'_(\d{6})\.wav$', re.IGNORECASE)
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
        print(f'   {c}: P={p_:.3f} R={r_:.3f} F1={f_:.3f}')


def voting(per_ckpt_argmax: dict, per_ckpt_lm: dict, subset: list, weights: dict = None):
    n = list(per_ckpt_argmax.values())[0].shape[0]
    votes = np.zeros((n, 4), dtype=np.float32)
    sum_lm = np.zeros((n, 4), dtype=np.float32)
    for s in subset:
        w = weights.get(s, 1.0) if weights else 1.0
        am = per_ckpt_argmax[s]
        for i in range(n):
            votes[i, am[i]] += w
        sum_lm += per_ckpt_lm[s] * w
    # Find max-vote count per row; tie-break by sum_lm
    pred = np.zeros(n, dtype=np.int64)
    for i in range(n):
        max_v = votes[i].max()
        cands = np.where(votes[i] == max_v)[0]
        if len(cands) == 1:
            pred[i] = cands[0]
        else:
            pred[i] = cands[np.argmax(sum_lm[i, cands])]
    return pred


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

    per_ckpt_lm = {s: src_logmean(P, GT)[0] for s, P in pool.items()}
    per_ckpt_argmax = {s: lm.argmax(1) for s, lm in per_ckpt_lm.items()}

    baseline = ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042']
    bp = baseline + ['precise-013-p0.7444']
    bp_069 = baseline + ['hydra-069-p0.7279', 'precise-013-p0.7444']
    bp_069_042 = baseline + ['hydra-069-p0.7279', 'hydra-042-p0.7114', 'precise-013-p0.7444']
    all8 = list(pool.keys())

    print('=== Hard voting (equal weight) ===')
    for name, sub in [('baseline_subset', baseline),
                      ('baseline+precise', bp),
                      ('baseline+069+precise', bp_069),
                      ('baseline+069+042+precise', bp_069_042),
                      ('all8', all8)]:
        pred = voting(per_ckpt_argmax, per_ckpt_lm, sub)
        report(name, yt, pred)

    print('\n=== Hard voting (precise weight = 2) ===')
    for name, sub in [('baseline+precise', bp),
                      ('baseline+069+precise', bp_069),
                      ('baseline+069+042+precise', bp_069_042),
                      ('all8+precise2x', all8)]:
        w = {'precise-013-p0.7444': 2.0}
        pred = voting(per_ckpt_argmax, per_ckpt_lm, sub, w)
        report(name + ' (precise×2)', yt, pred)

    print('\n=== Hard voting (precise weight = 3) ===')
    for name, sub in [('baseline+precise', bp),
                      ('baseline+069+042+precise', bp_069_042)]:
        w = {'precise-013-p0.7444': 3.0}
        pred = voting(per_ckpt_argmax, per_ckpt_lm, sub, w)
        report(name + ' (precise×3)', yt, pred)


if __name__ == '__main__':
    main()
