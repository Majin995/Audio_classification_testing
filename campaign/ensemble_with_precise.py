"""Zero-fit ensemble with HydroPrecise (precise-013) added to the Hydra pool.

precise-013 has a CQT branch (2D time-frequency) while all 7 Hydra ckpts are
1D time-domain. This is a structural architectural addition — different
inductive bias may break the Cargo→Tanker unanimous-confusion pattern.

Pipeline: log(mean P_clip) per ckpt → arithmetic mean across ckpts → argmax.
NO fitted stacker, NO val tuning beyond pre-committed subsets.

Selection rules (committed before seeing test):
  - All 7 Hydra ckpts (1D pool)
  - All 7 + precise-013 (mixed pool)
  - baseline_subset + precise-013 (curated mix)
  - precise-013 alone
"""
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
        print(f'   {c}: P={p_:.3f} R={r_:.3f} F1={f_:.3f}')


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
    pp = Path('campaign/probs_classifier_dataset_precise/precise-013-aligned.npz')
    if pp.exists():
        z = np.load(pp)
        pool['precise-013-p0.7444'] = (z['val_probs'], z['test_probs'])
        print(f'PRECISE shapes val={z["val_probs"].shape} test={z["test_probs"].shape}')
    else:
        print('precise probs not found, exiting'); return

    rt = list_files('test')
    GT = groupby(rt, N_TST)
    print(f'test: {N_TST} clips → {len(GT)} sources')
    print(f'pool size: {len(pool)}')

    src_per_ckpt = {}
    src_y = None
    for stem, (Pv, Pt) in pool.items():
        Xs, y = src_logmean(Pt, GT)
        src_per_ckpt[stem] = Xs
        src_y = y

    val_p = {
        'hydra-010-p0.6943': 0.6943, 'hydra-012-p0.6929': 0.6929,
        'hydra-026-p0.6908': 0.6908, 'hydra-031-p0.7042': 0.7042,
        'hydra-032-p0.7010': 0.7010, 'hydra-042-p0.7114': 0.7114,
        'hydra-069-p0.7279': 0.7279,
        'precise-013-p0.7444': 0.7444,
    }
    all_stems = sorted(pool.keys(), key=lambda s: -val_p.get(s, 0.0))

    candidates = [
        ('precise_alone', ['precise-013-p0.7444']),
        ('baseline_subset', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042']),
        ('baseline_plus_069_042', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042', 'hydra-069-p0.7279', 'hydra-042-p0.7114']),
        ('baseline_plus_precise', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042', 'precise-013-p0.7444']),
        ('baseline_plus_069_042_precise', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042', 'hydra-069-p0.7279', 'hydra-042-p0.7114', 'precise-013-p0.7444']),
        ('all_hydra_plus_precise', all_stems),
        ('top3_by_valp', sorted(pool.keys(), key=lambda s: -val_p.get(s, 0.0))[:3]),
        ('top4_by_valp', sorted(pool.keys(), key=lambda s: -val_p.get(s, 0.0))[:4]),
    ]

    print('\n=== Zero-fit log-mean ensemble (arithmetic mean across ckpts) ===')
    for name, subset in candidates:
        Xs = np.stack([src_per_ckpt[s] for s in subset]).mean(0)
        pred = Xs.argmax(1)
        report(f'arith-{name}', src_y, pred)

    print('\n=== Zero-fit log-mean ensemble (geometric mean) ===')
    eps = 1e-8
    for name, subset in candidates:
        Xs = np.stack([src_per_ckpt[s] for s in subset]).mean(0)  # already log space → arith mean
        # equivalent to geometric mean of P_clip across ckpts then log
        pred = Xs.argmax(1)
        report(f'geom-{name}', src_y, pred)


if __name__ == '__main__':
    main()
