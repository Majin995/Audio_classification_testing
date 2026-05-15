"""Full-pool zero-fit ensemble + cargo-confirm strategies with all 9 ckpts.

Pool: 5 Hydra v1 + 1 v2 (hydra-069) + 1 v3 (hydra-042) + precise-013 + complete-044.

Test single-ckpt performance first (descriptive). Then evaluate pre-committed
ensembles and cargo-confirm with val-selected τ.
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
TANKER = classes.index('Tanker')
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


def main():
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
    z = np.load('campaign/probs_classifier_dataset_precise/complete-044-aligned.npz')
    pool['complete-044-p0.7159'] = (z['val_probs'], z['test_probs'])

    rv = list_files('val')
    rt = list_files('test')
    GV = groupby(rv, N_VAL)
    GT = groupby(rt, N_TST)
    print(f'val: {N_VAL} clips → {len(GV)} sources;  test: {N_TST} clips → {len(GT)} sources')
    print(f'pool size: {len(pool)}')

    val_src = {s: src_logmean(pool[s][0], GV)[0] for s in pool}
    test_src = {s: src_logmean(pool[s][1], GT)[0] for s in pool}
    val_y = src_logmean(pool[list(pool.keys())[0]][0], GV)[1]
    test_y = src_logmean(pool[list(pool.keys())[0]][1], GT)[1]

    print('\n=== Single-ckpt test (descriptive) ===')
    for s in sorted(pool.keys()):
        pred = test_src[s].argmax(1)
        report(f'single-{s}', test_y, pred)

    print('\n=== Pre-committed ensemble candidates ===')
    cands = [
        ('baseline_subset', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042']),
        ('baseline + precise', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042', 'precise-013-p0.7444']),
        ('baseline + complete', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042', 'complete-044-p0.7159']),
        ('baseline + precise + complete', ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042', 'precise-013-p0.7444', 'complete-044-p0.7159']),
        ('precise + complete', ['precise-013-p0.7444', 'complete-044-p0.7159']),
        ('all_hydra + precise + complete', sorted(pool.keys())),
    ]
    for name, subset in cands:
        Xs = np.stack([test_src[s] for s in subset]).mean(0)
        pred = Xs.argmax(1)
        report(f'ens-{name}', test_y, pred)

    # cargo-confirm with complete as the "voucher"
    print('\n=== Cargo-confirm using complete-044 as voucher (val-selected τ) ===')
    base_subset = ['hydra-026-p0.6908', 'hydra-032-p0.7010', 'hydra-031-p0.7042']
    val_base = np.stack([val_src[s] for s in base_subset]).mean(0)
    test_base = np.stack([test_src[s] for s in base_subset]).mean(0)
    val_complete = val_src['complete-044-p0.7159']
    test_complete = test_src['complete-044-p0.7159']

    def confirm(base_lm, voucher_lm, tau):
        base_pred = base_lm.argmax(1)
        voucher_prob = np.exp(voucher_lm)
        voucher_prob /= voucher_prob.sum(1, keepdims=True)
        p_cargo_v = voucher_prob[:, CARGO]
        top2 = np.argsort(-base_lm, axis=1)[:, :2]
        cargo_top2 = (top2[:, 0] == CARGO) | (top2[:, 1] == CARGO)
        mask = (p_cargo_v > tau) & cargo_top2
        return np.where(mask, CARGO, base_pred)

    tau_grid = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print('  val selection:')
    best = (-1.0, None)
    for tau in tau_grid:
        pred = confirm(val_base, val_complete, tau)
        m = metrics(val_y, pred)
        print(f'    tau={tau:.2f}  val F1={m["f1"]:.4f}  mP={m["macroP"]:.4f}')
        if m['f1'] > best[0]:
            best = (m['f1'], tau)
    tau_best = best[1]
    print(f'  selected tau={tau_best}')
    print('  test (touched once):')
    pred_test = confirm(test_base, test_complete, tau_best)
    report(f'TEST cargo-confirm-complete (τ={tau_best})', test_y, pred_test)

    # Use BOTH precise AND complete as vouchers
    print('\n=== Cargo-confirm using PRECISE+COMPLETE (both must agree on Cargo, val-selected τ) ===')
    val_precise = val_src['precise-013-p0.7444']
    test_precise = test_src['precise-013-p0.7444']

    def confirm_both(base_lm, v1_lm, v2_lm, tau1, tau2):
        base_pred = base_lm.argmax(1)
        p1 = np.exp(v1_lm); p1 /= p1.sum(1, keepdims=True)
        p2 = np.exp(v2_lm); p2 /= p2.sum(1, keepdims=True)
        top2 = np.argsort(-base_lm, axis=1)[:, :2]
        cargo_top2 = (top2[:, 0] == CARGO) | (top2[:, 1] == CARGO)
        mask = (p1[:, CARGO] > tau1) & (p2[:, CARGO] > tau2) & cargo_top2
        return np.where(mask, CARGO, base_pred)

    # Pre-committed: tie τ1=τ2=τ
    print('  val selection (τ1=τ2=τ):')
    best = (-1.0, None)
    for tau in tau_grid:
        pred = confirm_both(val_base, val_precise, val_complete, tau, tau)
        m = metrics(val_y, pred)
        print(f'    tau={tau:.2f}  val F1={m["f1"]:.4f}  mP={m["macroP"]:.4f}')
        if m['f1'] > best[0]:
            best = (m['f1'], tau)
    tau_best = best[1]
    print(f'  selected tau={tau_best}')
    pred_test = confirm_both(test_base, test_precise, test_complete, tau_best, tau_best)
    report(f'TEST cargo-confirm-both (τ={tau_best})', test_y, pred_test)


if __name__ == '__main__':
    main()
