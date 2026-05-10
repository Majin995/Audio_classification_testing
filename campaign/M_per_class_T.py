"""Phase M: per-class (vector) temperature scaling.

Replace single scalar T with one T_c per class, fit on val by minimizing
cross-entropy. Compare F1 to single-T on the geom-2 ensemble AND on the
stacker output.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from campaign.lib import (
    fit_temperature,
    fmt_metrics,
    load_dump,
    load_meta,
    metrics,
    metrics_with_thresholds,
    search_thresholds_f1_with_p_floor,
    stack_probs,
    temperature_scale,
)


def get_full(meta):
    full_dir = Path("campaign/probs_split1s_full")
    P, y = [], None
    for entry in meta["ckpts"]:
        d = load_dump(full_dir / f"{entry['stem']}.npz")
        P.append(d["test_probs"])
        y = d["test_y"]
    return np.stack(P), y


def fit_per_class_T(probs, y, num_classes, max_iter=400):
    """Per-class temperature: scale logits by 1/T_c per class then softmax."""
    eps = 1e-8
    logp = torch.tensor(np.log(probs + eps), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    log_T = nn.Parameter(torch.zeros(num_classes))
    optim = torch.optim.LBFGS([log_T], lr=1e-2, max_iter=max_iter)

    def closure():
        optim.zero_grad()
        T = log_T.exp().clamp(min=1e-2, max=100.0)
        scaled = logp / T[None, :]
        loss = F.cross_entropy(scaled, yt)
        loss.backward()
        return loss

    optim.step(closure)
    T = log_T.exp().clamp(min=1e-2, max=100.0).detach().numpy()
    return T


def apply_per_class_T(probs, T):
    eps = 1e-8
    logp = np.log(probs + eps) / T[None, :]
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def main():
    meta = load_meta("campaign/probs_rapid_1s")
    nc = meta["num_classes"]
    val_P, val_y, test_P, test_y, stems = stack_probs("campaign/probs_rapid_1s")
    full_P, full_y = get_full(meta)

    # geom-2 base
    best_idx = [i for i, s in enumerate(stems) if "026" in s or "031" in s]
    eps = 1e-8
    v = np.exp(np.log(val_P[best_idx] + eps).mean(axis=0))
    v /= v.sum(axis=1, keepdims=True)
    t = np.exp(np.log(test_P[best_idx] + eps).mean(axis=0))
    t /= t.sum(axis=1, keepdims=True)
    f = np.exp(np.log(full_P[best_idx] + eps).mean(axis=0))
    f /= f.sum(axis=1, keepdims=True)

    grid = list(np.linspace(0.0, 0.99, 50))
    lines = ["# M. Per-class (vector) temperature", ""]

    # Geom-2 base
    T_scalar = fit_temperature(v, val_y)
    T_vec = fit_per_class_T(v, val_y, nc)
    lines.append(f"## Geom-2 ensemble base")
    lines.append(f"- Scalar T = {T_scalar:.3f}")
    lines.append(f"- Vector T = {[f'{x:.3f}' for x in T_vec.tolist()]}")
    lines.append("")

    for label, T_obj, apply_fn in [("scalar", T_scalar, lambda x: temperature_scale(x, T_obj)),
                                     ("vector", T_vec, lambda x: apply_per_class_T(x, T_vec))]:
        vT = apply_fn(v) if label == "scalar" else apply_per_class_T(v, T_vec)
        tT = apply_fn(t) if label == "scalar" else apply_per_class_T(t, T_vec)
        fT = apply_fn(f) if label == "scalar" else apply_per_class_T(f, T_vec)
        # raw + thr
        m_raw = metrics(fT, full_y, nc)
        res = search_thresholds_f1_with_p_floor(vT, val_y, nc, p_floor=0.65, grid=grid)
        thr = np.array(res["thresholds"])
        m_cal = metrics_with_thresholds(fT, full_y, thr, nc)
        lines.append(f"### {label.title()} T")
        lines.append(f"- thresholds = {[round(x,2) for x in thr.tolist()]}")
        lines.append(f"- raw full F1 = {m_raw['f1']:.4f}, raw full macroP = {m_raw['macro_P']:.4f}")
        lines.append(f"- cal full F1 = {m_cal['f1']:.4f}, cal full macroP = {m_cal['macro_P']:.4f}, cov = {m_cal['coverage']:.3f}")
        lines.append("")

    # Stacker base
    Xv = np.log(val_P + eps).transpose(1, 0, 2).reshape(len(val_y), -1)
    Xt = np.log(test_P + eps).transpose(1, 0, 2).reshape(len(test_y), -1)
    Xf = np.log(full_P + eps).transpose(1, 0, 2).reshape(len(full_y), -1)
    clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
    clf.fit(Xv, val_y)
    sv = clf.predict_proba(Xv)
    st = clf.predict_proba(Xt)
    sf = clf.predict_proba(Xf)

    T_scalar_s = fit_temperature(sv, val_y)
    T_vec_s = fit_per_class_T(sv, val_y, nc)
    lines.append(f"## Stacker base")
    lines.append(f"- Scalar T = {T_scalar_s:.3f}")
    lines.append(f"- Vector T = {[f'{x:.3f}' for x in T_vec_s.tolist()]}")
    lines.append("")
    for label, T_obj, apply_fn in [("scalar", T_scalar_s, lambda x: temperature_scale(x, T_obj)),
                                     ("vector", T_vec_s, lambda x: apply_per_class_T(x, T_vec_s))]:
        vT = apply_fn(sv) if label == "scalar" else apply_per_class_T(sv, T_vec_s)
        tT = apply_fn(st) if label == "scalar" else apply_per_class_T(st, T_vec_s)
        fT = apply_fn(sf) if label == "scalar" else apply_per_class_T(sf, T_vec_s)
        m_raw = metrics(fT, full_y, nc)
        res = search_thresholds_f1_with_p_floor(vT, val_y, nc, p_floor=0.65, grid=grid)
        thr = np.array(res["thresholds"])
        m_cal = metrics_with_thresholds(fT, full_y, thr, nc)
        lines.append(f"### Stacker + {label} T")
        lines.append(f"- thresholds = {[round(x,2) for x in thr.tolist()]}")
        lines.append(f"- raw full F1 = {m_raw['f1']:.4f}, raw full macroP = {m_raw['macro_P']:.4f}")
        lines.append(f"- cal full F1 = {m_cal['f1']:.4f}, cal full macroP = {m_cal['macro_P']:.4f}, cov = {m_cal['coverage']:.3f}")
        lines.append("```")
        lines.append(fmt_metrics(m_cal, nc, "  "))
        lines.append("```")
        lines.append("")

    Path("campaign/M_per_class_T.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
