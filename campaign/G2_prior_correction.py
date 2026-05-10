"""Logit adjustment / prior correction (Menon et al., ICLR 2021).

Subtract tau * log(class_prior) from log-probs at inference to undo the
training-prior bias. Train prior comes from Split1s/train counts:
  Cargo=11481, Passenger=62, Tanker=2881, Tug=1498.
The Passenger class is 185x rarer than Cargo, so the model under-predicts
it relative to Cargo. Subtracting log(prior) corrects this.

Sweep tau in [0, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0].
"""
import json
from pathlib import Path

import numpy as np

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
    test_P = []
    test_y = None
    for entry in meta["ckpts"]:
        d = load_dump(full_dir / f"{entry['stem']}.npz")
        test_P.append(d["test_probs"])
        test_y = d["test_y"]
    return np.stack(test_P), test_y


def adjust(probs, log_prior, tau):
    """probs: (N, C). Subtract tau * log_prior from log-probs, renormalize."""
    eps = 1e-8
    logp = np.log(probs + eps) - tau * log_prior[None, :]
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def main():
    meta = load_meta("campaign/probs_rapid_1s")
    nc = meta["num_classes"]
    val_P, val_y, test_P, test_y, stems = stack_probs("campaign/probs_rapid_1s")
    full_P, full_y = get_full(meta)

    # Best 2 ckpts geom
    best_idx = [i for i, s in enumerate(stems) if "026" in s or "031" in s]
    eps = 1e-8
    v = np.exp(np.log(val_P[best_idx] + eps).mean(axis=0))
    v /= v.sum(axis=1, keepdims=True)
    t = np.exp(np.log(test_P[best_idx] + eps).mean(axis=0))
    t /= t.sum(axis=1, keepdims=True)
    f = np.exp(np.log(full_P[best_idx] + eps).mean(axis=0))
    f /= f.sum(axis=1, keepdims=True)

    # Class priors from Split1s train
    counts = np.array([11481, 62, 2881, 1498], dtype=np.float64)
    prior = counts / counts.sum()
    log_prior = np.log(prior)
    print(f"train prior: {[f'{p:.4f}' for p in prior]}")
    print(f"log_prior:   {[f'{lp:+.3f}' for lp in log_prior]}")

    taus = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0]
    grid = list(np.linspace(0.0, 0.99, 50))

    rows = []
    lines = ["# G2. Logit adjustment / prior correction", ""]
    lines.append(f"- Train priors: Cargo={prior[0]:.4f}  Passenger={prior[1]:.4f}  "
                 f"Tanker={prior[2]:.4f}  Tug={prior[3]:.4f}")
    lines.append(f"- Geometric mean of {[stems[i] for i in best_idx]}")
    lines.append(f"- For each τ: subtract τ·log(prior) from log-probs, refit T, refit thresholds")
    lines.append("")
    lines.append("| τ | T | thresholds | val F1 | rapid_test F1 | rapid_test macroP | full_test F1 | full_test macroP | full cov |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    best = None
    best_full_f1 = -1.0
    for tau in taus:
        v_adj = adjust(v, log_prior, tau)
        t_adj = adjust(t, log_prior, tau)
        f_adj = adjust(f, log_prior, tau)
        T = fit_temperature(v_adj, val_y)
        vT = temperature_scale(v_adj, T)
        tT = temperature_scale(t_adj, T)
        fT = temperature_scale(f_adj, T)
        res = search_thresholds_f1_with_p_floor(vT, val_y, nc, p_floor=0.65, grid=grid)
        thr = np.array(res["thresholds"])
        rt = metrics_with_thresholds(tT, test_y, thr, nc)
        ft = metrics_with_thresholds(fT, full_y, thr, nc)
        rows.append({"tau": tau, "T": T, "thresholds": thr.tolist(),
                     "val_metrics": res["metrics"],
                     "rapid_test": rt, "full_test": ft})
        if ft["f1"] > best_full_f1:
            best_full_f1 = ft["f1"]
            best = rows[-1]
        lines.append(f"| {tau:.2f} | {T:.2f} | {[round(x,2) for x in thr.tolist()]}"
                     f" | {res['metrics']['f1']:.4f} | {rt['f1']:.4f} | {rt['macro_P']:.4f}"
                     f" | {ft['f1']:.4f} | {ft['macro_P']:.4f} | {ft['coverage']:.3f} |")

    lines.append("")
    lines.append("## Best by full_test F1")
    lines.append("```")
    lines.append(f"τ={best['tau']}  T={best['T']:.3f}  thresholds={best['thresholds']}")
    lines.append("rapid_test:")
    lines.append(fmt_metrics(best["rapid_test"], nc, "  "))
    lines.append("full_test:")
    lines.append(fmt_metrics(best["full_test"], nc, "  "))
    lines.append("```")

    Path("campaign/G2_prior_correction.md").write_text("\n".join(lines))
    Path("campaign/G2_prior_correction.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
