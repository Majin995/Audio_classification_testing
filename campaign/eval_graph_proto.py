"""Honest test eval of HydroGraphProto on Combined IARA+Deepship.

Mirrors eval_recurrent_stacker.py: full-test metrics + val-tuned
MSP-isotonic selective gate (Hendrycks-Gimpel baseline) + per-source-type
breakdown + cargo_confirm reference baselines.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

from models.hydro_graph_proto import HydroGraphProto
from campaign.train_recurrent import scan_split, read_batch_threaded, preprocess, macro_prf1, CLASSES
from campaign.train_recurrent_stacker import EnsembleCache
from campaign.train_graph_proto import eval_split


def metric_block(y, pred, K=4):
    P, R, F1, mp, mr, mf, cm = macro_prf1(y, pred, K)
    return {"macro_f1": float(mf), "macro_p": float(mp), "macro_r": float(mr),
            "per_class_p": P.tolist(), "per_class_r": R.tolist(),
            "per_class_f1": F1.tolist(), "cm": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/var/mnt/5A009BF8009BD8F9/Data/Combined_IARA_Deepship_1s")
    ap.add_argument("--ens_npz", default="campaign/probs_combined_recurrent_stacker.npz")
    ap.add_argument("--ckpt", default="lightning_logs/hydro_graph_proto_combined/best.pt")
    ap.add_argument("--out_dir", default="lightning_logs/hydro_graph_proto_combined")
    ap.add_argument("--K_eval", type=int, default=60)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["args"]
    tcn_dils = tuple(int(d) for d in cfg["tcn_dilations"].split(","))
    cache = EnsembleCache(Path(args.ens_npz), Path(args.data_dir))

    model = HydroGraphProto(
        num_classes=4, n_bands=cfg["n_bands"], embed_dim=cfg["embed_dim"],
        tcn_dilations=tcn_dils,
        ens_n_ckpts=cache.M, ens_n_classes=cache.C,
        gnn_dim=cfg["gnn_dim"], gnn_depth=cfg["gnn_depth"],
        gnn_heads=cfg["gnn_heads"], kNN=cfg["kNN"],
        temporal_ring=cfg["temporal_ring"],
        perceiver_queries=cfg["perceiver_queries"],
        perceiver_depth=cfg["perceiver_depth"],
        n_prototypes_per_class=cfg["proto_per_class"],
        proto_temp=cfg["proto_temp"], proto_dim=cfg["proto_dim"],
        dropout=cfg["dropout"], band_dropout=cfg["band_dropout"],
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    print(f"Loaded ckpt step={ck.get('step')} val_F1={ck.get('val_macro_f1'):.4f}")
    print(f"Params = {model.n_params():,}")

    executor = ThreadPoolExecutor(max_workers=12)

    # ---- TEST ----
    test_src = scan_split(args.data_dir, "Test")
    yt, pred_t, fp_t, fa_t, st_t = eval_split(
        model, test_src, executor, device, cache, args.K_eval)
    print(f"\nTest sources: {len(yt)}  (iara={int((st_t=='iara').sum())}, "
          f"deepship={int((st_t=='deepship').sum())})")

    m_full = metric_block(yt, pred_t)
    print(f"\n[FULL — no abstain]  macroF1={m_full['macro_f1']:.4f} "
          f"macroP={m_full['macro_p']:.4f} macroR={m_full['macro_r']:.4f}")
    print("  CM:")
    for i, c in enumerate(CLASSES):
        row = m_full["cm"][i]
        f1 = m_full["per_class_f1"][i]
        p_ = m_full["per_class_p"][i]; r_ = m_full["per_class_r"][i]
        print(f"    {c:>10s}  " + " ".join(f"{int(v):>6d}" for v in row)
              + f"   P={p_:.3f} R={r_:.3f} F1={f1:.3f}")
    print(f"  mean abstain prob: {fa_t.mean():.3f}")

    # ---- VAL MSP-isotonic calibration ----
    val_src = scan_split(args.data_dir, "Val")
    yv, pred_v, fp_v, fa_v, _ = eval_split(
        model, val_src, executor, device, cache, args.K_eval)
    correct_v = (pred_v == yv).astype(np.int32)
    msp_v = fp_v.max(axis=1)
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(msp_v, correct_v)
    p_correct_v = cal.predict(msp_v)
    grid = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]
    print("\nVal MSP-calibrated sweep:")
    best_cov, best_f1 = 1.0, -1.0; val_sweep = []
    for ct in grid:
        if ct >= 1.0:
            keep = np.ones_like(yv, dtype=bool)
        else:
            cut = np.quantile(p_correct_v, 1.0 - ct)
            keep = p_correct_v >= cut
        if keep.sum() == 0: continue
        m = metric_block(yv[keep], pred_v[keep])
        cov = float(keep.mean())
        val_sweep.append({"cov_target": ct, "cov": cov, **m})
        mark = ""
        if cov >= 0.85 and m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]; best_cov = ct; mark = "  ← cand"
        print(f"  cov_target={ct:.2f}  actual={cov:.3f}  macroF1={m['macro_f1']:.4f}{mark}")

    sel_cut = -np.inf if best_cov >= 1.0 else float(np.quantile(p_correct_v, 1.0 - best_cov))
    print(f"Val-selected coverage_target={best_cov:.2f}  cut={sel_cut:.4f}  val_F1={best_f1:.4f}")

    msp_t = fp_t.max(axis=1)
    p_correct_t = cal.predict(msp_t)
    keep_t = np.ones_like(yt, dtype=bool) if not np.isfinite(sel_cut) else p_correct_t >= sel_cut
    m_sel = metric_block(yt[keep_t], pred_t[keep_t]) if keep_t.sum() else None
    if m_sel:
        print(f"\n[SELECTIVE — val gate]  cov={keep_t.mean():.3f}  "
              f"macroF1={m_sel['macro_f1']:.4f}  macroP={m_sel['macro_p']:.4f}")
        for i, c in enumerate(CLASSES):
            n_kept = int(((yt == i) & keep_t).sum())
            print(f"   {c:>10s}  n_kept={n_kept:>3d}  "
                  f"P={m_sel['per_class_p'][i]:.3f} "
                  f"R={m_sel['per_class_r'][i]:.3f} "
                  f"F1={m_sel['per_class_f1'][i]:.3f}")

    print("\nPer-source-type:")
    breakdown = {}
    for s in ("iara", "deepship"):
        mask = st_t == s
        if mask.sum():
            m_full_s = metric_block(yt[mask], pred_t[mask])
            mask_kept = mask & keep_t
            m_sel_s = metric_block(yt[mask_kept], pred_t[mask_kept]) if mask_kept.sum() else None
            cov_s = float(mask_kept.sum() / max(mask.sum(), 1))
            print(f"  {s:>10s}  n={mask.sum():>3d}  full_F1={m_full_s['macro_f1']:.4f}  "
                  f"sel_F1={m_sel_s['macro_f1'] if m_sel_s else 0:.4f} (cov={cov_s:.2f})")
            breakdown[s] = {"full": m_full_s, "selective": m_sel_s, "cov": cov_s}

    np.savez(Path(args.out_dir) / "test_probs.npz",
             y_true=yt, pred=pred_t, final_probs=fp_t, final_abstain=fa_t,
             source_type=st_t)
    results = {
        "ckpt": args.ckpt, "val_macro_f1": ck.get("val_macro_f1"),
        "test_full": m_full,
        "val_sweep": val_sweep,
        "val_selected": {"cov_target": best_cov,
                         "cut": float(sel_cut) if np.isfinite(sel_cut) else None,
                         "val_f1": best_f1},
        "test_selective": m_sel,
        "per_source_type": breakdown,
        "test_mean_abstain": float(fa_t.mean()),
    }
    (Path(args.out_dir) / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {Path(args.out_dir)/'test_metrics.json'}")


if __name__ == "__main__":
    main()
