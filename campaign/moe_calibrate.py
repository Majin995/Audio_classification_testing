"""MoE post-hoc calibration: isotonic per expert + α-tuned gate prior.

Loads the gate + 4 binary experts from a finished moe_sweep run, fits
isotonic regression on val mapping each expert's positive-sigmoid prob to
empirical P(positive | x). Then re-scores val/test using two combiners:

  1. moe_soft_iso     : argmax_c gate[c] * iso_c(expert_pos_c[c])
  2. moe_prior_alpha  : argmax_c log(gate[c]) + α * log(iso_c(...))   (α on val)

Reads existing checkpoints, no retraining. ≈1h wall on the existing run.

Usage
-----
$ python -m campaign.moe_calibrate \\
    --run_dir lightning_logs/moesweep_full_1s_1780240079
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from campaign.train_moe import (  # noqa: E402
    _accuracy,
    _build_backbone,
    _macro_f1,
    _make_loader,
    _predict_split,
    _rest_merge,
)


# ───────────────────────────────────────────────────────────────────────
# Args shim so train_moe._make_loader can be reused
# ───────────────────────────────────────────────────────────────────────

class _ArgsLike:
    def __init__(self, m: dict, run_args: dict):
        self.loader = run_args.get("loader", "dali")
        self.batch_size = int(run_args.get("batch_size", 128))
        self.target_sr = int(run_args.get("target_sr", 5120))
        self.fixed_len = int(run_args.get("fixed_len", 5120))
        self.no_oversample = False  # calibration must respect natural distribution
        self.num_threads = int(run_args.get("num_threads", 8))
        self.device_id = int(run_args.get("device_id", 0))
        self.denoise_method = run_args.get("denoise_method", "off")
        self.window_sec = run_args.get("window_sec", None)
        self.hop_sec = run_args.get("hop_sec", None)
        self.num_workers = int(run_args.get("num_workers", 8))
        self.rms_normalize = bool(run_args.get("rms_normalize", False))
        self.target_rms = float(run_args.get("target_rms", 0.1))
        self.backbone = m["backbone"]
        self.data_dir = m["data_dir"]


def _load_model(backbone: str, num_classes: int, depth: int, lr: float,
                margin: float, sr: int, fixed_len: int, ckpt_path: str):
    m = _build_backbone(
        backbone=backbone, num_classes=num_classes, class_weights=None,
        sample_rate=sr, input_len=fixed_len, depth=depth,
        max_epochs=20, learning_rate=lr, warmup_epochs=3,
        extra={"lmf_margin": float(margin)},
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(state["state_dict"], strict=False)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    ap.add_argument("--gate_depth", type=int, default=3,
                    help="Depth used to build the gate at retrain (must match training).")
    ap.add_argument("--gate_lr", type=float, default=3e-4)
    args = ap.parse_args()

    run = Path(args.run_dir)
    manifest = json.load(open(run / "sweep_manifest.json"))
    run_args = manifest.get("args", {})
    arg_shim = _ArgsLike(manifest, run_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gate_c2i = manifest["gate_class_to_idx"]
    K = len(gate_c2i)
    class_list = sorted(gate_c2i.keys())

    # ── Load + predict gate ────────────────────────────────────────────
    print("[calib] loading gate")
    gate = _load_model(manifest["backbone"], K, args.gate_depth, args.gate_lr,
                       0.5, arg_shim.target_sr, arg_shim.fixed_len,
                       manifest["gate_ckpt"])
    gate_dm = _make_loader(arg_shim, merge_classes=None)
    gate_dm.setup()
    gate_val,  y_val,  _ = _predict_split(gate, gate_dm, "val",  device)
    gate_test, y_test, _ = _predict_split(gate, gate_dm, "test", device)
    del gate; torch.cuda.empty_cache()
    n_val, n_test = len(y_val), len(y_test)
    print(f"[calib] gate done. val={n_val} test={n_test}")

    # ── Per-expert: predict val/test, fit isotonic on val ─────────────
    expert_val  = np.zeros((n_val,  K), dtype=np.float64)   # raw positive probs
    expert_test = np.zeros((n_test, K), dtype=np.float64)
    iso_val     = np.zeros_like(expert_val)                  # calibrated
    iso_test    = np.zeros_like(expert_test)

    for rec in manifest["experts"]:
        cls = rec["class"]
        gidx = int(rec["gate_idx"])
        pos = int(rec["expert_pos_idx"])
        cfg = rec["best_cfg"]
        print(f"[calib] expert {cls} (gate_idx={gidx})")

        m = _load_model(manifest["backbone"], 2, int(cfg["depth"]),
                        float(cfg["lr"]), float(cfg["lmf_margin"]),
                        arg_shim.target_sr, arg_shim.fixed_len, rec["ckpt"])
        dm = _make_loader(arg_shim, merge_classes=_rest_merge(cls, class_list))
        dm.setup()
        # Expert returns 2-class softmax; pos column is what we calibrate.
        pv, yv, _ = _predict_split(m, dm, "val",  device)
        pt, _,  _ = _predict_split(m, dm, "test", device)
        del m; torch.cuda.empty_cache()

        raw_v = pv[:n_val,  pos]
        raw_t = pt[:n_test, pos]
        y_bin = (yv[:n_val] == pos).astype(np.int64)

        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(raw_v, y_bin)
        expert_val[:,  gidx] = raw_v
        expert_test[:, gidx] = raw_t
        iso_val[:,  gidx] = iso.transform(raw_v)
        iso_test[:, gidx] = iso.transform(raw_t)
        print(f"  raw  mean={raw_v.mean():.3f} pos-rate={y_bin.mean():.3f}")
        print(f"  iso  mean={iso_val[:, gidx].mean():.3f}")

    # ── Combiners ──────────────────────────────────────────────────────
    eps = 1e-9
    def f1_of(pred, y):
        f1, per = _macro_f1(y, pred, K)
        return f1, _accuracy(y, pred), per

    # Gate-only baseline
    g_pred_v = gate_val.argmax(1)
    g_pred_t = gate_test.argmax(1)

    # raw product (old soft routing)
    raw_v_pred = (gate_val  * expert_val ).argmax(1)
    raw_t_pred = (gate_test * expert_test).argmax(1)

    # isotonic product (new soft routing)
    iso_v_pred = (gate_val  * iso_val ).argmax(1)
    iso_t_pred = (gate_test * iso_test).argmax(1)

    # α-prior on val
    alpha_results = []
    best_alpha, best_val_f1 = None, -1.0
    for a in args.alphas:
        log_pred_v = np.log(gate_val + eps) + a * np.log(iso_val + eps)
        log_pred_t = np.log(gate_test + eps) + a * np.log(iso_test + eps)
        f1_v, _, _ = f1_of(log_pred_v.argmax(1), y_val)
        f1_t, _, _ = f1_of(log_pred_t.argmax(1), y_test)
        alpha_results.append({"alpha": a, "val_f1": f1_v, "test_f1": f1_t})
        if f1_v > best_val_f1:
            best_val_f1, best_alpha = f1_v, a
    log_pred_v_best = (np.log(gate_val + eps) + best_alpha * np.log(iso_val + eps)).argmax(1)
    log_pred_t_best = (np.log(gate_test + eps) + best_alpha * np.log(iso_test + eps)).argmax(1)

    out = {
        "alpha_sweep": alpha_results,
        "best_alpha": best_alpha,
        "metrics": {
            "gate_only":   {"val": f1_of(g_pred_v, y_val), "test": f1_of(g_pred_t, y_test)},
            "moe_soft_raw":{"val": f1_of(raw_v_pred, y_val),"test": f1_of(raw_t_pred, y_test)},
            "moe_soft_iso":{"val": f1_of(iso_v_pred, y_val),"test": f1_of(iso_t_pred, y_test)},
            "moe_prior_a": {"val": f1_of(log_pred_v_best, y_val),
                            "test": f1_of(log_pred_t_best, y_test)},
        },
    }

    # Console summary
    print("\n=== MoE calibration results ===")
    print(f"best α on val = {best_alpha}")
    for k, v in out["metrics"].items():
        vf, va, _ = v["val"]; tf, ta, _ = v["test"]
        print(f"  {k:14s}  val macroF1={vf:.4f} acc={va:.4f}  |  test macroF1={tf:.4f} acc={ta:.4f}")

    # Persist
    out_path = run / "calibration_results.json"
    def serial(x):
        if isinstance(x, tuple):
            return [float(x[0]), float(x[1]), [float(v) for v in x[2]]]
        return x
    flat = {k: {sp: serial(v[sp]) for sp in ("val", "test")} for k, v in out["metrics"].items()}
    json.dump({"alpha_sweep": out["alpha_sweep"], "best_alpha": out["best_alpha"],
               "metrics": flat}, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
