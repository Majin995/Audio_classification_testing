"""MoE with 6-ckpt cargo_confirm ensemble probs as gate.

Replaces the trained HydroComplete gate (~0.51 test macroF1) with the
log-mean of the 6-ckpt cargo_confirm pool (project SOTA 0.715 on this
dataset) cached in campaign/probs_combined_recurrent_stacker.npz.

Then multiplies by calibrated multi-mode expert probs from an existing
moe_strong run.

Usage
-----
$ python -m campaign.moe_stacker_gate --run_dir lightning_logs/moe_strong_1s_<ts>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from campaign.train_moe import (  # noqa: E402
    _accuracy, _build_backbone, _macro_f1, _make_loader, _predict_split,
)


class _Shim:
    def __init__(self, m):
        a = m.get("args", {})
        self.loader = a.get("loader", "dali")
        self.batch_size = 128
        self.target_sr = int(a.get("target_sr", 5120))
        self.fixed_len = int(a.get("fixed_len", 5120))
        self.no_oversample = True
        self.num_threads = 8; self.device_id = 0
        self.denoise_method = "off"
        self.window_sec = None; self.hop_sec = None
        self.num_workers = 8
        self.rms_normalize = False; self.target_rms = 0.1
        self.backbone = m["backbone"]
        self.data_dir = m.get("args", {}).get("data_dir") or m.get("data_dir")


def _load(backbone, K, depth, lr, ckpt):
    m = _build_backbone(backbone=backbone, num_classes=K, class_weights=None,
                        sample_rate=5120, input_len=5120, depth=depth,
                        max_epochs=20, learning_rate=lr, warmup_epochs=3)
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(s["state_dict"], strict=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--probs_npz",
                    default="campaign/probs_combined_recurrent_stacker.npz")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = Path(args.run_dir)
    mani = json.load(open(run / "moe_manifest.json"))
    sh = _Shim(mani)
    gate_c2i = mani["gate_class_to_idx"]
    K = len(gate_c2i)

    # Load cached ensemble probs and build a path → row index.
    print("[stk] loading ensemble npz")
    npz = np.load(args.probs_npz, allow_pickle=True)
    all_paths = np.array([str(p) for p in npz["paths"]])
    all_probs = npz["probs"]  # (6, N, 4)
    ens_classes = list(npz["classes"])
    # map ensemble class index → gate class index
    ens_to_gate = np.array([gate_c2i[c] for c in ens_classes], dtype=np.int64)
    path_idx = {p: i for i, p in enumerate(all_paths)}
    log_ens = np.log(np.clip(all_probs, 1e-9, 1.0))
    gate_logp = log_ens.mean(axis=0)  # (N, 4) log-mean ensemble
    # reorder gate cols into gate index space
    inv = np.empty(K, dtype=np.int64)
    inv[ens_to_gate] = np.arange(K)
    gate_logp = gate_logp[:, inv]
    gate_prob_all = np.exp(gate_logp)
    gate_prob_all /= gate_prob_all.sum(axis=1, keepdims=True)

    # ── Inference for experts on val/test (in dm order) ───────────────
    # We need the file paths per row so we can grab matching ensemble probs.
    # _predict_split returns probs only — refit to also return paths via
    # the loader's internal file list.
    expert_probs = {sp: np.zeros((0, K), dtype=np.float64) for sp in ("val", "test")}
    paths_per_split = {sp: None for sp in ("val", "test")}
    y_per_split = {sp: None for sp in ("val", "test")}

    # We will fetch paths from gate_dm._splits or fallback to expert_dm
    gate_dm = _make_loader(sh, merge_classes=None)
    gate_dm.setup()
    # DALI loader exposes train_files_override or _splits in extended; for DALI
    # build paths by walking data_dir.
    for sp in ("val", "test"):
        # Get path list from disk in same alphabetical-class order as DALI uses.
        cap = {"val": "Val", "test": "Test"}[sp]
        d = Path(sh.data_dir) / cap
        paths = []
        labels = []
        cls_sorted = sorted([c.name for c in d.iterdir() if c.is_dir()])
        c2i_disk = {c: i for i, c in enumerate(cls_sorted)}
        for c in cls_sorted:
            for f in sorted((d / c).rglob("*")):
                if f.suffix.lower() in {".wav", ".mp3", ".flac"}:
                    paths.append(str(f))
                    labels.append(gate_c2i[c])
        paths_per_split[sp] = np.array(paths)
        y_per_split[sp] = np.array(labels)
        print(f"[stk] {sp}: {len(paths)} files")

    # For each expert, run inference and stash class probabilities
    expert_cols = {sp: np.zeros((len(paths_per_split[sp]), K), dtype=np.float64)
                   for sp in ("val", "test")}
    for rec in mani["experts"]:
        cls = rec["class"]; gidx = int(rec["gate_idx"])
        pos = int(rec["expert_pos_idx"])
        # Multi-mode expert: full K-class softmax in same gate index space
        m = _load(mani["backbone"], K, mani["depth"]["expert"],
                  float(mani["args"]["learning_rate"]), rec["ckpt"])
        for sp in ("val", "test"):
            pv, yv, _ = _predict_split(m, gate_dm, sp, device)
            n = min(len(pv), len(paths_per_split[sp]))
            expert_cols[sp][:n, gidx] = pv[:n, pos]
        del m; torch.cuda.empty_cache()
        print(f"[stk] expert {cls} done")

    # ── Map ensemble probs by path → gate ─────────────────────────────
    stk_gate = {}
    root = str(sh.data_dir).rstrip("/") + "/"
    for sp in ("val", "test"):
        rel_paths = [p[len(root):] if p.startswith(root) else p
                     for p in paths_per_split[sp]]
        idx_list, keep_mask = [], []
        for p in rel_paths:
            i = path_idx.get(p, -1)
            keep_mask.append(i >= 0)
            if i >= 0:
                idx_list.append(i)
        rows = np.array(idx_list, dtype=np.int64)
        mask = np.array(keep_mask)
        if mask.sum() != len(paths_per_split[sp]):
            print(f"[stk] {sp}: matched {mask.sum()}/{len(paths_per_split[sp])}")
        stk_gate[sp] = gate_prob_all[rows]
        y_per_split[sp] = y_per_split[sp][mask]
        expert_cols[sp] = expert_cols[sp][mask]

    # ── Isotonic-calibrate expert columns on val ──────────────────────
    iso_cols = {sp: expert_cols[sp].copy() for sp in ("val", "test")}
    for gidx in range(K):
        y_bin = (y_per_split["val"] == gidx).astype(np.int64)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(expert_cols["val"][:, gidx], y_bin)
        iso_cols["val"][:, gidx]  = iso.transform(expert_cols["val"][:, gidx])
        iso_cols["test"][:, gidx] = iso.transform(expert_cols["test"][:, gidx])

    eps = 1e-9
    def score(gate_p, exp_p, a):
        return (np.log(gate_p + eps) + a * np.log(exp_p + eps)).argmax(1)

    # gate_only via ensemble
    g_val_pred  = stk_gate["val" ].argmax(1)
    g_test_pred = stk_gate["test"].argmax(1)
    gf1_v, gf1_v_per = _macro_f1(y_per_split["val"], g_val_pred, K)
    gf1_t, gf1_t_per = _macro_f1(y_per_split["test"], g_test_pred, K)

    rows = []
    for a in args.alphas:
        pv = score(stk_gate["val"], iso_cols["val"], a)
        pt = score(stk_gate["test"], iso_cols["test"], a)
        f1v, perv = _macro_f1(y_per_split["val"], pv, K)
        f1t, pert = _macro_f1(y_per_split["test"], pt, K)
        rows.append({"alpha": a, "val_f1": f1v, "test_f1": f1t,
                     "val_per": [float(x) for x in perv],
                     "test_per": [float(x) for x in pert]})

    best = max(rows, key=lambda r: r["val_f1"])
    out = {
        "ensemble_gate_only": {"val_f1": gf1_v, "test_f1": gf1_t,
                               "val_per": [float(x) for x in gf1_v_per],
                               "test_per": [float(x) for x in gf1_t_per]},
        "alpha_sweep": rows,
        "best_alpha": best["alpha"],
        "best_test_f1": best["test_f1"],
    }
    print("\n=== Stacker-gate MoE ===")
    print(f"ens gate only       val={gf1_v:.4f}  test={gf1_t:.4f}")
    for r in rows:
        print(f"  α={r['alpha']:.2f}  val={r['val_f1']:.4f}  test={r['test_f1']:.4f}")
    print(f"best α={best['alpha']} → test macroF1={best['test_f1']:.4f}")

    json.dump(out, open(run / "stacker_gate_results.json", "w"), indent=2)
    print(f"\nwrote {run/'stacker_gate_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
