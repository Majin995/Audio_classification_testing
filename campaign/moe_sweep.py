"""MoE grid-search orchestrator — tune each binary expert, then assemble.

For each user-supplied class, sweeps a grid of (depth, learning_rate,
lmf_margin) at short epochs and selects the cell with the best val
positive-class F1 (binary expert). Then retrains each winning expert at full
epochs, trains a gate, and evaluates the combined MoE on val/test.

Reads the train_moe helpers directly for backbone construction and training,
so the orchestration stays a thin script rather than another monolith.

Example
-------
$ python -m campaign.moe_sweep \\
    --data_dir /var/mnt/5A009BF8009BD8F9/Data/Combined_IARA_Deepship_1s \\
    --classes Cargo Passenger Tanker Tug \\
    --depths 1 2 3 --lrs 1e-4 3e-4 1e-3 --margins 0.3 0.5 0.7 \\
    --sweep_epochs 5 --final_epochs 25 --backbone complete
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from campaign.train_moe import (  # noqa: E402
    REST_LABEL,
    _accuracy,
    _build_backbone,
    _combine,
    _macro_f1,
    _make_loader,
    _predict_split,
    _rest_merge,
    _resolve_devices,
    _train_one,
)
from data.loader_factory import LOADER_CHOICES  # noqa: E402


# ───────────────────────────────────────────────────────────────────────
# Free CUDA memory between trials (Lightning leaves Trainer state behind)
# ───────────────────────────────────────────────────────────────────────

def _release(model) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ───────────────────────────────────────────────────────────────────────
# Train a single binary expert and return its val positive-class F1.
# ───────────────────────────────────────────────────────────────────────

def _train_binary_expert(
    *,
    cls_name: str,
    cell_tag: str,
    out_root: Path,
    args,
    depth: int,
    lr: float,
    lmf_margin: float,
    epochs: int,
    class_list: List[str],
    device: torch.device,
) -> Tuple[float, str, Dict[str, int]]:
    """Return (val_pos_F1, ckpt_path, expert_class_to_idx)."""
    dm = _make_loader(args, merge_classes=_rest_merge(cls_name, class_list))
    dm.setup()
    ec2i = dict(dm.class_to_idx)
    pos_idx = ec2i[cls_name]

    model = _build_backbone(
        backbone=args.backbone,
        num_classes=2,
        class_weights=None,
        sample_rate=args.target_sr,
        input_len=args.fixed_len,
        depth=depth,
        max_epochs=epochs,
        learning_rate=lr,
        warmup_epochs=min(args.warmup_epochs, max(1, epochs // 4)),
        extra={"lmf_margin": float(lmf_margin)},
    )

    name = f"sweep_{cls_name}/{cell_tag}"
    model, ckpt = _train_one(
        name=name,
        out_root=out_root,
        model=model,
        dm=dm,
        max_epochs=epochs,
        monitor_metric="val/f1",
        monitor_mode="max",
        precision=args.precision,
        accelerator=args.accelerator,
        devices=_resolve_devices(args.devices),
        grad_clip=args.grad_clip,
        log_every=args.log_every,
    )

    # Compute val positive-class F1 directly so the sweep score is the binary
    # signal we actually care about (not the 2-class macro which mixes Other).
    probs, labels, _ = _predict_split(model, dm, "val", device)
    pred = probs.argmax(axis=1)
    y_pos = (labels == pos_idx).astype(np.int64)
    p_pos = (pred == pos_idx).astype(np.int64)
    tp = int(((p_pos == 1) & (y_pos == 1)).sum())
    fp = int(((p_pos == 1) & (y_pos == 0)).sum())
    fn = int(((p_pos == 0) & (y_pos == 1)).sum())
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0

    _release(model)
    return float(f1), ckpt, ec2i


# ───────────────────────────────────────────────────────────────────────
# Args
# ───────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="moe_sweep",
        description="Per-expert grid search → final MoE assembly + eval.",
    )
    p.add_argument("--data_dir", required=True)
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--loader", choices=LOADER_CHOICES, default="dali")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--target_sr", type=int, default=5120)
    p.add_argument("--fixed_len", type=int, default=5120)
    p.add_argument("--no_oversample", action="store_true")
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--denoise_method", default="off")
    p.add_argument("--window_sec", type=float, default=None)
    p.add_argument("--hop_sec", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--rms_normalize", action="store_true")
    p.add_argument("--target_rms", type=float, default=0.1)
    # Backbone
    p.add_argument("--backbone", choices=("complete", "hydra"), default="complete")
    # Grid
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--margins", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    # Schedules
    p.add_argument("--sweep_epochs", type=int, default=5)
    p.add_argument("--final_epochs", type=int, default=25)
    p.add_argument("--gate_epochs", type=int, default=25)
    p.add_argument("--warmup_epochs", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    p.add_argument("--log_every", type=int, default=20)
    # MoE final
    p.add_argument("--routing", choices=("hard", "soft"), default="hard")
    p.add_argument("--tau", type=float, default=0.5)
    # Output
    p.add_argument("--out_root", default="lightning_logs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--skip_sweep", action="store_true",
                   help="Reuse sweep_summary.json from a prior run; just train finals.")
    p.add_argument("--seed", type=int, default=2024)
    return p


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    pl.seed_everything(args.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or (
        f"moesweep_{args.backbone}_d{'-'.join(map(str, args.depths))}"
        f"_lr{len(args.lrs)}_m{len(args.margins)}_{int(time.time())}"
    )
    out_root = Path(args.out_root) / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n[Sweep] run_name={run_name}  out={out_root}")

    # Probe class list via a no-merge loader so it matches the gate's space.
    probe = _make_loader(args, merge_classes=None)
    probe.setup()
    gate_class_to_idx = dict(probe.class_to_idx)
    class_list = sorted(gate_class_to_idx.keys())
    K = len(class_list)
    print(f"[Sweep] gate class_to_idx={gate_class_to_idx}")

    grid = list(itertools.product(args.depths, args.lrs, args.margins))
    print(f"[Sweep] grid size = {len(grid)} per class × {K} classes "
          f"= {len(grid)*K} sweep trainings")

    # ── Phase 1: grid per class ────────────────────────────────────────
    sweep_summary_path = out_root / "sweep_summary.json"
    if args.skip_sweep and sweep_summary_path.exists():
        sweep_summary = json.loads(sweep_summary_path.read_text())
        print(f"[Sweep] reusing existing summary: {sweep_summary_path}")
    else:
        sweep_summary: Dict[str, dict] = {}
        for cls_name in class_list:
            print(f"\n[Sweep] === class: {cls_name} ({len(grid)} cells) ===")
            cells: List[dict] = []
            best = None
            for (d, lr, m) in grid:
                cell_tag = f"d{d}_lr{lr:.0e}_m{m:.2f}"
                t0 = time.time()
                try:
                    f1, ckpt, ec2i = _train_binary_expert(
                        cls_name=cls_name, cell_tag=cell_tag, out_root=out_root,
                        args=args, depth=d, lr=lr, lmf_margin=m,
                        epochs=args.sweep_epochs,
                        class_list=class_list, device=device,
                    )
                    dt = time.time() - t0
                    cell = {"depth": d, "lr": lr, "lmf_margin": m,
                            "val_pos_f1": f1, "ckpt": ckpt, "seconds": dt,
                            "expert_class_to_idx": ec2i}
                    cells.append(cell)
                    print(f"[Sweep]   {cls_name}/{cell_tag}: val pos-F1={f1:.4f} "
                          f"({dt:.0f}s)")
                    if best is None or f1 > best["val_pos_f1"]:
                        best = cell
                except Exception as e:
                    print(f"[Sweep]   {cls_name}/{cell_tag}: FAILED ({e})")
                    cells.append({"depth": d, "lr": lr, "lmf_margin": m,
                                  "val_pos_f1": float("nan"), "error": str(e)})
                # Free between cells.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            sweep_summary[cls_name] = {"cells": cells, "best": best}
            # Persist incrementally so a crash doesn't lose state.
            sweep_summary_path.write_text(json.dumps(sweep_summary, indent=2, default=str))

    # ── Phase 2: retrain winners at full epochs ────────────────────────
    print(f"\n[Sweep] === Final retrain per class @ {args.final_epochs} ep ===")
    final_experts: List[dict] = []
    for cls_name in class_list:
        best = sweep_summary[cls_name]["best"]
        if best is None:
            raise RuntimeError(f"No successful sweep cell for {cls_name}")
        print(f"[Sweep] retraining {cls_name} with "
              f"depth={best['depth']}  lr={best['lr']}  m={best['lmf_margin']}")
        f1, ckpt, ec2i = _train_binary_expert(
            cls_name=cls_name, cell_tag=f"final_d{best['depth']}_lr{best['lr']:.0e}_m{best['lmf_margin']:.2f}",
            out_root=out_root, args=args,
            depth=best["depth"], lr=best["lr"], lmf_margin=best["lmf_margin"],
            epochs=args.final_epochs,
            class_list=class_list, device=device,
        )
        # Build & reload the final expert for inference.
        from campaign.train_moe import _build_backbone as build
        expert = build(
            backbone=args.backbone, num_classes=2, class_weights=None,
            sample_rate=args.target_sr, input_len=args.fixed_len,
            depth=best["depth"], max_epochs=args.final_epochs,
            learning_rate=best["lr"], warmup_epochs=args.warmup_epochs,
            extra={"lmf_margin": float(best["lmf_margin"])},
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        expert.load_state_dict(state["state_dict"], strict=False)
        dm = _make_loader(args, merge_classes=_rest_merge(cls_name, class_list))
        dm.setup()
        final_experts.append({
            "class": cls_name,
            "gate_idx": gate_class_to_idx[cls_name],
            "expert_class_to_idx": ec2i,
            "expert_pos_idx": ec2i[cls_name],
            "ckpt": ckpt,
            "val_pos_f1_final": f1,
            "best_cfg": best,
            "model": expert,
            "dm": dm,
        })

    # ── Phase 3: train the gate at full epochs ─────────────────────────
    print(f"\n[Sweep] === Train gate (K={K}) @ {args.gate_epochs} ep ===")
    gate_dm = _make_loader(args, merge_classes=None)
    gate_dm.setup()
    gate = _build_backbone(
        backbone=args.backbone, num_classes=K, class_weights=None,
        sample_rate=args.target_sr, input_len=args.fixed_len,
        depth=max(args.depths),  # gate gets the deepest sweep depth by default
        max_epochs=args.gate_epochs,
        learning_rate=args.lrs[len(args.lrs)//2],
        warmup_epochs=args.warmup_epochs,
    )
    gate, gate_ckpt = _train_one(
        name="gate",
        out_root=out_root,
        model=gate,
        dm=gate_dm,
        max_epochs=args.gate_epochs,
        monitor_metric="val/f1",
        monitor_mode="max",
        precision=args.precision,
        accelerator=args.accelerator,
        devices=_resolve_devices(args.devices),
        grad_clip=args.grad_clip,
        log_every=args.log_every,
    )

    # ── Phase 4: combined inference val + test ─────────────────────────
    eval_report: Dict[str, dict] = {}
    gate_idx_to_class = {v: k for k, v in gate_class_to_idx.items()}
    for split in ("val", "test"):
        try:
            gate_probs, gate_labels, _ = _predict_split(gate, gate_dm, split, device)
        except Exception as e:
            print(f"[Sweep] gate {split} prediction skipped: {e}")
            continue
        n = len(gate_labels)
        expert_pos_col = np.zeros((n, K), dtype=np.float64)
        ok = True
        for rec in final_experts:
            try:
                probs, _, _ = _predict_split(rec["model"], rec["dm"], split, device)
            except Exception as e:
                print(f"[Sweep] expert {rec['class']} {split} prediction failed: {e}")
                ok = False
                break
            if probs.shape[0] != n:
                m = min(probs.shape[0], n)
                gate_probs, gate_labels = gate_probs[:m], gate_labels[:m]
                expert_pos_col = expert_pos_col[:m]
                probs = probs[:m]
                n = m
            expert_pos_col[:, rec["gate_idx"]] = probs[:, rec["expert_pos_idx"]]
        if not ok:
            continue
        gate_pred = gate_probs.argmax(axis=1)
        gf1, gpc = _macro_f1(gate_labels, gate_pred, K)
        ph = _combine(gate_probs, expert_pos_col, "hard", tau=args.tau)
        ps = _combine(gate_probs, expert_pos_col, "soft")
        hf1, hpc = _macro_f1(gate_labels, ph, K)
        sf1, spc = _macro_f1(gate_labels, ps, K)
        eval_report[split] = {
            "n": int(n),
            "gate_only": {"macro_f1": gf1, "accuracy": _accuracy(gate_labels, gate_pred),
                          "per_class_f1": {gate_idx_to_class[i]: float(gpc[i]) for i in range(K)}},
            "moe_hard":  {"macro_f1": hf1, "accuracy": _accuracy(gate_labels, ph),
                          "per_class_f1": {gate_idx_to_class[i]: float(hpc[i]) for i in range(K)}},
            "moe_soft":  {"macro_f1": sf1, "accuracy": _accuracy(gate_labels, ps),
                          "per_class_f1": {gate_idx_to_class[i]: float(spc[i]) for i in range(K)}},
        }
        print(f"[Sweep] {split}: gate={gf1:.4f}  hard={hf1:.4f}  soft={sf1:.4f}")

    # ── Persist outputs ────────────────────────────────────────────────
    manifest = {
        "run_name": run_name,
        "backbone": args.backbone,
        "data_dir": args.data_dir,
        "classes": class_list,
        "gate_class_to_idx": gate_class_to_idx,
        "gate_ckpt": gate_ckpt,
        "grid": {"depths": args.depths, "lrs": args.lrs, "margins": args.margins,
                 "sweep_epochs": args.sweep_epochs, "final_epochs": args.final_epochs,
                 "gate_epochs": args.gate_epochs},
        "sweep_summary": sweep_summary,
        "experts": [{
            "class": r["class"], "ckpt": r["ckpt"],
            "best_cfg": r["best_cfg"], "val_pos_f1_final": r["val_pos_f1_final"],
            "expert_pos_idx": int(r["expert_pos_idx"]),
            "gate_idx": int(r["gate_idx"]),
        } for r in final_experts],
        "eval": eval_report,
        "args": vars(args),
    }
    (out_root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # Markdown report
    lines = [f"# MoE Sweep Report — {run_name}", "",
             f"- backbone: **{args.backbone}**",
             f"- data: `{args.data_dir}`",
             f"- classes: {class_list}",
             f"- grid: depths={args.depths}, lrs={args.lrs}, margins={args.margins}",
             f"- schedule: sweep_epochs={args.sweep_epochs}, final_epochs={args.final_epochs}, gate_epochs={args.gate_epochs}",
             "",
             "## Per-class sweep — best cell",
             "",
             "| class | depth | lr | margin | val pos-F1 (sweep) | val pos-F1 (final retrain) |",
             "|---|---|---|---|---|---|"]
    for r in final_experts:
        b = r["best_cfg"]
        lines.append(f"| {r['class']} | {b['depth']} | {b['lr']:.0e} | "
                     f"{b['lmf_margin']:.2f} | {b['val_pos_f1']:.4f} | "
                     f"{r['val_pos_f1_final']:.4f} |")
    lines += ["", "## Full sweep grid (val pos-F1 at sweep_epochs)", ""]
    for cls_name in class_list:
        lines.append(f"\n### {cls_name}\n")
        lines.append("| depth | lr | margin | val pos-F1 |")
        lines.append("|---|---|---|---|")
        for c in sweep_summary[cls_name]["cells"]:
            f1 = c.get("val_pos_f1", float("nan"))
            f1s = f"{f1:.4f}" if not (isinstance(f1, float) and np.isnan(f1)) else "ERR"
            lines.append(f"| {c['depth']} | {c['lr']:.0e} | {c['lmf_margin']:.2f} | {f1s} |")
    lines += ["", "## Combined MoE eval", ""]
    for split, m in eval_report.items():
        lines.append(f"\n### {split} (n={m['n']})\n")
        lines.append("| variant | macroF1 | accuracy |")
        lines.append("|---|---|---|")
        for key in ("gate_only", "moe_hard", "moe_soft"):
            r = m[key]
            lines.append(f"| {key} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} |")
        lines.append("\nPer-class F1 (MoE-soft):\n")
        for c, v in m["moe_soft"]["per_class_f1"].items():
            lines.append(f"- {c}: {v:.4f}")
    (out_root / "sweep_report.md").write_text("\n".join(lines))
    print(f"\n[Sweep] wrote {out_root/'sweep_manifest.json'} and {out_root/'sweep_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
