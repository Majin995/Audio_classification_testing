"""Mixture-of-Experts trainer — gate + per-class experts in one offline run.

Trains:
  * a multi-class GATE classifier over the user-supplied class folders, and
  * one EXPERT per class — either binary (one-vs-rest via loader merge) or
    multi (full K-class with target-class focal/weight bias).

Then evaluates the combined system on val/test with both hard and soft routing.

Backbones (configurable; both top performers on Combined_IARA_Deepship_1s):
  * ``complete`` — HydroComplete (default): Gabor+Scat+Sinc+TDSBE+CQT+DEMON,
    cross-attention fuser. Depth scales gabor/cqt/attn/s4d ``n_blocks``.
  * ``hydra``    — HydroHydra: strictly 1D (Gabor+Scat+Sinc+TDSBE → S4).
    Depth scales ``s4_n_blocks``.

Classes are passed as folder names — the loader resolves indices alphabetically.
Adjustable depth controls how many blocks each backbone branch stacks.

Example
-------
$ python -m campaign.train_moe \\
    --data_dir /path/Combined_IARA_Deepship_1s \\
    --classes Cargo Passenger Tanker Tug \\
    --expert_mode binary --routing soft --depth 2 \\
    --backbone complete --gate_epochs 30 --expert_epochs 30
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Repo root on path so models/, data/ imports work whether invoked as module
# or as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.loader_factory import LOADER_CHOICES, build_loader  # noqa: E402
from models.hydro_complete import HydroComplete  # noqa: E402
from models.hydro_hydra import HydroHydra  # noqa: E402


REST_LABEL = "__rest__"  # binary-mode merge target (sorts after capitals)


# ───────────────────────────────────────────────────────────────────────
# Depth presets — map a single --depth knob to per-branch n_blocks
# ───────────────────────────────────────────────────────────────────────

def _depth_overrides(backbone: str, depth: int) -> dict:
    """Map ``--depth`` (1..N) to per-branch block counts for the backbone.

    depth=1: shallow / fast; depth=2: backbone defaults; depth>=3: deeper.
    """
    d = max(1, int(depth))
    if backbone == "complete":
        return dict(
            gabor_n_blocks=d,
            cqt_n_blocks=d,
            demon_n_blocks=max(1, d - 1),
            n_attn_blocks=max(1, d - 1),
            n_s4d_blocks=d,
        )
    if backbone == "hydra":
        return dict(s4_n_blocks=d)
    raise ValueError(f"unknown backbone: {backbone!r}")


# ───────────────────────────────────────────────────────────────────────
# Backbone construction
# ───────────────────────────────────────────────────────────────────────

def _build_backbone(
    backbone: str,
    num_classes: int,
    class_weights: Optional[List[float]],
    sample_rate: int,
    input_len: int,
    depth: int,
    max_epochs: int,
    learning_rate: float,
    warmup_epochs: int,
    extra: Optional[dict] = None,
) -> pl.LightningModule:
    """Instantiate a HydroComplete or HydroHydra with depth overrides applied."""
    overrides = _depth_overrides(backbone, depth)
    common = dict(
        num_classes=num_classes,
        class_weights=class_weights,
        sample_rate=sample_rate,
        input_len=input_len,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        warmup_epochs=warmup_epochs,
    )
    if extra:
        common.update(extra)
    common.update(overrides)
    if backbone == "complete":
        return HydroComplete(**common)
    if backbone == "hydra":
        return HydroHydra(**common)
    raise ValueError(backbone)


# ───────────────────────────────────────────────────────────────────────
# Loader helpers
# ───────────────────────────────────────────────────────────────────────

def _make_loader(args, *, merge_classes: Optional[dict]):
    """Build a DataModule via the project's loader factory."""
    return build_loader(
        args.loader,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        target_sr=args.target_sr,
        fixed_len=args.fixed_len,
        oversample_train=not args.no_oversample,
        merge_classes=merge_classes,
        num_threads=args.num_threads,
        device_id=args.device_id,
        denoise_method=args.denoise_method,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        num_workers=args.num_workers,
        rms_normalize=args.rms_normalize,
        target_rms=args.target_rms,
    )


def _rest_merge(target_class: str, all_classes: List[str]) -> dict:
    """{non-target → REST_LABEL} so the loader produces a 2-class problem."""
    return {c: REST_LABEL for c in all_classes if c != target_class}


# ───────────────────────────────────────────────────────────────────────
# Inference utilities — collect probs from a single backbone over a split
# ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict_split(
    model: pl.LightningModule,
    dm,
    split: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs, labels, file_indices) for a split's dataloader.

    Indices come from the loader's iteration order; pairing across models
    only requires that the same dm is reused or that file lists/order match.
    """
    if split == "val":
        loader = dm.val_dataloader()
    elif split == "test":
        loader = dm.test_dataloader()
    else:
        loader = dm.train_dataloader()
    model.eval().to(device)
    probs_chunks: List[np.ndarray] = []
    labels_chunks: List[np.ndarray] = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            x, y = batch["x"], batch["y"]
        x = x.to(device, non_blocking=True)
        logits = model(x)
        if logits.ndim != 2:
            logits = logits.view(logits.shape[0], -1)
        n_cls = getattr(model, "num_classes", logits.shape[1])
        if logits.shape[1] > n_cls:
            logits = logits[:, :n_cls]
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        probs_chunks.append(probs)
        labels_chunks.append(y.detach().cpu().numpy())
    return (
        np.concatenate(probs_chunks, axis=0),
        np.concatenate(labels_chunks, axis=0),
        np.arange(sum(len(p) for p in probs_chunks)),
    )


# ───────────────────────────────────────────────────────────────────────
# Metrics
# ───────────────────────────────────────────────────────────────────────

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int) -> Tuple[float, np.ndarray]:
    f1s = np.zeros(n_cls, dtype=np.float64)
    for c in range(n_cls):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s[c] = (2 * tp / denom) if denom > 0 else 0.0
    return float(f1s.mean()), f1s


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


# ───────────────────────────────────────────────────────────────────────
# Training one model
# ───────────────────────────────────────────────────────────────────────

def _train_one(
    *,
    name: str,
    out_root: Path,
    model: pl.LightningModule,
    dm,
    max_epochs: int,
    monitor_metric: str,
    monitor_mode: str,
    precision: str,
    accelerator: str,
    devices,
    grad_clip: float,
    log_every: int,
) -> Tuple[pl.LightningModule, str]:
    """Train ``model`` against ``dm``; return (loaded_best_model, ckpt_path)."""
    log_dir = out_root / name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(log_dir / "checkpoints"),
        filename="best-{epoch:03d}-{" + monitor_metric.replace("/", "_") + ":.4f}",
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    logger = CSVLogger(save_dir=str(out_root), name=name, version=0)
    trainer = pl.Trainer(
        default_root_dir=str(log_dir),
        max_epochs=max_epochs,
        precision=precision,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=grad_clip,
        log_every_n_steps=log_every,
        enable_progress_bar=True,
        callbacks=[ckpt_cb],
        logger=logger,
    )
    trainer.fit(model, dm)
    best = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    # Reload best weights into the live model (cheaper than re-instantiating).
    if best and Path(best).exists():
        state = torch.load(best, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"], strict=False)
    return model, best


# ───────────────────────────────────────────────────────────────────────
# Combined inference — gate + experts → predictions
# ───────────────────────────────────────────────────────────────────────

def _combine(
    gate_probs: np.ndarray,          # (N, K) over gate's class indices
    expert_pos_probs: np.ndarray,    # (N, K) — column c = expert_c's P(class=c)
    routing: str,
    tau: float = 0.5,
) -> np.ndarray:
    """Return predicted class indices in the gate's class-index space.

    Hard: pick gate's argmax; if expert_c confirms (prob >= tau) use it, else
    mask that class in the gate and re-argmax (one re-try).
    Soft: argmax of product gate[c] * expert_c[c].
    """
    n, k = gate_probs.shape
    if routing == "soft":
        score = gate_probs * expert_pos_probs
        return score.argmax(axis=1)
    # hard
    yhat = gate_probs.argmax(axis=1)
    confirm = expert_pos_probs[np.arange(n), yhat] >= tau
    if confirm.all():
        return yhat
    # For unconfirmed rows, mask top-1 in gate and pick the next-best.
    backup = gate_probs.copy()
    backup[np.arange(n), yhat] = -np.inf
    yhat_alt = backup.argmax(axis=1)
    return np.where(confirm, yhat, yhat_alt)


# ───────────────────────────────────────────────────────────────────────
# Args
# ───────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_moe",
        description="Train a gate + per-class experts in one offline run.",
    )
    # Data / classes
    p.add_argument("--data_dir", required=True, type=str,
                   help="Dataset root containing Train/Val/Test (or train/val/test) class folders.")
    p.add_argument("--classes", nargs="+", required=True,
                   help="Class folder names. Order is for human-reading; the loader assigns indices alphabetically.")
    # Loader factory
    p.add_argument("--loader", choices=LOADER_CHOICES, default="dali")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--target_sr", type=int, default=5120)
    p.add_argument("--fixed_len", type=int, default=5120)
    p.add_argument("--no_oversample", action="store_true",
                   help="Disable oversample_train (default: enabled).")
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--denoise_method", default="off")
    p.add_argument("--window_sec", type=float, default=None)
    p.add_argument("--hop_sec", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--rms_normalize", action="store_true")
    p.add_argument("--target_rms", type=float, default=0.1)
    # Backbone & depth
    p.add_argument("--backbone", choices=("complete", "hydra"), default="complete")
    p.add_argument("--depth", type=int, default=2,
                   help="Per-branch block count multiplier. 1=shallow, 2=default, 3+=deeper.")
    p.add_argument("--gate_depth", type=int, default=None,
                   help="Override depth for the gate (default: --depth).")
    p.add_argument("--expert_depth", type=int, default=None,
                   help="Override depth for experts (default: --depth).")
    # MoE shape
    p.add_argument("--expert_mode", choices=("binary", "multi"), default="binary",
                   help="binary = one-vs-rest experts via loader merge; multi = full K-class experts with biased class_weights.")
    p.add_argument("--routing", choices=("hard", "soft"), default="hard")
    p.add_argument("--tau", type=float, default=0.5,
                   help="Hard-routing expert confirmation threshold.")
    p.add_argument("--target_weight", type=float, default=4.0,
                   help="multi-mode: class_weight assigned to the expert's target class (others=1.0).")
    # Training schedules
    p.add_argument("--gate_epochs", type=int, default=30)
    p.add_argument("--expert_epochs", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--warmup_epochs", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default="auto")
    p.add_argument("--log_every", type=int, default=20)
    # Output
    p.add_argument("--out_root", default="lightning_logs",
                   help="Base output dir; final dir is <out_root>/<run_name>.")
    p.add_argument("--run_name", default=None,
                   help="Defaults to moe_<backbone>_<expert_mode>_d<depth>_<timestamp>.")
    p.add_argument("--skip_gate", action="store_true",
                   help="Skip gate training (re-use existing ckpt under run_name).")
    p.add_argument("--skip_experts", action="store_true",
                   help="Skip expert training (re-use existing ckpts under run_name).")
    p.add_argument("--seed", type=int, default=2024)
    return p


def _resolve_devices(arg) -> object:
    if arg == "auto":
        return "auto"
    if isinstance(arg, str) and "," in arg:
        return [int(x) for x in arg.split(",")]
    try:
        return int(arg)
    except (TypeError, ValueError):
        return arg


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    pl.seed_everything(args.seed, workers=True)

    run_name = args.run_name or (
        f"moe_{args.backbone}_{args.expert_mode}_d{args.depth}_{int(time.time())}"
    )
    out_root = Path(args.out_root) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    devices = _resolve_devices(args.devices)
    gate_depth = args.gate_depth if args.gate_depth is not None else args.depth
    expert_depth = args.expert_depth if args.expert_depth is not None else args.depth

    print(f"\n[MoE] run_name={run_name}  out={out_root}")
    print(f"[MoE] classes={args.classes}  backbone={args.backbone}"
          f"  expert_mode={args.expert_mode}  routing={args.routing}"
          f"  gate_depth={gate_depth}  expert_depth={expert_depth}")

    # ── Gate loader: full K-class view ─────────────────────────────────
    gate_dm = _make_loader(args, merge_classes=None)
    gate_dm.setup()
    gate_class_to_idx: Dict[str, int] = dict(gate_dm.class_to_idx)
    gate_idx_to_class: Dict[int, str] = {v: k for k, v in gate_class_to_idx.items()}
    K = len(gate_class_to_idx)
    requested = sorted(args.classes)
    discovered = sorted(gate_class_to_idx.keys())
    if requested != discovered:
        print(f"[MoE] WARNING: --classes {requested} != discovered {discovered}; "
              f"using discovered set for experts.")
    class_list: List[str] = discovered  # the actual list we'll train experts for
    K = len(class_list)
    print(f"[MoE] gate class_to_idx = {gate_class_to_idx}  (K={K})")

    # ── Train gate ─────────────────────────────────────────────────────
    gate_ckpt_path = ""
    gate_model = None
    if not args.skip_gate:
        print(f"\n[MoE] === Training gate (K={K}) ===")
        gate_model = _build_backbone(
            backbone=args.backbone,
            num_classes=K,
            class_weights=None,
            sample_rate=args.target_sr,
            input_len=args.fixed_len,
            depth=gate_depth,
            max_epochs=args.gate_epochs,
            learning_rate=args.learning_rate,
            warmup_epochs=args.warmup_epochs,
        )
        gate_model, gate_ckpt_path = _train_one(
            name="gate",
            out_root=out_root,
            model=gate_model,
            dm=gate_dm,
            max_epochs=args.gate_epochs,
            monitor_metric="val/f1",
            monitor_mode="max",
            precision=args.precision,
            accelerator=args.accelerator,
            devices=devices,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
        )
    else:
        # Reload latest existing gate ckpt.
        gate_ckpts = sorted((out_root / "gate" / "checkpoints").glob("best-*.ckpt"))
        if not gate_ckpts:
            raise FileNotFoundError(f"--skip_gate set but no gate ckpt under {out_root/'gate'}")
        gate_ckpt_path = str(gate_ckpts[-1])
        gate_model = _build_backbone(
            backbone=args.backbone, num_classes=K, class_weights=None,
            sample_rate=args.target_sr, input_len=args.fixed_len,
            depth=gate_depth, max_epochs=args.gate_epochs,
            learning_rate=args.learning_rate, warmup_epochs=args.warmup_epochs,
        )
        state = torch.load(gate_ckpt_path, map_location="cpu", weights_only=False)
        gate_model.load_state_dict(state["state_dict"], strict=False)
        print(f"[MoE] gate reloaded from {gate_ckpt_path}")

    # ── Train experts ──────────────────────────────────────────────────
    expert_records: List[dict] = []  # one per class
    for cls_name in class_list:
        rec: dict = {"class": cls_name, "gate_idx": gate_class_to_idx[cls_name]}

        if args.expert_mode == "binary":
            merge = _rest_merge(cls_name, class_list)
            dm = _make_loader(args, merge_classes=merge)
            dm.setup()
            ec2i = dict(dm.class_to_idx)
            if cls_name not in ec2i:
                raise RuntimeError(
                    f"Expert loader for {cls_name!r} does not contain that class — "
                    f"got {ec2i}. Check folder names.")
            rec["expert_class_to_idx"] = ec2i
            rec["expert_pos_idx"] = ec2i[cls_name]
            num_classes = 2
            class_weights = None
        else:
            dm = _make_loader(args, merge_classes=None)
            dm.setup()
            rec["expert_class_to_idx"] = dict(dm.class_to_idx)
            rec["expert_pos_idx"] = rec["expert_class_to_idx"][cls_name]
            num_classes = K
            class_weights = [1.0] * K
            class_weights[rec["expert_pos_idx"]] = float(args.target_weight)

        if not args.skip_experts:
            print(f"\n[MoE] === Training expert: {cls_name} (mode={args.expert_mode}) ===")
            expert = _build_backbone(
                backbone=args.backbone,
                num_classes=num_classes,
                class_weights=class_weights,
                sample_rate=args.target_sr,
                input_len=args.fixed_len,
                depth=expert_depth,
                max_epochs=args.expert_epochs,
                learning_rate=args.learning_rate,
                warmup_epochs=args.warmup_epochs,
            )
            expert, ckpt = _train_one(
                name=f"expert_{cls_name}",
                out_root=out_root,
                model=expert,
                dm=dm,
                max_epochs=args.expert_epochs,
                monitor_metric="val/f1",
                monitor_mode="max",
                precision=args.precision,
                accelerator=args.accelerator,
                devices=devices,
                grad_clip=args.grad_clip,
                log_every=args.log_every,
            )
        else:
            ckpts = sorted((out_root / f"expert_{cls_name}" / "checkpoints").glob("best-*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"--skip_experts set but no ckpt for {cls_name}")
            ckpt = str(ckpts[-1])
            expert = _build_backbone(
                backbone=args.backbone, num_classes=num_classes, class_weights=class_weights,
                sample_rate=args.target_sr, input_len=args.fixed_len,
                depth=expert_depth, max_epochs=args.expert_epochs,
                learning_rate=args.learning_rate, warmup_epochs=args.warmup_epochs,
            )
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            expert.load_state_dict(state["state_dict"], strict=False)
            print(f"[MoE] expert {cls_name} reloaded from {ckpt}")
        rec["ckpt"] = ckpt
        rec["model"] = expert
        rec["dm"] = dm
        expert_records.append(rec)

    # ── Combined inference & metrics ───────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_report: Dict[str, dict] = {}
    for split in ("val", "test"):
        try:
            gate_probs, gate_labels, _ = _predict_split(gate_model, gate_dm, split, device)
        except Exception as e:
            print(f"[MoE] gate {split} prediction skipped: {e}")
            continue

        # Per-class expert positive-prob column in gate-index space.
        n = len(gate_labels)
        expert_pos_col = np.zeros((n, K), dtype=np.float64)
        skip = False
        for rec in expert_records:
            try:
                # For binary experts dm sees only 2 classes; labels here are
                # the expert's own merged labels, not the gate's. We only need
                # the positive-class column from probs to plug back into the
                # gate's index space.
                probs, _, _ = _predict_split(rec["model"], rec["dm"], split, device)
            except Exception as e:
                print(f"[MoE] expert {rec['class']} {split} prediction failed: {e}")
                skip = True
                break
            if probs.shape[0] != n:
                print(f"[MoE] WARN: row-count mismatch on {split} for "
                      f"expert {rec['class']} ({probs.shape[0]} vs gate {n}). "
                      f"Truncating to min().")
                m = min(probs.shape[0], n)
                if m != n:
                    gate_probs = gate_probs[:m]
                    gate_labels = gate_labels[:m]
                    expert_pos_col = expert_pos_col[:m]
                    n = m
                probs = probs[:m]
            expert_pos_col[:, rec["gate_idx"]] = probs[:, rec["expert_pos_idx"]]
        if skip:
            continue

        # Gate-only baseline
        gate_pred = gate_probs.argmax(axis=1)
        gate_macro_f1, gate_per_class_f1 = _macro_f1(gate_labels, gate_pred, K)
        gate_acc = _accuracy(gate_labels, gate_pred)

        # Combined predictions (both routing modes for the report)
        pred_hard = _combine(gate_probs, expert_pos_col, "hard", tau=args.tau)
        pred_soft = _combine(gate_probs, expert_pos_col, "soft")
        hard_f1, hard_pcf1 = _macro_f1(gate_labels, pred_hard, K)
        soft_f1, soft_pcf1 = _macro_f1(gate_labels, pred_soft, K)
        hard_acc = _accuracy(gate_labels, pred_hard)
        soft_acc = _accuracy(gate_labels, pred_soft)

        eval_report[split] = {
            "n": int(n),
            "gate_only": {"macro_f1": gate_macro_f1, "accuracy": gate_acc,
                          "per_class_f1": {gate_idx_to_class[i]: float(gate_per_class_f1[i]) for i in range(K)}},
            "moe_hard": {"macro_f1": hard_f1, "accuracy": hard_acc,
                         "per_class_f1": {gate_idx_to_class[i]: float(hard_pcf1[i]) for i in range(K)}},
            "moe_soft": {"macro_f1": soft_f1, "accuracy": soft_acc,
                         "per_class_f1": {gate_idx_to_class[i]: float(soft_pcf1[i]) for i in range(K)}},
        }
        print(f"\n[MoE] {split}: gate macroF1={gate_macro_f1:.4f}  "
              f"hard={hard_f1:.4f}  soft={soft_f1:.4f}")

    # ── Persist report + manifest ──────────────────────────────────────
    manifest = {
        "run_name": run_name,
        "backbone": args.backbone,
        "expert_mode": args.expert_mode,
        "routing_default": args.routing,
        "depth": {"gate": gate_depth, "expert": expert_depth},
        "tau": args.tau,
        "target_weight": args.target_weight,
        "gate_ckpt": gate_ckpt_path,
        "gate_class_to_idx": gate_class_to_idx,
        "experts": [
            {"class": r["class"], "ckpt": r["ckpt"],
             "expert_class_to_idx": r["expert_class_to_idx"],
             "expert_pos_idx": int(r["expert_pos_idx"]),
             "gate_idx": int(r["gate_idx"])}
            for r in expert_records
        ],
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "eval": eval_report,
    }
    with open(out_root / "moe_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Human-readable summary
    lines = [f"# MoE run — {run_name}", "",
             f"- backbone={args.backbone}  expert_mode={args.expert_mode}  "
             f"depth gate/expert = {gate_depth}/{expert_depth}",
             f"- gate ckpt: `{gate_ckpt_path}`",
             "", "## Experts", ""]
    for r in expert_records:
        lines.append(f"- **{r['class']}** (gate idx {r['gate_idx']}, "
                     f"expert pos {r['expert_pos_idx']}): `{r['ckpt']}`")
    lines.append("")
    lines.append("## Eval")
    for split, m in eval_report.items():
        lines.append(f"\n### {split} (n={m['n']})\n")
        lines.append("| variant | macroF1 | accuracy |")
        lines.append("|---|---|---|")
        for key in ("gate_only", "moe_hard", "moe_soft"):
            r = m[key]
            lines.append(f"| {key} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} |")
        lines.append("\nPer-class F1 (MoE-soft):")
        for c, v in m["moe_soft"]["per_class_f1"].items():
            lines.append(f"- {c}: {v:.4f}")
    (out_root / "moe_report.md").write_text("\n".join(lines))
    print(f"\n[MoE] wrote {out_root/'moe_manifest.json'} and {out_root/'moe_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
