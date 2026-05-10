"""
Training script for HydroAudioLLM — Whisper + Llama hybrid classifier.

Usage:
    export DATA_DIR=/path/to/Split1s
    python training/train_hydro_audio_llm.py [options]

Monitors ``val/micro_precision`` for checkpointing.  Defaults freeze both
backbones (only the adapter and the classification head receive
gradients); pass ``--unfreeze_encoder`` and/or ``--unfreeze_llm`` for
partial fine-tuning.

Optionally performs post-hoc temperature scaling on the validation set
after training — disabled with ``--skip_calibration``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from data.audio_lightning_loader import DALIAudioDataModule
from models.hydro_audio_llm      import HydroAudioLLM


# ═══════════════════════════════════════════════════════════════════════
#  Post-hoc temperature scaling (mirrors train_precise.py:40-67)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_logits(model: HydroAudioLLM, dataloader, device: torch.device):
    model.eval()
    logits_all, targets_all = [], []
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits_all.append(model(x).cpu())
        targets_all.append(y.cpu())
    return torch.cat(logits_all), torch.cat(targets_all)


def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor,
                     lr: float = 1e-2, max_iter: int = 200) -> float:
    log_T = nn.Parameter(torch.zeros(1))
    optim = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        optim.zero_grad()
        T = log_T.exp().clamp(min=1e-2, max=100.0)
        loss = F.cross_entropy(logits / T, targets)
        loss.backward()
        return loss

    optim.step(closure)
    return float(log_T.exp().clamp(min=1e-2, max=100.0).item())


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description="Train HydroAudioLLM (Whisper + Llama)")

    # Data
    p.add_argument("--data_dir",      default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--num_threads",   type=int, default=4)
    p.add_argument("--no_oversample", action="store_true")

    # Audio
    p.add_argument("--sample_rate",   type=int, default=5_120)
    p.add_argument("--fixed_len",     type=int, default=5_120)

    # Backbones
    p.add_argument("--whisper_model_id", default="openai/whisper-large-v3")
    p.add_argument("--llm_model_id",     default="meta-llama/Llama-3.2-1B")

    # Adapter / head
    p.add_argument("--adapter_type",    default="linear", choices=["linear"])
    p.add_argument("--adapter_dropout", type=float, default=0.1)
    p.add_argument("--head_hidden",     type=int,   default=512)
    p.add_argument("--head_dropout",    type=float, default=0.2)

    # Freezing flags (default: freeze both; flip with --unfreeze_*)
    enc = p.add_mutually_exclusive_group()
    enc.add_argument("--freeze_encoder",   dest="freeze_encoder", action="store_true",
                     help="Freeze Whisper encoder (default).")
    enc.add_argument("--unfreeze_encoder", dest="freeze_encoder", action="store_false",
                     help="Allow Whisper encoder to receive gradients.")
    p.set_defaults(freeze_encoder=True)

    llm = p.add_mutually_exclusive_group()
    llm.add_argument("--freeze_llm",   dest="freeze_llm", action="store_true",
                     help="Freeze Llama backbone (default).")
    llm.add_argument("--unfreeze_llm", dest="freeze_llm", action="store_false",
                     help="Allow Llama backbone to receive gradients.")
    p.set_defaults(freeze_llm=True)

    # Loss
    p.add_argument("--lmf_gamma",       type=float, default=2.0)
    p.add_argument("--lmf_margin",      type=float, default=0.3)
    p.add_argument("--label_smoothing", type=float, default=0.05)

    # Training
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--max_epochs",    type=int,   default=50)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--precision",     default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--accelerator",   default="auto")
    p.add_argument("--devices",       default=1)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--run_name",      default="hydro_audio_llm")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)

    # Post-hoc
    p.add_argument("--skip_calibration", action="store_true")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main(args=None):
    if args is None:
        args = get_args()

    if not args.data_dir:
        raise ValueError("Set --data_dir or export DATA_DIR=/path/to/Split1s")

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        oversample_train=not args.no_oversample,
    )
    data.setup()

    model = HydroAudioLLM(
        num_classes=data.num_classes,
        class_weights=data.class_weights,
        sample_rate=args.sample_rate,
        whisper_model_id=args.whisper_model_id,
        llm_model_id=args.llm_model_id,
        adapter_type=args.adapter_type,
        adapter_dropout=args.adapter_dropout,
        freeze_encoder=args.freeze_encoder,
        freeze_llm=args.freeze_llm,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        lmf_gamma=args.lmf_gamma,
        lmf_margin=args.lmf_margin,
        label_smoothing=args.label_smoothing,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total  = sum(p.numel() for p in model.parameters())
    print(f"\nHydroAudioLLM — {n_params / 1e6:.2f}M trainable / "
          f"{n_total / 1e6:.2f}M total parameters")
    print(f"  Whisper: {args.whisper_model_id}  (frozen={args.freeze_encoder})")
    print(f"  LLM:     {args.llm_model_id}      (frozen={args.freeze_llm})")

    ckpt_cb = ModelCheckpoint(
        monitor="val/micro_precision", mode="max", save_top_k=3,
        filename="audio_llm-{epoch:03d}-p{val/micro_precision:.4f}",
        auto_insert_metric_name=False, verbose=True,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val/micro_precision", patience=args.patience,
                      mode="max", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator if args.accelerator != "auto"
                    else ("gpu" if torch.cuda.is_available() else "cpu"),
        devices=args.devices,
        precision=args.precision if torch.cuda.is_available() else 32,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=[
            CSVLogger("lightning_logs", name=args.run_name),
            TensorBoardLogger("lightning_logs", name=args.run_name),
        ],
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        deterministic=False,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    trainer.fit(model, data)
    trainer.test(model, data, ckpt_path="best")

    best_path  = ckpt_cb.best_model_path
    best_score = ckpt_cb.best_model_score
    print(f"\nBest checkpoint:       {best_path}")
    if best_score is not None:
        print(f"Best val/micro_prec:   {best_score:.4f}")

    # ── Post-hoc calibration ─────────────────────────────────────────────
    if not args.skip_calibration and best_path:
        print("\n── Post-hoc temperature scaling ──")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cal_model = HydroAudioLLM.load_from_checkpoint(
            best_path, map_location=device, strict=False,
        )
        cal_model.to(device).eval()

        logits, targets = _collect_logits(cal_model, data.val_dataloader(), device)
        T = _fit_temperature(logits, targets)
        probs = F.softmax(logits / T, dim=-1)
        print(f"  temperature = {T:.3f}")
        print(f"  temp-scaled val accuracy = "
              f"{float((probs.argmax(-1) == targets).float().mean()):.4f}")

        out_dir = Path(best_path).parent
        torch.save({"temperature": T}, out_dir / "temperature.pt")
        with open(out_dir / "calibration.json", "w") as f:
            json.dump({"temperature": T}, f, indent=2)
        print(f"  saved → {out_dir / 'temperature.pt'}")

    return float(best_score) if best_score is not None else float("nan"), best_path


if __name__ == "__main__":
    main()
