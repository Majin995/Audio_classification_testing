"""Exhaustive autotune for HydroRecurrent on Combined_IARA_Deepship_1s.

Optuna study (TPE sampler + MedianPruner), sqlite-resumable. Each trial
trains a HydroRecurrent in-process and reports intermediate val macro-F1
every `val_every` steps so unpromising trials get pruned early.

Honest contract: val sources only for selection; test is NEVER touched here.

Usage:
    PYTHONPATH=. python campaign/autotune_recurrent.py \
        --hours 12 --steps 2500 --val_every 250

Resumable: re-run the same command after a kill and Optuna picks up where
it left off via the sqlite study DB.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import optuna
import torch

from campaign.train_recurrent import (
    CLASSES,
    GamblerCE,
    SourceSampler,
    eval_split,
    macro_prf1,
    preprocess,
    read_batch_threaded,
    scan_split,
)
from models.hydro_recurrent import HydroRecurrent


# Loaded once, shared across trials.
_DATA_CACHE: dict = {}


def _load_data(data_dir: str):
    if "train" not in _DATA_CACHE:
        _DATA_CACHE["train"] = scan_split(data_dir, "Train")
        _DATA_CACHE["val"] = scan_split(data_dir, "Val")
        for label in ("train", "val"):
            d = _DATA_CACHE[label]
            cnt = {c: len(d[c]) for c in CLASSES}
            print(f"{label:>5s} sources: {cnt}", flush=True)
    return _DATA_CACHE["train"], _DATA_CACHE["val"]


# Discrete TCN-dilation presets the search picks among.
TCN_PRESETS = {
    "1,2,4,8":      (1, 2, 4, 8),
    "1,4,16,64":    (1, 4, 16, 64),
    "1,2,4,8,16":   (1, 2, 4, 8, 16),
    "1,3,9,27":     (1, 3, 9, 27),
    "1,4,16":       (1, 4, 16),
    "1,2,4,8,16,32":(1, 2, 4, 8, 16, 32),
}


def sample_config(trial: optuna.Trial) -> dict:
    return dict(
        # Optimizer
        lr            = trial.suggest_float("lr", 3e-4, 6e-3, log=True),
        wd            = trial.suggest_float("wd", 1e-4, 5e-2, log=True),
        # Model
        n_bands       = trial.suggest_categorical("n_bands", [16, 24, 32, 48]),
        embed_dim     = trial.suggest_categorical("embed_dim", [32, 48, 64, 96]),
        gru_hidden    = trial.suggest_categorical("gru_hidden", [32, 48, 64, 96, 128]),
        gru_layers    = trial.suggest_int("gru_layers", 1, 2),
        head_hidden   = trial.suggest_categorical("head_hidden", [32, 64, 96, 128]),
        dropout       = trial.suggest_float("dropout", 0.05, 0.45),
        band_dropout  = trial.suggest_float("band_dropout", 0.0, 0.30),
        bidirectional = trial.suggest_categorical("bidirectional", [False, True]),
        tcn_preset    = trial.suggest_categorical("tcn_preset", list(TCN_PRESETS.keys())),
        # Sampling
        per_class     = trial.suggest_categorical("per_class", [3, 4, 6]),
        K_train       = trial.suggest_categorical("K_train", [12, 16, 24, 32]),
        # Loss
        smoothing      = trial.suggest_float("smoothing", 0.0, 0.15),
        gambler_w      = trial.suggest_float("gambler_w", 0.0, 0.30),
        abstain_o      = trial.suggest_float("abstain_o", 1.5, 3.5),
        focal_gamma    = trial.suggest_float("focal_gamma", 0.0, 3.0),
        deep_aux_w     = trial.suggest_float("deep_aux_w", 0.0, 0.6),
        class_weight_pow = trial.suggest_float("class_weight_pow", 0.0, 1.0),
    )


def run_trial(trial: optuna.Trial, args, device, executor, train_src, val_src) -> float:
    cfg = sample_config(trial)
    tcn_dils = TCN_PRESETS[cfg["tcn_preset"]]
    seed = args.seed + trial.number
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model = HydroRecurrent(
        num_classes=4, n_bands=cfg["n_bands"], embed_dim=cfg["embed_dim"],
        gru_hidden=cfg["gru_hidden"], gru_layers=cfg["gru_layers"],
        head_hidden=cfg["head_hidden"], dropout=cfg["dropout"],
        band_dropout=cfg["band_dropout"], tcn_dilations=tcn_dils,
        bidirectional=cfg["bidirectional"], gambler=True,
    ).to(device)
    n_params = model.n_params()
    trial.set_user_attr("n_params", n_params)
    print(f"\n=== trial {trial.number}  params={n_params:,}  cfg={cfg}", flush=True)

    counts = np.array([len(train_src[c]) for c in CLASSES], dtype=np.float64)
    cw = (counts.sum() / (counts * len(CLASSES))) ** cfg["class_weight_pow"]
    class_weight = torch.tensor(cw, dtype=torch.float32, device=device)
    criterion = GamblerCE(
        K=4, smoothing=cfg["smoothing"], gambler_w=cfg["gambler_w"],
        abstain_o=cfg["abstain_o"], class_weight=class_weight,
        focal_gamma=cfg["focal_gamma"],
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    sampler = SourceSampler(train_src, cfg["per_class"], cfg["K_train"], random.Random(seed))

    best_f1 = -1.0
    best_step = -1
    patience_left = args.patience
    history = []
    t_trial = time.time()

    for step in range(1, args.steps + 1):
        if time.time() - args.t0 > args.hours * 3600:
            print(f"  [wall budget hit during trial {trial.number}]", flush=True)
            trial.set_user_attr("walltime_truncated", True)
            break
        model.train()
        labels, clip_paths = sampler.draw()
        flat = [p for lst in clip_paths for p in lst]
        audio = read_batch_threaded(flat, executor)
        wav = torch.from_numpy(audio).to(device, non_blocking=True)
        wav = preprocess(wav, args.target_rms, args.hpf_hz)
        N = len(labels); K = cfg["K_train"]
        wav = wav.view(N, K, -1)
        targets = torch.from_numpy(labels).to(device)

        logits_seq = model(wav, mask=None, return_seq=True)
        final_logits = logits_seq[:, -1]
        loss = criterion(final_logits, targets)
        if cfg["deep_aux_w"] > 0:
            tgt_rep = targets.unsqueeze(1).expand(-1, K).reshape(-1)
            seq_flat = logits_seq.reshape(N * K, -1)
            loss = loss + cfg["deep_aux_w"] * criterion(seq_flat, tgt_rep)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()

        if step % args.val_every == 0 or step == args.steps:
            y, yp, _, _ = eval_split(
                model, val_src, executor, device,
                K_eval=args.K_eval, target_rms=args.target_rms,
                hpf_hz=args.hpf_hz, batch_size_sources=4,
            )
            _, _, _, mp, mr, mf, _ = macro_prf1(y, yp, 4)
            history.append({"step": step, "f1": float(mf), "p": float(mp), "r": float(mr)})
            print(f"  [t{trial.number} step {step}] f1={mf:.4f} p={mp:.4f} r={mr:.4f} "
                  f"loss={loss.item():.3f}  ({time.time()-t_trial:.0f}s)", flush=True)
            trial.report(float(mf), step)
            if mf > best_f1:
                best_f1 = float(mf); best_step = step; patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"  early stop (patience)", flush=True)
                    break
            if trial.should_prune():
                trial.set_user_attr("pruned_at_step", step)
                trial.set_user_attr("best_f1_before_prune", best_f1)
                trial.set_user_attr("history", history)
                # Free GPU before raising.
                del model, opt, sched, criterion
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()

    trial.set_user_attr("best_step", best_step)
    trial.set_user_attr("history", history)
    trial.set_user_attr("trial_seconds", time.time() - t_trial)

    # Save best ckpt for top trials.
    if best_f1 >= args.save_threshold:
        ckpt_dir = Path(args.out_dir) / f"trial_{trial.number:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": cfg, "tcn_dilations": list(tcn_dils),
            "best_val_macro_f1": best_f1, "best_step": best_step,
            "n_params": n_params,
        }, ckpt_dir / "best.pt")
        (ckpt_dir / "trial.json").write_text(json.dumps({
            "trial": trial.number, "best_f1": best_f1, "best_step": best_step,
            "config": cfg, "history": history,
        }, indent=2))

    del model, opt, sched, criterion
    torch.cuda.empty_cache()
    return float(best_f1)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",
                   default="/var/mnt/5A009BF8009BD8F9/Data/Combined_IARA_Deepship_1s")
    p.add_argument("--out_dir", default="lightning_logs/autotune_recurrent")
    p.add_argument("--loader", choices=("dali", "dali_split", "threaded", "threaded_split"),
                   default="threaded",
                   help="Audio backend. Stacker uses internal threaded I/O; "
                        "'dali*' choices are accepted for uniformity but treated as 'threaded'.")
    p.add_argument("--class_depth", type=int, default=1,
                   help="Model depth knob. Scales gru_layers and tcn_dilations repeats.")
    p.add_argument("--study_name", default="hydro_recurrent_autotune_v1")
    p.add_argument("--hours", type=float, default=12.0)
    p.add_argument("--n_trials", type=int, default=10_000,
                   help="Hard cap; wall budget is the real stopper.")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--K_eval", type=int, default=48)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--target_rms", type=float, default=0.1)
    p.add_argument("--hpf_hz", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_workers", type=int, default=12)
    p.add_argument("--save_threshold", type=float, default=0.50,
                   help="Save ckpt if val macro-F1 >= this.")
    p.add_argument("--startup_trials", type=int, default=8,
                   help="Random trials before TPE kicks in.")
    p.add_argument("--pruner_warmup_steps", type=int, default=500,
                   help="No pruning before this many training steps within a trial.")
    return p.parse_args()


def main():
    args = get_args()
    args.t0 = time.time()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_src, val_src = _load_data(args.data_dir)
    executor = ThreadPoolExecutor(max_workers=args.n_workers)

    storage = f"sqlite:///{(out_dir / 'study.db').resolve()}"
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=args.startup_trials, multivariate=True, seed=args.seed,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.startup_trials,
        n_warmup_steps=args.pruner_warmup_steps,
        interval_steps=args.val_every,
    )
    study = optuna.create_study(
        study_name=args.study_name, storage=storage,
        direction="maximize", sampler=sampler, pruner=pruner,
        load_if_exists=True,
    )

    def _objective(trial):
        return run_trial(trial, args, device, executor, train_src, val_src)

    timeout = max(60, int(args.hours * 3600))
    print(f"\nStudy {args.study_name}  storage={storage}\n"
          f"Wall budget = {args.hours}h ({timeout}s)\n"
          f"Existing trials = {len(study.trials)}", flush=True)

    try:
        study.optimize(_objective, n_trials=args.n_trials, timeout=timeout,
                       gc_after_trial=True, show_progress_bar=False,
                       catch=(RuntimeError,))
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)

    # Report
    print("\n" + "=" * 70)
    print(f"FINISHED  trials_total={len(study.trials)}  "
          f"completed={sum(t.state.name=='COMPLETE' for t in study.trials)}  "
          f"pruned={sum(t.state.name=='PRUNED' for t in study.trials)}  "
          f"failed={sum(t.state.name=='FAIL' for t in study.trials)}")
    if study.best_trial is not None:
        bt = study.best_trial
        print(f"BEST  trial={bt.number}  val_macro_f1={bt.value:.4f}  params={bt.user_attrs.get('n_params','?')}")
        print(f"  config = {bt.params}")
    summary = {
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_value": study.best_value if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else None,
        "top10": [
            {"trial": t.number, "value": t.value, "params": t.params,
             "n_params": t.user_attrs.get("n_params")}
            for t in sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda t: t.value, reverse=True)[:10]
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
