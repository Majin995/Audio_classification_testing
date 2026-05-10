"""Train a HydroHydra student via 4-teacher distillation.

Mirrors ``training/train_hydra.py`` but:
  * Builds a ``HydroHydraStudent`` instead of plain ``HydroHydra``.
  * Loads 4 frozen binary specialist teachers from ``--teacher_ckpts``.
  * Adds ``--kd_alpha`` / ``--kd_temperature`` hyperparameters.

The teachers must be passed in alphabetical class order:
``cargo_ckpt passenger_ckpt tanker_ckpt tug_ckpt``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Re-use train_hydra.py's argument parser + main flow as much as possible.
import training.train_hydra as base_trainer  # type: ignore
from models.hydro_hydra import HydroHydra
from models.hydro_hydra_student import HydroHydraStudent


def _augment_args() -> argparse.Namespace:
    """Augment train_hydra's arg parser with KD flags, then parse."""
    p = argparse.ArgumentParser(parents=[], description="Train HydroHydra student",
                                conflict_handler="resolve")
    # Re-import the base parser from train_hydra (it constructs and returns
    # an args object; we patch the parser by hand).
    base_p = base_trainer.get_args.__wrapped__ if hasattr(base_trainer.get_args, '__wrapped__') else None
    # Simpler approach: just call get_args() to get the base namespace, then
    # parse the KD flags off the remaining argv.
    # The student inherits every flag from train_hydra.py.
    import sys as _sys
    # Pull out KD flags before delegating to base get_args.
    kd_flags_p = argparse.ArgumentParser(add_help=False)
    # Single-teacher-per-class (legacy, 4 paths in alphabetical order).
    kd_flags_p.add_argument("--teacher_ckpts", nargs=4, default=None,
                            metavar=("CARGO", "PASSENGER", "TANKER", "TUG"),
                            help="Legacy: 4 teacher checkpoints in alpha class "
                                 "order. Mutually exclusive with --teacher_<cls>.")
    # Multi-teacher-per-class (preferred when stacking v1/v2/v3/v4).
    kd_flags_p.add_argument("--teacher_cargo",     nargs="+", default=None)
    kd_flags_p.add_argument("--teacher_passenger", nargs="+", default=None)
    kd_flags_p.add_argument("--teacher_tanker",    nargs="+", default=None)
    kd_flags_p.add_argument("--teacher_tug",       nargs="+", default=None)
    kd_flags_p.add_argument("--kd_alpha", type=float, default=0.7,
                            help="Weight on KD loss vs supervised CE.")
    kd_flags_p.add_argument("--kd_temperature", type=float, default=4.0,
                            help="Softmax temperature applied to both "
                                 "teacher and student during KD.")
    kd_args, remaining = kd_flags_p.parse_known_args()

    # Now hand the rest to the base trainer.
    _sys.argv = [_sys.argv[0]] + remaining
    base_args = base_trainer.get_args()
    # Merge KD args onto the base namespace.
    base_args.teacher_ckpts     = kd_args.teacher_ckpts
    base_args.teacher_cargo     = kd_args.teacher_cargo
    base_args.teacher_passenger = kd_args.teacher_passenger
    base_args.teacher_tanker    = kd_args.teacher_tanker
    base_args.teacher_tug       = kd_args.teacher_tug
    base_args.kd_alpha          = kd_args.kd_alpha
    base_args.kd_temperature    = kd_args.kd_temperature

    # Validate teacher specification: exactly one of {legacy, per-class}.
    has_legacy = base_args.teacher_ckpts is not None
    has_groups = any(getattr(base_args, f"teacher_{c}") is not None
                     for c in ("cargo", "passenger", "tanker", "tug"))
    if has_legacy and has_groups:
        raise ValueError("Use either --teacher_ckpts (4 paths) OR "
                         "--teacher_<class> per-class lists, not both.")
    if not has_legacy and not has_groups:
        raise ValueError("Pass --teacher_ckpts (legacy 4 paths) or per-class "
                         "--teacher_cargo / --teacher_passenger / "
                         "--teacher_tanker / --teacher_tug.")
    return base_args


def main():
    args = _augment_args()
    if not args.data_dir:
        raise ValueError("Set --data_dir or export DATA_DIR")

    # Resolve teacher list -- single legacy path or per-class groups.
    if args.teacher_ckpts is not None:
        teacher_groups_spec: list[list[str]] = [[p] for p in args.teacher_ckpts]
    else:
        teacher_groups_spec = [
            args.teacher_cargo or [],
            args.teacher_passenger or [],
            args.teacher_tanker or [],
            args.teacher_tug or [],
        ]
        for cls_name, group in zip(("Cargo", "Passenger", "Tanker", "Tug"),
                                   teacher_groups_spec):
            if not group:
                raise ValueError(
                    f"--teacher_{cls_name.lower()} must have ≥1 ckpt"
                )
    # Existence check + summary.
    for cls_name, group in zip(("Cargo", "Passenger", "Tanker", "Tug"),
                               teacher_groups_spec):
        for p in group:
            if not Path(p).exists():
                raise FileNotFoundError(f"teacher ckpt not found: {p}")
        print(f"[student] {cls_name:10s}: {len(group)} teacher(s) — {group}")
    print(f"[student] kd_alpha={args.kd_alpha}  kd_temperature={args.kd_temperature}")

    # Monkey-patch the base trainer's HydroHydra reference so its main()
    # builds a HydroHydraStudent — but we need to inject teachers + KD
    # hparams. Cleanest: replicate base_trainer.main() inline with the
    # student class swap.

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    )
    from data.audio_lightning_loader import DALIAudioDataModule

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        oversample_train=not args.no_oversample,
        denoise_method=args.denoise,
    )
    data.setup()

    if data.train_files_override is not None:
        _train_by_cls = data._apply_merge(data.train_files_override)
    else:
        _train_by_cls = data._apply_merge(data._scan_split("train"))
    classes_sorted = sorted(_train_by_cls.keys(), key=lambda c: data.class_to_idx[c])
    cls_num_list = [len(_train_by_cls[c]) for c in classes_sorted]
    print(f"[cls_num_list] {dict(zip(classes_sorted, cls_num_list))}")

    if data.num_classes != 4:
        raise RuntimeError(
            f"student expects 4 classes, got {data.num_classes}. Did you "
            f"accidentally pass --positive_class?"
        )

    # ── Build student ─────────────────────────────────────────────────
    student = HydroHydraStudent(
        num_classes=data.num_classes,
        class_weights=data.class_weights,
        sample_rate=args.sample_rate,
        input_len=args.fixed_len,
        use_gabor=args.use_gabor,
        use_scattering=args.use_scattering,
        use_sincnet=args.use_sincnet,
        use_tdsbe=args.use_tdsbe,
        use_w2v=args.use_w2v,
        use_lpc=args.use_lpc,
        use_rp=args.use_rp,
        gabor_n_filters=args.gabor_n_filters,
        gabor_kernel=args.gabor_kernel,
        gabor_ch=args.gabor_ch,
        scat_J=args.scat_J,
        scat_Q=args.scat_Q,
        scat_ch=args.scat_ch,
        use_jtfs=args.use_jtfs,
        sinc_n_filters=args.sinc_n_filters,
        sinc_kernel=args.sinc_kernel,
        sinc_ch=args.sinc_ch,
        tdsbe_ch=args.tdsbe_ch,
        w2v_model=args.w2v_model,
        w2v_ch=args.w2v_ch,
        w2v_target_sr=args.w2v_target_sr,
        lpc_order=args.lpc_order,
        lpc_frame=args.lpc_frame,
        lpc_hop=args.lpc_hop,
        lpc_ch=args.lpc_ch,
        rp_downsample=args.rp_downsample,
        rp_dim=args.rp_dim,
        rp_delay=args.rp_delay,
        rp_eps_quantile=args.rp_eps_quantile,
        rp_ch=args.rp_ch,
        fusion_T=args.fusion_T,
        fusion_dim=args.fusion_dim,
        s4_n_blocks=args.s4_n_blocks,
        s4_d_state=args.s4_d_state,
        use_global_attn=args.use_global_attn,
        global_attn_heads=args.global_attn_heads,
        dropout=args.dropout,
        drop_path=args.drop_path,
        head_type=args.head_type,
        feature_norm=args.feature_norm,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        arcface_subcenters=args.arcface_subcenters,
        moe_n_experts=args.moe_n_experts,
        moe_gate_temperature=args.moe_gate_temperature,
        moe_aux_weight=args.moe_aux_weight,
        loss=args.loss,
        lmf_gamma=args.lmf_gamma,
        lmf_margin=args.lmf_margin,
        label_smoothing=args.label_smoothing,
        gambler_o=args.gambler_o,
        gambler_weight=args.gambler_weight,
        cls_num_list=cls_num_list,
        ldam_max_m=args.ldam_max_m,
        ldam_s=args.ldam_s,
        ldam_drw_epoch=args.ldam_drw_epoch,
        ldam_drw_beta=args.ldam_drw_beta,
        noise_prob=args.noise_prob,
        noise_snr_min=args.noise_snr_min,
        noise_snr_max=args.noise_snr_max,
        gain_prob=args.gain_prob,
        gain_range=args.gain_range,
        ocean_noise_pool=None,
        corpus_noise_prob=0.0,
        corpus_noise_snr_min=args.corpus_noise_snr_min,
        corpus_noise_snr_max=args.corpus_noise_snr_max,
        rir_prob=args.rir_prob,
        rir_max_delay_s=args.rir_max_delay_s,
        pitch_prob=args.pitch_prob,
        pitch_range=args.pitch_range,
        branch_dropout_p=args.branch_dropout_p,
        manifold_mixup_alpha=args.manifold_mixup_alpha,
        manifold_mixup_prob=args.manifold_mixup_prob,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
    )

    # ── Load teachers (per class) ─────────────────────────────────────
    teacher_groups: list[list[HydroHydra]] = []
    total_teachers = 0
    for cls_name, group_paths in zip(("Cargo", "Passenger", "Tanker", "Tug"),
                                     teacher_groups_spec):
        group = []
        for ck in group_paths:
            t = HydroHydra.load_from_checkpoint(ck, strict=False)
            group.append(t)
            total_teachers += 1
        teacher_groups.append(group)
    student.set_teacher_groups(teacher_groups)
    print(f"[student] loaded {total_teachers} teachers across 4 classes, all frozen.")

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"\nHydroHydraStudent — {n_params / 1e6:.2f}M trainable params")

    # ── Callbacks + trainer ───────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        monitor="val/macro_precision", mode="max", save_top_k=3,
        filename="student-{epoch:03d}-p{val/macro_precision:.4f}",
        auto_insert_metric_name=False, verbose=True,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val/macro_precision", patience=args.patience,
                      mode="max", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        default_root_dir=f"lightning_logs/{args.run_name}",
        enable_progress_bar=True,
        log_every_n_steps=50,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    trainer.fit(student, datamodule=data)
    trainer.test(student, datamodule=data, ckpt_path="best")

    # Run the same post-cal as the base trainer.
    if not args.skip_calibration:
        from inference.postcal_one import _collect, _fit_temperature
        from inference.calibrate import (
            search_thresholds_macroP_coverage,
            search_thresholds_recall_floor,
        )
        from inference.selective_pr import selective_pr_curve, write_markdown
        import json
        import torch.nn.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_path = ckpt_cb.best_model_path
        if not best_path:
            print("[student] no best ckpt — skipping post-cal.")
            return
        print(f"[student] post-cal on best ckpt: {best_path}")
        student_eval = HydroHydraStudent.load_from_checkpoint(
            best_path, strict=False, map_location=device,
        )
        student_eval = student_eval.to(device).eval()

        val_logits, val_targets = _collect(student_eval, data.val_dataloader(),
                                            device, data.num_classes)
        T = _fit_temperature(val_logits, val_targets)
        out_dir = Path(best_path).parent
        torch.save({"T": T}, out_dir / "temperature.pt")
        print(f"  temperature = {T:.3f}")

        val_probs = F.softmax(val_logits / T, dim=-1).numpy()
        coverages = [float(c) for c in args.selective_coverages.split(",")]
        rows = selective_pr_curve(val_probs, val_targets.numpy(),
                                   data.num_classes, coverages)
        class_names = [data.idx_to_class[i] for i in range(data.num_classes)]
        write_markdown(out_dir / "selective_pr.md", rows, data.num_classes,
                        class_names)
        th = search_thresholds_macroP_coverage(
            val_probs, val_targets.numpy(), data.num_classes,
            target_coverage=args.target_coverage,
        )
        rf = search_thresholds_recall_floor(
            val_probs, val_targets.numpy(), data.num_classes,
            recall_floor=args.recall_floor,
        )
        with open(out_dir / "thresholds.json", "w") as f:
            json.dump({"macroP_coverage": th, "recall_floor": rf}, f, indent=2)
        print(f"  MP@cov{args.target_coverage} = {th['macro_precision']:.4f}")


if __name__ == "__main__":
    main()
