"""
Training script for HydroHydra — unified spectrogram-free UATR classifier.

Monitors ``val/macro_precision``; on completion, fits a temperature scalar and
sweeps per-class softmax thresholds for gated-precision reporting.  Saves
``temperature.pt`` and ``thresholds.json`` alongside the best checkpoint.

Usage:
    export DATA_DIR=/abs/path/to/Split1s
    python training/train_hydra.py --run_name hydro_hydra_s42 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import CSVLogger

from data.audio_lightning_loader import DALIAudioDataModule
from data.loader_factory import LOADER_CHOICES, build_loader
from models.hydro_hydra          import HydroHydra
from training.train_precise      import _collect_logits, _fit_temperature
from inference.calibrate         import (
    search_thresholds_macroP_coverage,
    search_thresholds_recall_floor,
)
from inference.selective_pr      import selective_pr_curve, write_markdown


def get_args():
    p = argparse.ArgumentParser(description="Train HydroHydra")

    # Data
    p.add_argument("--data_dir",        default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--num_threads",     type=int, default=8)
    p.add_argument("--no_oversample",   action="store_true")
    p.add_argument("--denoise",         default="off",
                   choices=["off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"])
    # Loader selection — DALI vs threaded backend, splitting vs non-splitting.
    # `dali` requires files at exactly --fixed_len samples; `*_split` chunk
    # long files at load time using --window_sec / --hop_sec. `denoise` is
    # only honoured by the `dali` loader.
    p.add_argument("--loader",          default="dali", choices=list(LOADER_CHOICES))
    p.add_argument("--window_sec",      type=float, default=None,
                   help="Splitting loaders only. Defaults to fixed_len/sample_rate.")
    p.add_argument("--hop_sec",         type=float, default=None,
                   help="Splitting loaders only. Defaults to window_sec.")
    p.add_argument("--num_workers",     type=int, default=8,
                   help="Threaded loaders only. PyTorch DataLoader workers.")
    # Specialist binary teacher: collapse all non-positive classes into "neg",
    # turning the K-way problem into a 2-class {pos, neg} problem. The
    # gamblers loss still emits a +1 abstain output per usual.
    p.add_argument("--positive_class",  default="",
                   help="If set (e.g. 'Cargo'), build a binary teacher that "
                        "treats the named class as positive and folds all "
                        "other classes into a single 'neg' class.")

    # Audio
    p.add_argument("--sample_rate",     type=int, default=5_120)
    p.add_argument("--fixed_len",       type=int, default=5_120)

    # Stream toggles
    p.add_argument("--use_gabor",       action="store_true", default=True)
    p.add_argument("--no_gabor",        dest="use_gabor", action="store_false")
    p.add_argument("--use_scattering",  action="store_true", default=True)
    p.add_argument("--no_scattering",   dest="use_scattering", action="store_false")
    p.add_argument("--use_sincnet",     action="store_true", default=False)
    p.add_argument("--use_tdsbe",       action="store_true", default=False)
    p.add_argument("--use_w2v",         action="store_true", default=False)
    # Phase I γ.1 — LPC residual branch
    p.add_argument("--use_lpc",         action="store_true", default=False)
    p.add_argument("--lpc_order",       type=int, default=12)
    p.add_argument("--lpc_frame",       type=int, default=256)
    p.add_argument("--lpc_hop",         type=int, default=128)
    p.add_argument("--lpc_ch",          type=int, default=64)
    # Phase I γ.2 — Recurrence plot branch
    p.add_argument("--use_rp",          action="store_true", default=False)
    p.add_argument("--rp_downsample",   type=int, default=1024)
    p.add_argument("--rp_dim",          type=int, default=3)
    p.add_argument("--rp_delay",        type=int, default=4)
    p.add_argument("--rp_eps_quantile", type=float, default=0.10)
    p.add_argument("--rp_ch",           type=int, default=64)

    # Stream widths
    p.add_argument("--gabor_n_filters", type=int, default=64)
    p.add_argument("--gabor_kernel",    type=int, default=257)
    p.add_argument("--gabor_ch",        type=int, default=128)
    p.add_argument("--scat_J",          type=int, default=6)
    p.add_argument("--scat_Q",          type=int, default=8)
    p.add_argument("--scat_ch",         type=int, default=128)
    p.add_argument("--use_jtfs",        action="store_true", default=False,
                   help="JTFS-lite: stack Δ and ΔΔ over scattering coeffs")
    p.add_argument("--sinc_n_filters",  type=int, default=64)
    p.add_argument("--sinc_kernel",     type=int, default=251)
    p.add_argument("--sinc_ch",         type=int, default=128)
    p.add_argument("--tdsbe_ch",        type=int, default=64)
    p.add_argument("--w2v_model",       default="facebook/wav2vec2-base")
    p.add_argument("--w2v_ch",          type=int, default=128)
    p.add_argument("--w2v_target_sr",   type=int, default=16_000)

    # Fusion + backbone
    p.add_argument("--fusion_T",        type=int, default=80)
    p.add_argument("--fusion_dim",      type=int, default=192)
    p.add_argument("--s4_n_blocks",     type=int, default=2)
    p.add_argument("--s4_d_state",      type=int, default=64)
    p.add_argument("--use_global_attn", action="store_true", default=False)
    p.add_argument("--global_attn_heads", type=int, default=4)
    p.add_argument("--dropout",         type=float, default=0.15)
    p.add_argument("--drop_path",       type=float, default=0.0,
                   help="Stochastic depth in SaShiMi blocks (Phase I α.3).")
    # Phase I α.2 — Stochastic Weight Averaging
    p.add_argument("--swa", action="store_true", default=False,
                   help="Enable SWA callback (Phase I α.2).")
    p.add_argument("--swa_lrs",         type=float, default=1e-4)
    p.add_argument("--swa_epoch_start", type=float, default=0.75,
                   help="As fraction (0,1] or absolute epoch index.")
    p.add_argument("--swa_anneal_epochs", type=int, default=10)

    # Head
    p.add_argument("--head_type",       default="mlp",
                   choices=["mlp", "mlp_wide", "cosine", "prototype",
                            "arcface", "subcenter_arcface", "demon_moe"])
    p.add_argument("--feature_norm",    default="none",
                   choices=["none", "layernorm_l2"])
    p.add_argument("--arcface_margin",  type=float, default=0.2)
    p.add_argument("--arcface_scale",   type=float, default=30.0)
    p.add_argument("--arcface_subcenters", type=int, default=1)
    p.add_argument("--moe_n_experts",   type=int, default=4)
    p.add_argument("--moe_gate_temperature", type=float, default=1.0)
    p.add_argument("--moe_aux_weight",  type=float, default=0.05)

    # Loss
    p.add_argument("--loss",            default="lmf", choices=["lmf", "focal", "ldam"])
    p.add_argument("--lmf_gamma",       type=float, default=2.0)
    p.add_argument("--lmf_margin",      type=float, default=0.5)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--gambler_o",       type=float, default=0.3)
    p.add_argument("--gambler_weight",  type=float, default=0.1)
    p.add_argument("--ldam_max_m",      type=float, default=0.5)
    p.add_argument("--ldam_s",          type=float, default=30.0)
    p.add_argument("--ldam_drw_epoch",  type=int,   default=40)
    p.add_argument("--ldam_drw_beta",   type=float, default=0.99999)

    # Aug
    p.add_argument("--noise_prob",      type=float, default=0.5)
    p.add_argument("--noise_snr_min",   type=float, default=15.0)
    p.add_argument("--noise_snr_max",   type=float, default=30.0)
    p.add_argument("--gain_prob",       type=float, default=0.5)
    p.add_argument("--gain_range",      type=float, default=0.3)

    # Domain-aware aug (V2 _WaveformAug)
    p.add_argument("--corpus_noise_prob",     type=float, default=0.0)
    p.add_argument("--corpus_noise_snr_min",  type=float, default=-3.0)
    p.add_argument("--corpus_noise_snr_max",  type=float, default=15.0)
    p.add_argument("--noise_pool_max",        type=int,   default=4096)
    p.add_argument("--noise_pool_quantile",   type=float, default=0.25)
    p.add_argument("--noise_pool_seed",       type=int,   default=0)
    p.add_argument("--rir_prob",              type=float, default=0.0)
    p.add_argument("--rir_max_delay_s",       type=float, default=0.030)
    p.add_argument("--pitch_prob",            type=float, default=0.0)
    p.add_argument("--pitch_range",           type=float, default=0.015)
    p.add_argument("--branch_dropout_p",      type=float, default=0.0)
    p.add_argument("--manifold_mixup_alpha",  type=float, default=0.0,
                   help="Beta(α,α) for manifold mixup at pooled features. 0 disables.")
    p.add_argument("--manifold_mixup_prob",   type=float, default=0.5)

    # Training
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--max_epochs",      type=int,   default=80)
    p.add_argument("--warmup_epochs",   type=int,   default=8)
    p.add_argument("--patience",        type=int,   default=15)
    p.add_argument("--precision",       default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--run_name",        default="hydro_hydra")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)

    # Post-hoc
    p.add_argument("--target_coverage", type=float, default=0.85)
    p.add_argument("--recall_floor",    type=float, default=0.6)
    p.add_argument("--selective_coverages", type=str,
                   default="0.70,0.75,0.80,0.85,0.90,0.95")
    p.add_argument("--skip_calibration", action="store_true")

    return p.parse_args()


def main(args=None):
    if args is None:
        args = get_args()

    if not args.data_dir:
        raise ValueError("Set --data_dir or export DATA_DIR")

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # Specialist binary mode: build a merge_classes dict that folds every
    # non-target class into a single 'neg' class.
    merge_classes = None
    if args.positive_class:
        all_cls = ("Cargo", "Passenger", "Tanker", "Tug")
        if args.positive_class not in all_cls:
            raise ValueError(
                f"--positive_class {args.positive_class} not in {all_cls}"
            )
        merge_classes = {c: "neg" for c in all_cls if c != args.positive_class}
        print(f"[binary teacher] positive='{args.positive_class}'  merge={merge_classes}")

    data = build_loader(
        args.loader,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_workers=args.num_workers,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        oversample_train=not args.no_oversample,
        denoise_method=args.denoise,
        merge_classes=merge_classes,
    )
    data.setup()

    # Per-class counts (needed for LDAM). Compute once from the active
    # train file scan; reuses the same merge logic as the ocean-noise pool.
    if data.train_files_override is not None:
        _train_by_cls = data._apply_merge(data.train_files_override)
    else:
        _train_by_cls = data._apply_merge(data._scan_split("train"))
    classes_sorted = sorted(_train_by_cls.keys(), key=lambda c: data.class_to_idx[c])
    cls_num_list = [len(_train_by_cls[c]) for c in classes_sorted]
    print(f"[cls_num_list] {dict(zip(classes_sorted, cls_num_list))}")

    # Build OceanNoisePool only if corpus noise is requested.
    ocean_noise_pool = None
    if args.corpus_noise_prob > 0.0:
        from processing.aug.ocean_noise import OceanNoisePool
        if data.train_files_override is not None:
            train_files_by_class = data._apply_merge(data.train_files_override)
        else:
            train_files_by_class = data._apply_merge(data._scan_split("train"))
        all_train_files = [p for fs in train_files_by_class.values() for p in fs]
        print(f"[ocean_noise_pool] scanning {len(all_train_files)} train files for "
              f"low-energy crops (q={args.noise_pool_quantile}, "
              f"max={args.noise_pool_max}) — built lazily on first batch")
        ocean_noise_pool = OceanNoisePool(
            files=all_train_files,
            target_sr=args.sample_rate,
            fixed_len=args.fixed_len,
            max_clips=args.noise_pool_max,
            energy_quantile=args.noise_pool_quantile,
            seed=args.noise_pool_seed,
        )

    model = HydroHydra(
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
        ocean_noise_pool=ocean_noise_pool,
        corpus_noise_prob=args.corpus_noise_prob,
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
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHydroHydra — {n_params / 1e6:.2f}M params  "
          f"[gabor={args.use_gabor}  scattering={args.use_scattering}]")
    print(f"Loss: {args.loss}  (margin={args.lmf_margin}, γ={args.lmf_gamma}, "
          f"gambler_w={args.gambler_weight})")

    ckpt_cb = ModelCheckpoint(
        monitor="val/macro_precision", mode="max", save_top_k=3,
        filename="hydra-{epoch:03d}-p{val/macro_precision:.4f}",
        auto_insert_metric_name=False, verbose=True,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val/macro_precision", patience=args.patience,
                      mode="max", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if args.swa:
        # Phase I α.2: Stochastic Weight Averaging targets the val/test gap
        # by averaging weights across late-epoch SGD iterates inside a flatter
        # basin than any single iterate.
        callbacks.append(StochasticWeightAveraging(
            swa_lrs=args.swa_lrs,
            swa_epoch_start=args.swa_epoch_start,
            annealing_epochs=args.swa_anneal_epochs,
        ))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision if torch.cuda.is_available() else 32,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=CSVLogger("lightning_logs", name=args.run_name),
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        deterministic=False,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    trainer.fit(model, data)

    # If SWA was used, the post-SWA averaged weights are in-memory now.
    # Save them as a separate ckpt before trainer.test reloads "best"
    # (which is the *pre-SWA* top-val ckpt — without this save the
    # averaged weights are lost).
    if args.swa:
        swa_dir = Path(ckpt_cb.dirpath if ckpt_cb.dirpath else f"lightning_logs/{args.run_name}/version_0/checkpoints")
        swa_path = swa_dir / "swa_final.ckpt"
        try:
            trainer.save_checkpoint(swa_path)
            print(f"\n[SWA] post-SWA averaged weights saved → {swa_path}")
        except Exception as e:
            print(f"\n[SWA] failed to save post-SWA ckpt: {e}")

    trainer.test(model, data, ckpt_path="best")

    best_path = ckpt_cb.best_model_path
    best_score = ckpt_cb.best_model_score
    print(f"\nBest checkpoint:       {best_path}")
    if best_score is not None:
        print(f"Best val/macro_prec:   {best_score:.4f}")

    # ── Post-hoc calibration + threshold sweep ───────────────────────────
    if not args.skip_calibration and best_path:
        print("\n── Post-hoc calibration + threshold sweep ──")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cal_model = HydroHydra.load_from_checkpoint(
            best_path, map_location=device, strict=False,
        ).to(device).eval()

        # _collect_logits slices to model.num_classes already, so this is
        # safe for both the legacy MLP+Gamblers head (num_classes+1) and
        # the build_head heads (num_classes).
        logits, targets = _collect_logits(cal_model, data.val_dataloader(), device)
        nc = data.num_classes
        T = _fit_temperature(logits, targets)
        probs_t = F.softmax(logits / T, dim=-1)
        print(f"  temperature    = {T:.3f}")

        probs_np   = probs_t.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        res_macro = search_thresholds_macroP_coverage(
            probs_np, targets_np,
            num_classes=nc, target_coverage=args.target_coverage,
        )
        print(f"  macroP@cov{args.target_coverage}: "
              f"macro_P={res_macro['macro_precision']:.4f}  "
              f"thresholds={['%.2f' % t for t in res_macro['thresholds']]}")

        res_floor = search_thresholds_recall_floor(
            probs_np, targets_np,
            num_classes=nc, recall_floor=args.recall_floor,
        )
        tag = "OK" if res_floor["feasible"] else "INFEASIBLE"
        print(f"  recall_floor={args.recall_floor}: {tag}  "
              f"micro_P={res_floor.get('micro_precision', 0):.4f}")

        coverages = [float(x) for x in args.selective_coverages.split(",") if x.strip()]
        sel_rows = selective_pr_curve(
            probs_np, targets_np, num_classes=nc, coverages=coverages,
        )
        class_names = [data.idx_to_class[i] for i in range(nc)]

        out_dir = Path(best_path).parent
        torch.save({"temperature": T}, out_dir / "temperature.pt")
        thr_payload = {
            "macroP_coverage": res_macro,
            "recall_floor":    res_floor,
        }
        (out_dir / "thresholds.json").write_text(json.dumps(thr_payload, indent=2))
        write_markdown(out_dir / "selective_pr.md", sel_rows,
                       num_classes=nc, class_names=class_names)
        print(f"  saved → {out_dir / 'temperature.pt'}")
        print(f"  saved → {out_dir / 'thresholds.json'}")
        print(f"  saved → {out_dir / 'selective_pr.md'}")

    return float(best_score) if best_score is not None else float("nan"), best_path


if __name__ == "__main__":
    main()
