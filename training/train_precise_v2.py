"""
Training script for HydroPreciseV2.

Same flow as train_precise.py (CSV logger, EarlyStopping, post-hoc temperature
+ per-class threshold calibration) plus:
  • SWA (Stochastic Weight Averaging) over the last fraction of epochs
  • LDAM / CB-Focal loss families (driven by --loss)
  • cls_num_list derived from the DataModule class_weights
  • All v2 architectural / training flags exposed via CLI

Mean-Teacher EMA + consistency, mixup, SupCon, logit adjustment all live inside
HydroPreciseV2 itself; this trainer just wires the CLI through.
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
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from data.audio_lightning_loader import DALIAudioDataModule
from data._subset                import scan_train_pool, stratified_init
from models.hydro_precise_v2     import HydroPreciseV2
from training.callbacks          import LatentMetricsCallback


# ═══════════════════════════════════════════════════════════════════════
#  Post-hoc calibration  (copied from train_precise.py)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_logits(model, dataloader, device):
    model.eval()
    logits_all, targets_all = [], []
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits_all.append(model(x).cpu())
        targets_all.append(y.cpu())
    return torch.cat(logits_all), torch.cat(targets_all)


def _fit_temperature(logits, targets, lr=1e-2, max_iter=200) -> float:
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


def _search_thresholds(probs, targets, num_classes, target_coverage=0.85):
    import numpy as np
    probs_np   = probs.numpy()
    targets_np = targets.numpy()
    argmax     = probs_np.argmax(axis=1)
    p_max      = probs_np.max(axis=1)
    grid       = np.linspace(0.0, 0.95, 20)

    def score(thr):
        keep = p_max >= thr[argmax]
        if keep.sum() == 0:
            return 0.0, 0.0
        preds, gts = argmax[keep], targets_np[keep]
        precs = []
        for c in range(num_classes):
            mask = preds == c
            if mask.sum() == 0:
                continue
            precs.append((gts[mask] == c).mean())
        if not precs:
            return 0.0, float(keep.mean())
        return float(np.mean(precs)), float(keep.mean())

    thr = np.zeros(num_classes)
    best_prec, best_cov = score(thr)
    best_thr = thr.copy()
    improved = True
    while improved:
        improved = False
        for c in range(num_classes):
            for g in grid:
                if g <= thr[c]:
                    continue
                cand = thr.copy(); cand[c] = g
                prec, cov = score(cand)
                if cov < target_coverage:
                    continue
                if prec > best_prec + 1e-6:
                    best_prec, best_cov, best_thr = prec, cov, cand
                    thr = cand
                    improved = True
    return {
        "thresholds":      best_thr.tolist(),
        "macro_precision": best_prec,
        "coverage":        best_cov,
        "target_coverage": target_coverage,
    }


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description="Train HydroPreciseV2")

    # Data
    p.add_argument("--data_dir",        default=os.environ.get("DATA_DIR", ""))
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--num_threads",     type=int, default=8)
    p.add_argument("--no_oversample",   action="store_true")
    p.add_argument("--denoise",         default="off",
                   choices=["off", "emd_wavelet", "nmf", "ica", "nmf_ica", "emd_nmf"])
    p.add_argument("--sample_rate",     type=int, default=5_120)
    p.add_argument("--fixed_len",       type=int, default=5_120)

    # Waveform preprocessing (sub-task β)
    p.add_argument("--rms_normalize",   action="store_true",
                   help="Per-clip zero-mean + unit-std + scale to target_rms "
                        "in DALI before yielding to the model.")
    p.add_argument("--target_rms",      type=float, default=0.1,
                   help="Target RMS magnitude when --rms_normalize is on. "
                        "Use 1.0 if pairing with the wav2vec2 pretrained branch.")
    p.add_argument("--hpf_hz",          type=float, default=0.0,
                   help="Cutoff Hz for a Butterworth-magnitude high-pass filter "
                        "applied via FFT-mask post-DALI. 0 disables.")
    p.add_argument("--hpf_order",       type=int,   default=4,
                   help="Butterworth HPF order (only used when --hpf_hz > 0).")

    # Train-pool subset (for cheap Optuna trials — borrows the AL framework's
    # stratified subset trick from train_active.py).
    p.add_argument("--train_subset_size", type=int, default=0,
                   help="If >0, train on a stratified subset of this many "
                        "files (split evenly across classes). Val/test "
                        "splits are unaffected. 0 = full train pool.")
    p.add_argument("--train_subset_per_class", type=int, default=0,
                   help="If >0, take exactly this many files per class "
                        "(overrides --train_subset_size).")
    p.add_argument("--train_subset_seed", type=int, default=0,
                   help="Seed for the subset draw (independent of --seed so "
                        "the same model seed can sweep across subset draws).")

    # Branches
    p.add_argument("--gabor_n_filters", type=int, default=96)
    p.add_argument("--gabor_kernel",    type=int, default=257)
    p.add_argument("--gabor_ch",        type=int, default=192)
    p.add_argument("--cqt_n_bins",      type=int, default=96)
    p.add_argument("--cqt_bpo",         type=int, default=12)
    p.add_argument("--cqt_hop",         type=int, default=64)
    p.add_argument("--cqt_ch",          type=int, default=192)
    p.add_argument("--no_pcen_on_cqt",  action="store_true")
    p.add_argument("--demon_hop",       type=int, default=64)
    p.add_argument("--demon_ch",        type=int, default=128)
    p.add_argument("--demon_subbands",  type=str, default="",
                   help='Semicolon-separated "lo-hi" Hz pairs, e.g. "600-1200;1200-2000;2000-2560". '
                        'Default empty → single (800, Nyquist) band.')
    p.add_argument("--demon_n_fft",     type=int, default=2048,
                   help="FFT size for the linear DEMON modulation spectrogram.")
    p.add_argument("--demon_mod_f_min", type=float, default=0.0,
                   help="Lower edge (Hz) of kept modulation band.")
    p.add_argument("--demon_envelope", default="square",
                   choices=["square", "hilbert", "fwr"],
                   help="DEMON envelope detector. 'square' = legacy back-compat; "
                        "'hilbert' = canonical analytic-signal magnitude; "
                        "'fwr' = full-wave rectification.")
    p.add_argument("--demon_decimate",  type=int,   default=1,
                   help="Anti-aliased decimation factor for the DEMON envelope. "
                        "Only meaningful when --demon_envelope != 'square'.")
    p.add_argument("--demon_mod_f_max", type=float, default=50.0,
                   help="Upper edge (Hz) of kept modulation band. Default 50 Hz "
                        "covers BPF + first few harmonics; raise toward 250 Hz "
                        "to test wider modulation content.")
    p.add_argument("--use_gammatone_branch", action="store_true")
    p.add_argument("--gammatone_n_bands", type=int, default=64)
    p.add_argument("--gammatone_ch",    type=int, default=128)
    p.add_argument("--use_lofar_branch", action="store_true",
                   help="Enable a 5th LOFAR branch (high-resolution linear-narrowband STFT).")
    p.add_argument("--lofar_n_bins",    type=int,   default=256)
    p.add_argument("--lofar_n_fft",     type=int,   default=4096)
    p.add_argument("--lofar_hop",       type=int,   default=160)
    p.add_argument("--lofar_max_freq",  type=float, default=2560.0)
    p.add_argument("--lofar_ch",        type=int,   default=128)
    p.add_argument("--lofar_n_blocks",  type=int,   default=1)
    p.add_argument("--use_pretrained_branch", action="store_true",
                   help="Enable a 6th branch using a frozen wav2vec2 conv "
                        "extractor as a pretrained acoustic prior.")
    p.add_argument("--pretrained_model", type=str,
                   default="facebook/wav2vec2-base",
                   help="HuggingFace model name (or local path) for the "
                        "pretrained branch's feature extractor.")
    p.add_argument("--pretrained_ch",    type=int,   default=128)
    p.add_argument("--pretrained_target_sr", type=int, default=16_000)
    p.add_argument("--seres2_blocks",   type=str, default="2,2,1,1",
                   help="Comma-separated SE-Res2 block counts: gabor,cqt,demon,gammatone")
    p.add_argument("--no_spec_aug_all", action="store_true")

    # Fusion
    p.add_argument("--fusion_T",        type=int, default=64)
    p.add_argument("--fusion_dim",      type=int, default=256)
    p.add_argument("--n_heads",         type=int, default=4)
    p.add_argument("--n_attn_blocks",   type=int, default=1)
    p.add_argument("--no_boundary_attn", action="store_true")
    p.add_argument("--use_dart_block",  action="store_true")
    p.add_argument("--n_s4d_blocks",    type=int, default=1)
    p.add_argument("--s4d_d_state",     type=int, default=64)
    p.add_argument("--dropout",         type=float, default=0.25)
    p.add_argument("--drop_path",       type=float, default=0.10)

    # Loss
    p.add_argument("--loss",            default="focal",
                   choices=["focal", "lmf", "ldam", "cb_focal"])
    p.add_argument("--focal_gamma",     type=float, default=2.0)
    p.add_argument("--lmf_margin",      type=float, default=0.5)
    p.add_argument("--ldam_max_m",      type=float, default=0.5)
    p.add_argument("--ldam_s",          type=float, default=30.0)
    p.add_argument("--cb_beta",         type=float, default=0.999)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--aux_supcon_weight", type=float, default=0.1)
    p.add_argument("--supcon_temp",     type=float, default=0.07)
    p.add_argument("--logit_adjust_tau", type=float, default=0.0)

    # Mixup
    p.add_argument("--mixup_alpha",     type=float, default=0.2)

    # Mean-Teacher
    p.add_argument("--mean_teacher_weight", type=float, default=0.5)
    p.add_argument("--mt_ema_decay",        type=float, default=0.999)
    p.add_argument("--mt_rampup_epochs",    type=int,   default=10)

    # Waveform aug
    p.add_argument("--noise_prob",      type=float, default=0.3)
    p.add_argument("--noise_snr_min",   type=float, default=15.0)
    p.add_argument("--noise_snr_max",   type=float, default=30.0)
    p.add_argument("--gain_prob",       type=float, default=0.6)
    p.add_argument("--gain_range",      type=float, default=0.3)

    # Domain-aware aug (sub-task α)
    p.add_argument("--corpus_noise_prob",     type=float, default=0.0,
                   help="Probability of mixing in a corpus-derived ambient "
                        "noise crop. 0 disables (no noise pool built).")
    p.add_argument("--corpus_noise_snr_min",  type=float, default=-3.0)
    p.add_argument("--corpus_noise_snr_max",  type=float, default=15.0)
    p.add_argument("--noise_pool_quantile",   type=float, default=0.25,
                   help="Energy quantile cutoff for the noise pool (lowest "
                        "fraction kept).")
    p.add_argument("--noise_pool_max",        type=int,   default=4096)
    p.add_argument("--noise_pool_seed",       type=int,   default=0)
    p.add_argument("--rir_prob",              type=float, default=0.0,
                   help="Probability of applying a random 3-5 tap multipath "
                        "convolution per batch. 0 disables.")
    p.add_argument("--rir_max_delay_s",       type=float, default=0.030)
    p.add_argument("--pitch_prob",            type=float, default=0.0,
                   help="Probability of ±pitch_range resample-pitch shift.")
    p.add_argument("--pitch_range",           type=float, default=0.015)
    p.add_argument("--branch_dropout_p",      type=float, default=0.0,
                   help="Per-batch probability of zeroing one branch's "
                        "pooled features in _features.")

    # Training
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-3)
    p.add_argument("--max_epochs",      type=int,   default=100)
    p.add_argument("--warmup_epochs",   type=int,   default=5)
    p.add_argument("--patience",        type=int,   default=20)
    p.add_argument("--precision",       default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--run_name",        default="hydro_precise_v2")
    p.add_argument("--limit_train_batches", type=float, default=1.0)
    p.add_argument("--limit_val_batches",   type=float, default=1.0)

    # SWA
    p.add_argument("--swa",             action="store_true",
                   help="Enable Stochastic Weight Averaging.")
    p.add_argument("--swa_start_frac",  type=float, default=0.8,
                   help="Start SWA at this fraction of max_epochs.")
    p.add_argument("--swa_lr",          type=float, default=1e-4)

    # Monitoring
    p.add_argument("--monitor",         default="val/micro_precision",
                   help="Metric for ModelCheckpoint + EarlyStopping. Examples: "
                        "val/micro_precision, val/macro_precision, val/f1, val/f1_p_score")

    # Latent-space diagnostics
    p.add_argument("--latent_metrics",  action="store_true",
                   help="Log silhouette / Fisher / k-NN purity / centroid cos sim "
                        "on val embeddings (requires sklearn).")
    p.add_argument("--latent_every_n",  type=int, default=1)
    p.add_argument("--latent_sample",   type=int, default=2000,
                   help="Subsample for the O(n²) silhouette and k-NN steps.")

    # Post-hoc
    p.add_argument("--target_coverage", type=float, default=0.85)
    p.add_argument("--skip_calibration", action="store_true")
    p.add_argument("--recall_floor",    type=float, default=0.6,
                   help="Per-class recall floor for the precision-targeted "
                        "threshold sweep (sub-task δ). Set 0 to skip.")
    p.add_argument("--selective_coverages", type=str,
                   default="0.70,0.75,0.80,0.85,0.90,0.95",
                   help="Comma-separated coverage targets for the selective-PR "
                        "curve (sub-task δ).")

    # Head + embedding-norm surface (ties to models/heads.py)
    p.add_argument("--head_type", default="mlp",
                   choices=["mlp", "cosine", "prototype", "arcface", "mlp_wide"])
    p.add_argument("--feature_norm", default="none",
                   choices=["none", "layernorm_l2"],
                   help="If 'layernorm_l2', apply LayerNorm + L2 to the post-pool "
                        "embedding before the head (and before SupCon).")
    p.add_argument("--arcface_margin",    type=float, default=0.2)
    p.add_argument("--arcface_scale",     type=float, default=30.0)
    p.add_argument("--cosine_scale_init", type=float, default=10.0)
    p.add_argument("--prototype_init",    default="random",
                   choices=["random", "centroid"],
                   help="If 'centroid', after model build run train embeddings "
                        "through _features and copy per-class centroids into the "
                        "PrototypeHead weight (only valid with --head_type prototype).")

    # Backbone init / freeze for head-only retraining
    p.add_argument("--init_from_ckpt", default="",
                   help="Lightning .ckpt to load weights from (strict=False). "
                        "Use with --freeze_backbone for head-only training.")
    p.add_argument("--resume_from_ckpt", default="",
                   help="Lightning .ckpt to resume training from. Restores "
                        "optimizer, scheduler, and epoch counter (true resume, "
                        "unlike --init_from_ckpt which only loads weights).")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze everything except head.* / feat_norm.* / supcon_head.* "
                        "and put backbone modules in eval() so BN running stats stay frozen.")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _parse_subbands(s: str):
    if not s.strip():
        return None
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        lo_str, hi_str = part.split("-")
        out.append((float(lo_str), float(hi_str)))
    return out


def _cls_num_list_from_weights(class_weights):
    """class_weights[i] = total / (num_classes * counts[i]) → counts ∝ 1/class_weights[i]."""
    inv = [1.0 / max(w, 1e-9) for w in class_weights]
    s = sum(inv)
    # Scale to a meaningful magnitude (won't matter for LDAM; matters for CB beta).
    # Treat as relative counts, multiplied by 1000 for numerical stability.
    scale = 1000.0 / s if s > 0 else 1.0
    return [v * scale for v in inv]


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

    train_files_override = None
    if args.train_subset_size > 0 or args.train_subset_per_class > 0:
        import random
        rng = random.Random(args.train_subset_seed)
        pool = scan_train_pool(Path(args.data_dir))
        train_files_override = stratified_init(
            pool,
            total=args.train_subset_size,
            per_class=args.train_subset_per_class,
            rng=rng,
        )
        n_subset = sum(len(v) for v in train_files_override.values())
        print(f"\n[train_subset] using {n_subset} files "
              f"(per-class: {[len(v) for v in train_files_override.values()]}) "
              f"seed={args.train_subset_seed}")

    data = DALIAudioDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        target_sr=args.sample_rate,
        fixed_len=args.fixed_len,
        oversample_train=not args.no_oversample,
        denoise_method=args.denoise,
        train_files_override=train_files_override,
        rms_normalize=args.rms_normalize,
        target_rms=args.target_rms,
        hpf_hz=args.hpf_hz,
        hpf_order=args.hpf_order,
    )
    data.setup()

    seres2 = tuple(int(v) for v in args.seres2_blocks.split(","))
    if len(seres2) != 4:
        raise ValueError("--seres2_blocks must have exactly 4 comma-separated ints")

    cls_num_list = _cls_num_list_from_weights(data.class_weights)

    # Build OceanNoisePool (lazy on cuda) only if corpus noise is requested.
    ocean_noise_pool = None
    if args.corpus_noise_prob > 0.0:
        from processing.aug.ocean_noise import OceanNoisePool
        # Pull the train file list from the datamodule. After data.setup() the
        # train pool is determined either by override or by scanning Split1s/train.
        if data.train_files_override is not None:
            train_files_by_class = data._apply_merge(data.train_files_override)
        else:
            train_files_by_class = data._apply_merge(data._scan_split("train"))
        # Cargo dominates by recording length so the lowest-energy quartile
        # is overwhelmingly background. Use ALL train files; the energy filter
        # picks the quiet crops regardless of class.
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

    model = HydroPreciseV2(
        num_classes=data.num_classes,
        class_weights=data.class_weights,
        cls_num_list=cls_num_list,
        sample_rate=args.sample_rate,
        # branches
        gabor_n_filters=args.gabor_n_filters, gabor_kernel=args.gabor_kernel,
        gabor_ch=args.gabor_ch,
        cqt_n_bins=args.cqt_n_bins, cqt_bpo=args.cqt_bpo,
        cqt_hop=args.cqt_hop, cqt_ch=args.cqt_ch,
        pcen_on_cqt=not args.no_pcen_on_cqt,
        demon_hop=args.demon_hop,
        demon_ch=args.demon_ch, demon_subbands=_parse_subbands(args.demon_subbands),
        demon_n_fft=args.demon_n_fft,
        demon_mod_f_min=args.demon_mod_f_min,
        demon_mod_f_max=args.demon_mod_f_max,
        demon_envelope=args.demon_envelope,
        demon_decimate=args.demon_decimate,
        use_gammatone_branch=args.use_gammatone_branch,
        gammatone_n_bands=args.gammatone_n_bands, gammatone_ch=args.gammatone_ch,
        use_lofar_branch=args.use_lofar_branch,
        lofar_n_bins=args.lofar_n_bins, lofar_n_fft=args.lofar_n_fft,
        lofar_hop=args.lofar_hop, lofar_max_freq=args.lofar_max_freq,
        lofar_ch=args.lofar_ch, lofar_n_blocks=args.lofar_n_blocks,
        use_pretrained_branch=args.use_pretrained_branch,
        pretrained_model=args.pretrained_model,
        pretrained_ch=args.pretrained_ch,
        pretrained_target_sr=args.pretrained_target_sr,
        seres2_blocks_per_branch=seres2,
        spec_aug_all_branches=not args.no_spec_aug_all,
        # fusion
        fusion_T=args.fusion_T, fusion_dim=args.fusion_dim,
        n_heads=args.n_heads, n_attn_blocks=args.n_attn_blocks,
        use_boundary_attn=not args.no_boundary_attn,
        use_dart_block=args.use_dart_block,
        n_s4d_blocks=args.n_s4d_blocks, s4d_d_state=args.s4d_d_state,
        dropout=args.dropout, drop_path=args.drop_path,
        # loss
        loss=args.loss, focal_gamma=args.focal_gamma,
        lmf_margin=args.lmf_margin, ldam_max_m=args.ldam_max_m,
        ldam_s=args.ldam_s, cb_beta=args.cb_beta,
        label_smoothing=args.label_smoothing,
        aux_supcon_weight=args.aux_supcon_weight,
        supcon_temperature=args.supcon_temp,
        logit_adjust_tau=args.logit_adjust_tau,
        # mixup
        mixup_alpha=args.mixup_alpha,
        # mean-teacher
        mean_teacher_weight=args.mean_teacher_weight,
        mean_teacher_ema_decay=args.mt_ema_decay,
        mean_teacher_rampup_epochs=args.mt_rampup_epochs,
        # wave aug
        noise_prob=args.noise_prob, noise_snr_min=args.noise_snr_min,
        noise_snr_max=args.noise_snr_max,
        gain_prob=args.gain_prob, gain_range=args.gain_range,
        # domain-aware aug
        ocean_noise_pool=ocean_noise_pool,
        corpus_noise_prob=args.corpus_noise_prob,
        corpus_noise_snr_min=args.corpus_noise_snr_min,
        corpus_noise_snr_max=args.corpus_noise_snr_max,
        rir_prob=args.rir_prob, rir_max_delay_s=args.rir_max_delay_s,
        pitch_prob=args.pitch_prob, pitch_range=args.pitch_range,
        branch_dropout_p=args.branch_dropout_p,
        # optim
        learning_rate=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
        # head + embedding norm
        head_type=args.head_type,
        feature_norm=args.feature_norm,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        cosine_scale_init=args.cosine_scale_init,
    )

    # ── Optional: load backbone from a previous ckpt and/or freeze it ──────
    if args.init_from_ckpt:
        sd = torch.load(args.init_from_ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        # Strip any keys that target a head/feat_norm shape we no longer have.
        own = model.state_dict()
        compat = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        missing = sorted(set(own) - set(compat))
        unexpected = sorted(set(sd) - set(compat))
        model.load_state_dict(compat, strict=False)
        print(f"\n[init_from_ckpt] loaded {len(compat)}/{len(sd)} tensors from "
              f"{args.init_from_ckpt}\n  missing  ({len(missing)}): "
              f"{missing[:6]}{'...' if len(missing)>6 else ''}\n  unexpected ({len(unexpected)}): "
              f"{unexpected[:6]}{'...' if len(unexpected)>6 else ''}")

        # If using a prototype head with centroid init, compute centroids from
        # train embeddings and seed the head weights.
        if args.head_type == "prototype" and args.prototype_init == "centroid":
            print("[prototype_init=centroid] computing per-class train centroids …")
            model.eval()
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(dev)
            sums = torch.zeros(data.num_classes, args.fusion_dim * 2, device=dev)
            counts = torch.zeros(data.num_classes, device=dev)
            with torch.no_grad():
                for batch in data.train_dataloader():
                    x, y = batch
                    feat = model._features(x.to(dev))
                    for c in range(data.num_classes):
                        m = (y == c)
                        if m.any():
                            sums[c] += feat[m.to(dev)].sum(dim=0)
                            counts[c] += int(m.sum())
            cents = sums / counts.clamp(min=1).unsqueeze(1)
            model.head.init_from_centroids(cents.cpu())
            print("[prototype_init=centroid] done.")

    if args.freeze_backbone:
        n_frozen = n_train = 0
        keep_prefix = ("head.", "feat_norm.", "supcon_head.")
        for name, p in model.named_parameters():
            if name.startswith(keep_prefix):
                n_train += p.numel()
            else:
                p.requires_grad_(False)
                n_frozen += p.numel()
        # Freeze BN running stats too: put backbone modules in eval(). The
        # head subtree stays trainable.
        for name, mod in model.named_modules():
            if not name:
                continue
            if any(name == p[:-1] or name.startswith(p) for p in keep_prefix):
                continue
            mod.eval()
        # Disable mean-teacher consistency (teacher would also be frozen).
        model.mean_teacher_weight = 0.0
        model._teacher = None
        print(f"\n[freeze_backbone] frozen={n_frozen/1e6:.2f}M  trainable={n_train/1e3:.1f}k")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nHydroPreciseV2 — {n_params / 1e6:.2f}M trainable parameters")
    print(f"Loss: {args.loss}  Mixup α={args.mixup_alpha}  "
          f"MT λ={args.mean_teacher_weight}  SupCon w={args.aux_supcon_weight}  "
          f"S4D={args.n_s4d_blocks}  Boundary={'on' if not args.no_boundary_attn else 'off'}  "
          f"Gammatone={'on' if args.use_gammatone_branch else 'off'}  "
          f"DART={'on' if args.use_dart_block else 'off'}")
    print(f"Head: {args.head_type}  feature_norm={args.feature_norm}  "
          f"freeze_backbone={args.freeze_backbone}  init_from_ckpt={'yes' if args.init_from_ckpt else 'no'}")

    # Sanitise the monitor metric for use in filenames (no slashes).
    safe_metric = args.monitor.replace("/", "_")
    ckpt_cb = ModelCheckpoint(
        monitor=args.monitor, mode="max", save_top_k=3,
        filename=f"precisev2-{{epoch:03d}}-{{{args.monitor}:.4f}}",
        auto_insert_metric_name=False, verbose=True,
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor=args.monitor, patience=args.patience,
                      mode="max", verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if args.swa:
        swa_start_epoch = max(1, int(args.max_epochs * args.swa_start_frac))
        callbacks.append(StochasticWeightAveraging(
            swa_lrs=args.swa_lr, swa_epoch_start=swa_start_epoch,
            annealing_epochs=max(1, args.max_epochs - swa_start_epoch),
        ))
    if args.latent_metrics:
        callbacks.append(LatentMetricsCallback(
            every_n_epochs=args.latent_every_n,
            sample_size=args.latent_sample,
        ))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
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

    fit_kwargs = {}
    if args.resume_from_ckpt:
        fit_kwargs["ckpt_path"] = args.resume_from_ckpt
        print(f"\n[resume] continuing from {args.resume_from_ckpt} "
              "(optimizer + scheduler + epoch state restored)")
    trainer.fit(model, data, **fit_kwargs)
    trainer.test(model, data, ckpt_path="best")

    best_path  = ckpt_cb.best_model_path
    best_score = ckpt_cb.best_model_score
    print(f"\nBest checkpoint:     {best_path}")
    if best_score is not None:
        print(f"Best {args.monitor}: {best_score:.4f}")
        # Keep the historical "Best val/micro_prec:" line so existing log greppers
        # (e.g. /tmp/v2_ablations/run_ablations.sh) still see the result.
        if args.monitor != "val/micro_precision":
            print(f"Best val/micro_prec: {best_score:.4f}  (monitor={args.monitor})")

    if not args.skip_calibration and best_path:
        print("\n── Post-hoc calibration + threshold sweep ──")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cal_model = HydroPreciseV2.load_from_checkpoint(
            best_path, map_location=device, strict=False,
        ).to(device).eval()

        logits, targets = _collect_logits(cal_model, data.val_dataloader(), device)
        T = _fit_temperature(logits, targets)
        probs = F.softmax(logits / T, dim=-1)
        print(f"  temperature    = {T:.3f}")

        # Legacy macro-P / coverage objective.
        from inference.calibrate import (
            search_thresholds_macroP_coverage,
            search_thresholds_recall_floor,
        )
        from inference.selective_pr import selective_pr_curve, write_markdown

        probs_np   = probs.cpu().numpy()
        targets_np = targets.cpu().numpy()

        res_macro = search_thresholds_macroP_coverage(
            probs_np, targets_np, num_classes=data.num_classes,
            target_coverage=args.target_coverage,
        )
        print(f"  [macroP@cov]   thresholds  = {['%.2f' % t for t in res_macro['thresholds']]}")
        print(f"                 macro_P     = {res_macro['macro_precision']:.4f}  "
              f"coverage = {res_macro['coverage']:.4f}")

        # Precision-targeted recall-floor objective (sub-task δ).
        res_floor = None
        if args.recall_floor > 0:
            res_floor = search_thresholds_recall_floor(
                probs_np, targets_np, num_classes=data.num_classes,
                recall_floor=args.recall_floor,
            )
            tag = "OK" if res_floor["feasible"] else "INFEASIBLE"
            print(f"  [recall_floor={args.recall_floor:.2f}]  thresholds = "
                  f"{['%.2f' % t for t in res_floor['thresholds']]}  ({tag})")
            if res_floor["feasible"]:
                print(f"                 micro_P    = {res_floor['micro_precision']:.4f}  "
                      f"coverage = {res_floor['coverage']:.4f}  "
                      f"per_class_recall = "
                      f"{['%.2f' % r for r in res_floor['per_class_recall']]}")

        # Selective-PR curve (sub-task δ).
        coverages = [float(x) for x in args.selective_coverages.split(",") if x.strip()]
        sel_rows = selective_pr_curve(
            probs_np, targets_np, num_classes=data.num_classes,
            coverages=coverages,
        )

        out_dir = Path(best_path).parent
        torch.save({"temperature": T}, out_dir / "temperature.pt")
        thr_payload = {"macroP_coverage": res_macro}
        if res_floor is not None:
            thr_payload["recall_floor"] = res_floor
        with open(out_dir / "thresholds.json", "w") as f:
            json.dump(thr_payload, f, indent=2)
        try:
            class_names = [data.idx_to_class[i] for i in range(data.num_classes)]
        except Exception:
            class_names = None
        write_markdown(out_dir / "selective_pr.md", sel_rows,
                       num_classes=data.num_classes, class_names=class_names)
        print(f"  saved → {out_dir / 'temperature.pt'}")
        print(f"  saved → {out_dir / 'thresholds.json'}")
        print(f"  saved → {out_dir / 'selective_pr.md'}")

    return float(best_score) if best_score is not None else float("nan"), best_path


if __name__ == "__main__":
    main()
