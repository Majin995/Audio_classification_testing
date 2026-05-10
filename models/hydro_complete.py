"""
HydroComplete — Unified Hydra + Precise UATR classifier.

Combines the two production-tested ship configs into one model:

  * **HydroHydra Phase G R1** (waveform-only): Gabor + Scattering1D +
    SincNet + TDSBE → 1×1 fuse → SaShiMi (S4D) ×2 → AttentiveStatPool →
    LMF + Deep-Gamblers abstain head. Best single-seed val/μP=0.6908,
    test/μP=0.6620, post-cal MP@cov0.85=0.7518.

  * **HydroPrecise v1 verify-B** + **HydroPreciseV2** (spectrogram-domain):
    CQT+PCEN + multi-band DEMON+PCEN with SE-Res2 stacks, optional
    Gammatone / LOFAR / pretrained-w2v / boundary-aware attention.
    v1 verify-B ceiling 0.7444 with m=0.30 g=2.0 s=0.05 gw=0.0.

Default config enables the union of those two ship recipes; everything
else is a flag-gated ablation (Phase H showed compositional stacking can
hurt, so additions are opt-in).

Targets jointly: macro-precision (precision-aware abstain logit + LMF
margin + label smoothing) and macro-F1 (cross-domain stream diversity +
branch dropout + manifold mixup at the pooled feature, optional SupCon
auxiliary). Mean-Teacher EMA self-distillation is delegated to the
trainer (matches HydroPreciseV2's separation of concerns).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision,
    MulticlassRecall, MulticlassAUROC, MulticlassConfusionMatrix,
    MulticlassMatthewsCorrCoef,
)

from models.hydro_conformer import FocalLoss
from models.hydro_fusion    import _AttentiveStatisticsPool
from models.hydro_s4        import SaShiMiBlock
from models.hydro_bahtnet   import SpectralFluxOnset
from models.heads           import build_head
from processing.losses      import (
    LargeMarginFocalLoss, LDAMLoss, ClassBalancedFocalLoss,
)

# Reuse Hydra waveform streams.
from models.hydro_hydra import (
    _ScatteringStream, _SincNetStream, _TDSubBandEnvelopeStream,
    _LPCStream, _RecurrencePlotStream, _GlobalAttnBlock,
)

# Reuse V2 branches + augmentation infrastructure (waveform aug, branch
# dropout, SE-Res2 Gabor stack, CQT/DEMON/Gammatone/LOFAR/Pretrained,
# boundary-aware fused attn, supcon).
from models.hydro_precise_v2 import (
    _WaveformAug, _BranchDropout,
    _GaborBranch, _CQTBranch, _DEMONBranch,
    _GammatoneBranch, _LOFARBranch, _PretrainedBranch,
    _FusedAttnBlock, _supcon_loss,
)


# ═══════════════════════════════════════════════════════════════════════
#  HydroComplete
# ═══════════════════════════════════════════════════════════════════════

class HydroComplete(pl.LightningModule):
    """Unified multi-stream UATR classifier (Hydra ⊕ Precise).

    The default configuration enables the union of the two ship recipes:
    Gabor + Scattering + SincNet + TDSBE on the waveform side, CQT+PCEN
    + multi-band DEMON+PCEN on the spectrogram side, S4D ×2 backbone,
    boundary-aware fused attention, MLP head with Deep-Gamblers abstain
    logit, LMF loss, manifold mixup at the pooled feature.

    The MLP+abstain head emits ``num_classes + 1`` logits; all other heads
    emit ``num_classes`` logits (the abstain channel is dropped under
    those configs to avoid the ArcFace/Gambler interaction documented in
    HydroHydra).
    """

    # ── Construction ─────────────────────────────────────────────────────

    def __init__(
        self,
        num_classes:      int = 4,
        class_weights:    Optional[List[float]] = None,
        cls_num_list:     Optional[List[float]] = None,
        sample_rate:      int = 5_120,
        input_len:        int = 5_120,
        # ── Stream toggles ───────────────────────────────────────────────
        # Waveform-domain (Hydra R1)
        use_gabor:        bool = True,
        use_scattering:   bool = True,
        use_sincnet:      bool = True,
        use_tdsbe:        bool = True,
        use_lpc:          bool = False,
        use_rp:           bool = False,
        # Spectrogram-domain (Precise/V2)
        use_cqt:          bool = True,
        use_demon:        bool = True,
        use_gammatone:    bool = False,
        use_lofar:        bool = False,
        use_pretrained:   bool = False,
        # ── Branch hyperparams (waveform) ────────────────────────────────
        gabor_n_filters:  int = 96,
        gabor_kernel:     int = 257,
        gabor_ch:         int = 192,
        gabor_n_blocks:   int = 2,
        scat_J:           int = 6,
        scat_Q:           int = 8,
        scat_ch:          int = 128,
        use_jtfs:         bool = False,
        sinc_n_filters:   int = 64,
        sinc_kernel:      int = 251,
        sinc_ch:          int = 128,
        tdsbe_ch:         int = 64,
        lpc_order:        int = 12,
        lpc_frame:        int = 256,
        lpc_hop:          int = 128,
        lpc_ch:           int = 64,
        rp_downsample:    int = 1024,
        rp_dim:           int = 3,
        rp_delay:         int = 4,
        rp_eps_quantile:  float = 0.10,
        rp_ch:            int = 64,
        # ── Branch hyperparams (spectrogram) ─────────────────────────────
        cqt_n_bins:       int = 96,
        cqt_bpo:          int = 12,
        cqt_hop:          int = 64,
        cqt_ch:           int = 192,
        cqt_n_blocks:     int = 2,
        cqt_pcen:         bool = True,
        demon_hop:        int = 64,
        demon_ch:         int = 128,
        demon_n_blocks:   int = 1,
        demon_subbands:   Optional[List[Tuple[float, float]]] = None,
        demon_n_fft:      int = 2048,
        demon_mod_f_min:  float = 0.0,
        demon_mod_f_max:  float = 50.0,
        demon_envelope:   str   = "square",
        demon_decimate:   int   = 1,
        gammatone_n_bands: int = 64,
        gammatone_ch:     int = 128,
        gammatone_n_blocks: int = 1,
        lofar_n_bins:     int = 256,
        lofar_n_fft:      int = 4_096,
        lofar_hop:        int = 160,
        lofar_max_freq:   float = 2_560.0,
        lofar_ch:         int = 128,
        lofar_n_blocks:   int = 1,
        pretrained_model: str = "facebook/wav2vec2-base",
        pretrained_ch:    int = 128,
        pretrained_target_sr: int = 16_000,
        spec_aug_all_branches: bool = True,
        # ── Fusion ───────────────────────────────────────────────────────
        fusion_T:         int = 80,
        fusion_dim:       int = 256,
        n_attn_heads:     int = 4,
        n_attn_blocks:    int = 1,
        use_boundary_attn: bool = True,
        n_s4d_blocks:     int = 2,
        s4d_d_state:      int = 64,
        use_global_attn:  bool = False,
        global_attn_heads: int = 4,
        dropout:          float = 0.20,
        drop_path:        float = 0.05,
        # ── Head ─────────────────────────────────────────────────────────
        head_type:        str = "mlp",
        feature_norm:     str = "none",
        arcface_margin:   float = 0.2,
        arcface_scale:    float = 30.0,
        arcface_subcenters: int = 1,
        moe_n_experts:    int = 4,
        moe_gate_temperature: float = 1.0,
        moe_aux_weight:   float = 0.05,
        cosine_scale_init: float = 10.0,
        # ── Loss ─────────────────────────────────────────────────────────
        loss:             str = "lmf",      # "lmf" | "ldam" | "focal" | "cb_focal"
        lmf_gamma:        float = 2.0,
        lmf_margin:       float = 0.5,      # Phase G R1 winner
        focal_gamma:      float = 2.0,
        ldam_max_m:       float = 0.5,
        ldam_s:           float = 30.0,
        ldam_drw_epoch:   int   = 40,
        ldam_drw_beta:    float = 0.99999,
        cb_beta:          float = 0.999,
        label_smoothing:  float = 0.05,     # both verify-B and R1 use 0.05
        # ── Auxiliary objectives ─────────────────────────────────────────
        gambler_o:        float = 0.3,
        gambler_weight:   float = 0.1,      # R1 winner; verify-B used 0.0
        aux_supcon_weight: float = 0.0,
        supcon_temperature: float = 0.07,
        supcon_proj_dim:  int = 128,
        # ── Augmentation ─────────────────────────────────────────────────
        noise_prob:       float = 0.5,
        noise_snr_min:    float = 15.0,
        noise_snr_max:    float = 30.0,
        gain_prob:        float = 0.5,
        gain_range:       float = 0.3,
        ocean_noise_pool       = None,
        corpus_noise_prob:    float = 0.0,
        corpus_noise_snr_min: float = -3.0,
        corpus_noise_snr_max: float = 15.0,
        rir_prob:        float = 0.0,
        rir_max_delay_s: float = 0.030,
        pitch_prob:      float = 0.0,
        pitch_range:     float = 0.015,
        branch_dropout_p: float = 0.10,
        manifold_mixup_alpha: float = 0.2,
        manifold_mixup_prob:  float = 0.5,
        mixup_alpha:     float = 0.0,        # waveform-mixup; off by default
        # ── Optimiser ────────────────────────────────────────────────────
        learning_rate:   float = 3e-4,
        weight_decay:    float = 1e-2,
        warmup_epochs:   int = 8,
        max_epochs:      int = 100,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["class_weights", "cls_num_list", "ocean_noise_pool"],
        )

        if not (use_gabor or use_scattering or use_sincnet or use_tdsbe
                or use_lpc or use_rp
                or use_cqt or use_demon or use_gammatone
                or use_lofar or use_pretrained):
            raise ValueError(
                "HydroComplete needs at least one stream enabled "
                "(waveform: gabor/scattering/sincnet/tdsbe/lpc/rp; "
                "spectrogram: cqt/demon/gammatone/lofar/pretrained)."
            )

        if demon_subbands is None:
            demon_subbands = [(800.0, sample_rate / 2.0)]

        self.num_classes      = int(num_classes)
        self.fusion_T         = int(fusion_T)
        self.gambler_o        = float(gambler_o)
        self.head_type        = head_type
        self.aux_supcon_weight = float(aux_supcon_weight)
        self.supcon_temperature = float(supcon_temperature)
        self.manifold_mixup_alpha = float(manifold_mixup_alpha)
        self.manifold_mixup_prob  = float(manifold_mixup_prob)
        self.mixup_alpha          = float(mixup_alpha)

        # ArcFace/MoE × Gambler interaction is unresolved (see Hydra docs).
        # Force gambler_weight=0 if the head emits exactly num_classes.
        if head_type != "mlp" and gambler_weight > 0.0:
            gambler_weight = 0.0
        self.gambler_weight = float(gambler_weight)
        self._has_abstain_logit = (head_type == "mlp" and self.gambler_weight > 0.0)
        self.moe_aux_weight = float(moe_aux_weight)

        # ── Augmentation (waveform-domain) ───────────────────────────────
        self.wave_aug = _WaveformAug(
            noise_prob=noise_prob,
            noise_snr_min=noise_snr_min, noise_snr_max=noise_snr_max,
            gain_prob=gain_prob, gain_range=gain_range,
            ocean_noise_pool=ocean_noise_pool,
            corpus_noise_prob=corpus_noise_prob,
            corpus_noise_snr_min=corpus_noise_snr_min,
            corpus_noise_snr_max=corpus_noise_snr_max,
            rir_prob=rir_prob, rir_max_delay_s=rir_max_delay_s,
            sample_rate=sample_rate,
            pitch_prob=pitch_prob, pitch_range=pitch_range,
        )
        self.branch_drop = _BranchDropout(p=branch_dropout_p)

        # ── Streams ──────────────────────────────────────────────────────
        cat_ch = 0
        # Waveform-domain
        self.stream_gabor = (
            _GaborBranch(
                sample_rate=sample_rate, n_filters=gabor_n_filters,
                out_ch=gabor_ch, kernel_size=gabor_kernel,
                n_blocks=gabor_n_blocks, spec_aug=spec_aug_all_branches,
                dropout=dropout, drop_path=drop_path,
            ) if use_gabor else None
        )
        if use_gabor:
            cat_ch += gabor_ch

        self.stream_scattering = (
            _ScatteringStream(
                sample_rate=sample_rate, input_len=input_len,
                J=scat_J, Q=scat_Q, out_ch=scat_ch,
                use_jtfs=use_jtfs,
            ) if use_scattering else None
        )
        if use_scattering:
            cat_ch += scat_ch

        self.stream_sincnet = (
            _SincNetStream(
                sample_rate=sample_rate, n_filters=sinc_n_filters,
                kernel_size=sinc_kernel, out_ch=sinc_ch, dropout=dropout,
            ) if use_sincnet else None
        )
        if use_sincnet:
            cat_ch += sinc_ch

        self.stream_tdsbe = (
            _TDSubBandEnvelopeStream(
                sample_rate=sample_rate, out_ch=tdsbe_ch, dropout=dropout,
            ) if use_tdsbe else None
        )
        if use_tdsbe:
            cat_ch += tdsbe_ch

        self.stream_lpc = (
            _LPCStream(
                sample_rate=sample_rate, order=lpc_order,
                frame=lpc_frame, hop=lpc_hop, out_ch=lpc_ch, dropout=dropout,
            ) if use_lpc else None
        )
        if use_lpc:
            cat_ch += lpc_ch

        self.stream_rp = (
            _RecurrencePlotStream(
                input_len=input_len, downsample=rp_downsample,
                embed_dim=rp_dim, delay=rp_delay,
                eps_quantile=rp_eps_quantile, out_ch=rp_ch, dropout=dropout,
            ) if use_rp else None
        )
        if use_rp:
            cat_ch += rp_ch

        # Spectrogram-domain
        self.stream_cqt = (
            _CQTBranch(
                sample_rate=sample_rate, n_bins=cqt_n_bins,
                bins_per_octave=cqt_bpo, hop_length=cqt_hop,
                out_ch=cqt_ch, n_blocks=cqt_n_blocks,
                spec_aug=spec_aug_all_branches, use_pcen=cqt_pcen,
                dropout=dropout, drop_path=drop_path,
            ) if use_cqt else None
        )
        if use_cqt:
            cat_ch += cqt_ch

        if use_demon:
            self.stream_demon = _DEMONBranch(
                sample_rate=sample_rate, hop_length=demon_hop,
                out_ch=demon_ch, n_blocks=demon_n_blocks,
                subbands=demon_subbands, n_fft=demon_n_fft,
                mod_f_min=demon_mod_f_min, mod_f_max=demon_mod_f_max,
                envelope=demon_envelope, decimate=demon_decimate,
                spec_aug=spec_aug_all_branches,
                dropout=dropout, drop_path=drop_path,
            )
            cat_ch += self.stream_demon.out_ch
        else:
            self.stream_demon = None

        self.stream_gammatone = (
            _GammatoneBranch(
                sample_rate=sample_rate, n_bands=gammatone_n_bands,
                out_ch=gammatone_ch, n_blocks=gammatone_n_blocks,
                spec_aug=spec_aug_all_branches,
                dropout=dropout, drop_path=drop_path,
            ) if use_gammatone else None
        )
        if use_gammatone:
            cat_ch += gammatone_ch

        self.stream_lofar = (
            _LOFARBranch(
                sample_rate=sample_rate, n_bins=lofar_n_bins,
                n_fft=lofar_n_fft, hop_length=lofar_hop,
                max_freq=lofar_max_freq, out_ch=lofar_ch,
                n_blocks=lofar_n_blocks,
                spec_aug=spec_aug_all_branches,
                dropout=dropout, drop_path=drop_path,
            ) if use_lofar else None
        )
        if use_lofar:
            cat_ch += lofar_ch

        self.stream_pretrained = (
            _PretrainedBranch(
                sample_rate=sample_rate, out_ch=pretrained_ch,
                model_name=pretrained_model,
                spec_aug=spec_aug_all_branches,
                dropout=dropout, drop_path=drop_path,
                target_sr=pretrained_target_sr,
            ) if use_pretrained else None
        )
        if use_pretrained:
            cat_ch += pretrained_ch

        # ── Onset detector for boundary attention ────────────────────────
        self.onset = (
            SpectralFluxOnset(patch_size=1, std_mult=1.5)
            if use_boundary_attn else None
        )

        # ── Fusion ───────────────────────────────────────────────────────
        self.fuse_proj = nn.Sequential(
            nn.Conv1d(cat_ch, fusion_dim, 1, bias=False),
            nn.BatchNorm1d(fusion_dim), nn.GELU(),
        )

        # SaShiMi (S4D) backbone — Hydra's spine.
        self.s4d_blocks = nn.ModuleList([
            SaShiMiBlock(
                d_model=fusion_dim, d_state=s4d_d_state,
                expansion=4, dropout=dropout, drop_path=drop_path,
            )
            for _ in range(n_s4d_blocks)
        ])

        # Boundary-aware fused attention (Precise V2 layer).
        self.attn_blocks = nn.ModuleList([
            _FusedAttnBlock(
                d_model=fusion_dim, n_heads=n_attn_heads,
                dropout=dropout, use_boundary=use_boundary_attn,
            )
            for _ in range(n_attn_blocks)
        ])

        # Optional global attention block (HELIX-inspired).
        self.global_attn = (
            _GlobalAttnBlock(
                d_model=fusion_dim, n_heads=global_attn_heads, dropout=dropout,
            ) if use_global_attn else None
        )
        self.final_norm = nn.LayerNorm(fusion_dim)

        # ── Pool + (optional) feature norm + head ────────────────────────
        self.pool = _AttentiveStatisticsPool(fusion_dim)

        self.feature_norm_kind = feature_norm
        if feature_norm == "layernorm_l2":
            self.feat_norm = nn.LayerNorm(fusion_dim * 2)
        elif feature_norm == "none":
            self.feat_norm = None
        else:
            raise ValueError(f"Unknown feature_norm: {feature_norm!r}")

        if head_type == "mlp" and self._has_abstain_logit:
            # Legacy MLP head: emit num_classes+1 logits, last is abstain.
            self.head = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, num_classes + 1),
            )
        else:
            self.head = build_head(
                name=head_type,
                in_dim=fusion_dim * 2,
                num_classes=num_classes,
                fusion_dim=fusion_dim,
                dropout=dropout,
                arcface_margin=arcface_margin,
                arcface_scale=arcface_scale,
                arcface_subcenters=arcface_subcenters,
                cosine_scale_init=cosine_scale_init,
                moe_n_experts=moe_n_experts,
                moe_gate_temperature=moe_gate_temperature,
            )

        # ── SupCon projection head ───────────────────────────────────────
        if self.aux_supcon_weight > 0.0:
            self.supcon_head = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.GELU(),
                nn.Linear(fusion_dim, supcon_proj_dim),
            )
        else:
            self.supcon_head = None

        # ── Loss ─────────────────────────────────────────────────────────
        self.criterion = self._build_loss(
            loss=loss, num_classes=num_classes, class_weights=class_weights,
            cls_num_list=cls_num_list,
            lmf_gamma=lmf_gamma, lmf_margin=lmf_margin,
            focal_gamma=focal_gamma,
            ldam_max_m=ldam_max_m, ldam_s=ldam_s,
            cb_beta=cb_beta, label_smoothing=label_smoothing,
        )
        # LDAM/DRW bookkeeping (matches Hydra).
        self._cls_num_list = list(cls_num_list) if cls_num_list else None
        self._ldam_drw_epoch = int(ldam_drw_epoch)
        self._ldam_drw_beta  = float(ldam_drw_beta)

        # ── Metrics ──────────────────────────────────────────────────────
        m_macro = dict(num_classes=num_classes, average="macro")
        m_micro = dict(num_classes=num_classes, average="micro")
        self.train_acc            = MulticlassAccuracy(**m_macro)
        self.val_acc              = MulticlassAccuracy(**m_macro)
        self.val_f1               = MulticlassF1Score(**m_macro)
        self.val_recall           = MulticlassRecall(**m_macro)
        self.val_precision_macro  = MulticlassPrecision(**m_macro)
        self.val_precision_micro  = MulticlassPrecision(**m_micro)
        self.val_precision_per    = MulticlassPrecision(num_classes=num_classes, average=None)
        self.val_mcc              = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.val_auroc            = MulticlassAUROC(num_classes=num_classes)
        self.test_acc             = MulticlassAccuracy(**m_macro)
        self.test_f1              = MulticlassF1Score(**m_macro)
        self.test_precision       = MulticlassPrecision(**m_macro)
        self.test_precision_micro = MulticlassPrecision(**m_micro)
        self.test_recall          = MulticlassRecall(**m_macro)
        self.test_mcc             = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_auroc           = MulticlassAUROC(num_classes=num_classes)
        self.test_cm              = MulticlassConfusionMatrix(num_classes=num_classes)

    # ── Loss factory ────────────────────────────────────────────────────

    @staticmethod
    def _build_loss(*, loss, num_classes, class_weights, cls_num_list,
                    lmf_gamma, lmf_margin, focal_gamma,
                    ldam_max_m, ldam_s, cb_beta, label_smoothing):
        if loss == "lmf":
            return LargeMarginFocalLoss(
                num_classes=num_classes, alpha=class_weights,
                gamma=lmf_gamma, margin=lmf_margin,
                label_smoothing=label_smoothing,
            )
        if loss == "ldam":
            if cls_num_list is None:
                cls_num_list = [1.0] * num_classes
            return LDAMLoss(
                cls_num_list=cls_num_list,
                max_m=ldam_max_m, s=ldam_s,
                weight=None, label_smoothing=label_smoothing,
            )
        if loss == "cb_focal":
            if cls_num_list is None:
                cls_num_list = [1.0] * num_classes
            return ClassBalancedFocalLoss(
                cls_num_list=cls_num_list, beta=cb_beta,
                gamma=focal_gamma, label_smoothing=label_smoothing,
            )
        if loss == "focal":
            return FocalLoss(
                class_weights=class_weights,
                gamma=focal_gamma, label_smoothing=label_smoothing,
            )
        raise ValueError(f"Unknown loss: {loss!r}")

    # ── Forward ──────────────────────────────────────────────────────────

    def _stream_features(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Run all enabled streams; return their (B, C_i, T_i) outputs."""
        feats: List[torch.Tensor] = []
        # Waveform branches
        if self.stream_gabor is not None:
            feats.append(self.stream_gabor(waveform, self.training))
        if self.stream_scattering is not None:
            feats.append(self.stream_scattering(waveform))
        if self.stream_sincnet is not None:
            feats.append(self.stream_sincnet(waveform, self.training))
        if self.stream_tdsbe is not None:
            feats.append(self.stream_tdsbe(waveform, self.training))
        if self.stream_lpc is not None:
            feats.append(self.stream_lpc(waveform, self.training))
        if self.stream_rp is not None:
            feats.append(self.stream_rp(waveform, self.training))
        # Spectrogram branches
        if self.stream_cqt is not None:
            feats.append(self.stream_cqt(waveform, self.training))
        if self.stream_demon is not None:
            feats.append(self.stream_demon(waveform, self.training))
        if self.stream_gammatone is not None:
            feats.append(self.stream_gammatone(waveform, self.training))
        if self.stream_lofar is not None:
            feats.append(self.stream_lofar(waveform, self.training))
        if self.stream_pretrained is not None:
            feats.append(self.stream_pretrained(waveform, self.training))
        return feats

    def _onset_mask(self, waveform: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute an onset mask for boundary-aware attention.

        Uses spectral flux on the Gabor branch's input envelope when Gabor
        is enabled; otherwise returns None and downstream attention falls
        back to vanilla MHSA.
        """
        if self.onset is None or self.stream_gabor is None:
            return None
        # Run the gabor filterbank without spec-aug, take log envelope, feed
        # into the onset detector. SpectralFluxOnset returns a (B, T') mask.
        with torch.no_grad():
            g = self.stream_gabor.filterbank(waveform)
            g = torch.log1p(g.abs())
            mask = self.onset(g)
        return mask

    def _features(self, waveform: torch.Tensor) -> torch.Tensor:
        feats = self._stream_features(waveform)
        # Align all streams to the common fusion length T_f.
        feats = [F.adaptive_avg_pool1d(f, self.fusion_T) for f in feats]
        feats = self.branch_drop(feats)
        z = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]   # (B, sum_C, T_f)
        z = self.fuse_proj(z)                                          # (B, D, T_f)

        z = z.transpose(1, 2)                                          # (B, T_f, D)
        # S4D backbone first (sequence model), then attention re-weights.
        for blk in self.s4d_blocks:
            z = blk(z)
        # Onset-mask, downsampled to fusion_T.
        onset_mask = None
        if self.onset is not None:
            m = self._onset_mask(waveform)
            if m is not None:
                # Adaptive-pool the onset mask to the fusion grid.
                onset_mask = F.adaptive_max_pool1d(
                    m.float().unsqueeze(1), self.fusion_T,
                ).squeeze(1).bool()
        for blk in self.attn_blocks:
            z = blk(z, onset_mask=onset_mask)
        if self.global_attn is not None:
            z = self.global_attn(z)
        z = self.final_norm(z)
        z = z.transpose(1, 2)                                          # (B, D, T_f)

        feat = self.pool(z)                                            # (B, 2D)
        if self.feat_norm is not None:
            feat = self.feat_norm(feat)
            feat = F.normalize(feat, dim=-1)
        return feat

    def forward(self, waveform: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self._features(waveform)
        if isinstance(self.head, nn.Sequential):
            return self.head(feat)
        if self.training:
            return self.head(feat, labels)
        return self.head(feat, None)

    # ── Loss helpers ─────────────────────────────────────────────────────

    def _split_logits(self, logits: torch.Tensor):
        """Returns (class_logits, full_softmax_or_None)."""
        if self._has_abstain_logit:
            class_logits = logits[:, :self.num_classes]
            full = F.softmax(logits, dim=-1)
            return class_logits, full
        return logits, None

    def _gambler_loss(self, full: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p_y       = full.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_abstain = full[:, -1]
        return -torch.log(p_y + self.gambler_o * p_abstain + 1e-8).mean()

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor,
                      feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        class_logits, full = self._split_logits(logits)
        loss = self.criterion(class_logits, y)
        if self._has_abstain_logit and self.gambler_weight > 0.0:
            loss = loss + self.gambler_weight * self._gambler_loss(full, y)
        if (self.head_type == "demon_moe" and self.moe_aux_weight > 0.0
                and self.training):
            loss = loss + self.moe_aux_weight * self.head.aux_loss
        if (self.supcon_head is not None and feat is not None and self.training):
            proj = F.normalize(self.supcon_head(feat), dim=-1)
            loss = loss + self.aux_supcon_weight * _supcon_loss(
                proj, y, temperature=self.supcon_temperature,
            )
        return loss

    def _class_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, :self.num_classes] if self._has_abstain_logit else logits

    def _head_forward(self, feat: torch.Tensor,
                      labels: Optional[torch.Tensor]) -> torch.Tensor:
        if isinstance(self.head, nn.Sequential):
            return self.head(feat)
        return self.head(feat, labels)

    # ── Lightning steps ──────────────────────────────────────────────────

    def on_train_epoch_start(self):
        # Deferred Re-Weighting for LDAM (matches Hydra).
        if not isinstance(self.criterion, LDAMLoss):
            return
        if self._cls_num_list is None:
            return
        if self.criterion.weight is None and self.current_epoch >= self._ldam_drw_epoch:
            beta = self._ldam_drw_beta
            n = torch.tensor(self._cls_num_list, dtype=torch.float64)
            eff_n = (1.0 - beta ** n) / (1.0 - beta)
            if eff_n.std() / eff_n.mean().clamp(min=1e-9) < 1e-3:
                w = (n.sum() / (len(n) * n.clamp(min=1.0)))
            else:
                w = 1.0 / eff_n
                w = w * len(self._cls_num_list) / w.sum()
            self.criterion.weight = w.float().to(self.device)
            self.print(
                f"[LDAM-DRW] epoch={self.current_epoch}: swapped to weights "
                f"{w.tolist()}  (β={beta})",
                flush=True,
            )

    def _waveform_mixup(self, x: torch.Tensor, y: torch.Tensor):
        """Standard waveform-level mixup. Returns (x_mix, y_a, y_b, lam)."""
        if self.mixup_alpha <= 0.0:
            return x, y, y, 1.0
        lam = float(torch.distributions.Beta(
            self.mixup_alpha, self.mixup_alpha
        ).sample().clamp(min=0.05, max=0.95).item())
        perm = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.wave_aug(x)
        # Optional input-space mixup.
        x, y_a, y_b, lam_input = self._waveform_mixup(x, y)
        head_supports_feat_mixup = isinstance(self.head, nn.Sequential) \
            or self.head_type in ("mlp", "mlp_wide")
        do_feat_mixup = (
            self.manifold_mixup_alpha > 0.0
            and head_supports_feat_mixup
            and torch.rand(()).item() < self.manifold_mixup_prob
        )
        if do_feat_mixup:
            feat = self._features(x)
            lam = float(torch.distributions.Beta(
                self.manifold_mixup_alpha, self.manifold_mixup_alpha,
            ).sample().clamp(min=0.05, max=0.95).item())
            perm = torch.randperm(feat.size(0), device=feat.device)
            feat_mix = lam * feat + (1.0 - lam) * feat[perm]
            logits = self._head_forward(feat_mix, None)
            y_b2 = y_a[perm]
            loss = (
                lam * self._compute_loss(logits, y_a, feat=feat_mix)
                + (1.0 - lam) * self._compute_loss(logits, y_b2, feat=feat_mix)
            )
            class_logits = self._class_logits(logits)
            self.train_acc(class_logits, y_a)
        else:
            feat = self._features(x)
            logits = self._head_forward(feat, y_a)
            if lam_input < 1.0:
                loss = (
                    lam_input * self._compute_loss(logits, y_a, feat=feat)
                    + (1.0 - lam_input) * self._compute_loss(logits, y_b, feat=feat)
                )
            else:
                loss = self._compute_loss(logits, y_a, feat=feat)
            class_logits = self._class_logits(logits)
            self.train_acc(class_logits, y_a)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        feat   = self._features(x)
        logits = self._head_forward(feat, None)
        loss   = self._compute_loss(logits, y, feat=feat)
        class_logits = self._class_logits(logits)
        probs  = F.softmax(class_logits, dim=-1)

        self.val_acc(class_logits, y)
        self.val_f1(class_logits, y)
        self.val_recall(class_logits, y)
        self.val_precision_macro(class_logits, y)
        self.val_precision_micro(class_logits, y)
        self.val_precision_per(class_logits, y)
        self.val_mcc(class_logits, y)
        self.val_auroc(probs, y)

        self.log("val/loss",            loss,                     on_epoch=True, prog_bar=True)
        self.log("val/acc",             self.val_acc,             on_epoch=True, prog_bar=True)
        self.log("val/f1",              self.val_f1,              on_epoch=True, prog_bar=True)
        self.log("val/recall",          self.val_recall,          on_epoch=True)
        self.log("val/macro_precision", self.val_precision_macro, on_epoch=True, prog_bar=True)
        self.log("val/micro_precision", self.val_precision_micro, on_epoch=True, prog_bar=True)
        self.log("val/mcc",             self.val_mcc,             on_epoch=True)
        self.log("val/auroc",           self.val_auroc,           on_epoch=True)

    def on_validation_epoch_end(self):
        per = self.val_precision_per.compute()
        for i, v in enumerate(per):
            self.log(f"val/precision_c{i}", v, prog_bar=False)
        self.val_precision_per.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        feat   = self._features(x)
        logits = self._head_forward(feat, None)
        loss   = self._compute_loss(logits, y, feat=feat)
        class_logits = self._class_logits(logits)
        probs  = F.softmax(class_logits, dim=-1)

        self.test_acc(class_logits, y)
        self.test_f1(class_logits, y)
        self.test_precision(class_logits, y)
        self.test_precision_micro(class_logits, y)
        self.test_recall(class_logits, y)
        self.test_mcc(class_logits, y)
        self.test_auroc(probs, y)
        self.test_cm(class_logits, y)

        self.log("test/loss",            loss,                      on_epoch=True)
        self.log("test/acc",             self.test_acc,             on_epoch=True)
        self.log("test/f1",              self.test_f1,              on_epoch=True)
        self.log("test/macro_precision", self.test_precision,       on_epoch=True)
        self.log("test/micro_precision", self.test_precision_micro, on_epoch=True)
        self.log("test/recall",          self.test_recall,          on_epoch=True)
        self.log("test/mcc",             self.test_mcc,             on_epoch=True)
        self.log("test/auroc",           self.test_auroc,           on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix:\n{cm.cpu().numpy()}")
        self.test_cm.reset()

    # ── Optimiser ────────────────────────────────────────────────────────

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [{"params": decay,    "weight_decay": self.hparams.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.hparams.learning_rate, betas=(0.9, 0.98), eps=1e-8,
        )

        def lr_lambda(epoch):
            wu, total = self.hparams.warmup_epochs, self.hparams.max_epochs
            if epoch < wu:
                return (epoch + 1) / max(wu, 1)
            p = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
