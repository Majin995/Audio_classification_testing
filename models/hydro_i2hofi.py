"""
HydroI2HOFI — PyTorch / Lightning port of I2-HOFI.

Original paper:
  "I2-HOFI: Intra-Inter Region Graph Neural Network for Fine-Grained Recognition"
  Pal et al., International Journal of Computer Vision, 2024.
  https://github.com/Arindam-1991/I2-HOFI

Two input modes
---------------
  Image mode  (waveform_input=False, default)
    RGB image  (B, 3, H, W)  →  backbone  →  ROI GNN  →  classifier

  Waveform mode  (waveform_input=True)
    Raw waveform  (B, T)
        ↓
    WaveformFrontend
      MelSpectrogram (n_fft=32_000, n_mels=8_092, hop_length=160)
      log1p compression  →  InstanceNorm2d
      Conv2d(1→3, 1×1) learned channel projection
        ↓  (B, 3, 8_092, T_frames)
    backbone  →  ROI GNN  →  classifier

    The very high n_mels (8 092) gives near-STFT frequency resolution.
    The ROI grid partitions the spectrogram into frequency-band regions;
    intra-ROI APPNP models fine structure within each band, inter-ROI GAT
    models cross-band interactions — an acoustic analogue of the spatial
    region graph in the original paper.

Architecture (shared path after frontend)
------------------------------------------
  (B, 3, H, W)
      ↓
  CNN Backbone (torchvision pretrained)  →  feat_map  (B, C, Hf, Wf)
      ↓
  ROI Grid Extraction
    divide feat_map into grid_h × grid_w regions,
    AdaptivePool each to pool_h × pool_w          →  (B, n_rois, C, pool_h, pool_w)
      ↓
  Intra-ROI branch (APPNP)
    pool_h*pool_w spatial positions per ROI as a graph,
    project C → gcn_out, run K APPNP propagation steps,
    aggregate via GlobalAttentionPool              →  (B, n_rois, gcn_out)
      ↓
  Inter-ROI branch (GAT)
    n_rois ROI vectors as a grid graph,
    single-layer multi-head GAT,
    aggregate via GlobalAttentionPool              →  (B, gat_out)
      ↓
  Concat [intra_global, inter_global]
      ↓
  Classifier  Linear → BN → GELU → Dropout → Linear  →  (B, num_classes)

Key differences from the TF / Spektral original
-------------------------------------------------
  • Pure PyTorch — no TF / Keras / Spektral dependency
  • torchvision backbones instead of tf.keras.applications
  • APPNP and GAT reimplemented from scratch (small graphs → no torch_geometric needed)
  • LightningModule wrapper with AdamW + cosine-warmup, FocalLoss, torchmetrics
  • Optional waveform input with high-resolution mel spectrogram frontend
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as tvm
import torchaudio.transforms as TA
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassMatthewsCorrCoef,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
)

from .hydro_conformer import FocalLoss


# ═══════════════════════════════════════════════════════════════════════════════
#  Waveform → spectrogram frontend
# ═══════════════════════════════════════════════════════════════════════════════

class WaveformFrontend(nn.Module):
    """
    Convert a raw waveform to a 3-channel pseudo-image for the CNN backbone.

    Pipeline
    --------
      (B, T)  waveform  at  sample_rate  Hz
          ↓
      MelSpectrogram(n_fft, hop_length, n_mels)   →  (B, n_mels, T_frames)
          ↓
      log1p compression                            →  (B, n_mels, T_frames)
          ↓
      unsqueeze → InstanceNorm2d                   →  (B, 1, n_mels, T_frames)
          ↓
      Conv2d(1, 3, 1×1) — learned channel proj.   →  (B, 3, n_mels, T_frames)

    The very high default n_mels (8 092) gives near-STFT frequency resolution.
    InstanceNorm2d normalises each spectrogram individually, removing any
    global amplitude bias before the backbone.

    Args:
        sample_rate: audio sample rate in Hz (default 32 000)
        n_fft:       FFT window length in samples (default 32 000 = 1 s at 32 kHz)
        hop_length:  STFT hop in samples (default 160 = 5 ms at 32 kHz)
        n_mels:      number of mel filterbank bins (default 8 092)
        f_min:       lowest mel filter centre frequency in Hz
        f_max:       highest mel filter centre frequency in Hz (None = Nyquist)
    """

    def __init__(
        self,
        sample_rate: int   = 32_000,
        n_fft:       int   = 32_000,
        hop_length:  int   = 160,
        n_mels:      int   = 8_092,
        f_min:       float = 0.0,
        f_max:       Optional[float] = None,
    ):
        super().__init__()

        # torchaudio MelSpectrogram returns power spectrum (magnitude²)
        self.mel = TA.MelSpectrogram(
            sample_rate = sample_rate,
            n_fft       = n_fft,
            hop_length  = hop_length,
            n_mels      = n_mels,
            f_min       = f_min,
            f_max       = f_max if f_max is not None else sample_rate / 2,
            power       = 2.0,           # power spectrogram
            normalized  = False,
            norm        = "slaney",      # area-normalised mel filters
            mel_scale   = "slaney",
        )

        # Per-sample normalisation: zero-mean / unit-var in (freq, time) dims
        self.norm = nn.InstanceNorm2d(1, affine=True)

        # Learn the mapping from the single-channel spectrogram to 3 channels
        # that the pretrained RGB backbone expects.
        # kernel_size=1 so spatial structure is preserved exactly.
        self.channel_proj = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        nn.init.ones_(self.channel_proj.weight)
        nn.init.zeros_(self.channel_proj.bias)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T)  float32 at self.sample_rate Hz

        Returns:
            (B, 3, n_mels, T_frames)  ready for backbone
        """
        # (B, n_mels, T_frames)
        spec = self.mel(waveform)

        # Compressive log normalisation — avoids vanishing gradients from
        # the very large dynamic range in raw power spectrograms.
        spec = torch.log1p(spec)

        # (B, 1, n_mels, T_frames) → InstanceNorm → (B, 1, n_mels, T_frames)
        spec = self.norm(spec.unsqueeze(1))

        # (B, 3, n_mels, T_frames)
        return self.channel_proj(spec)


class STFTLOFARFrontend(nn.Module):
    """
    Raw waveform → log-power STFT (LOFARgram) → 3-channel pseudo-image.

    Unlike WaveformFrontend which uses a mel filterbank, this keeps the linear
    STFT bin axis so that 1 STFT bin == ``sample_rate / n_fft`` Hz exactly.
    The frequency axis is band-limited to ``[0, stft_f_max_hz]`` then resized
    (via adaptive avg-pool / bilinear interpolation) to ``output_hw`` so the
    backbone receives a tensor with enough spatial extent for its downsampling
    factor.  ``output_hw`` should be set so that ``H / backbone_stride >=
    grid_h`` and ``W / backbone_stride >= grid_w`` (32× for ResNet/EfficientNet).
    """

    def __init__(
        self,
        sample_rate:     int   = 5_120,
        n_fft:           int   = 5_120,
        hop_length:      int   = 160,
        win_length:      Optional[int] = None,
        stft_f_max_hz:   float = 2_560.0,
        output_hw:       Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.sample_rate   = sample_rate
        self.stft_f_max_hz = stft_f_max_hz
        self.output_hw     = output_hw

        # Hann window over win_length samples (= 1 full snippet for 1 s clips
        # at sample_rate Hz), then zero-padded to n_fft for FFT.  This gives
        # the maximum real frequency resolution (1/T Hz) for the snippet
        # length, with optional zero-pad interpolation when n_fft > win_length.
        self.spec = TA.Spectrogram(
            n_fft       = n_fft,
            hop_length  = hop_length,
            win_length  = win_length if win_length is not None else n_fft,
            window_fn   = torch.hann_window,
            power       = 2.0,
            normalized  = False,
            center      = True,
        )

        # rfft gives n_fft//2 + 1 bins; band-limit by hz/bin = sample_rate/n_fft.
        hz_per_bin = sample_rate / n_fft
        n_bins_full = n_fft // 2 + 1
        self.n_keep = min(int(round(stft_f_max_hz / hz_per_bin)) + 1, n_bins_full)

        self.norm         = nn.InstanceNorm2d(1, affine=True)
        self.channel_proj = nn.Conv2d(1, 3, kernel_size=1, bias=True)
        nn.init.ones_(self.channel_proj.weight)
        nn.init.zeros_(self.channel_proj.bias)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.spec(waveform)                       # (B, n_bins_full, T_frames)
        spec = spec[:, : self.n_keep, :]                 # band-limit to f_max
        spec = torch.log1p(spec)
        spec = spec.unsqueeze(1)                         # (B, 1, n_freq, T_frames)
        H_out, W_out = self.output_hw
        # Use bilinear up-sample when output > input on either axis, otherwise
        # adaptive avg-pool (anti-aliasing for downsampling).
        if spec.shape[-2] >= H_out and spec.shape[-1] >= W_out:
            spec = F.adaptive_avg_pool2d(spec, (H_out, W_out))
        else:
            spec = F.interpolate(spec, size=(H_out, W_out), mode="bilinear",
                                 align_corners=False)
        spec = self.norm(spec)
        return self.channel_proj(spec)                   # (B, 3, H_out, W_out)


# ═══════════════════════════════════════════════════════════════════════════════
#  Backbone factory
# ═══════════════════════════════════════════════════════════════════════════════

_BACKBONE_REGISTRY = {
    "resnet18":        (tvm.resnet18,        tvm.ResNet18_Weights.DEFAULT,         512),
    "resnet50":        (tvm.resnet50,        tvm.ResNet50_Weights.DEFAULT,        2048),
    "resnet101":       (tvm.resnet101,       tvm.ResNet101_Weights.DEFAULT,       2048),
    "resnet152":       (tvm.resnet152,       tvm.ResNet152_Weights.DEFAULT,       2048),
    "densenet121":     (tvm.densenet121,     tvm.DenseNet121_Weights.DEFAULT,     1024),
    "densenet169":     (tvm.densenet169,     tvm.DenseNet169_Weights.DEFAULT,     1664),
    "densenet201":     (tvm.densenet201,     tvm.DenseNet201_Weights.DEFAULT,     1920),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.DEFAULT, 1280),
    "efficientnet_b4": (tvm.efficientnet_b4, tvm.EfficientNet_B4_Weights.DEFAULT, 1792),
    "efficientnet_b7": (tvm.efficientnet_b7, tvm.EfficientNet_B7_Weights.DEFAULT, 2560),
}


def _build_backbone(
    name:       str,
    pretrained: bool = True,
    freeze:     bool = False,
) -> Tuple[nn.Module, int]:
    """
    Return (feature_extractor, out_channels).

    The feature extractor outputs (B, out_channels, Hf, Wf) —
    the spatial feature map before global pooling.

    Supported names: resnet18, resnet50, resnet101, resnet152,
                     densenet121, densenet169, densenet201,
                     efficientnet_b0, efficientnet_b4, efficientnet_b7
    """
    name = name.lower().replace("-", "_")
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Supported: {sorted(_BACKBONE_REGISTRY.keys())}"
        )
    factory, weights, feat_ch = _BACKBONE_REGISTRY[name]
    model = factory(weights=weights if pretrained else None)

    if name.startswith("resnet"):
        # Drop avgpool + fc — keeps (B, feat_ch, 7, 7) for 224-px input
        backbone = nn.Sequential(*list(model.children())[:-2])
    elif name.startswith("densenet"):
        # model.features is the conv tower; add ReLU missing at end of DenseNet
        backbone = nn.Sequential(model.features, nn.ReLU(inplace=True))
    else:  # efficientnet
        # model.features outputs (B, feat_ch, 7, 7) for 224-px input
        backbone = model.features

    if freeze:
        for p in backbone.parameters():
            p.requires_grad_(False)

    return backbone, feat_ch


# ═══════════════════════════════════════════════════════════════════════════════
#  Adjacency helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_normalized_adj(n: int, mode: str = "full") -> torch.Tensor:
    """
    Build symmetric normalized adjacency  Â = D^{-½} (A + I) D^{-½}.

    Args:
        n:    number of nodes
        mode: "full"  — fully connected (complete graph)
              "grid"  — 4-connected 2D grid (n must be a perfect square)
              "grid8" — 8-connected 2D grid (n must be a perfect square)

    Returns:
        Tensor of shape (n, n), values in [0, 1].
    """
    A = torch.zeros(n, n)

    if mode == "full":
        A = torch.ones(n, n)

    elif mode in ("grid", "grid8"):
        side = int(math.isqrt(n))
        assert side * side == n, f"grid mode requires n to be a perfect square, got {n}"
        for i in range(side):
            for j in range(side):
                node = i * side + j
                neighbors = []
                if mode == "grid8":
                    offsets = [(-1, -1), (-1, 0), (-1, 1),
                               (0,  -1),           (0,  1),
                               (1,  -1),  (1, 0),  (1,  1)]
                else:
                    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for di, dj in offsets:
                    ni_, nj_ = i + di, j + dj
                    if 0 <= ni_ < side and 0 <= nj_ < side:
                        neighbors.append(ni_ * side + nj_)
                for nb in neighbors:
                    A[node, nb] = 1.0

    else:
        raise ValueError(f"Unknown adjacency mode '{mode}'")

    # Add self-loops
    A = A + torch.eye(n)

    # Symmetric normalisation: Â = D^{-½} A D^{-½}
    deg  = A.sum(dim=1)                           # (n,)
    d_inv_sqrt = torch.pow(deg.clamp(min=1e-9), -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)            # (n, n)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt            # (n, n)

    return A_hat


# ═══════════════════════════════════════════════════════════════════════════════
#  GNN layers
# ═══════════════════════════════════════════════════════════════════════════════

class APPNPLayer(nn.Module):
    """
    APPNP — Approximate Personalized PageRank propagation.

    Projects node features (in_features → out_features) with a learned linear
    transform + activation, then runs K propagation steps:

        H^(0) = σ(W x)
        H^(k) = (1 − α) Â H^(k−1) + α H^(0)

    Reference:
        Klicpera et al. "Predict then Propagate: Graph Neural Networks meet
        Personalized PageRank." ICLR 2019.

    Args:
        in_features:  input feature dimension per node
        out_features: output feature dimension per node
        K:            propagation steps
        alpha:        teleport (restart) probability — keeps initial representation
        dropout:      node feature dropout applied before each propagation step
        activation:   activation on the initial linear projection
    """

    _ACT = {
        "sigmoid": nn.Sigmoid,
        "relu":    nn.ReLU,
        "elu":     nn.ELU,
        "gelu":    nn.GELU,
        "tanh":    nn.Tanh,
    }

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        K:            int   = 3,
        alpha:        float = 0.3,
        dropout:      float = 0.2,
        activation:   str   = "sigmoid",
    ):
        super().__init__()
        self.K     = K
        self.alpha = alpha

        self.proj    = nn.Linear(in_features, out_features)
        self.act     = self._ACT.get(activation, nn.Sigmoid)()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, N, in_features)  — batched node features
            adj_hat: (N, N)               — precomputed normalised adjacency

        Returns:
            (B, N, out_features)
        """
        h0 = self.act(self.proj(x))       # (B, N, out_features)
        h  = h0
        for _ in range(self.K):
            h = self.dropout(h)
            # (B, N, F) ← Â (N,N) @ (B, N, F)  — broadcast over batch
            # A (N,N) @ H (B,N,F) → (B,N,F): sum over source-node axis j
            h_prop = torch.einsum("ij,bjf->bif", adj_hat, h)
            h      = (1.0 - self.alpha) * h_prop + self.alpha * h0
        return h


class GATLayer(nn.Module):
    """
    Single-layer Graph Attention Network.

    h'_i = act( sum_j  α_{ij}  W h_j )
    where  α_{ij} = softmax_j( LeakyReLU( a^T [ W h_i ‖ W h_j ] ) )
    and masked to zero for non-adjacent pairs.

    Multi-head outputs are concatenated (not averaged) to give out_features total.

    Reference:
        Veličković et al. "Graph Attention Networks." ICLR 2018.

    Args:
        in_features:  input feature dimension
        out_features: total output features (must be divisible by heads)
        heads:        number of attention heads
        dropout:      attention coefficient dropout
        activation:   per-head nonlinearity ('elu', 'relu', etc.)
        concat_heads: if True concatenate head outputs; if False average them
    """

    _ACT = {
        "sigmoid": nn.Sigmoid,
        "relu":    nn.ReLU,
        "elu":     nn.ELU,
        "gelu":    nn.GELU,
    }

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        heads:        int   = 1,
        dropout:      float = 0.2,
        activation:   str   = "elu",
        concat_heads: bool  = True,
    ):
        super().__init__()
        assert out_features % heads == 0, \
            f"out_features ({out_features}) must be divisible by heads ({heads})"
        self.heads        = heads
        self.concat_heads = concat_heads
        self.d            = out_features // heads          # features per head

        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention vector: one per head, operates on [h_i ‖ h_j]  (size 2 * d)
        self.a        = nn.Parameter(torch.empty(heads, 2 * self.d))
        self.leaky    = nn.LeakyReLU(negative_slope=0.2)
        self.dropout  = nn.Dropout(dropout)
        self.act      = self._ACT.get(activation, nn.ELU)()

        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, N, in_features)
            adj: (N, N)  — binary adjacency (0/1); non-edges are masked to −∞

        Returns:
            (B, N, out_features)
        """
        B, N, _ = x.shape
        H = self.heads
        d = self.d

        # Linear transform → reshape for multi-head
        h = self.W(x).view(B, N, H, d)          # (B, N, H, d)

        # Pairwise attention logits
        # h_i: (B, N, 1, H, d) → broadcast to (B, N, N, H, d)
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)
        cat = torch.cat([h_i, h_j], dim=-1)      # (B, N, N, H, 2d)

        # e_{ij} = LeakyReLU( a^T [h_i ‖ h_j] )
        # self.a: (H, 2d) → (1, 1, 1, H, 2d)
        e = self.leaky(
            (cat * self.a.view(1, 1, 1, H, 2 * d)).sum(dim=-1)
        )                                         # (B, N, N, H)

        # Mask non-edges to −∞ so they vanish under softmax
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)   # (1, N, N, 1)
        e    = e.masked_fill(mask, float("-inf"))

        alpha = F.softmax(e, dim=2)               # (B, N, N, H)
        alpha = self.dropout(alpha)

        # Aggregation: sum_j α_{ij} h_j   per head
        # h: (B, N, H, d) → (B, 1, N, H, d) for aggregation over j
        h_agg = (alpha.unsqueeze(-1) * h.unsqueeze(1)).sum(dim=2)
        #        (B, N, N, H, 1)   * (B, 1, N, H, d)   → sum_j → (B, N, H, d)

        if self.concat_heads:
            h_agg = h_agg.reshape(B, N, H * d)   # (B, N, out_features)
        else:
            h_agg = h_agg.mean(dim=2)             # (B, N, d)

        return self.act(h_agg)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pooling
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalAttentionPool(nn.Module):
    """
    Soft global pooling: learned scalar gate per node → weighted sum.

    Input:  (..., N, F)
    Output: (..., F)
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.Tanh(),
            nn.Linear(in_features // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.gate(x), dim=-2)   # (..., N, 1)
        return (w * x).sum(dim=-2)                 # (..., F)


# ═══════════════════════════════════════════════════════════════════════════════
#  I2HOFI Lightning Module
# ═══════════════════════════════════════════════════════════════════════════════

class I2HOFI(pl.LightningModule):
    """
    I2-HOFI — Intra-Inter Region Graph Neural Network for fine-grained recognition.

    Ported from TF/Keras/Spektral to PyTorch Lightning.

    Args:
        num_classes:        number of target classes.
        class_weights:      per-class inverse-frequency weights for FocalLoss.
        backbone:           torchvision backbone name (see _BACKBONE_REGISTRY).
        pretrained:         load ImageNet weights for backbone.
        freeze_backbone:    freeze all backbone parameters.
        grid_size:          (grid_h, grid_w) — number of ROI rows and columns.
        pool_size:          (pool_h, pool_w) — spatial dims of each ROI after pooling.
        gcn_out_features:   APPNP output channels (= intra-ROI node feature size).
        gat_out_features:   GAT output channels (= inter-ROI node feature size).
        appnp_K:            APPNP propagation steps.
        alpha:              APPNP teleport probability.
        gat_heads:          GAT attention heads (gat_out_features must divide evenly).
        intra_adj_mode:     adjacency for intra-ROI graph ("full", "grid", "grid8").
        inter_adj_mode:     adjacency for inter-ROI graph ("full", "grid", "grid8").
        dropout:            dropout rate in GNN layers and classifier.
        activation_appnp:   activation after APPNP linear projection.
        activation_gat:     per-head activation in GAT.
        focal_gamma:        focal loss focusing parameter.
        label_smoothing:    cross-entropy label smoothing.
        learning_rate:      peak AdamW learning rate.
        weight_decay:       AdamW weight decay.
        warmup_epochs:      linear LR warmup length.
        max_epochs:         total epochs for cosine schedule.
        img_size:           expected input spatial size (used for smoke test only).
        waveform_input:     if True, accept raw waveforms (B, T) instead of images.
                            Activates the WaveformFrontend spectrogram pipeline.
        sample_rate:        audio sample rate in Hz — waveform mode only.
        n_fft:              MelSpectrogram FFT window length — waveform mode only.
        hop_length:         MelSpectrogram hop in samples — waveform mode only.
        n_mels:             number of mel bins — waveform mode only.
        f_min:              lowest mel filter frequency — waveform mode only.
        f_max:              highest mel filter frequency (None = Nyquist).
    """

    def __init__(
        self,
        num_classes:       int            = 200,
        class_weights:     Optional[list] = None,
        backbone:          str            = "resnet50",
        pretrained:        bool           = True,
        freeze_backbone:   bool           = False,
        grid_size:         Tuple[int,int] = (3, 3),
        pool_size:         Tuple[int,int] = (3, 3),
        gcn_out_features:  int            = 512,
        gat_out_features:  int            = 512,
        appnp_K:           int            = 3,
        alpha:             float          = 0.3,
        gat_heads:         int            = 1,
        intra_adj_mode:    str            = "full",
        inter_adj_mode:    str            = "grid8",
        dropout:           float          = 0.2,
        activation_appnp:  str            = "sigmoid",
        activation_gat:    str            = "elu",
        focal_gamma:       float          = 2.0,
        label_smoothing:   float          = 0.0,
        learning_rate:     float          = 1e-3,
        weight_decay:      float          = 2.5e-4,
        warmup_epochs:     int            = 5,
        max_epochs:        int            = 150,
        img_size:          int            = 224,
        # ── Waveform input ──────────────────────────────────────────────
        waveform_input:    bool           = False,
        sample_rate:       int            = 32_000,
        n_fft:             int            = 32_000,
        hop_length:        int            = 160,
        n_mels:            int            = 8_092,
        f_min:             float          = 0.0,
        f_max:             Optional[float] = None,
        # ── LOFAR (raw STFT) input ──────────────────────────────────────
        lofar_input:       bool           = False,
        win_length:        Optional[int]  = None,
        stft_f_max_hz:     float          = 2_560.0,
        # ── Per-layer dropout overrides (default to ``dropout``) ────────
        dropout_appnp:      Optional[float] = None,
        dropout_gat:        Optional[float] = None,
        dropout_classifier: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        grid_h, grid_w = grid_size
        pool_h, pool_w = pool_size
        n_rois        = grid_h * grid_w
        n_intra_nodes = pool_h * pool_w

        d_appnp = dropout_appnp      if dropout_appnp      is not None else dropout
        d_gat   = dropout_gat        if dropout_gat        is not None else dropout
        d_clf   = dropout_classifier if dropout_classifier is not None else dropout

        if waveform_input and lofar_input:
            raise ValueError("Set only one of waveform_input / lofar_input, not both.")

        # ── Optional waveform → spectrogram frontend ──────────────────────
        if waveform_input:
            self.spec_frontend = WaveformFrontend(
                sample_rate = sample_rate,
                n_fft       = n_fft,
                hop_length  = hop_length,
                n_mels      = n_mels,
                f_min       = f_min,
                f_max       = f_max,
            )
        elif lofar_input:
            # ResNet/EfficientNet backbones downsample by 32×; pad output_hw so
            # the post-backbone feature map has at least grid_h × grid_w cells.
            BACKBONE_STRIDE = 32
            out_h = max(BACKBONE_STRIDE * 2, grid_h * BACKBONE_STRIDE)
            out_w = max(BACKBONE_STRIDE * 2, grid_w * BACKBONE_STRIDE)
            self.spec_frontend = STFTLOFARFrontend(
                sample_rate   = sample_rate,
                n_fft         = n_fft,
                hop_length    = hop_length,
                win_length    = win_length,
                stft_f_max_hz = stft_f_max_hz,
                output_hw     = (out_h, out_w),
            )

        # ── Backbone ──────────────────────────────────────────────────────
        self.backbone, feat_ch = _build_backbone(backbone, pretrained, freeze_backbone)

        # ── Intra-ROI branch (APPNP) ──────────────────────────────────────
        self.intra_appnp = APPNPLayer(
            in_features  = feat_ch,
            out_features = gcn_out_features,
            K            = appnp_K,
            alpha        = alpha,
            dropout      = d_appnp,
            activation   = activation_appnp,
        )
        self.intra_pool = GlobalAttentionPool(gcn_out_features)

        # ── Inter-ROI branch (GAT) ────────────────────────────────────────
        self.inter_gat = GATLayer(
            in_features  = gcn_out_features,
            out_features = gat_out_features,
            heads        = gat_heads,
            dropout      = d_gat,
            activation   = activation_gat,
        )
        self.inter_pool = GlobalAttentionPool(gat_out_features)

        # ── Classifier ───────────────────────────────────────────────────
        fused_dim = gcn_out_features + gat_out_features
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.BatchNorm1d(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(d_clf),
            nn.Linear(fused_dim // 2, num_classes),
        )

        # ── Adjacency matrices (fixed buffers) ───────────────────────────
        intra_adj = _make_normalized_adj(n_intra_nodes, mode=intra_adj_mode)
        inter_adj = _make_normalized_adj(n_rois, mode=inter_adj_mode)
        self.register_buffer("intra_adj", intra_adj)  # (pool_h*pool_w, pool_h*pool_w)
        self.register_buffer("inter_adj", inter_adj)  # (n_rois, n_rois)
        # Binary version of inter_adj for GAT masking
        inter_adj_bin = (inter_adj > 0).float()
        self.register_buffer("inter_adj_bin", inter_adj_bin)

        # ── Loss ─────────────────────────────────────────────────────────
        self.criterion = FocalLoss(
            class_weights  = class_weights,
            gamma          = focal_gamma,
            label_smoothing= label_smoothing,
        )

        # ── Metrics ──────────────────────────────────────────────────────
        m_kw = dict(num_classes=num_classes, average="macro")
        self.train_acc     = MulticlassAccuracy(**m_kw)
        self.val_acc       = MulticlassAccuracy(**m_kw)
        self.val_f1        = MulticlassF1Score(**m_kw)
        self.val_precision = MulticlassPrecision(**m_kw)
        self.val_recall    = MulticlassRecall(**m_kw)
        self.val_mcc       = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_acc      = MulticlassAccuracy(**m_kw)
        self.test_f1       = MulticlassF1Score(**m_kw)
        self.test_mcc      = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.test_auroc    = MulticlassAUROC(num_classes=num_classes)
        self.test_cm       = MulticlassConfusionMatrix(num_classes=num_classes)

    # ── ROI extraction ──────────────────────────────────────────────────

    def _extract_roi_features(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Divide backbone feature map into a spatial grid of ROIs.

        Args:
            feat: (B, C, Hf, Wf) — backbone feature map

        Returns:
            (B, n_rois, C, pool_h, pool_w)
        """
        B, C, Hf, Wf = feat.shape
        grid_h, grid_w = self.hparams.grid_size
        pool_h, pool_w = self.hparams.pool_size

        rois: List[torch.Tensor] = []
        for i in range(grid_h):
            for j in range(grid_w):
                h0 = int(i * Hf / grid_h)
                h1 = int((i + 1) * Hf / grid_h)
                w0 = int(j * Wf / grid_w)
                w1 = int((j + 1) * Wf / grid_w)
                roi = feat[:, :, h0:h1, w0:w1]                   # (B, C, roi_h, roi_w)
                roi = F.adaptive_avg_pool2d(roi, (pool_h, pool_w))  # (B, C, pool_h, pool_w)
                rois.append(roi)

        return torch.stack(rois, dim=1)   # (B, n_rois, C, pool_h, pool_w)

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  normalised RGB images  — image mode
               (B, T)         float32 waveform       — waveform mode

        Returns:
            logits: (B, num_classes)
        """
        grid_h, grid_w = self.hparams.grid_size
        pool_h, pool_w = self.hparams.pool_size
        n_rois         = grid_h * grid_w
        n_intra        = pool_h * pool_w

        # ── Waveform / LOFAR → spectrogram (optional) ────────────────────
        if self.hparams.waveform_input or self.hparams.lofar_input:
            x = self.spec_frontend(x)             # (B, T) → (B, 3, n_freq, T_frames)

        # ── Backbone ──────────────────────────────────────────────────────
        feat = self.backbone(x)                   # (B, C, Hf, Wf)
        B    = feat.size(0)

        # ── ROI extraction ────────────────────────────────────────────────
        roi_feats = self._extract_roi_features(feat)   # (B, n_rois, C, pool_h, pool_w)
        # Reshape to node format: (B*n_rois, n_intra, C)
        x_intra = (
            roi_feats
            .view(B * n_rois, feat.size(1), n_intra)  # (B*n_rois, C, n_intra)
            .permute(0, 2, 1)                          # (B*n_rois, n_intra, C)
        )

        # ── Intra-ROI APPNP ───────────────────────────────────────────────
        x_intra = self.intra_appnp(x_intra, self.intra_adj)   # (B*n_rois, n_intra, gcn_out)
        # Attention-pool over spatial nodes within each ROI → one vector per ROI
        x_intra = self.intra_pool(x_intra)                    # (B*n_rois, gcn_out)
        x_intra = x_intra.view(B, n_rois, -1)                 # (B, n_rois, gcn_out)

        # Global intra representation (pool over ROIs for residual)
        intra_global = x_intra.mean(dim=1)                    # (B, gcn_out)

        # ── Inter-ROI GAT ─────────────────────────────────────────────────
        x_inter = self.inter_gat(x_intra, self.inter_adj_bin)  # (B, n_rois, gat_out)
        inter_global = self.inter_pool(x_inter)                 # (B, gat_out)

        # ── Fusion + classify ─────────────────────────────────────────────
        fused = torch.cat([intra_global, inter_global], dim=-1)  # (B, gcn+gat)
        return self.classifier(fused)                             # (B, num_classes)

    # ── Lightning steps ─────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log("train/loss", loss,           on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)
        self.val_mcc(logits, y)
        self.log("val/loss",      loss,               on_epoch=True, prog_bar=True)
        self.log("val/acc",       self.val_acc,       on_epoch=True, prog_bar=True)
        self.log("val/f1",        self.val_f1,        on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_epoch=True)
        self.log("val/recall",    self.val_recall,    on_epoch=True)
        self.log("val/mcc",       self.val_mcc,       on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        probs  = F.softmax(logits, dim=-1)
        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.test_mcc(logits, y)
        self.test_auroc(probs, y)
        self.test_cm(logits, y)
        self.log("test/acc",   self.test_acc,   on_epoch=True)
        self.log("test/f1",    self.test_f1,    on_epoch=True, prog_bar=True)
        self.log("test/mcc",   self.test_mcc,   on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix (rows=true, cols=pred):\n{cm.cpu().numpy()}")
        self.test_cm.reset()

    # ── Optimiser ───────────────────────────────────────────────────────

    def configure_optimizers(self):
        # Separate weight-decayed and non-decayed params (biases / 1-D tensors)
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr     = self.hparams.learning_rate,
            betas  = (0.9, 0.999),
            eps    = 1e-8,
        )

        def _lr_lambda(epoch: int) -> float:
            wu    = self.hparams.warmup_epochs
            total = self.hparams.max_epochs
            if epoch < wu:
                return max(epoch, 1) / max(wu, 1)
            progress = (epoch - wu) / max(total - wu, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = I2HOFI(
        num_classes      = 200,
        backbone         = "resnet50",
        pretrained       = False,    # fast smoke test
        grid_size        = (3, 3),
        pool_size        = (3, 3),
        gcn_out_features = 512,
        gat_out_features = 512,
        alpha            = 0.3,
        gat_heads        = 1,
    ).to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"I2HOFI  |  {total:,} parameters")
    print(f"  backbone={model.hparams.backbone}, "
          f"grid={model.hparams.grid_size}, pool={model.hparams.pool_size}")

    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(x)
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(logits.shape)}")
