# UATR Integration Map

Integration of five state-of-the-art Underwater Acoustic Target Recognition (UATR)
architectures, advanced DSP denoising, LMF loss, and DART-MT semi-supervised training
into the HydroVision codebase.

---

## 1. Pipeline Overview

```
                 ┌───────────────────────────────────────────────────────────┐
                 │              Audio Data Pipelines                         │
                 │                                                           │
  data/Split1s   │  DALI raw-audio path (5120 Hz, 1 s clips)                │
  ──────────┐    │   audio_lightning_loader.py                               │
            │    │     _DALIWrapper.__iter__()                               │
            │    │       │                                                   │
            │    │       ▼  (if --denoise != off)                            │
            │    │     _DenoisingBatchHook  ◄── processing/denoise/         │
            │    │       │  EMD-Wavelet or NMF-ICA (CPU)                    │
            │    │       │                                                   │
            │    │       ▼                                                   │
            │    │     (B, T) GPU tensors ──────────────────────────────┐   │
            │    │                                                       │   │
            │    │  Cached-feature path                                 │   │
            │    │   cached_feature_dataset.py                          │   │
            │    │     _worker_precompute()                             │   │
            │    │       │                                              │   │
            │    │       ▼  (if denoise_cfg['method'] != 'off')        │   │
            │    │     EMD-Wavelet / NMF-ICA (precompute time)         │   │
            │    │       │                                              │   │
            │    │       ▼  UnderwaterFeatureExtractor                 │   │
            │    │     12 representations → .pt cache                  │   │
            │    │                                                      │   │
            └────┘                                                      │   │
                                                                        ▼   │
                                                              ┌─────────────┤
                                                              │  Models     │
                                                              └─────────────┘
```

---

## 2. Model → File → Trainer Mapping

| Model         | File                              | Trainer                              | Input      | Loss            |
|---------------|-----------------------------------|--------------------------------------|------------|-----------------|
| HydroCATFISH  | `models/hydro_catfish.py`         | `training/train_catfish.py`          | raw (B,T)  | FocalLoss       |
| HydroALSI     | `models/hydro_alsi.py`            | `training/train_alsi.py`             | raw (B,T)  | FocalLoss       |
| HydroDCN      | `models/hydro_dcn.py`             | `training/train_dcn.py`              | raw (B,T)  | FocalLoss       |
| HydroBAHTNet  | `models/hydro_bahtnet.py`         | `training/train_bahtnet.py`          | raw (B,T)  | **LMF**         |
| HydroSSCPMobile | `models/hydro_sscp_mobile.py`   | `training/train_sscp_mobile.py`      | raw (B,T)  | FocalLoss + KD  |
| HydroDARTMT   | `models/hydro_dart_mt.py`         | `training/train_dart_mt.py`          | raw (B,T)  | FocalLoss (+MT) |

### Architecture Summary

| Model | Frontend | Backbone | Key Innovation |
|---|---|---|---|
| CATFISH | Learnable Gabor filterbank | Dilated TCN × 6 | Replaces fixed STFT with jointly-optimised filters |
| ALSI | Wav2Vec2 (frozen, 16kHz) + CQT-ResNet | Multi-Head Cross-Attention | Dual-stream temporal×frequency fusion |
| DCN | Raw STFT → complex (B,2,F,T) | Complex Conv × 4 + DCMF | Complex-valued arithmetic + matched filter front-end |
| BAHTNet | LogMelPCEN | Boundary-Aware Transformer × 6 | Onset-gated attention bias + LMF loss |
| SSCP-Mobile | LogMelPCEN (n_mels=32) | DW-Sep × 2 + grouped PW | <128 kB budget + BAHTNet teacher KD |

---

## 3. DSP / Denoising Components

| Module | File | Description |
|---|---|---|
| `emd_wavelet_denoise` | `processing/denoise/emd_wavelet.py` | EMD → IMF noise classification → SURE wavelet threshold |
| `nmf_ica_separate` | `processing/denoise/bss.py` | NMF spectral factorisation + FastICA decorrelation → dominant source |
| `estimate_snr_noise_floor` | `processing/denoise/snr.py` | Spectrogram-percentile SNR estimator for ablation logging |
| `DenoiseTransform` | `processing/denoise/transform.py` | `nn.Module` wrapper — batch CPU denoising; composable |

### Denoising Insertion Points

1. **DALI raw pipeline** (`data/audio_lightning_loader.py:_DenoisingBatchHook`):
   - Activated by `--denoise {emd_wavelet,nmf,ica,nmf_ica,emd_nmf}` on any `train_*.py`.
   - Runs after DALI decodes + resamples, before GPU model forward.

2. **Cached-feature pipeline** (`data/cached_feature_dataset.py:_worker_precompute:114`):
   - Activated by `denoise_cfg={'method': 'emd_wavelet'}` passed to `CachedFeatureDataset`.
   - Runs at precompute time → zero cost during training epochs.

---

## 4. Loss Matrix

| Model | Primary Loss | Notes |
|---|---|---|
| HydroCATFISH | `FocalLoss(γ=2, margin=0)` | From `models/hydro_conformer.py:359` |
| HydroALSI | `FocalLoss(γ=2, margin=0)` | Same |
| HydroDCN | `FocalLoss(γ=2, margin=0)` | Same |
| HydroBAHTNet | `LargeMarginFocalLoss(γ=2, m=0.35)` | From `processing/losses/lmf.py` |
| HydroSSCPMobile | `FocalLoss + KD(KLDiv, T=4)` | CE term weighted by `kd_alpha=0.3` |
| HydroDARTMT (semi-sup) | `FocalLoss + MSE(consistency)` | Consistency weight ramps up over `rampup_epochs` |

### LargeMarginFocalLoss API
```python
from processing.losses import LargeMarginFocalLoss
criterion = LargeMarginFocalLoss(
    num_classes=4,
    alpha=class_weights,   # list[float] or None
    gamma=2.0,
    margin=0.35,           # set 0.0 to reduce to standard Focal
    label_smoothing=0.05,
)
```

---

## 5. DART-MT Semi-Supervised Data Flow

```
Epoch start
    │
    ├─── Labeled batch (x_l, y_l)  ←── DALI train split
    │       │
    │       ├── CE loss: FocalLoss(student(x_l), y_l)
    │
    ├─── Unlabeled batch x_u  ←── data/Unlabeled/ via _UnlabeledDALIWrapper
    │       │
    │       ├── Teacher (EMA copy) forward: t_logits = teacher(x_u)  [no grad]
    │       └── Student forward:           s_logits = student(x_u)
    │           Consistency loss: MSE(s_logits, t_logits) × λ(epoch)
    │
    └── Total loss = CE + λ(epoch) × MSE
    │
    EMA update: teacher ← m·teacher + (1-m)·student   (m=0.999)
```

**Enabling semi-supervised mode:**
```bash
python training/train_dart_mt.py \
    --use_mean_teacher \
    --unlabeled_dir data/Unlabeled \
    --unlabeled_ratio 2 \
    --consistency_max 1.0 \
    --rampup_epochs 10 \
    --ema_momentum 0.999
```

---

## 6. Model Registry

```python
from processing.registry import build_model, list_models

print(list_models())
# ['alsi', 'bahtnet', 'catfish', 'conformer', 'dart_mt', 'dcn', 'resnet', 'sscp_mobile']

model = build_model("catfish", num_classes=4, class_weights=[1.0, 2.5, 1.2, 3.1])
```

File: `processing/registry.py` — lazy-loaded, no circular-import issues.

---

## 7. Grid Search Usage

### Single model
```bash
export DATA_DIR=/path/to/Split1s
bash scripts/grid_catfish.sh
```

### All models sequentially (recommended)
```bash
export DATA_DIR=/path/to/Split1s
bash scripts/run_all_grids.sh 2>&1 | tee all_grids.log
```

### Dry run (verify commands before launching)
```bash
DRY_RUN=1 bash scripts/run_all_grids.sh
```

### SSCP-Mobile with explicit teacher
```bash
BAHTNET_BEST_CKPT=lightning_logs/hydro_bahtnet/.../best.ckpt \
  bash scripts/grid_sscp_mobile.sh
```

### Grid axes summary
| Model | Axes swept | Runs |
|---|---|---|
| catfish | seed(2)×lr(2)×bs(2)×denoise(2)×gabor_filters(2) | 32 |
| alsi | seed(2)×lr(2)×denoise(2)×freeze_w2v(2)×fusion_heads(2) | 32 |
| dcn | seed(2)×lr(2)×bs(2)×denoise(2)×dcmf_templates(2)×depth(2) | 64 |
| bahtnet | seed(2)×lr(2)×bs(2)×denoise(2)×loss(2)×margin(3) | 96 |
| sscp_mobile | seed(2)×lr(2)×bs(2)×denoise(2)×kd_temp(2)×kd_alpha(2) | 64 |

---

## 8. Known Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| Wav2Vec2 expects 16 kHz; dataset is 5120 Hz | ALSI resamples in `forward` — ~3× time axis expansion | Use `--batch_size 16` on <24 GB GPU |
| EMD-Wavelet is CPU-only | Cannot run inside DALI pipeline | Applied post-DALI via `_DenoisingBatchHook` |
| SSCP-Mobile 128 kB constraint | Fails at `on_train_start` if exceeded | Keep `n_mels≤32`, `groups=8` in block3 |
| BAHTNet must complete before SSCP-Mobile | Grid runner enforces order | See `run_all_grids.sh` auto-detection logic |
| No unlabeled data shipped | DART-MT semi-sup is a no-op until `data/Unlabeled/` is populated | Falls back to supervised silently |
| Complex tensor interleaving (re/im) | HydroDCN uses channel stride 2 convention | Use `ComplexConv2d`/`ComplexBN`/`ModReLU` from `hydro_dcn.py` |

---

## 9. New File Index

```
processing/
├── __init__.py
├── denoise/
│   ├── __init__.py
│   ├── emd_wavelet.py       EMD + SURE wavelet denoising
│   ├── bss.py               NMF + FastICA blind source separation
│   ├── snr.py               SNR estimation utility
│   └── transform.py         DenoiseTransform nn.Module
├── losses/
│   ├── __init__.py
│   └── lmf.py               LargeMarginFocalLoss
└── registry.py              Lazy model registry

models/
├── hydro_catfish.py         Learnable Gabor + TCN
├── hydro_alsi.py            Wav2Vec2 × CQT-ResNet dual-stream
├── hydro_dcn.py             Complex CNN + DCMF
├── hydro_bahtnet.py         Boundary-Aware Transformer + LMF
├── hydro_sscp_mobile.py     Edge CNN + KD (BAHTNet teacher)
└── __init__.py              Updated to re-export all models

training/
├── train_catfish.py
├── train_alsi.py
├── train_dcn.py
├── train_bahtnet.py
└── train_sscp_mobile.py

scripts/
├── grid_catfish.sh
├── grid_alsi.sh
├── grid_dcn.sh
├── grid_bahtnet.sh
├── grid_sscp_mobile.sh
└── run_all_grids.sh         Sequential driver with BAHTNet→SSCP ordering
```

---

## 10. Modified Existing Files

| File | Change |
|---|---|
| `requirements.txt` | Added EMD-signal, PyWavelets, librosa, nnAudio; uncommented optuna |
| `models/__init__.py` | Added HydroDARTMT, HydroCNN1D, AcousticOmniResNet, HydroUAST3D, + 5 new |
| `models/hydro_dart_mt.py` | Added `use_mean_teacher`, EMA teacher, `_update_teacher`, `_consistency_lambda`, unlabeled batch handling in `training_step` |
| `data/audio_lightning_loader.py` | Added `denoise_method`, `unlabeled_dir` params; `_DenoisingBatchHook`, `_UnlabeledDALIWrapper`, `_scan_unlabeled`, `_make_unlabeled_loader`, `unlabeled_dataloader()` |
| `data/cached_feature_dataset.py` | Added `denoise_cfg` to `CachedFeatureDataset.__init__`, `_worker_precompute`, `OmniFeatureDataModule.__init__` and `setup()` |

---

## 11. Quantum / Hybrid Models

Three hybrid quantum-classical models classify the same 5120 Hz hydrophone clips
using a parameterised quantum circuit as the classification head. All three
share `models/quantum/_qcircuits.py::QuantumHead` — a `Linear → AngleEncoder →
VariationalAnsatz → MeasureAll(PauliZ) → Linear` block built on TorchQuantum
(pure-PyTorch, GPU-capable, autograd-native).

### Shared QuantumHead contract
`QuantumHead(in_dim, n_qubits=8, n_layers=4, n_reuploads=1, num_classes=4)` —
forward shape `(B, in_dim) → (B, num_classes)`. Default uses an 8-qubit
register with a 4-layer hardware-efficient ansatz (RY+RZ + CNOT-ring),
small-std initialisation (mitigates barren plateaus), and 1 data re-uploading
pass. Gradients flow end-to-end via TorchQuantum's autograd integration.

### Models
| Registry key | File | Frontend | Classifier | Trainable params |
|---|---|---|---|---|
| `cnnlstm_qc`       | `models/hydro_cnnlstm_qc.py`       | 4-block strided 1-D CNN → Bi-LSTM (2 layers, h=64) on raw waveform `(B, 5120)` | QuantumHead(128 → 4) | ~250 k |
| `quantum_transfer` | `models/hydro_quantum_transfer.py` | Frozen UATR backbone (e.g. BAHTNet) — `classifier` replaced with Identity to expose `(B, feat_dim)` penultimate features | Linear → tanh → QuantumHead(32 → 4) → Linear | ~5 k trainable (head only) |
| `vqc_features`     | `models/hydro_vqc_features.py`     | LayerNorm + Linear(138 → 16) on cached scalar+PSD+spectral features                | QuantumHead(16 → 4)            | ~3 k |

### CLI examples
```bash
# A — CNN-LSTM-QC on raw waveform
python training/train_cnnlstm_qc.py \
    --data_dir $DATA_DIR --n_qubits 8 --q_layers 4 --max_epochs 100

# B — Quantum Transfer Learning (requires a trained BAHTNet teacher)
python training/train_quantum_transfer.py \
    --data_dir $DATA_DIR \
    --teacher_arch bahtnet \
    --teacher_ckpt lightning_logs/hydro_bahtnet/version_0/checkpoints/best.ckpt \
    --n_qubits 8 --q_layers 4 --max_epochs 30

# D — VQC on cached 138-D features
python training/train_vqc_features.py \
    --data_dir $DATA_DIR --cache_dir data/cache --n_qubits 8 --q_layers 6
```

### Smoke tests
`pytest tests/smoke/test_quantum_models.py -v` — validates QuantumHead shape
and gradient flow through the variational ansatz, all-trainable for Models A
and D, frozen-teacher invariant for Model B, and registry round-trip.

### Known limitations
- TorchQuantum simulation is roughly 3–10× slower than a comparable classical
  head on CPU at 8 qubits × 4 layers. GPU sim helps; mixed-precision training
  is not recommended (quantum gates expect fp32).
- Model B requires a trained backbone checkpoint; running without
  `--teacher_ckpt` only smoke-tests the wiring.
- TorchQuantum is simulation-only — to target a real QPU, swap the
  `QuantumHead` backend for Qiskit + TorchConnector. The interface is
  isolated to one file (`models/quantum/_qcircuits.py`).
