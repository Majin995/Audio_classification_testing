# Architectural Analysis Report — USW Audio Classification Models

## 1. Survey Scope

All 30 model files under `models/hydro_*.py` were analysed.  The goal was to
identify the architectural ingredients most valuable for **Undersea Warfare (USW)
acoustic classification** of raw hydrophone signals at 5120 Hz — particularly for:

- **Acoustic transients** (cavitation bursts, hull slams, propeller blade-rate pulses)
- **Narrow-band tonals** (shaft harmonics, engine lines, propeller blade-rate)
- **Low-SNR robustness** (ocean ambient noise, recording distance, reverberation)

---

## 2. Models by Input Modality

| Modality | Models |
|---|---|
| **Pure 1D raw waveform** | HydroCATFISH, HydroLEAF, HydroSpikingLEAF, HydroCNN1D, WaveletScatteringClassifier |
| **Raw waveform via deep CNN filterbank** | HydroXLSR, HydroALSI, HydroFusion (Stream B) |
| **2D spectrogram (mel/PCEN/LOFAR)** | HydroConformer, HydroResNet, HydroNet, HydroDARTMT, HydroBAHTNet, HydroSSCPMobile, HydroPANNs, HydroS4, HydroS5, HydroSSAMBA, HydroBioMamba, HydroEAT, HydroMAE, HydroUAST3D, HydroLofarResNet, I2HOFI, HydroFusion (Stream A) |
| **Complex STFT / phase-aware** | HydroDCN |
| **Ensemble / hybrid / feature-level** | HydroEnsemble, HydroBinaryEnsemble, HydroEnsembleTrio, HydroStudentFusion, AcousticOmniResNet, HydroClassical |

---

## 3. Key Architectural Ingredients — Inventory

### 3.1 Learnable Front-Ends (Raw Waveform → Feature Maps)

| Module | File | Mechanism | USW Advantage |
|---|---|---|---|
| `LearnableGaborFilterbank` | `hydro_catfish.py:52` | Per-filter (f0, log_σ) Gabor; grouped Conv1d; log1p\|·\| | Interprets data as analytic signal envelopes; learns to concentrate energy at blade-rate and shaft-line frequencies; no fixed mel bias |
| `LEAFFrontend` | `hydro_leaf.py` | Complex Gabor + GaussianLowpass (per-filter learnable stride) + TrainablePCEN | Best low-SNR: PCEN suppresses noise floor before features reach the backbone |
| `TrainablePCEN` | `hydro_conformer.py:55` | Learnable per-channel AGC via causal EMA; compresses dynamic range | Equalises ocean ambient vs signal level; critical for variable recording conditions |

### 3.2 USW-Physics Spectrogram Channels

| Module | File | Mechanism | USW Advantage |
|---|---|---|---|
| `EnhancedFrontEnd` | `hydro_net.py:289` | 4 parallel channels: wideband mel+PCEN, narrowband mel+PCEN, Gammatone+PCEN, DEMON+PCEN | One module provides all physically motivated views |
| `GammatoneSpectrogram` | `hydro_net.py:65` | ERB-spaced Gaussian weights over STFT magnitude; fixed | Narrower bands than mel at low frequencies; resolves shaft harmonics (<200 Hz) |
| `DEMONChannel` | `hydro_net.py:123` | Bandpass ≥ f_cav=800 Hz → squaring → mel → PCEN | Directly extracts blade-rate modulation — the primary UATR discriminator |
| `CepstralLifter` | `hydro_net.py:185` | Differentiable FFT lifter; keeps quefrency [low_q, high_q] | Separates harmonic ridge structure from smooth transmission-path envelope |
| `SubBandEnvelope` | `hydro_net.py:234` | RMS + variance + kurtosis over 0–100, 100–800, 800–2560 Hz | High kurtosis = impulsive/tonal; 9 interpretable scalars for free |
| `LofarFrontend` | `hydro_lofar_resnet.py:62` | STFT → bandlimit ≤2560 Hz → log-power → InstanceNorm → AdaptiveAvgPool2d(32,256) | The standard passive sonar display: fine frequency resolution for tonal line tracking |
| `MultiScalePCEN` | `hydro_conformer.py` | Wide + narrow STFT mel + PCEN, stacked along channel dim | Covers both blade-rate (short window) and shaft-rate (long window) simultaneously |

### 3.3 1D Backbone Architectures

| Module | File | Mechanism | USW Advantage |
|---|---|---|---|
| `TCNBlock` | `hydro_catfish.py:174` | Dilated DW-sep Conv1d, exponential dilation (1,2,4,8,16,32) | Efficient; covers 1 s receptive field at 5120 Hz in 6 blocks |
| `_SERes2Block` | `hydro_fusion.py:125` | Res2Net multi-scale dilated conv + SE attention + DropPath | Multi-scale within each block captures simultaneous transient + tonal features |
| `ResBlock1d` | `hydro_resnet.py` | Pre-activation BN→ReLU→Conv1d(dil)→BN→ReLU→Drop→Conv1d + DropPath | Standard ECAPA-TDNN residual; proven for speaker/sound recognition |

### 3.4 Transient-Aware Attention

| Module | File | Mechanism | USW Advantage |
|---|---|---|---|
| `SpectralFluxOnset` | `hydro_bahtnet.py:62` | Half-wave rectified spectral flux → adaptive threshold → binary patch mask | Detects cavitation onset events from filterbank energy rises |
| `BoundaryAwareAttention` | `hydro_bahtnet.py:107` | MHSA + additive scalar-gated bias `B_ij = max(onset_i, onset_j) × gate` | Focuses attention on token pairs that straddle transient boundaries |
| `BAHTBlock` | `hydro_bahtnet.py:189` | BoundaryAwareAttention + SwiGLUFFN + LayerNorm residual + DropPath | Ready-to-use BAHT block |

### 3.5 Long-Range / Narrow-Band Temporal Models (SSM)

| Module | File | Mechanism | USW Advantage |
|---|---|---|---|
| `SaShiMiBlock` | `hydro_s4.py:309` | S4D diagonal complex SSM; FFT kernel; GLU gate; pre-norm FFN | Infinite-horizon receptive field → can track slowly drifting shaft harmonics over 1 s+ |
| `BidirMambaBlock` | `hydro_ssamba.py:204` | Forward + backward Mamba (selective SSM) summed; pre-norm residual | Bidirectional context; selective compression attends to informative regions |
| `S5Layer` | `hydro_s5.py` | MIMO diagonal SSM with bidirectional FFT scan | Strong tonal tracker; LayerScale for stable training |

### 3.6 Pooling

| Module | File | Mechanism | USW Advantage |
|---|---|---|---|
| `_AttentiveStatisticsPool` | `hydro_fusion.py:170` | Softmax-weighted mean + std over time; channel-first | Short transients dominate if their attention weight is high; better than GAP |
| `AttentiveStatisticsPool` | `hydro_net.py:468` | Identical but also exported from `hydro_net` | Same |

### 3.7 Training Robustness Techniques

| Technique | Source | Details |
|---|---|---|
| **CSSD** | `hydro_fusion.py` | EMA teacher + resample-to-low-SR student; KL distillation. Forces backbone to rely on low-frequency tonal structure rather than high-SNR broadband cues |
| **Waveform mixup** | All models | Beta(α,α) interpolation in waveform space |
| **Gaussian noise @ SNR 20–40 dB** | `hydro_fusion.py:467` | USW-realistic SNR augmentation |
| **Random gain ±40 %** | `hydro_fusion.py:483` | Simulates variable source level and recording range |
| **FocalLoss (γ=2, label_smoothing)** | `hydro_conformer.py` | Down-weights easy samples; critical for class-imbalanced UATR datasets |
| **SpecAugment1D** | `hydro_catfish.py:143` | Frequency-band and time masking on filterbank output |
| **DropPath (stochastic depth)** | `hydro_conformer.py` | Regularises deep transformer/SSM stacks |
| **SSM parameter LR split** | `hydro_fusion.py:636` | S4D poles (`log_a_real`, `log_a_imag`, `log_dt`) and Mamba `dt_proj` get 0.1× LR — critical for stable SSM training |

---

## 4. Analysis: Most Valuable Ingredients for USW

### 4.1 Why multi-view time-frequency is essential

Undersea acoustic targets produce simultaneously:
- **Blade-rate modulations** at 1–30 Hz (slow, requires wide STFT window or DEMON envelope)
- **Shaft harmonics** at 30–300 Hz (requires moderate frequency resolution)
- **Propeller cavitation broadband** at 500–2560 Hz (requires short window for envelope detection)
- **Machinery lines** (fixed harmonics that drift slowly over 1 s)

No single STFT window optimises for all four. The `EnhancedFrontEnd` directly addresses this by computing four parallel representations:
- Wide mel (n_fft=256 at 5120 Hz, ~50 ms) — cavitation and blade-rate transients
- Narrow mel (n_fft=1024, ~200 ms) — shaft harmonics
- Gammatone — ERB-spaced, narrower than mel below 500 Hz, better machinery line resolution
- DEMON — bandpassed-then-squared, extracts blade-rate AM directly

The Gabor 1D stream adds per-sample temporal resolution (no STFT windowing artefact) and the LOFAR view adds fine-frequency tracking for tonals.

### 4.2 Why S4D + Mamba for narrow-band tracking

Classical CNNs have limited receptive fields. Dilated TCNs reach 1 s with 6 blocks but at O(N·k) cost. S4D processes the entire sequence in O(N·H·log(N)) via FFT convolution with a globally parameterised SSM kernel — the kernel can represent periodic components with arbitrarily long periods. For shaft harmonics that repeat at 2–10 Hz, S4D's ability to maintain memory over hundreds of frames is a genuine advantage over dilated CNNs.

Bidirectional Mamba adds selective state compression — it can "forget" sections of ambient noise while preserving tonal events, which the S4D fixed kernel cannot do.

### 4.3 Why onset-biased attention complements the SSM

The SSM handles slow tonal tracking. Transient events (cavitation pops, thruster blasts) require attention to be focused at the onset boundary. `BoundaryAwareAttention` + `SpectralFluxOnset` provides exactly this: the onset mask biases the attention matrix so that tokens adjacent to energy-rise events attend strongly to each other. The learnable `boundary_gate` scalar lets the network decay this bias if it is unhelpful for a particular class.

---

## 5. SuperModel1D Design Summary

The recommended "Super Standalone" model fuses all the above into a single cohesive graph:

```
Raw waveform (B, T=5120) @ 5120 Hz
  ├─ Stream G  LearnableGaborFilterbank → log1p|·| → SpecAugment1D → stride-4 stem
  ├─ Stream S  EnhancedFrontEnd (wb+nb mel+Gammatone+DEMON, PCEN) → CepstralLifter → CNNInput → interpolate
  ├─ Stream L  LofarFrontend (bandlimited log-power, fixed 32×256) → Conv1d → interpolate
  └─ Stream F  SubBandEnvelope (9 scalar DSP statistics) — concatenated at head only

  Onset: SpectralFluxOnset(Gabor envelope) → binary transient patch mask

  Fusion gate → 3× SE-Res2Block (dilations 2,4,8) + MFALayer
  → [permute] → 2× BAHTBlock (boundary-aware attention, onset-biased)
  → n_s4× SaShiMiBlock (S4D, long-range tonals)
  → n_mamba× BidirMambaBlock (selective global context)
  → [permute] → AttentiveStatisticsPool → concat(SubBandEnvelope) → classifier

  Loss:  FocalLoss + optional CSSD (EMA teacher, 2048 Hz degradation)
  Augs:  mixup, Gaussian noise @SNR 20–40 dB, random gain ±40 %, SpecAugment1D
  ~2.7–3.1 M trainable parameters
```

See `models/super_model.py` for the full implementation,
`training/train_super_model.py` for the training script, and
`training/tune_super_model.py` for the Optuna hyperparameter sweep.
