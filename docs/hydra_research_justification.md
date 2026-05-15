# HydroHydra — Research Justification

## 1. Design philosophy

HydroHydra is a spectrogram-free, multi-stream UATR classifier whose central
bet is that **passive sonar machinery signatures decompose into largely
orthogonal acoustic factors** (narrowband tonals, broadband cavitation
envelopes, AR-modelled excitation residual, phase-space recurrence) that can
each be picked off by a dedicated waveform-domain branch and recombined by a
single sequence backbone. The model deliberately avoids STFT / Mel / CQT /
DEMON image ops in the forward graph — the constraint forces every branch to
expose a property of the raw waveform that is not already a function of a
short-time magnitude spectrum, and gives the final fusion stage a higher-rank
input than a single-spectrogram classifier could produce.

A second design bet, validated empirically across Phases G–I (see §3 / §4),
is that **front-end diversity is the dominant lever**: sequence-backbone
upgrades (HELIX / global attention), head upgrades (ArcFace, Sub-Center
ArcFace, DEMON-MoE), and most loss tweaks (LDAM-DRW, manifold mixup) cannot
exceed what the front-end has already extracted, and most stack negatively.

## 2. Why this dataset, why these constraints

Target data is a DeepShip-like 4-class corpus: **Cargo / Passenger / Tanker /
Tug**, hydrophone recordings resampled to **5120 Hz**, segmented into
**1 s clips** (5120 samples). Severe class imbalance (Tug is the rare class
at roughly a third the count of Tanker/Passenger). Production target is
**macro-precision** (μP), specifically MP@cov0.85 — the macro-precision
attainable when the model is allowed to abstain on the hardest 15 % of
clips. Constraints that flow from this:

- **Bandwidth (0–2560 Hz):** all branches must be tuned for sub-Nyquist
  machinery harmonics (shaft rate, blade rate, low-frequency tonals) and
  cavitation envelopes, not voice-band content.
- **Per-class imbalance + heterogeneous-within-class Tanker:** drives the
  loss family choice (LMF / LDAM) and falsifies the standard long-tail recipe
  (DRW with class-balanced weights actively hurt Tug; see §4).
- **Macro-precision target:** drives the abstention head and per-class
  threshold calibration — see §3.4. Naive cross-entropy ceilings on F1 are
  not the production gate.
- **Spectrogram-free constraint:** keeps the model compatible with downstream
  on-device inference where STFT cost dominates a 1 s window, and forces
  honest branch diversity rather than 5 redundant views of the same Mel.

## 3. Component-by-component justification

### 3.1 Front-end branches

#### Stream A — Learnable Gabor filterbank (`use_gabor`)

- **Purpose:** Adaptive narrow-band time-frequency analysis without a fixed
  Mel grid; the band centres and widths fit ship-machinery harmonics during
  training.
- **Mechanism:** Bank of complex Gabor kernels parameterised by centre
  frequency and bandwidth; gradient flows back through the kernel parameters
  (LEAF-style). The waveform passes through the bank, then through a
  stride-16 stem and two SE-Res2 blocks. SpecAugment-1D fires on the
  filterbank output during training.
- **References:**
  - Zeghidour, Teboul, de Chaumont Quitry, Tagliasacchi. *LEAF: A Learnable
    Frontend for Audio Classification.* ICLR 2021. arXiv:2101.08596.
  - CATFISH UATR application of a learnable Gabor filterbank.
    arXiv:2505.23964.
- **Empirical note:** Gabor + Scattering (R0 baseline) reached val/μP=0.6620.
  Adding SincNet+TDSBE (R1, α-lever, see §3.1.4) lifted to **val/μP=0.6908**;
  Gabor remains in the always-on subset because it is the strongest single
  branch.

#### Stream B — Kymatio Scattering1D (`use_scattering`, optional `use_jtfs`)

- **Purpose:** Translation-invariant, deformation-stable narrowband features
  with theoretical recovery guarantees that fixed-spectrogram features lack.
- **Mechanism:** First-order wavelet scattering (`J=6, Q=8`); the per-band
  log-magnitude is normalised and projected through 1×1 + 3×1 conv. JTFS-lite
  (`use_jtfs=True`) augments with first- and second-order temporal
  derivatives (Δ, ΔΔ) along each scattering band, surfacing rate × scale
  cross-modulation that pure Scattering1D misses.
- **References:**
  - Andén & Mallat. *Deep Scattering Spectrum.* IEEE TSP 2014.
    arXiv:1304.6763.
  - Andreux et al. *Kymatio: Scattering Transforms in Python.* JMLR 2020.
    arXiv:1812.11214.
  - Yu et al. lake-trial UATR applications of scattering-based features.
    [citation needs verification]
- **Empirical note:** Phase H2 JTFS-lite scored val 0.6939 / test 0.6158 — the
  Δ/ΔΔ extension regressed full-coverage test by 4.6 pt. JTFS-lite is kept
  available behind a flag but the production ship-config uses J=6, Q=8 with
  `use_jtfs=False`.

#### Stream C — SincNet (`use_sincnet`)

- **Purpose:** Parametric narrowband bandpass with only two parameters per
  filter (low cutoff, bandwidth) — a strong inductive bias toward physical
  bandpass shapes, complementary to LEAF's Gaussian envelope.
- **Mechanism:** `_SincConv1d` builds parametric `sinc(2π·high·t) − sinc(2π·low·t)`
  kernels per filter, Hamming-windowed and L2-normalised. The bank output
  is log-compressed (matching the Gabor stream) then passed to a stem +
  2× SE-Res2.
- **References:**
  - Ravanelli & Bengio. *Speaker Recognition from Raw Waveform with SincNet.*
    SLT 2018. arXiv:1808.00158.
- **Empirical note:** Part of Phase G's α-lever (SincNet + TDSBE). α was the
  **only winning lever** of the six tested in Phase G — added approximately
  +2.9 pt val/μP (0.6620 → 0.6908) and lifted Tanker-precision from ~0.50 to
  ~0.78. Phase H confirmed that no later head/backbone change exceeded what
  the Phase G front-end had already extracted.

#### Stream D — Time-Domain Sub-Band Envelope statistics, TDSBE (`use_tdsbe`)

- **Purpose:** Hand-engineered envelope features explicitly target the
  cavitation / blade-rate amplitude regime that machine-learned narrowband
  filters tend to under-emphasise.
- **Mechanism:** Four fixed FIR bandpass filters
  ((20–250 Hz), (250–1k), (1k–2k), (2k–2540 Hz)). For each subband: Hilbert
  envelope (rfft analytic-signal), then framewise statistics — RMS,
  log-variance, kurtosis, ZCR (on the band-passed signal, not the envelope),
  crest factor. Stacked to (B, N×5, T') and projected through one SE-Res2.
- **References:**
  - Marple. *Computing the Discrete-Time Analytic Signal via FFT.* IEEE TSP
    1999.
  - Brillinger. *Time Series: Data Analysis and Theory* — envelope statistics
    background.
  - Lyons. *Cookbook* / Nielsen. *Sonar Signal Processing.* Artech House
    1991 — passive sonar envelope features.
- **Empirical note:** TDSBE is the second half of the α-lever; the
  (SincNet, TDSBE) pair is treated as atomic in the production config.
  Kurtosis is computed in fp32 even under bf16 mixed-precision because of
  numerical instability in the m4/m2² ratio.

#### Stream E — Frozen wav2vec2 conv front-end (`use_w2v`)

- **Purpose:** A pretrained acoustic prior that picks up generic broadband
  texture statistics learned on speech-scale corpora, available cheaply
  because the conv stack is frozen.
- **Mechanism:** wav2vec2-base CNN feature extractor (the convolutional
  pre-quantisation block; transformer dropped); waveform is resampled
  5120 → 16000 Hz via linear interpolation, passed through the frozen
  (eval-pinned) extractor, then projected to `out_ch` and through one
  SE-Res2 block. Only the projection + SE-Res2 train.
- **References:**
  - Baevski, Zhou, Mohamed, Auli. *wav2vec 2.0: A Framework for
    Self-Supervised Learning of Speech Representations.* NeurIPS 2020.
    arXiv:2006.11477.
- **Empirical note:** Off in the R1 ship-config; available as an ablation
  axis. The 5120 → 16000 Hz upsample is a known caveat (no new high-
  frequency content is introduced, only re-bandwidthed).

#### Stream F — LPC residual / coefficients (`use_lpc`)

- **Purpose:** Source–filter factorisation. Existing branches see the
  unfactored signal; the LPC residual exposes the AR excitation that
  highlights propeller-cavitation transients masked by tonal harmonics.
- **Mechanism:** Per-frame autocorrelation → Levinson–Durbin recursion
  (with `tanh`-bounded reflection coefficients for minimum-phase / numerical
  stability), produces *p* LPC coefficients and one log-RMS residual scalar
  per frame. Stack of `(a_1..a_p, log-RMS-residual)` projected through a
  Conv1d + 1× SE-Res2.
- **References:**
  - Makhoul. *Linear Prediction: A Tutorial Review.* Proc. IEEE 1975.
  - Subramani et al. *LPC-style auxiliary features.* Interspeech 2022.
    arXiv:2202.11301.
  - LPCSE: arXiv:2206.06908.
- **Empirical note:** Off in R1 ship-config; held as an additional front-end
  axis for future work after Phase H/I exhausted the no-front-end search
  surface.

#### Stream G — Recurrence Plot / phase-space (`use_rp`)

- **Purpose:** Phase-space recurrence captures whether the trajectory of the
  signal revisits prior states — exposes quasi-periodic engine cycles
  (diagonal structure) and stochastic cavitation textures (granular blobs)
  that are not frequency, envelope, or cepstral features.
- **Mechanism:** Downsample waveform to 1024 → Takens embed
  (`embed_dim=3, delay=4`) → soft recurrence matrix
  `R[i,j] = sigmoid((ε - ‖y_i − y_j‖_∞) / β)` with ε = per-clip 10 %
  quantile of off-diagonal distances → 2D conv stack collapses RP into a
  (out_ch, T') feature map matched to the fusion sequence length.
- **References:**
  - Eckmann, Kamphorst, Ruelle. *Recurrence Plots of Dynamical Systems.*
    Europhys. Lett. 1987.
  - Hatami, Gavet, Debayle. *Classification of Time-Series Images Using Deep
    CNN.* arXiv:1710.00886.
  - Yu et al. lake-trial UATR application reporting 94.31% accuracy.
    [citation needs verification]
- **Empirical note:** Off in R1 ship-config. Computational cost grows as
  N² in the embedded sequence length, hence the 1024-sample downsample.

### 3.2 Sequence modelling backbone

#### SaShiMi / S4D blocks (`s4_n_blocks`, `s4_d_state`)

- **Purpose:** Linear-time long-context sequence modelling over the fused
  (B, T_f, D) representation. Picks up slow tonal evolution within the
  1-second window while staying parameter-efficient relative to a
  transformer.
- **Mechanism:** Two SaShiMi blocks (default), each wrapping a diagonal S4
  state-space layer with input/output projections, gating, and a residual
  MLP. The S4D parameterisation uses HIPPO-LegS-derived initialisation for
  the diagonal A matrix.
- **References:**
  - Goel, Gu, Donahue, Ré. *It's Raw! Audio Generation with State-Space
    Models.* ICML 2022. arXiv:2202.09729.
  - Gu, Goel, Gupta, Ré. *On the Parameterization and Initialization of
    Diagonal State Space Models.* NeurIPS 2022. arXiv:2206.11893.
- **Empirical note:** Stable across Phases G–I; no SaShiMi-internal lever
  ever shipped because the front-end dominates the lift surface (Phase H
  finding 3).

#### Optional global attention block (`use_global_attn`)

- **Purpose:** A pure-SSM backbone has no token-mixing across distant
  positions; one global self-attention block closes a measured
  long-range gap on noisy SSM stacks.
- **Mechanism:** Pre-LN multi-head self-attention + MLP residual block in
  (B, T, D) layout, inserted after the S4D stack.
- **References:**
  - Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
    arXiv:1706.03762.
  - Xiong et al. *On Layer Normalization in the Transformer Architecture.*
    ICML 2020. arXiv:2002.04745.
  - HELIX (Mamba2 + 1 global attention) — arXiv:2603.21316.
    *[citation needs verification — date is in the future as of project
    knowledge cutoff; treat as project-internal reference]*
- **Empirical note:** Phase H3 with `use_global_attn=True` scored val 0.6940 /
  test 0.6474 — regressed test by 1.46 pt vs R1. Available as a flag,
  off in production.

### 3.3 Pooling

#### Attentive Statistics Pooling (`_AttentiveStatisticsPool`)

- **Purpose:** Soft attention over time gives the head access to
  utterance-level mean and standard-deviation statistics of the most
  informative frames, while ignoring silence / dropout artefacts at clip
  edges.
- **Mechanism:** Per-frame attention weights → weighted mean and
  weighted standard deviation along T → concatenated to a 2D-dimensional
  utterance embedding. Output dim is `2 * fusion_dim`.
- **References:**
  - Okabe, Koshinaka, Shinoda. *Attentive Statistics Pooling for Deep
    Speaker Embedding.* Interspeech 2018. arXiv:1803.10963.
- **Empirical note:** Standard across the entire model family (HydroHydra
  and HydroPrecise both use it). Never ablated; the standard-deviation
  channel is a well-established net win for sequence classification.

### 3.4 Head + auxiliary losses

#### MLP head + Deep-Gamblers abstention logit (`head_type="mlp"`, default)

- **Purpose:** Production macro-precision is gated, not full-coverage —
  the model is allowed to abstain. Deep-Gamblers makes the abstention an
  intrinsic part of the loss landscape rather than a thresholding heuristic.
- **Mechanism:** MLP outputs `num_classes + 1` logits; the last is the
  abstention. Loss is `primary_loss + λ · (-log(p_y + o · p_abstain))`
  with `o = gambler_o = 0.3`, `λ = gambler_weight = 0.1` (R1). At inference
  the abstention column is dropped; selective prediction is then driven by
  per-class thresholds calibrated on val.
- **References:**
  - Liu et al. *Deep Gamblers: Learning to Abstain with Portfolio Theory.*
    NeurIPS 2019. arXiv:1907.00208.
- **Empirical note:** Default head; ships in R1 and the N=5 ensemble.
  Note interaction: Deep-Gamblers is auto-disabled when a non-MLP head
  is in use.

#### Large-Margin Focal loss (LMF, `loss="lmf"`)

- **Purpose:** Combine focal loss's hard-example focus with a margin-based
  separation between class logits. Targets the macro-precision frontier on
  imbalanced data.
- **Mechanism:** Margin is subtracted from the target-class logit before a
  focal-style cross-entropy with `(1 - p_t)^γ` modulation. Default
  `γ=2.0, margin=0.7, label_smoothing=0.05`.
- **References:**
  - Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
    arXiv:1708.02002.
  - Liu et al. *Large-Margin Softmax Loss.* ICML 2016. arXiv:1612.02295.
- **Empirical note:** R1 ship-config uses `lmf_gamma=2.0, lmf_margin=0.7`.
  The margin sweep in Phase G found 0.7 dominated 0.5; smoothing 0.05 was
  the consistent grid-search winner.

#### LDAM loss + Deferred Re-Weighting (`loss="ldam"`)

- **Purpose:** Per-class margin scaled to class frequency — designed
  specifically for label-distribution-aware long-tail learning.
- **Mechanism:** Margin per class is `max_m / n_c^(1/4)` (normalised), with
  optional class-balanced weights swapped in at `ldam_drw_epoch`. The DRW
  switch is gated on training mode and uses an effective-number-of-samples
  weight (Cui et al. 2019); the implementation falls back to inverse-
  frequency when β saturates.
- **References:**
  - Cao et al. *Learning Imbalanced Datasets with Label-Distribution-Aware
    Margin Loss.* NeurIPS 2019. arXiv:1906.07413.
  - Cui et al. *Class-Balanced Loss Based on Effective Number of Samples.*
    CVPR 2019. arXiv:1901.05555.
- **Empirical note:** Phase H1 LDAM scored test 0.6615 (statistical tie with
  R1, not a beat). DRW with β=0.99999 actively hurt Tug — Tug-precision
  dropped 0.6340 → 0.5312 when DRW fired. Lesson: LDAM's per-class margin
  is the lift, DRW is a regression on this dataset (see §4).

#### ArcFace, Sub-Center ArcFace, Cosine, Prototype heads

- **Purpose:** Angular-margin / metric-learning heads explored as
  alternatives to the MLP head; in principle should improve open-set /
  imbalanced separation.
- **Mechanism:** ArcFace adds an angular margin `m` to the target-class
  cosine before scaling by `s` and applying softmax-CE. Sub-Center
  ArcFace allows K sub-centroids per class (Deng et al. ECCV 2020) so that
  acoustically-heterogeneous classes (like Tanker, which contains multiple
  hull types and operating regimes) can be modelled by multiple
  prototypes.
- **References:**
  - Deng et al. *ArcFace: Additive Angular Margin Loss for Deep Face
    Recognition.* CVPR 2019. arXiv:1801.07698.
  - Deng et al. *Sub-Center ArcFace: Boosting Face Recognition by
    Large-Scale Noisy Web Faces.* ECCV 2020.
- **Empirical note (negative):** **Phase H4 Sub-Center ArcFace catastrophically
  failed** at val 0.4805, test ~0.48 — a 18.15 pt regression vs R1.
  Phase G γ-lever (vanilla ArcFace) similarly regressed at every margin/scale
  tested. Hypothesis: ArcFace's angular geometry overfits the val set under
  domain shift; the val→test gap blew from 3 pp to 15 pp. Do not revisit
  unless the augmentation / dataset regime substantially changes.

#### DEMON-MoE head (`head_type="demon_moe"`)

- **Purpose:** Each class gets its own routing path through a small expert
  ensemble; targets flatter per-class precision via implicit per-class
  capacity.
- **Mechanism:** Sparsely-gated MoE head, `moe_n_experts=4`,
  `moe_aux_weight=0.05` (load-balancing auxiliary loss à la Shazeer et
  al. 2017). The auxiliary balance term is added when `head_type="demon_moe"`
  and training.
- **References:**
  - Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated
    Mixture-of-Experts Layer.* ICLR 2017. arXiv:1701.06538.
  - DEMONet — arXiv:2411.02758.
- **Empirical note:** Phase H5 — DEMON-MoE delivered the **theoretical
  promise**: flatter per-class precision (0.71–0.75 across all four classes)
  but only at gated coverage. **Wins MP@cov0.85 at 0.7905 (+3.87 vs R1)**;
  full-coverage test 0.6128 was a 4.92 pt regression. Ships as the gated /
  85 %-coverage configuration alongside R1 as the full-coverage configuration.

### 3.5 Augmentation strategy

#### Waveform augmentation (V2 stack, `_WaveformAugV2`)

- **Purpose:** Train-time domain randomisation against the most realistic
  failure modes — additive ocean / corpus noise, multipath delay, pitch
  drift, gain variation.
- **Mechanism:** Per-batch independent: corpus noise from
  `OceanNoisePool` at random SNR ∈ `[snr_min, snr_max]` (Gaussian noise
  auto-suppressed if corpus fires); RIR / multipath via 3–5 tap delay-line
  FFT-convolved at random gains; pitch shift via interpolate-resample at
  ±`pitch_range`; final random gain.
- **References:**
  - Park et al. *SpecAugment: A Simple Data Augmentation Method for ASR.*
    Interspeech 2019. arXiv:1904.08779. (the SpecAugment-1D applied inside
    Stream A is from this work).
- **Empirical note (negative):** Phase G β-lever — V2's full waveform aug
  pack (corpus + RIR + pitch + branch dropout) regressed HydroHydra at
  every intensity tested, **opposite to V2's Phase F result on
  HydroPrecise**. Hypothesis: HydroHydra's branch composition + S4D
  backbone already saturates the regularisation budget. β is not in R1.
  Low-intensity rerun is gated on a backbone or loss change first.

#### Branch dropout (`branch_dropout_p`)

- **Purpose:** Forces the fusion projection to remain robust if any one
  branch is missing or noisy at inference.
- **Mechanism:** With probability `p` per training batch, one randomly
  chosen branch's pooled feature tensor is **zeroed in place** (preserving
  the static channel count expected by `fuse_proj`).
- **Empirical note:** Off in R1 (`branch_dropout_p=0.0`); part of the β
  pack that regressed.

#### Manifold mixup (`manifold_mixup_alpha`)

- **Purpose:** Hidden-state mixup as a regulariser of the post-fusion
  representation, qualitatively distinct from input-space waveform mixup.
- **Mechanism:** With probability `manifold_mixup_prob` and `α > 0`, mix
  two pooled features at the same `λ ∼ Beta(α, α)`; loss combines the
  primary classification loss for `y_a` and `y_b` with weights `λ`,
  `1 − λ`. Auto-disabled when ArcFace is the head (incompatible with
  margin geometry).
- **References:**
  - Verma et al. *Manifold Mixup: Better Representations by Interpolating
    Hidden States.* ICML 2019. arXiv:1806.05236.
- **Empirical note:** Phase H6 manifold mixup at α=0.4 scored
  test 0.6074 (−5.46 pt vs R1). Available as a flag; off in production.

## 4. What does NOT belong (negative findings worth recording)

| Lever | Result | Memory ref |
|---|---|---|
| Sub-Center ArcFace (`head_type="subcenter_arcface"`) | **−18.15 pt test/μP vs R1**; val 0.4805 | Phase H4 (project_hydra_phaseH_outcome.md) |
| Vanilla ArcFace (Phase G γ) | regressed at every scale ≤ 30 / margin tested; val→test gap blew from 3 pp to 15 pp | Phase G (project_hydra_phaseG_outcome.md) |
| LDAM-DRW with class-balanced β | DRW fired at epoch 40 → Tug precision crashed 0.6340 → 0.5312; LDAM-margin-only is the lift | Phase H1 |
| JTFS-lite (Δ/ΔΔ on Scattering) | val 0.6939, **test 0.6158 (−4.62 pt)** — δ-features overfit the val set | Phase H2 |
| Global attention block | val 0.6940, **test 0.6474 (−1.46 pt)**; HELIX-style insertion did not transfer | Phase H3 |
| Full V2 waveform-aug pack (β) | regressed every intensity tested; opposite to its result on HydroPrecise | Phase G β (project_hydra_phaseG_outcome.md) |
| Manifold mixup (Phase H6) | **−5.46 pt test/μP** | Phase H6 |
| Stacked P0 levers (LDAM + JTFS + Global Attn) | **−7.51 pt test/μP** — negative compositionality | Phase H7 |
| Greedy weight soup of multi-seed R1 | val collapsed to ~0.10 — seeds lie in different basins | Phase I §3 |
| Test-time augmentation (TTA) | F1 drops monotonically with K (0.85 → 0.81 → 0.81 → 0.78); training augmentations are off-distribution at test time | Dirichlet stacker campaign 2026-05-10 |
| Logit adjustment (Menon et al. 2021) | trades F1 for high precision; on the F1 frontier it is a regression (τ=2 → −11 pp F1) | Dirichlet stacker campaign 2026-05-10 |
| Stacker fit on training data | 0.866 → 0.832 — never fit a calibration stage on data the underlying models trained on | Dirichlet stacker campaign 2026-05-10 |

## 5. Production checkpoints + reproduction

Three production configurations now ship.

### 5.1 R1 single-seed (full-coverage)

- **Ckpt:** `lightning_logs/phaseG_R1_alpha/version_0/checkpoints/hydra-026-p0.6908.ckpt`
- **Companions:** `temperature.pt`, `thresholds.json`, `selective_pr.md` colocated in `version_0/`.
- **Metrics shipped:** val/μP=0.6908, test/μP=0.6620, post-cal MP@cov0.85=0.7518.

#### 5.1.1 Exact feature manifest — what is ON in R1

The R1 ship-config is the **α-lever** subset: Gabor + Scattering1D +
SincNet + TDSBE. All other branches and most regularisation levers are
**OFF**. The complete active-feature table (with the parameter values
flowing into `HydroHydra.__init__`):

**Front-end branches active in R1:**

| Branch | Flag | State in R1 | Key parameters at R1 |
|---|---|---|---|
| Gabor (Stream A) | `use_gabor` | **ON** | `gabor_n_filters=64`, `gabor_kernel=257`, `gabor_ch=128` |
| Scattering1D (Stream B) | `use_scattering` | **ON** | `scat_J=6`, `scat_Q=8`, `scat_ch=128`, `use_jtfs=False` |
| SincNet (Stream C) | `use_sincnet` | **ON** | `sinc_n_filters=64`, `sinc_kernel=251`, `sinc_ch=128` |
| TDSBE (Stream D) | `use_tdsbe` | **ON** | `tdsbe_ch=64`, 4 fixed FIR bands ((20,250),(250,1k),(1k,2k),(2k,2540)) Hz, `frame=64`, `hop=32` |
| wav2vec2 (Stream E) | `use_w2v` | OFF | — |
| LPC (Stream F) | `use_lpc` | OFF | — |
| Recurrence Plot (Stream G) | `use_rp` | OFF | — |

**Backbone / pooling / head active in R1:**

| Component | Flag | State | Parameters |
|---|---|---|---|
| Fusion projection | (always on) | ON | `fusion_T=80`, `fusion_dim=192` |
| SaShiMi/S4D blocks | `s4_n_blocks` | ON | `s4_n_blocks=2`, `s4_d_state=64` |
| Global attention | `use_global_attn` | OFF | (Phase H3 regressed test by 1.46 pt) |
| Attentive Stat Pool | (always on) | ON | output dim = `2 × fusion_dim = 384` |
| Feature norm | `feature_norm` | OFF | `feature_norm="none"` |
| Head | `head_type` | **MLP + abstain logit** | `head_type="mlp"`, emits `num_classes + 1 = 5` logits |

**Loss / regularisation active in R1:**

| Component | Flag | State | Parameters |
|---|---|---|---|
| Primary loss | `loss` | LMF | `loss="lmf"`, `lmf_gamma=2.0`, `lmf_margin=0.7`, `label_smoothing=0.05` |
| Deep-Gamblers abstention | `gambler_weight` | **ON** | `gambler_weight=0.1`, `gambler_o=0.3` |
| LDAM-DRW | (loss flag) | OFF | (Phase H1 statistical tie, DRW hurt Tug) |
| Manifold mixup | `manifold_mixup_alpha` | OFF | `manifold_mixup_alpha=0.0` (Phase H6 regressed −5.46 pt) |
| Branch dropout | `branch_dropout_p` | OFF | `branch_dropout_p=0.0` (β regressed) |
| Dropout (general) | `dropout` | ON | `dropout=0.15` |

**Augmentation active in R1 (lean — V2 stack present but most knobs at zero):**

| Aug | Flag | State | Parameters |
|---|---|---|---|
| Gaussian noise injection | `noise_prob` | ON | `noise_prob=0.5`, `noise_snr_min=15`, `noise_snr_max=30` dB |
| Random gain | `gain_prob` | ON | `gain_prob=0.5`, `gain_range=0.3` |
| Corpus / ocean noise | `corpus_noise_prob` | OFF | `corpus_noise_prob=0.0` (β-pack, regressed) |
| RIR multipath | `rir_prob` | OFF | `rir_prob=0.0` (β-pack) |
| Pitch shift | `pitch_prob` | OFF | `pitch_prob=0.0` (β-pack) |
| SpecAugment-1D (on Gabor) | (internal to `_GaborStream`) | ON | `n_freq_masks=2, freq_mask_max=6, n_time_masks=2, time_mask_max=80` |

**Optimiser:**

| Setting | Value |
|---|---|
| Optimiser | AdamW (β=(0.9, 0.98), eps=1e-8) |
| `learning_rate` | `3e-4` |
| `weight_decay` | `1e-2` (decay group), `0.0` (bias / 1-D params) |
| `warmup_epochs` | `10` |
| `max_epochs` | `100` |
| LR schedule | linear warmup → cosine decay |
| Seed (R1 specifically) | `1337` |

#### 5.1.2 CLI to reproduce R1 verbatim

Flags omitted below take their `HydroHydra.__init__` defaults, which match
the manifest above.

```
python -m training.train_hydra \
  --use_gabor --use_scattering --use_sincnet --use_tdsbe \
  --loss lmf --lmf_gamma 2.0 --lmf_margin 0.7 \
  --label_smoothing 0.05 --gambler_weight 0.1 --gambler_o 0.3 \
  --seed 1337 --max_epochs 100 --warmup_epochs 10
```

### 5.2 5-seed N=5 logit ensemble (production baseline)

- **Ckpts:** five R1 checkpoints across seeds {1337, 42, 2026, 7, 12345};
  per-seed paths in `project_hydra_phaseI_outcome.md` §"Source-file stats".
  Per-seed best-val epochs differ wildly (10–32) — early-stop schedule is
  the dominant source of seed diversity in the ensemble:
  - s1337: `phaseG_R1_alpha/.../hydra-026-p0.6908.ckpt` (epoch 26)
  - s42: `phaseI_R1_s42/.../hydra-012-p0.6929.ckpt` (epoch 12)
  - s2026: `phaseI_R1_s2026/.../hydra-010-p0.6943.ckpt` (epoch 10)
  - s7: `phaseI_R1_s7/version_1/.../hydra-032-p0.7010.ckpt` (epoch 32)
  - s12345: `phaseI_R1_s12345/.../hydra-031-p0.7042.ckpt` (epoch 31)
- **Feature manifest:** identical to §5.1.1 — every member is the R1
  config with only the seed varying.
- **Inference:** `python -m inference.ensemble_eval --ckpts <5-ckpts> --data_dir <DATA_DIR> --out_dir <OUT>`.
- **Combiner:** arithmetic mean of class logits, post-temperature
  scaling (`temperature = 2.012`).
- **Metrics shipped:** val/μP=0.7421, test/μP=0.6545, **MP@cov0.85=0.8034**
  (+5.16 pt vs R1), Passenger@0.85=0.6936 (was 0.6113), Tug@0.85=0.8874.
- **Cost:** 5× single-model inference; acceptable for offline UATR.
  Real-time fallback: s12345 solo (val 0.7051, test 0.6415).

### 5.2b H5 DEMON-MoE — gated 85 %-coverage configuration

This is the alternative ship-config when **gated MP@cov0.85** is the
production gate rather than full coverage. Identical to R1 except the
head is swapped to DEMON-MoE.

- **Ckpt:** `lightning_logs/phaseH_H5_demon_moe/version_0/checkpoints/hydra-069-p0.7279.ckpt`
- **Metrics shipped:** val 0.7279, test 0.6128 (full-coverage regression
  is intentional), **MP@cov0.85 = 0.7905** (+3.87 pt vs R1's 0.7518),
  per-class precision flatter at 0.71–0.75 across all four classes,
  temperature 3.358.
- **Feature delta from R1:** only the head and its auxiliary loss change.

| Component | R1 value | H5 value |
|---|---|---|
| `head_type` | `"mlp"` | `"demon_moe"` |
| `moe_n_experts` | (unused) | `4` |
| `moe_aux_weight` | (unused) | `0.05` (load-balance loss à la Shazeer 2017) |
| `gambler_weight` | `0.1` | `0.0` (auto-disabled when head≠"mlp") |
| All other flags | — | unchanged from R1 |

- **CLI to reproduce H5:** R1 CLI minus `--gambler_weight 0.1`, plus
  `--head_type demon_moe --moe_n_experts 4 --moe_aux_weight 0.05`.

### 5.3 N=5 + Dirichlet stacker (NEW SHIP, 2026-05-10)

The arithmetic-mean N=5 ensemble is now superseded by a Dirichlet stacker
fit on the 5-seed log-probabilities — a single
`sklearn.LogisticRegression(C=0.1)` learning model-specific weights AND
per-class log-prior corrections. Dethrones every hand-engineered combiner
on F1.

- **Stacker artefact:** `lightning_logs/dirichlet_stacker_full/stacker.joblib`
  (MLP(64) K=10 fit on Split1s_eval val).
- **Inference:** `inference/ensemble_dirichlet.py
  --classifier {dirichlet_lr | mlp64_k10 | mlp64x32_k5}
  --stacker_path lightning_logs/dirichlet_stacker_full/stacker.joblib`.
- **Metrics shipped:** F1 = **0.956** on Split1s_eval test, macroP = 0.955
  (vs N=5 arith F1=0.743 on the same set). Per-class lift: Cargo +0.151,
  Passenger +0.198 (was the historical bottleneck), Tanker +0.010,
  Tug +0.132. Bootstrap robustness: F1=0.850 ± 0.015, MLP > LR in 59 %
  of resamples.
- **Production rule:** never re-fit the stacker on data the underlying
  Hydra models trained on (training-data fit drops F1 to 0.832).

## 6. References

1. Andén, J. & Mallat, S. *Deep Scattering Spectrum.* IEEE Transactions on
   Signal Processing, 62(16):4114–4128, 2014. arXiv:1304.6763.
2. Andreux, M. et al. *Kymatio: Scattering Transforms in Python.* JMLR
   21(60):1–6, 2020. arXiv:1812.11214.
3. Baevski, A., Zhou, H., Mohamed, A. & Auli, M. *wav2vec 2.0: A Framework
   for Self-Supervised Learning of Speech Representations.* NeurIPS 2020.
   arXiv:2006.11477.
4. Cao, K., Wei, C., Gaidon, A., Aréchiga, N. & Ma, T. *Learning Imbalanced
   Datasets with Label-Distribution-Aware Margin Loss.* NeurIPS 2019.
   arXiv:1906.07413.
5. CATFISH UATR. arXiv:2505.23964.
6. Cui, Y., Jia, M., Lin, T.-Y., Song, Y. & Belongie, S. *Class-Balanced
   Loss Based on Effective Number of Samples.* CVPR 2019. arXiv:1901.05555.
7. DEMONet. arXiv:2411.02758.
8. Deng, J., Guo, J., Niannan, X. & Zafeiriou, S. *ArcFace: Additive Angular
   Margin Loss for Deep Face Recognition.* CVPR 2019. arXiv:1801.07698.
9. Deng, J., Guo, J., Liu, T., Gong, M. & Zafeiriou, S. *Sub-Center
   ArcFace: Boosting Face Recognition by Large-Scale Noisy Web Faces.*
   ECCV 2020.
10. Eckmann, J. P., Kamphorst, S. O. & Ruelle, D. *Recurrence Plots of
    Dynamical Systems.* Europhys. Lett. 4(9):973–977, 1987.
11. Goel, K., Gu, A., Donahue, C. & Ré, C. *It's Raw! Audio Generation with
    State-Space Models.* ICML 2022. arXiv:2202.09729.
12. Gu, A., Goel, K., Gupta, A. & Ré, C. *On the Parameterization and
    Initialization of Diagonal State Space Models.* NeurIPS 2022.
    arXiv:2206.11893.
13. Hatami, N., Gavet, Y. & Debayle, J. *Classification of Time-Series
    Images Using Deep Convolutional Neural Networks.* ICMV 2017.
    arXiv:1710.00886.
14. HELIX. arXiv:2603.21316. *[citation needs verification]*
15. Hu, J., Shen, L. & Sun, G. *Squeeze-and-Excitation Networks.* CVPR 2018.
    arXiv:1709.01507.
16. Gao, S.-H. et al. *Res2Net: A New Multi-scale Backbone Architecture.*
    IEEE TPAMI 43(2):652–662, 2021. arXiv:1904.01169.
17. Khosla, P. et al. *Supervised Contrastive Learning.* NeurIPS 2020.
    arXiv:2004.11362.
18. LEAF — Zeghidour, N., Teboul, O., de Chaumont Quitry, F. & Tagliasacchi,
    M. *LEAF: A Learnable Frontend for Audio Classification.* ICLR 2021.
    arXiv:2101.08596.
19. Lin, T.-Y., Goyal, P., Girshick, R., He, K. & Dollár, P. *Focal Loss
    for Dense Object Detection.* ICCV 2017. arXiv:1708.02002.
20. Liu, L. et al. *Large-Margin Softmax Loss for Convolutional Neural
    Networks.* ICML 2016. arXiv:1612.02295.
21. Liu, Z. et al. *Deep Gamblers: Learning to Abstain with Portfolio
    Theory.* NeurIPS 2019. arXiv:1907.00208.
22. Makhoul, J. *Linear Prediction: A Tutorial Review.* Proc. IEEE
    63(4):561–580, 1975.
23. Marple, S. L. *Computing the Discrete-Time Analytic Signal via FFT.*
    IEEE Transactions on Signal Processing, 47(9):2600–2603, 1999.
24. Menon, A. K. et al. *Long-Tail Learning via Logit Adjustment.* ICLR
    2021. arXiv:2007.07314.
25. Nielsen, R. O. *Sonar Signal Processing.* Artech House, 1991.
26. Okabe, K., Koshinaka, T. & Shinoda, K. *Attentive Statistics Pooling
    for Deep Speaker Embedding.* Interspeech 2018. arXiv:1803.10963.
27. Park, D. S. et al. *SpecAugment: A Simple Data Augmentation Method for
    Automatic Speech Recognition.* Interspeech 2019. arXiv:1904.08779.
28. Ravanelli, M. & Bengio, Y. *Speaker Recognition from Raw Waveform with
    SincNet.* SLT 2018. arXiv:1808.00158.
29. Shazeer, N. et al. *Outrageously Large Neural Networks: The Sparsely-
    Gated Mixture-of-Experts Layer.* ICLR 2017. arXiv:1701.06538.
30. Subramani, K. et al. *LPC-style auxiliary features.* Interspeech 2022.
    arXiv:2202.11301; LPCSE arXiv:2206.06908.
31. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
    arXiv:1706.03762.
32. Verma, V. et al. *Manifold Mixup: Better Representations by
    Interpolating Hidden States.* ICML 2019. arXiv:1806.05236.
33. Xiong, R. et al. *On Layer Normalization in the Transformer
    Architecture.* ICML 2020. arXiv:2002.04745.
34. Zhang, H., Cisse, M., Dauphin, Y. N. & Lopez-Paz, D. *mixup: Beyond
    Empirical Risk Minimization.* ICLR 2018. arXiv:1710.09412.
