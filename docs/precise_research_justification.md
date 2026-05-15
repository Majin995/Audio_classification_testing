# HydroPrecise / V2 — Research Justification

## 1. Design philosophy

HydroPrecise is a **spectrogram-domain** UATR classifier that takes the
opposite architectural bet to HydroHydra: instead of waveform-only branch
diversity, it commits to a small number (3 → 6) of complementary
*time–frequency* views — adaptive narrowband (Gabor + PCEN), constant-Q
log-spaced (CQT), and modulation-domain (DEMON) — and invests
representation budget in a heavier per-branch SE-Res2 stack and a
cross-attention fuser. v2 generalises the v1 design into an ablation
surface (v2-A → v2-G) that adds Gammatone, LOFAR, wav2vec2, optional S4D,
optional DART parallel-residual, optional Mean-Teacher EMA self-distillation,
SupCon auxiliary, and SpecAugment-on-all-branches.

The model targets **macro-precision** under the same 4-class Lexar /
Split1s constraints as HydroHydra. v1's verify-grid pinned the ceiling at
**val/μP = 0.7444 (best ckpt) / 0.7353 (2-seed mean)**; v2 surfaces are a
research/ablation track around that baseline.

## 2. Why this dataset, why these constraints

Identical to HydroHydra: DeepShip-like 4-class corpus
(Cargo / Passenger / Tanker / Tug), 5120 Hz hydrophone, 1 s clips.
Macro-precision is the production metric. Imbalance is severe (Tug ≪
Cargo).

The spectrogram-domain choice is justified by:

- **Narrowband shaft / blade harmonics** dominate machinery signatures; CQT's
  log-spaced bins give octave-uniform resolution where these tonals live.
- **DEMON modulation lives at 1–~30 Hz** with harmonics out to ~250 Hz; a
  dedicated linear-modulation branch surfaces this content in a way no
  raw-waveform 1D conv reaches without an explicit envelope step.
- **PCEN** stabilises the long-time mean of high-Q tonals against
  channel-gain variation and far-field roll-off — both endemic in the
  recording corpus.
- The 5120 Hz / 1 s window is short enough that the parameter budget for
  a per-branch SE-Res2 stack stays under ~4 M params (verified) —
  affordable for production inference.

## 3. Component-by-component justification

### 3.1 Front-end branches

#### Branch A — Learnable Gabor + (optional) PCEN + SE-Res2 stack

- **Purpose:** Adaptive narrowband front-end that picks up cavitation
  bursts, blade-rate transients and tonal onsets at filterbank centre
  frequencies that train end-to-end with the rest of the model.
- **Mechanism:** `LearnableGaborFilterbank` (96 filters, kernel 257),
  optional `SpecAugment1D` (frequency + time masking on the filterbank
  output), stride-16 strided stem, then `_seres2_stack(n_blocks=2)` of
  SE-Res2 blocks with linearly increasing dilation.
- **References:**
  - Zeghidour et al. *LEAF: A Learnable Frontend for Audio Classification.*
    ICLR 2021. arXiv:2101.08596.
  - CATFISH UATR. arXiv:2505.23964.
  - Hu, Shen, Sun. *Squeeze-and-Excitation Networks.* CVPR 2018.
    arXiv:1709.01507.
  - Gao et al. *Res2Net: A New Multi-scale Backbone Architecture.*
    TPAMI 2021. arXiv:1904.01169.
- **Empirical note:** Always-on across v1 and v2; v1's verify_B winner
  uses default Gabor settings; in v2 the same branch carries n_blocks=2
  and additionally provides the spectral-flux signal to the
  boundary-aware attention (§3.2).

#### Branch B — CQT (Constant-Q Transform) + Trainable PCEN + 2D backbone

- **Purpose:** Octave-uniform log-spaced narrowband resolution targeting
  shaft / blade harmonic structure that has constant fractional bandwidth.
- **Mechanism:** `CQT1992v2` (`fmin=20 Hz, n_bins=84/96, bpo=12,
  hop=64`); falls back to MelSpectrogram when `nnAudio` is missing. Output
  is normalised by either Trainable PCEN (default in v2) or
  `InstanceNorm1d` (v1). 2D conv backbone has two stride-(2,1) stages —
  collapses frequency into channels — then a 1×1 projection into
  (B, out_ch, T). v2 appends `_seres2_stack(n_blocks=2)`.
- **References:**
  - Brown, J. C. *Calculation of a constant Q spectral transform.* JASA
    89(1):425–434, 1991.
  - Schörkhuber, C. & Klapuri, A. *Constant-Q Transform Toolbox for Music
    Processing.* Sound and Music Computing 2010.
  - Wang et al. *Trainable Frontend For Robust and Far-Field Keyword
    Spotting.* ICASSP 2017. arXiv:1607.05666.
  - Lostanlen et al. *Per-Channel Energy Normalization: Why and How.*
    IEEE Signal Processing Letters 2019.
- **Empirical note:** Always-on. PCEN-on-CQT is the default in v2-B
  onward (`pcen_on_cqt=True`). v1 verify_B with `label_smoothing=0.05,
  gambler_weight=0.0, lmf_margin=0.30, lmf_gamma=2.0` is the v1 ceiling
  configuration at val/μP=0.7444.

#### Branch C — Multi-band DEMON (linear modulation spectrogram)

- **Purpose:** **Detection of Envelope Modulation On Noise** — the
  classical passive-sonar approach to extract blade-rate amplitude
  modulation from cavitation broadband. Designed to dominate the
  Tug / Tanker discrimination channel.
- **Mechanism:** Per-subband (default `(800, Nyquist)` Hz, configurable
  multi-band) FFT-mask bandpass → squared envelope (or Hilbert / FWR
  options) → `_LinearModulationSpec` (linear-frequency STFT magnitude
  truncated to `[mod_f_min, mod_f_max]`, default 0–50 Hz) → Trainable
  PCEN per subband. Concatenated bins go through a 1D conv stem → SE-Res2
  stack → bidirectional GRU (slow temporal smoothing of modulation
  trajectories).
- **References:**
  - Nielsen, R. O. *Sonar Signal Processing.* Artech House, 1991.
  - Vaccaro, R. J. *Passive Sonar Signal Processing.* IEEE Signal
    Processing Magazine, 1998.
  - de Moura et al. *Passive sonar signal detection and classification
    based on independent component analysis.* In *Sonar Systems*, 2011.
- **Empirical note (rewrite 2026-04-28):** the previous mel-bank DEMON
  wasted ~31/32 bins above the BPF range; the linear-modulation rewrite
  packs the bins into the discriminative 0–50 Hz region (BPF + harmonics
  to ~250 Hz). `demon_n_mels` was removed. v2-A's first lindemon Optuna
  rerun (`hydro_precise_v2A_lindemon`) launched 2026-04-28 with
  `mod_f_max=50, n_fft=2048` → 21 linear bins. v2-A linear-DEMON
  underperformed v1 verify_B by ~4 pt val/μP (peak v2-A 0.7069 vs verify_B
  0.7444) — a regression versus the v1 best.

#### Branch D (v2 optional) — Gammatone filterbank (`use_gammatone_branch`)

- **Purpose:** ERB-spaced auditory filtering — biologically-motivated
  resolution complementary to CQT's musical-octave spacing, particularly
  in the low-frequency tonal range.
- **Mechanism:** `GammatoneSpectrogram` (`f_min=20 Hz`, 64 ERB-spaced
  bands) → Trainable PCEN → SpecAugment-1D → conv stem → SE-Res2 stack.
- **References:**
  - Patterson, R. D. et al. *An Efficient Auditory Filterbank Based on
    the Gammatone Function.* APU report 2341, 1988.
  - Slaney, M. *An Efficient Implementation of the Patterson-Holdsworth
    Auditory Filter Bank.* Apple TR #35, 1993.
- **Empirical note:** Off by default; enabled in v2-F. The v2-A surface
  showed v2-F's Gammatone + DART pair did not lift past the v1 verify_B
  ceiling on the focused 60-trial scaleup.

#### Branch E (v2 optional) — LOFAR (`use_lofar_branch`)

- **Purpose:** **LOw-Frequency Analysis and Recording** — the standard
  passive sonar high-resolution narrowband display, providing uniform
  linear-Hz resolution at low frequencies where machinery tonals
  cluster and where the log-spaced CQT bins are sparse.
- **Mechanism:** `_LOFARSpec` (`n_fft=4096, hop=160, max_freq=2560 Hz`,
  256 linear bins) → SpecAugment → Conv1d projection → SE-Res2 stack.
- **References:**
  - Maranda, B. H. *LOFARgram-based passive ranging.* US Naval Research
    Laboratory technical work; standard sonar processing.
  - Nielsen, R. O. *Sonar Signal Processing.* Artech House, 1991.
- **Empirical note:** Available as a 5th stream behind the
  `use_lofar_branch` flag; not in default v2-G.

#### Branch F (v2 optional) — Frozen wav2vec2 conv front-end

- **Purpose:** Pretrained acoustic prior (generic broadband texture) at
  ~0.3 M trainable params; the transformer is dropped, only the conv
  stem trains.
- **Mechanism:** `_PretrainedBranch` resamples 5120 → 16000 Hz, runs the
  frozen wav2vec2-base feature extractor, then projects to `out_ch` and
  through one SE-Res2 block. Frozen extractor stays in `eval()` even
  when the parent is in `train()` — its BN running stats must not
  update.
- **References:**
  - Baevski et al. *wav2vec 2.0.* NeurIPS 2020. arXiv:2006.11477.
- **Empirical note:** Optional; `use_pretrained_branch=False` by default.
  Adds ~95 M frozen params + ~0.3 M trainable.

### 3.2 Sequence modelling backbone (fusion stage)

#### Cross-attention fusion (`_CrossAttnFuser` in v1, `_FusedAttnBlock` in v2)

- **Purpose:** After branch features are pooled to a common `T_f` and
  channel-projected through `fuse_proj`, multi-head self-attention lets
  any time-step query any other across branches — addresses the
  "ship-machinery feature is rare and time-localised" failure mode of
  pure pooling.
- **Mechanism:** Pre-norm or post-norm MHSA + FFN block at `n_heads=2`
  (v1) or `n_heads=4` (v2), residual + LayerNorm. v2 stacks
  `n_attn_blocks` (default 1).
- **References:**
  - Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
    arXiv:1706.03762.
- **Empirical note:** Always on; never ablated separately because it is
  the only generic token-mixing in the architecture.

#### Boundary-aware attention (BAHTNet, `use_boundary_attn`)

- **Purpose:** Onset events (cavitation bursts, blade-pass impacts) carry
  disproportionate discriminative content; a boundary mask biases the
  self-attention toward those time-steps.
- **Mechanism:** `SpectralFluxOnset` is computed from the pooled Gabor
  features (B, gabor_ch, T_f) → emits a per-frame onset score; that mask
  is fed into `BoundaryAwareAttention` as an additive attention prior.
- **References:**
  - Bello, J. P. et al. *A Tutorial on Onset Detection in Music Signals.*
    IEEE TSAP 13(5):1035–1047, 2005.
  - Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
    arXiv:1706.03762.
  - BAHTNet paper [citation needs verification — internal reference;
    treat as "spectral-flux-gated cross-attention" if the formal cite is
    not located].
- **Empirical note:** On in v2-C onward; off in v2-A. Combined with
  `n_s4d_blocks=1` constitutes the v2-C surface delta over v2-B.

#### S4D / SaShiMi blocks (`n_s4d_blocks`)

- **Purpose:** Linear-time long-context state-space modelling for slow
  spectral evolution within the 1 s window.
- **Mechanism:** `SaShiMiBlock` instances (default 1 in v2, 2 in v1+
  variants) wrapping a diagonal S4 layer with HIPPO-LegS-derived
  initialisation and a separate SSM learning-rate group via
  `ssm_lr_mult=0.1` in `configure_optimizers`.
- **References:**
  - Goel et al. *It's Raw! Audio Generation with State-Space Models.*
    ICML 2022. arXiv:2202.09729.
  - Gu, Goel, Gupta, Ré. *On the Parameterization and Initialization of
    Diagonal State Space Models.* NeurIPS 2022. arXiv:2206.11893.
- **Empirical note:** `n_s4d_blocks=1` is the v2 default (v2-C onward).
  Lean v2-A drops S4D entirely (`n_s4d_blocks=0`).

#### DART parallel-residual block (`use_dart_block`)

- **Purpose:** A parallel depthwise-conv ‖ multi-head attention block
  that captures local temporal patterns and global dependencies in one
  residual step.
- **Mechanism:** `DARTBlock` (kernel_size=15, ff_expansion=2) — depthwise
  conv branch and MHA branch fed in parallel and summed back into the
  residual stream.
- **References:**
  - DART block — internal name; cited generically as parallel
    depthwise-conv + MHA fusion in the same family as Conformer
    (Gulati et al. Interspeech 2020). [citation needs verification —
    if formal DART paper unavailable, treat as Conformer-family
    ablation.]
- **Empirical note:** Off by default; on in v2-F. Ships as part of the
  v2-F (Gammatone + DART) surface combination.

### 3.3 Pooling

#### Attentive Statistics Pooling (`_AttentiveStatisticsPool`)

- **Purpose:** Soft-attention weighted mean + std along the time axis,
  producing a `2 * fusion_dim` utterance embedding.
- **Mechanism:** Same as in HydroHydra — see
  `models/hydro_fusion._AttentiveStatisticsPool`. Consumed by every head.
- **References:**
  - Okabe, K., Koshinaka, T. & Shinoda, K. *Attentive Statistics Pooling
    for Deep Speaker Embedding.* Interspeech 2018. arXiv:1803.10963.
- **Empirical note:** Standard, never ablated.

### 3.4 Head + auxiliary losses

#### MLP head + Deep-Gamblers abstention logit (v1 default; optional v2)

- **Purpose:** Same rationale as HydroHydra — production target is gated
  macro-precision, abstention is intrinsic to the loss.
- **Mechanism:** v1's MLP head emits `num_classes + 1` logits (last is
  the abstention). Loss is
  `LMF(class_logits, y) + λ · (-log(p_y + o · p_abstain))` with
  `o = gambler_o = 0.3`. v2 drops the abstention logit by default
  (`head_type="mlp"` returns `num_classes` logits via `build_head`)
  but the abstention path is reachable behind the legacy MLP+Gamblers
  configuration.
- **References:**
  - Liu et al. *Deep Gamblers.* NeurIPS 2019. arXiv:1907.00208.
- **Empirical note:** **v1 verify_B winner uses `gambler_weight=0.0`
  (gambler off) with `label_smoothing=0.05`** — at the v1 ceiling, the
  gambler auxiliary did not help beyond label smoothing alone. The
  preceding verify_A (`gambler_weight=0.1, label_smoothing=0.0`) was
  ~0.5 pt below verify_B on both seeds. **v1 production CLI:**
  ```
  training/train_precise.py --lmf_margin 0.30 --lmf_gamma 2.0 \
    --label_smoothing 0.05 --gambler_weight 0.0 \
    --lr 3e-4 --batch_size 64 --max_epochs 60 --warmup_epochs 8 \
    --patience 15 --target_coverage 0.85 --precision bf16-mixed
  ```

#### Cosine / Prototype / ArcFace heads (v2 `head_type` flag)

- **Purpose:** Surface alternative metric-learning heads.
- **Mechanism:** `build_head` resolves `head_type ∈ {mlp, cosine,
  prototype, arcface, mlp_wide}`. ArcFace consumes labels at training
  time (margin geometry); inference uses `labels=None`.
- **References:**
  - Deng et al. *ArcFace.* CVPR 2019. arXiv:1801.07698.
- **Empirical note:** No v2 surface used a non-MLP head as the ship
  config — same conclusion as HydroHydra Phases G/H: ArcFace family is
  fragile under domain shift on this dataset.

#### Loss family flag (v2: `loss ∈ {focal, lmf, ldam, cb_focal}`)

- **Mechanism:**
  - `lmf` — Large-Margin Focal: `LargeMarginFocalLoss(num_classes,
    α=class_weights, γ=focal_gamma, margin=lmf_margin,
    label_smoothing)`.
  - `ldam` — LDAM with margin `max_m / n_c^(1/4)`, scale `s`, optional
    DRW (handled in trainer).
  - `cb_focal` — Class-Balanced Focal with effective-number-of-samples
    weighting, β=0.999.
  - `focal` — vanilla focal loss.
- **References:**
  - Lin et al. *Focal Loss.* ICCV 2017. arXiv:1708.02002.
  - Liu et al. *Large-Margin Softmax.* ICML 2016. arXiv:1612.02295.
  - Cao et al. *LDAM.* NeurIPS 2019. arXiv:1906.07413.
  - Cui et al. *Class-Balanced Loss.* CVPR 2019. arXiv:1901.05555.
- **Empirical note:** **v1 verify_B uses `loss="lmf",
  lmf_margin=0.30, lmf_gamma=2.0`**, dethroning the Phase F grid-winner
  (`m=0.30 g=2.0 s=0.00 gw=0.1` verify_A at 0.7308) by ~0.45 pt mean.
  The Optuna `uatr_hydro_precise_demon_gambler_v2` sweep peaked at
  trial #34 (val 0.7452) with `lmf_margin=0.7, lmf_gamma=1.5,
  label_smoothing=0.0, gambler_weight=0.1, lr=1e-3` — **caveat:** trial
  ran only 10 epochs vs verify_B's 60-epoch / patience-15 schedule, so
  the 0.7452 is not directly comparable until reproduced over 60 epochs
  with 2 seeds.

#### Supervised contrastive auxiliary (`aux_supcon_weight`)

- **Purpose:** Pull same-class embeddings together, push different-class
  apart in the projection space — a representation regulariser
  orthogonal to the classification loss.
- **Mechanism:** `_supcon_loss(features, labels, temperature=0.07)` on
  L2-normalised projections from a 2-layer MLP `supcon_head`. Default
  weight 0.1.
- **References:**
  - Khosla, P. et al. *Supervised Contrastive Learning.* NeurIPS 2020.
    arXiv:2004.11362.
- **Empirical note:** Caveat (file-level): when `mixup_alpha > 0`, the
  SupCon loss uses *mixed* features but *un-mixed* labels — noisy by
  design, not worth special-casing. Enabled in v2-E.

#### Mean-Teacher EMA self-distillation (`mean_teacher_weight`)

- **Purpose:** Self-distillation regulariser that uses an EMA copy of
  the model as a stable teacher; the consistency loss between two
  stochastic augmentation views replaces unlabeled-data semi-supervision
  when no unlabeled corpus exists.
- **Mechanism:** Eager `copy.deepcopy(self)` in `__init__` (required for
  SWA + checkpoint compatibility — doubles VRAM when on); EMA decay
  `mean_teacher_ema_decay=0.999`; consistency loss is
  `MSE(softmax(student), softmax(teacher.detach()))` on a second
  stochastic augmentation view; weight ramps up sigmoidally over
  `mean_teacher_rampup_epochs=10`.
- **References:**
  - Tarvainen, A. & Valpola, H. *Mean Teachers are Better Role Models.*
    NeurIPS 2017. arXiv:1703.01780.
- **Empirical note:** Eager teacher creation is a deliberate caveat (not
  a bug). Teacher's own `mean_teacher_weight=0` to prevent recursive
  teacher creation. Enabled in v2-G; disabled in v2-A through v2-F.

#### Logit adjustment at evaluation (`logit_adjust_tau`)

- **Mechanism:** At eval (`not self.training`), subtracts
  `τ · log_prior` from logits to correct for class-prior bias.
  `log_prior` is registered from `class_weights` (treated as inverse-
  frequency).
- **References:**
  - Menon et al. *Long-Tail Learning via Logit Adjustment.* ICLR 2021.
    arXiv:2007.07314.
- **Empirical note (negative — see HydroHydra §4):** prior correction
  trades F1 for precision; on the F1 frontier it is a regression. Useful
  only as a precision knob.

### 3.5 Augmentation strategy

#### Waveform augmentation (`_WaveformAug`)

- **Purpose:** Train-time domain randomisation (matches HydroHydra V2
  stack — corpus noise, RIR, pitch, gain).
- **Mechanism:** Identical pipeline to HydroHydra's `_WaveformAugV2`:
  corpus noise from `OceanNoisePool` (auto-suppresses synthetic Gaussian
  if corpus fires), RIR (3–5 tap delay-line FFT-conv), pitch via
  interpolate-resample, final random gain.
- **References:** see Park et al. SpecAugment for the SpecAug-1D inside
  individual branches.

#### Waveform Mixup (`mixup_alpha`)

- **Purpose:** Standard mixup at the waveform input — convex combinations
  of two clips and their labels regularise both representation and
  decision boundary.
- **Mechanism:** With `α > 0`, `λ ∼ Beta(α, α)`, `x_m = λ·x + (1-λ)·x[π]`
  with random permutation π; loss is
  `λ · L(logits, y) + (1-λ) · L(logits, y[π])`. Default `α=0.2`. ArcFace
  receives mixed features but unmixed primary labels (head needs labels
  at train time).
- **References:**
  - Zhang, H. et al. *mixup: Beyond Empirical Risk Minimization.*
    ICLR 2018. arXiv:1710.09412.
- **Empirical note:** Enabled in v2-D onward. v1 verify_B does **not**
  use mixup.

#### SpecAugment on every branch (`spec_aug_all_branches`)

- **Purpose:** Branch-specific frequency / time masking on the 2D and 1D
  representations, prevents the heavier per-branch SE-Res2 stacks from
  overfitting to dominant tonal lines.
- **Mechanism:** `SpecAugment1D(n_freq_masks=2, freq_mask_max=...,
  n_time_masks=2, time_mask_max=...)` with branch-specific mask widths
  matched to each branch's frequency-axis resolution.
- **References:**
  - Park, D. S. et al. *SpecAugment.* Interspeech 2019. arXiv:1904.08779.
- **Empirical note:** On by default in v2 (`spec_aug_all_branches=True`,
  v2-B onward); v1 had SpecAugment only on the Gabor branch.

#### Branch dropout (`branch_dropout_p`)

- **Purpose:** Forces the fusion layer to remain robust to a missing
  branch.
- **Mechanism:** With probability `p` per training batch, one randomly
  chosen branch's pooled tensor is zeroed in place (channel count
  preserved).
- **Empirical note:** Off by default. The v1→v2 migration kept it
  available behind a flag because it fired clean in V2 Phase F but
  regressed HydroHydra in Phase G β.

## 4. What does NOT belong (negative findings worth recording)

| Lever | Result | Memory ref |
|---|---|---|
| v2-A linear-DEMON config space | peak val/μP=0.7069 vs v1 verify_B 0.7444 → **−4 pt regression**; v2-A is currently a regression vs v1 | project_hydro_precise_v1_grid.md |
| v1 Optuna trial #34 (lmf_margin=0.7, lmf_gamma=1.5, label_smoothing=0.0, gambler_weight=0.1, lr=1e-3, demon_n_fft=4096) | peak val/μP=0.7452 at 10 epochs only; **not directly comparable** to verify_B's 60-epoch schedule. Reproduce over 60 epochs / 2 seeds before treating as a beat. | project_hydro_precise_v1_grid.md |
| Gambler loss at v1 ceiling | verify_A (gw=0.1, s=0.0) at 0.7308 vs verify_B (gw=0.0, s=0.05) at 0.7353 — **gambler off + label smoothing on dominates** at the v1 ceiling | project_hydro_precise_v1_grid.md |
| Mel-bank DEMON | Wasted ~31/32 bins above the BPF range; rewritten to linear modulation spectrogram 2026-04-28 | project_hydro_precise_v2.md |
| `demon_n_mels` argument | Removed everywhere in DEMON rewrite | project_hydro_precise_v2.md |
| Plan's 1.5–2.0 M parameter budget | Optimistic; real budget is 2.0 M (lean v2-A) → 4.1 M (full v2-G) | project_hydro_precise_v2.md |
| Mean-Teacher VRAM cost | doubles VRAM when `mean_teacher_weight > 0` due to eager teacher deepcopy in `__init__`; deliberate (SWA + checkpoint compatibility) | project_hydro_precise_v2.md |
| HydroPrecise + HydroHydra cross-arch ensembling | blocked by ckpt head-size mismatch (5-output abstain head vs 4-class loader); not pursued after Dirichlet stacker dethroned every hand-engineered combiner on Hydra alone | project_dirichlet_stacker_campaign.md |

## 5. Production checkpoints + reproduction

### 5.1 v1 verify_B (current production v1 baseline)

- **Ckpt:** `lightning_logs/grid_precise_verify/verify_B_m0.30_g2.0_s0.05_gw0.0_seed2026/version_0/checkpoints/precise-013-p0.7444.ckpt`
- **Companions:** `temperature.pt`, `thresholds.json` colocated.
- **Metrics shipped:** val/μP=0.7444 (best ckpt) / 0.7353 (2-seed mean
  across seeds 1337, 2026).

#### 5.1.1 Exact feature manifest — what is ON in v1 verify_B

verify_B is the standard 3-branch HydroPrecise (Gabor + CQT + DEMON) at
its grid-search winner. v2-only branches and v2-only auxiliary heads do
not exist in v1 and are listed N/A below.

**Front-end branches active in v1 verify_B:**

| Branch | State | Parameters at verify_B |
|---|---|---|
| Gabor (Branch A) | **ON** | `gabor_n_filters=64`, `gabor_kernel=257`, `gabor_ch=128`; SpecAugment-1D on filterbank output (`n_freq_masks=2, freq_mask_max=6, n_time_masks=2, time_mask_max=80`); 2× SE-Res2 blocks |
| CQT (Branch B) | **ON** | `cqt_n_bins=84`, `cqt_bpo=12`, `cqt_hop=64`, `fmin=20.0` Hz, `cqt_ch=128`; **InstanceNorm** (v1 — PCEN is a v2 default); 2D-conv backbone collapses freq into channels |
| DEMON (Branch C) | **ON** | `demon_hop=64`, `demon_ch=64`, `demon_n_fft=2048`, `demon_mod_f_min=0.0`, `demon_mod_f_max=50.0`, `f_cav=800 Hz` BPF; Conv1d stem → bidirectional GRU (`gru_hidden=64`, output dim `2*64=128`) |
| Gammatone | N/A in v1 | (v2-F flag) |
| LOFAR | N/A in v1 | (v2 flag) |
| wav2vec2 | N/A in v1 | (v2 flag) |

**Backbone / pooling / head active in v1 verify_B:**

| Component | State | Parameters |
|---|---|---|
| Fusion projection | ON | `fusion_T=64`, `fusion_dim=192`, `cat_ch = 128 + 128 + 128 = 384` |
| Cross-attention fuser | ON | `_CrossAttnFuser`, `n_heads=2`, post-norm MHSA + FFN |
| S4D / SaShiMi | OFF | (v1 has no S4D in the fusion stage; v2-only flag) |
| Boundary-aware attention | OFF | (v2 flag) |
| DART block | OFF | (v2 flag) |
| Attentive Stat Pool | ON | output dim `2 × fusion_dim = 384` |
| Head | **MLP + abstain logit** | 2-layer MLP, emits `num_classes + 1 = 5` logits; abstain column dropped at inference |

**Loss / regularisation active in v1 verify_B (the grid-search winner):**

| Component | State | Parameters |
|---|---|---|
| Primary loss | LMF | `loss="lmf"`, `lmf_gamma=2.0`, **`lmf_margin=0.30`**, **`label_smoothing=0.05`** |
| Deep-Gamblers | **OFF** | **`gambler_weight=0.0`** (this is the verify_B winner over verify_A's `gw=0.1, s=0.0`) |
| `gambler_o` | (unused) | `0.3` (would be used if gw>0) |
| SupCon auxiliary | N/A in v1 | (v2 flag) |
| Mean-Teacher | N/A in v1 | (v2 flag) |
| Logit adjustment | N/A in v1 | (v2 flag) |
| Mixup | N/A in v1 | (v2 flag, v2-D onward) |
| Dropout | ON | `dropout=0.15` |

**Augmentation active in v1 verify_B:**

| Aug | State | Parameters |
|---|---|---|
| Gaussian noise injection | ON | `noise_prob=0.5`, `noise_snr_min=15`, `noise_snr_max=30` dB |
| Random gain | ON | `gain_prob=0.5`, `gain_range=0.3` |
| SpecAugment-1D (Gabor only) | ON | as per filterbank parameters above |
| SpecAugment on CQT / DEMON | N/A in v1 | (v2 `spec_aug_all_branches` flag) |
| Corpus noise / RIR / pitch | N/A in v1 | (v2 _WaveformAug additions) |
| Branch dropout | N/A in v1 | (v2 flag) |

**Optimiser:**

| Setting | Value |
|---|---|
| Optimiser | AdamW (β=(0.9, 0.98), eps=1e-8) |
| `learning_rate` | `3e-4` |
| `weight_decay` | `1e-2` |
| `warmup_epochs` | `8` |
| `max_epochs` | `60` |
| `patience` (early-stop) | `15` |
| LR schedule | linear warmup → cosine decay |
| `batch_size` | `64` |
| `precision` | `bf16-mixed` |
| `target_coverage` (post-cal) | `0.85` |
| Best seed (verify_B winner) | `2026` (epoch 13, val 0.7444); s1337 mean partner reached 0.7261 |

#### 5.1.2 CLI to reproduce verify_B verbatim

```
python -m training.train_precise \
  --lmf_margin 0.30 --lmf_gamma 2.0 \
  --label_smoothing 0.05 --gambler_weight 0.0 \
  --lr 3e-4 --batch_size 64 --max_epochs 60 --warmup_epochs 8 \
  --patience 15 --target_coverage 0.85 --precision bf16-mixed
```

### 5.2 v2 ablation surface

v2 is currently a **research / ablation track**, not a production ship.
The flag map:

| Surface | Toggle delta from defaults |
|---|---|
| v2-A (lean) | `--n_s4d_blocks 0 --aux_supcon_weight 0 --mean_teacher_weight 0 --no_boundary_attn` |
| v2-B | + SE-Res2-on-all + PCEN-on-CQT (defaults) |
| v2-C | + boundary attn + S4D=1 (defaults; remove `--no_boundary_attn`) |
| v2-D | + Mixup + SpecAug-all (defaults; `--mixup_alpha 0.2`) |
| v2-E | + SupCon + SWA (`--aux_supcon_weight 0.1 --swa`) |
| v2-F | + Gammatone + DART (`--use_gammatone_branch --use_dart_block`) |
| v2-G | + Mean-Teacher (`--mean_teacher_weight 0.5`) |

#### 5.2.1 Cumulative feature manifest per v2 surface

Each surface inherits everything from the row above, so the manifest is
cumulative. ON / OFF columns refer to whether the listed component is
active in that surface. Best v2-A val/μP observed on the focused
60-trial scaleup was **0.7069** — a regression vs v1 verify_B's 0.7444,
which is why no v2 surface has shipped to production yet.

| Component | v2-A (lean) | v2-B | v2-C | v2-D | v2-E | v2-F | v2-G (full) |
|---|---|---|---|---|---|---|---|
| **Branches** | | | | | | | |
| Gabor + SE-Res2 stack | ON (n_blocks=2) | same | same | same | same | same | same |
| CQT + 2D backbone + SE-Res2 stack | ON (n_blocks=2) | same | same | same | same | same | same |
| DEMON multi-band + SE-Res2 + GRU | ON (n_blocks=1) | same | same | same | same | same | same |
| PCEN-on-CQT (`pcen_on_cqt`) | OFF (InstanceNorm) | **ON** | ON | ON | ON | ON | ON |
| SE-Res2-on-all (already on by default) | ON | ON | ON | ON | ON | ON | ON |
| Gammatone branch (`use_gammatone_branch`) | OFF | OFF | OFF | OFF | OFF | **ON** | ON |
| LOFAR branch (`use_lofar_branch`) | OFF | OFF | OFF | OFF | OFF | OFF | OFF (still off in v2-G default) |
| wav2vec2 branch (`use_pretrained_branch`) | OFF | OFF | OFF | OFF | OFF | OFF | OFF |
| **Fusion stage** | | | | | | | |
| Cross-attention fuser (`_FusedAttnBlock`) | ON | ON | ON | ON | ON | ON | ON |
| Boundary-aware attention (`use_boundary_attn`) | OFF (`--no_boundary_attn`) | OFF | **ON** | ON | ON | ON | ON |
| S4D blocks (`n_s4d_blocks`) | `0` | `0` | **`1`** | `1` | `1` | `1` | `1` |
| DART parallel-residual (`use_dart_block`) | OFF | OFF | OFF | OFF | OFF | **ON** | ON |
| **Head + auxiliary** | | | | | | | |
| MLP head | ON (no abstain) | ON | ON | ON | ON | ON | ON |
| Deep-Gamblers abstention | OFF | OFF | OFF | OFF | OFF | OFF | OFF |
| SupCon auxiliary (`aux_supcon_weight`) | `0.0` | `0.0` | `0.0` | `0.0` | **`0.1`** | `0.1` | `0.1` |
| SWA | OFF | OFF | OFF | OFF | **ON** | ON | ON |
| Mean-Teacher (`mean_teacher_weight`) | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | **`0.5`** |
| **Augmentation** | | | | | | | |
| Mixup (`mixup_alpha`) | `0.0` | `0.0` | `0.0` | **`0.2`** | `0.2` | `0.2` | `0.2` |
| SpecAugment on every branch (`spec_aug_all_branches`) | OFF (Gabor only) | OFF | OFF | **ON** | ON | ON | ON |
| Waveform aug (noise, gain) | ON | ON | ON | ON | ON | ON | ON |
| Corpus noise / RIR / pitch | OFF (defaults; flag-gated separately) | OFF | OFF | OFF | OFF | OFF | OFF |

**Defaults shared across all v2 surfaces (from `HydroPreciseV2.__init__`):**

| Setting | Value |
|---|---|
| `gabor_n_filters / kernel / ch` | `96 / 257 / 192` |
| `cqt_n_bins / bpo / hop / ch` | `96 / 12 / 64 / 192` |
| `demon_hop / ch` | `64 / 128` |
| `demon_subbands` | `[(800.0, sample_rate/2.0)]` (single-band default) |
| `demon_n_fft / mod_f_min / mod_f_max` | `2048 / 0.0 / 50.0` (linear modulation spec) |
| `demon_envelope` | `"square"` (alts: `"hilbert"`, `"fwr"`) |
| `seres2_blocks_per_branch` | `(2, 2, 1, 1)` (Gabor, CQT, DEMON, Gammatone) |
| `fusion_T / fusion_dim / n_heads / n_attn_blocks` | `64 / 256 / 4 / 1` |
| `s4d_d_state` | `64` |
| `dropout / drop_path` | `0.25 / 0.10` |
| `loss / focal_gamma / lmf_margin / label_smoothing` | `"focal" / 2.0 / 0.5 / 0.05` (defaults; LDAM and CB-Focal need `cls_num_list`) |
| `head_type / feature_norm` | `"mlp" / "none"` |
| `learning_rate / weight_decay / ssm_lr_mult` | `3e-4 / 1e-3 / 0.1` |
| `warmup_epochs / max_epochs` | `5 / 100` |
| `mean_teacher_ema_decay / rampup_epochs` | `0.999 / 10` |

**Parameter budgets (verified via smoke test, see `project_hydro_precise_v2.md`):**

| Surface | Trainable params |
|---|---|
| v2-A (lean) | ~2.0 M |
| v2-C / D / E (default) | ~3.1 M |
| v2-G (full + Mean-Teacher) | ~4.1 M (teacher EMA copy doubles VRAM at runtime) |

- **Tuning entry point:** `training/tune_hydro_precise_v2.py` (study
  `uatr_hydro_precise_v2`).
- **Best metric for early-stop & ckpt selection:** `val/micro_precision`
  (matches v1 to keep cross-version comparisons honest).
- **Output locations:** `lightning_logs/hydro_precise_v2*` for full
  ablations, `lightning_logs/optuna_precise_v2_t*` for Optuna trials.

## 6. References

1. Baevski, A., Zhou, H., Mohamed, A. & Auli, M. *wav2vec 2.0: A Framework
   for Self-Supervised Learning of Speech Representations.* NeurIPS 2020.
   arXiv:2006.11477.
2. Bello, J. P. et al. *A Tutorial on Onset Detection in Music Signals.*
   IEEE TSAP 13(5):1035–1047, 2005.
3. Brown, J. C. *Calculation of a constant Q spectral transform.* JASA
   89(1):425–434, 1991.
4. Cao, K., Wei, C., Gaidon, A., Aréchiga, N. & Ma, T. *Learning Imbalanced
   Datasets with Label-Distribution-Aware Margin Loss.* NeurIPS 2019.
   arXiv:1906.07413.
5. CATFISH UATR. arXiv:2505.23964.
6. Cui, Y. et al. *Class-Balanced Loss Based on Effective Number of
   Samples.* CVPR 2019. arXiv:1901.05555.
7. de Moura, N. N. et al. *Passive sonar signal detection and
   classification based on independent component analysis.* In *Sonar
   Systems*, ed. N. Z. Kolev, IntechOpen, 2011.
8. Deng, J. et al. *ArcFace: Additive Angular Margin Loss for Deep Face
   Recognition.* CVPR 2019. arXiv:1801.07698.
9. Gao, S.-H. et al. *Res2Net: A New Multi-scale Backbone Architecture.*
   IEEE TPAMI 43(2):652–662, 2021. arXiv:1904.01169.
10. Goel, K., Gu, A., Donahue, C. & Ré, C. *It's Raw! Audio Generation
    with State-Space Models.* ICML 2022. arXiv:2202.09729.
11. Gu, A., Goel, K., Gupta, A. & Ré, C. *On the Parameterization and
    Initialization of Diagonal State Space Models.* NeurIPS 2022.
    arXiv:2206.11893.
12. Gulati, A. et al. *Conformer: Convolution-augmented Transformer for
    Speech Recognition.* Interspeech 2020. arXiv:2005.08100.
13. Hu, J., Shen, L. & Sun, G. *Squeeze-and-Excitation Networks.* CVPR
    2018. arXiv:1709.01507.
14. Khosla, P. et al. *Supervised Contrastive Learning.* NeurIPS 2020.
    arXiv:2004.11362.
15. Lin, T.-Y. et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
    arXiv:1708.02002.
16. Liu, L. et al. *Large-Margin Softmax Loss.* ICML 2016.
    arXiv:1612.02295.
17. Liu, Z. et al. *Deep Gamblers: Learning to Abstain with Portfolio
    Theory.* NeurIPS 2019. arXiv:1907.00208.
18. Lostanlen, V. et al. *Per-Channel Energy Normalization: Why and How.*
    IEEE Signal Processing Letters, 2019.
19. Maranda, B. H. *LOFARgram-based passive ranging.* Naval Research
    Laboratory technical work; standard sonar processing.
20. Menon, A. K. et al. *Long-Tail Learning via Logit Adjustment.* ICLR
    2021. arXiv:2007.07314.
21. Nielsen, R. O. *Sonar Signal Processing.* Artech House, 1991.
22. Okabe, K., Koshinaka, T. & Shinoda, K. *Attentive Statistics Pooling
    for Deep Speaker Embedding.* Interspeech 2018. arXiv:1803.10963.
23. Park, D. S. et al. *SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition.* Interspeech 2019. arXiv:1904.08779.
24. Patterson, R. D. et al. *An Efficient Auditory Filterbank Based on
    the Gammatone Function.* APU report 2341, 1988.
25. Schörkhuber, C. & Klapuri, A. *Constant-Q Transform Toolbox for Music
    Processing.* Sound and Music Computing 2010.
26. Slaney, M. *An Efficient Implementation of the Patterson-Holdsworth
    Auditory Filter Bank.* Apple TR #35, 1993.
27. Tarvainen, A. & Valpola, H. *Mean Teachers are Better Role Models:
    Weight-Averaged Consistency Targets Improve Semi-Supervised Deep
    Learning Results.* NeurIPS 2017. arXiv:1703.01780.
28. Vaccaro, R. J. *Passive Sonar Signal Processing.* IEEE Signal
    Processing Magazine, 1998.
29. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
    arXiv:1706.03762.
30. Wang, Y. et al. *Trainable Frontend For Robust and Far-Field Keyword
    Spotting.* ICASSP 2017. arXiv:1607.05666.
31. Zeghidour, N., Teboul, O., de Chaumont Quitry, F. & Tagliasacchi, M.
    *LEAF: A Learnable Frontend for Audio Classification.* ICLR 2021.
    arXiv:2101.08596.
32. Zhang, H., Cisse, M., Dauphin, Y. N. & Lopez-Paz, D. *mixup: Beyond
    Empirical Risk Minimization.* ICLR 2018. arXiv:1710.09412.
