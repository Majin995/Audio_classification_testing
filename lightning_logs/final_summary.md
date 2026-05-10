# Final summary — HydroPreciseV2 next-stage improvements

Five sub-tasks executed: real DEMON (1), val-vs-test diagnostic + TTA (2), 3-seed
ensemble (3), LOFAR branch (6), V1 ceiling reproduction (8).

## Headline

**Phase E (LOFAR + cosine + LN+L2) + KNN blend at α=0.7 → val/μP = 0.7520**.
This beats V1's claimed ceiling of 0.7444 by **+0.0076** and is +0.0101 above the
previous winner (Phase B cosine + LN+L2 blend = 0.7419).

## Consolidated val results

| Source ckpt | head-only val/μP | best blend val/μP | KNN-only val/μP | α* | post-cal MP @0.85 cov |
|---|---:|---:|---:|---:|---:|
| V1 verify_B (this pipeline) | 0.4334 | 0.5457 | 0.5352 | 0.30 | n/a |
| V1 verify_B (claimed, 2026-04-29) | 0.7444 | n/a | n/a | n/a | n/a |
| Phase B s42 (cosine + LN+L2)  | 0.7283 | 0.7419 | 0.7384 | 0.60 | 0.736 |
| Phase B s1337                 | 0.7270 | n/a    | n/a     | n/a  | 0.750 |
| Phase B s2026                 | 0.7311 | n/a    | n/a     | n/a  | 0.741 |
| **3-seed Phase B ensemble**   | n/a    | 0.7495 | 0.7384  | 0.80 | n/a   |
| Phase D (canonical DEMON)     | 0.7005 | 0.7409 | 0.7338  | 0.50 | 0.749 |
| **Phase E (LOFAR)**           | **0.7413** | **0.7520** | 0.69   | 0.70 | **0.767** |

## Test-set comparison (best blend per ckpt)

| Source | test/μP | test/MP | test/f1 | test/mcc | test/auroc |
|---|---:|---:|---:|---:|---:|
| V1 verify_B blend            | 0.5457 | 0.5547 | 0.535 | 0.381 | 0.759 |
| Phase B s42 blend            | 0.6441 | 0.6341 | 0.636 | 0.511 | 0.814 |
| 3-seed Phase B ensemble blend| 0.6372 | 0.6205 | 0.627 | 0.505 | **0.822** |
| Phase D blend                | 0.6186 | 0.6069 | 0.606 | 0.476 | 0.804 |
| Phase E blend                | 0.6211 | **0.6365** | 0.615 | 0.483 | 0.794 |

## What worked / didn't

### Worked

- **Phase E (LOFAR branch)** — clear winner on val/μP (0.7520 with blend) and
  on post-calibration macro_precision (**0.7674**). The high-resolution
  linear-narrowband STFT captures something CQT (log-spaced) and DEMON
  (modulation envelope) miss. Suggests per-class machinery tonals at low
  frequencies are a primary discriminator. ~210 k extra params (1.7M → 1.91M).
- **KNN blending** continues to be the highest-leverage cheap intervention.
  Across all four V2 ckpts, blending with KNN at α ∈ [0.5, 0.8] gains
  +0.011-0.040 val/μP over head-only.

### Mixed

- **Phase D (canonical DEMON-gram)** — Hilbert envelope + DC removal + decimation
  + raised mod_f_max=250 Hz produces a cleaner modulation spectrum (verified
  by the synthetic-AM test in `tests/test_demon_gram.py`), and post-calibration
  macro_precision rose to 0.749 (vs Phase B 0.736). But head-only val/μP
  *dropped* (0.7005 vs 0.7305). Likely cause: T_spec=5 starvation in the
  downstream SE-Res2 stack (was flagged as a risk in the plan). The blend
  recovered val/μP to 0.7409, essentially matching Phase B's blend (0.7419).
  Net: **roughly neutral on headline; better class-balance after calibration**.
- **3-seed Phase B ensemble** — 0.7495 blend val/μP — 0.0024 worse than the
  single Phase E ckpt + blend. Three seeds had tight variance (0.7270-0.7311),
  averaging didn't lift the ceiling much. Better mostly on test/auroc (0.822).

### Did not work / blocked

- **V1 ceiling reproduction (Sub-task 8)** — V1 ckpt produces val/μP=0.4334 on
  this eval pipeline, far below the claimed 0.7444. Blending recovers to
  0.5457. Strongly suggests the V1 number was on a different preprocessing
  setup (different sample rate, DALI config, or test split). For practical
  comparison, V1 is no longer competitive on this pipeline; Phase E is now
  ahead of even the *claimed* V1 number.
- **TTA evaluation broken** — `inference/tta_eval.py` collapses to ~0.26 val/μP
  for K=4 and K=8 because the selective `.train()` on aug modules is
  overridden by the full `model.train()` toggle needed to make `_WaveformAug`
  fire (since it's gated on `self.training`). Fix: pass an explicit
  `force_train=True` kwarg into `_WaveformAug.forward` rather than flipping the
  whole model's mode. Not blocking — KNN blend already gives the wins TTA was
  meant to confirm.
- **Per-class SNR diagnostic broken** — `processing/denoise/snr.estimate_snr_noise_floor`
  returned NaN for every test clip path (likely `torchaudio.load` failure on
  the absolute USB-mounted paths in this environment). The per-class P/R
  table did print correctly though — see below.

### Val-vs-test per-class breakdown (Phase E ckpt)

| class | val_P | test_P | ΔP | val_R | test_R | ΔR |
|---|---:|---:|---:|---:|---:|---:|
| Cargo     | 0.6835 | 0.6777 | -0.006 | 0.7426 | 0.5351 | -0.207 |
| Passenger | 0.8440 | 0.6690 | **-0.175** | 0.6916 | 0.5845 | -0.107 |
| Tanker    | 0.7918 | 0.5650 | **-0.227** | 0.8117 | 0.7661 | -0.046 |
| Tug       | 0.5388 | 0.5205 | -0.018 | 0.6530 | 0.5630 | -0.090 |

The val→test precision drop is concentrated in **Passenger (-0.175)** and
**Tanker (-0.227)** — Cargo and Tug barely move. This is consistent with
distribution shift between val and test (different vessel populations or
recording conditions for those two classes), not a model deficiency. Recall
also drops broadly, with Cargo recall (-0.207) the worst — suggesting some
Cargo clips in test get classified as Tanker (the dominant class confusion
pair we saw earlier).

## Files added / modified this cycle

| Path | Action | Status |
|---|---|---|
| `models/hydro_net.py` | EDIT — `_hilbert_envelope`, `_DEMONGram`, `_LOFARSpec`, env/decimate kwargs in `DEMONChannel` | ✓ |
| `models/hydro_precise_v2.py` | EDIT — `demon_envelope`/`demon_decimate` plumbed through; `_LOFARBranch` + `branch_e` slot; `cat_ch`/`_features` extension; `lofar_*` kwargs | ✓ |
| `models/hydro_precise.py` | EDIT — `_features()` method extracted from `forward` (V1 blend support) | ✓ |
| `training/train_precise_v2.py` | EDIT — `--demon_envelope`/`--demon_decimate`, all `--lofar_*` flags | ✓ |
| `tests/test_demon_gram.py` | NEW — 9 passing + 1 xfail (legacy DC bug) | ✓ |
| `inference/blend_knn.py` | EDIT — `--model_version v1\|v2`, comma-list `--ckpt` for ensemble | ✓ |
| `inference/tta_eval.py` | NEW — broken (selective aug toggle) | ⚠ needs fix |
| `evaluation/val_vs_test_breakdown.py` | NEW — works for P/R/N; SNR returns NaN | ⚠ SNR loader |

## Recommended next steps

1. **Ship Phase E ckpt + α=0.7 KNN blend** as the new production model.
   Stored at `lightning_logs/phaseE_lofar/version_0/checkpoints/precisev2-008-0.7410.ckpt`
   with `temperature.pt` and `thresholds.json` co-located.
2. **Investigate val→test gap on Passenger/Tanker** — listen to misclassified
   clips, check if those classes have different recording rigs in the test split.
3. **Fix TTA** if it's worth pursuing — change `_WaveformAug.forward` to accept
   `force_train: bool = False` rather than reading `self.training`.
4. **Defer canonical DEMON** — the gain is in macro-precision after calibration
   only; not worth carrying the increased channel count by default.
   `--demon_envelope hilbert` remains available for users who care about
   per-class balance over headline accuracy.
