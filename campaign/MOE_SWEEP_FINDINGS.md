# MoE Sweep — Findings on Combined_IARA_Deepship_1s

Run: `lightning_logs/moesweep_full_1s_1780240079/` (≈ 16.5 h wall-clock).
Backbone: **HydroComplete**. Dataset: 350k clips / 4 classes (Cargo, Passenger, Tanker, Tug).

## TL;DR

| variant            | val macroF1 | test macroF1 | val acc | test acc |
|---|---|---|---|---|
| **gate_only** (K=4) | **0.578**   | **0.519**    | 0.638   | 0.555    |
| moe_soft           | 0.448       | 0.416        | 0.527   | 0.480    |
| moe_hard           | 0.255       | 0.277        | 0.349   | 0.365    |

**The MoE does not beat its own gate** — combining a gate with one-vs-rest binary
experts loses 0.13 (soft) to 0.32 (hard) macroF1 vs the gate alone. Reasons
discussed in *Why MoE lost* below.

## Grid search — what won per class

3 × 3 × 3 sweep at `sweep_epochs=3` (108 sweep trainings; Tug skipped — see
note). Best cell per class on val *positive*-class F1:

| class     | best depth | best lr   | best margin | sweep val pos-F1 | final retrain val pos-F1 |
|---|---|---|---|---|---|
| Cargo     | 2          | 1e-04     | 0.70        | 0.6781           | 0.6588 |
| Passenger | 2          | 1e-04     | 0.50        | 0.6811           | 0.6803 |
| Tanker    | 1          | 1e-04     | 0.70        | 0.6390           | 0.6168 |
| Tug       | 2          | 1e-04     | 0.70 ¹      | (skipped)        | 0.4197 |

¹ Tug grid was deliberately skipped after Cargo / Passenger / Tanker; the
cross-class consensus winner (`lr=1e-04`, `d=2`, `m=0.70`) was applied
directly. Tug retrain hit val pos-F1 **0.4197** — a full 0.20 below the other
binary experts (matches the well-known Combined-1s Tug ceiling).

### Cross-cell observations

- **`lr=1e-04` is unanimously the best LR.** Across all three swept classes
  the top cell sits at `1e-04`. Sweep evidence is consistent: when LR ≥ 3e-4
  the val pos-F1 either flatlines or regresses, especially at higher depths.
- **Depth matters only weakly.** d2 wins for Cargo/Passenger but d1 wins for
  Tanker by 0.012. d3 is never best and often worst (extra capacity overfits
  the 3-epoch budget).
- **Margin (`lmf_margin`) is class-dependent.** Cargo and Tanker prefer 0.70,
  Passenger prefers 0.50. Lower margins look noisy across the grid.
- **Sweep → retrain drift exists.** Retraining the best cell at 20 ep on the
  same loader gave −0.01 to −0.02 vs the 3-ep sweep score for Cargo/Tanker;
  Passenger held up. So 3-epoch grid scores rank reasonably but slightly
  *over*-estimate the final ceiling.

## Combined MoE eval — per class

### val (n = 66,944 clips)

| class | gate_only | moe_hard | moe_soft |
|---|---|---|---|
| Cargo     | 0.691 | 0.632 | 0.661 |
| Passenger | 0.684 | 0.104 | 0.380 |
| Tanker    | 0.621 | 0.193 | 0.510 |
| Tug       | 0.316 | 0.093 | 0.241 |

### test (n = 61,440 clips)

| class | gate_only | moe_hard | moe_soft |
|---|---|---|---|
| Cargo     | 0.637 | 0.601 | 0.626 |
| Passenger | 0.625 | 0.171 | 0.361 |
| Tanker    | 0.480 | 0.181 | 0.398 |
| Tug       | 0.334 | 0.154 | 0.278 |

## Why MoE lost

The hard router demolishes Passenger / Tanker / Tug F1 by a factor of 4-6×
while leaving Cargo only mildly worse. The pattern is consistent with this
failure mode:

- The binary experts emit a **strong positive sigmoid for their own class**,
  so any sample the gate routes (correctly or incorrectly) to non-Cargo
  experts gets aggressively pulled toward the expert's positive class.
- The hard rule (`gate.argmax → confirm if expert ≥ 0.5 else fall to next-best`)
  too often *rejects* the gate's good answer and falls back to second-best.
- The soft rule (`P(c) ∝ gate[c] × expert_c[c]`) is also asymmetric:
  experts with miscalibrated high positives systematically promote a single
  class regardless of the gate's confidence.

In short: independently trained one-vs-rest experts are **not calibrated to
each other**, so their positive sigmoid sits on a different scale than the
gate's softmax. Product-of-experts amplifies the mismatch.

Cargo survives because the Cargo expert reaches the highest absolute val
pos-F1 (0.68 sweep / 0.66 final) and its gate also picks Cargo most reliably
of all four classes — so the two systems agree and reinforce.

## How this run compares to the project's SOTA

`CLAUDE.md` reports the current Combined-1s SOTA as
`HydroRecurrentStacker + MSP-isotonic gate` at test macroF1 **0.715**
(`campaign/eval_recurrent_stacker.py`).

| approach                      | test macroF1 |
|---|---|
| HydroRecurrentStacker + MSP gate (project SOTA) | 0.715 |
| **This run — gate_only HydroComplete d=3**      | 0.519 |
| This run — moe_soft                             | 0.416 |
| This run — moe_hard                             | 0.277 |

The gate alone is **−0.20 vs project SOTA** even before considering the MoE
penalty. Two clear reasons:

1. The gate here is a single HydroComplete with no stacking, no isotonic
   gate, no IARA-length-stratified fine-tune — the four ingredients that
   actually drive SOTA.
2. 20 epochs at lr=3e-4 (the middle of the swept LRs) on `--oversample_train`
   is well short of the typical project training schedule (50–100 ep with
   per-class scheduling).

So this run is best read as a **MoE methodology study** rather than an
attempt at SOTA: it answers "what does an offline-tuned MoE on this data
buy us?" and the answer is *less than the gate alone*.

## What would unlock MoE here

Three concrete fixes worth trying if we revisit:

1. **Calibrate experts to gate scale.** Fit isotonic regression on val per
   expert mapping `sigmoid(expert_logit) → P(correct positive)`; then route
   with calibrated probs. Mirrors the MSP-isotonic move that lifted
   HydroRecurrentStacker from 0.653 → 0.715.
2. **Train experts with class-balanced softmax over K, not one-vs-rest.**
   Keep the multi-class head but heavily upweight the target class
   (`--expert_mode multi --target_weight 4`). Same hyperparameter discovery
   pipeline, but the expert's output lives in the gate's index space and
   doesn't need separate calibration.
3. **Use the gate's softmax as a router prior.** Re-derive
   `P(c|x) = softmax(gate_logits + alpha · expert_logits)` with α tuned on
   val (start at α≈0.3). This caps how much an expert can override the gate
   and prevents the rare-class blow-up seen here.

## Operational notes

- **Total compute:** ≈ 16.5 h on an RTX 5090 (32 GB), bs=128, AMP 16-mixed.
- **Sweep:** 81 of 108 grid cells trained (Tug grid skipped after consensus
  was clear by the third class). Total grid wall ≈ 12 h.
- **Final phase:** 4 expert retrains (20 ep) + gate (20 ep) ≈ 4.5 h.
- **All artifacts** under `lightning_logs/moesweep_full_1s_1780240079/`:
  `sweep_summary.json` (raw grid), `sweep_manifest.json` (full eval),
  `sweep_report.md` (auto-generated tables), per-cell + per-final checkpoints.

## Decision

- **Don't ship MoE as currently assembled** — it loses to its own gate.
- The grid result *is* useful: `lr=1e-04` for HydroComplete binary heads is
  now established; `d=2` is the default to beat; `lmf_margin` should be
  class-tuned (0.70 vs 0.50 trade).
- Next experiment: try `--expert_mode multi --target_weight 4` with the same
  consensus winners — likely the cheapest path to having the MoE actually
  help, before reaching for isotonic calibration.
