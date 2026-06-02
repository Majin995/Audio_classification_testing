# SOTA Stacker — Per-Class Confidence Distributions

Source: `campaign/sota_confidence_dist.py` on `lightning_logs/hydro_recurrent_stacker_combined/best.pt`.
Test split: 236 sources, max-softmax-prob (MSP) as the confidence signal.

## Per-class summary

| class     | n  | conf μ | conf σ | p(true) μ | acc   | correct μ | incorrect μ |
|-----------|----|--------|--------|-----------|-------|-----------|-------------|
| Cargo     | 88 | 0.648  | 0.176  | 0.536     | 0.591 | 0.704     | 0.567       |
| Passenger | 43 | **0.885** | 0.197 | 0.787   | **0.814** | 0.955 | 0.577       |
| Tanker    | 74 | 0.810  | 0.212  | 0.666     | 0.689 | 0.884     | 0.647       |
| Tug       | 31 | 0.685  | 0.145  | 0.447     | 0.548 | 0.708     | 0.657       |

## Confidence percentiles per true class (MSP)

| class     | 10%   | 25%   | 50%   | 75%   | 90%   |
|-----------|-------|-------|-------|-------|-------|
| Cargo     | 0.462 | 0.510 | 0.617 | 0.749 | 0.942 |
| Passenger | 0.510 | 0.878 | 0.996 | 0.999 | 1.000 |
| Tanker    | 0.520 | 0.581 | 0.932 | 0.999 | 1.000 |
| Tug       | 0.509 | 0.552 | 0.668 | 0.800 | 0.891 |

Passenger and Tanker are bimodal at "either confidently right or not" — the
≥0.93 median is what makes MSP-isotonic selective so cheap on them.
Cargo and Tug both sit in a low-confidence band (μ ≈ 0.65–0.69) with little
spread — selective at cov0.85 can only thin them by a few sources.

## Confusion matrix (rows=true, cols=pred)

```
              Cargo  Passe  Tanke    Tug
   Cargo       52       3     18     15
Passenger       4      35      1      3
   Tanker      14       1     51      8
      Tug     10       2      2     17
```

Top error modes are Cargo↔Tanker (32 mis-routes) and Cargo↔Tug (25). These
are the same source-level confusions that bottleneck `cargo_confirm_5base`
and the AST runs.

## Why Tug is unrescuable by MSP-selective

Tug's *correct* vs *incorrect* confidence means (0.708 vs 0.657) are **only
0.05 apart** — there is essentially no MSP gap to threshold on. Any
selective rule fails to lift Tug F1 above ~0.51 on this checkpoint.

Compare Passenger: correct μ=0.955 vs incorrect μ=0.577 → 0.38 gap, which is
why MSP-iso boosts Passenger to F1 0.864 once we abstain on the cov0.15
low-confidence tail.

## Implications

- **Don't tune selective for Tug.** Coverage 0.85 already gives back what's
  reachable here; further abstention only sheds correct Tug calls.
- **Per-class iso helps because Cargo's MSP is mis-scaled.** Cargo MSP μ
  (0.648) is lower than Passenger/Tanker's (0.81–0.89), so the stacker is
  under-confident on the largest test class. Per-class isotonic rebalances
  this and is what nudges raw 0.6526 → 0.6590 → 0.7046 with selective.
- **The remaining gain lever is the Cargo-Tanker confusion, not calibration.**
  See `campaign/MOE_SWEEP_FINDINGS.md` for the MoE-route attempt and why it
  did not help — Cargo's expert was the only one that survived because the
  base ensemble's Cargo confidence is already the strongest of the four.

## Artifact

`lightning_logs/hydro_recurrent_stacker_combined/confidence_dump.npz` —
`y`, `pred`, `probs (236,4)`, `abstain (236,)`, `source_type (236,)`.

Re-run any time with:
```
python -m campaign.sota_confidence_dist
```
