# Phase I — I2 LPC residual branch: training failure

## Outcome

| metric | I2 | R1 baseline | Δ |
|---|---|---|---|
| val/μP | 0.0652 | 0.6908 | −0.6256 (random-class chance) |
| test/μP | 0.0865 | 0.6620 | −0.5755 |
| temperature | NaN | 3.29 | (calibration fit failed) |

Training collapsed at epoch 0 with `train/loss = nan` and never recovered.
The model output was effectively constant — `val/macro_precision` stuck at
0.06520 for 16 epochs before EarlyStopping fired. The LPC branch produced
NaN gradients and broke the entire model.

## Root cause (suspected)

The Levinson-Durbin recursion in `_LPCStream._levinson` uses prediction-
error update:

```python
E = E * (1.0 - k.unsqueeze(-1).pow(2)).clamp(min=eps)
```

For ship-machinery signals with strong tonal harmonics, the reflection
coefficients `k_i` can exceed 1 in magnitude when the autocorrelation
matrix is ill-conditioned. When `|k| > 1`, the term `(1 - k²)` becomes
negative; the `.clamp(min=eps)` floors it at 1e-6. The next iteration's
`k_{i+1} = -acc / E` then divides by the tiny floor and explodes,
producing NaN values that propagate through the conv1d residual filter
to the loss.

Smoke tests passed on synthetic Gaussian noise (which has bounded
autocorrelation by construction) but Lexar's coherent ship signals
trigger the unstable case.

## Fix (deferred — not implementing during the sweep)

Replace the unstable Levinson recursion with **reflection-coefficient
parameterization with tanh squashing**, equivalent to what librosa's LPC
does internally:

```python
# In place of: k = -acc / E
# Use:        k = torch.tanh(-acc / E.clamp(min=1e-3))
```

The tanh forces `|k| < 1`, guaranteeing a numerically stable filter.
Combined with Levinson's standard recursion, this produces minimum-phase
LPC filters that are always invertible.

Alternative (cheaper): drop the residual stream entirely and use only
the LPC coefficients (which are stable when bounded by tanh). The
"source/filter factorisation" claim still holds — coefficients are the
"filter" half — but we lose the residual transient highlighting.

## Effort to fix

~30 LOC in `models/hydro_hydra.py:_LPCStream._levinson`. After fix, smoke
test on a real-data batch (not just synthetic noise) before re-launching
the run. Estimated: 1h debug + 6h re-train.

## Phase I sweep impact

The orchestrator continued to I3 (RP) after I2 finished cleanly with
exit=0 (the failure was silent — no exception, just NaN loss). I3 and I4
(LPC + RP) are unaffected by the LPC bug except for I4, which will
likely also collapse since it includes `--use_lpc`. I4's NaN may
prematurely halt that run too; will check when it reaches.

## Action

- Marked I2 task remains "completed" (it ran end-to-end without crashing).
- Documented this failure in `phaseI_I2_findings.md`.
- Phase I orchestrator continues with I3 RP and (probably-broken) I4
  LPC+RP. Will re-evaluate the LPC fix priority after the rest of Phase I
  lands.
