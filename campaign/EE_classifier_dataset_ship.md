# Classifier_Dataset production stacker

## Setup
- 5 HydroHydra Phase-I seed ckpts → log-prob features (5×4=20 dims)
- Stacker: LR(C=0.1, class_weight='balanced')
- Per-class thresholds: [0.364, 0.303, 0.303, 0.606]
- Fit val: n=50752 (Cargo 13236, Passenger 15045, Tanker 17087, Tug 5384)
- Test: n=14208

## Test metrics
```
raw (no thresholds):
  acc=0.6255  f1=0.5997  macroP=0.6027  microP=0.6255  recall=0.6160  mcc=0.4926
    P0=0.636  P1=0.815  P2=0.609  P3=0.352
    R0=0.494  R1=0.688  R2=0.768  R3=0.514
    F1_0=0.556  F1_1=0.746  F1_2=0.679  F1_3=0.418
with thresholds:
  acc=0.6545  f1=0.6042  macroP=0.6326  microP=0.6545  recall=0.5890  mcc=0.5244  cov=0.935
    P0=0.637  P1=0.816  P2=0.609  P3=0.468
    R0=0.485  R1=0.687  R2=0.767  R3=0.417
    F1_0=0.551  F1_1=0.746  F1_2=0.679  F1_3=0.441
```

## Comparison vs baseline
- N=5 arith baseline: F1=0.5621, macroP=0.6545
- Production stacker (raw): F1=0.5997, macroP=0.6027
- Production stacker (cal): F1=0.6042, macroP=0.6326
- **Δ F1 (cal vs baseline) = +0.0421**

## Artifacts
- `lightning_logs/dirichlet_stacker_classifier_dataset/stacker.joblib` — fitted LR
- `lightning_logs/dirichlet_stacker_classifier_dataset/thresholds.npy` — per-class thresholds (pre-applied)
- `lightning_logs/dirichlet_stacker_classifier_dataset/stacker_meta.json` — full metrics + provenance
- `lightning_logs/dirichlet_stacker_classifier_dataset/final_test_probs.npy` — raw (pre-threshold) softmax probs