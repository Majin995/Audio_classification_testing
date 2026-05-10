# HydroVision Model Evaluation Card

**Model:** hydro_precise  
**Generated:** 2026-05-01T14:36:57  
**Dataset:** `reports/model_cards/hydro_precise_smoke/manifest.csv` (40 samples)  
**Deployment ready:** Yes

## Performance Summary (holdout)

| Metric | Score | Plain English |
|--------|-------|---------------|
| Accuracy | 0.5500 | What fraction of all predictions were correct?  e. |
| Macro F1 | 0.5355 | Average F1 score across all classes, weighted equally. |
| Macro Precision | 0.7346 | When the model predicts a class, how often is it correct?  High precision means few false alarms. |
| Macro Recall | 0.5500 | Of all samples that truly belong to a class, how many did the model find?  High recall means the model misses few real events. |
| Mcc | 0.4377 | Matthews Correlation Coefficient — a balanced single-number summary ranging from -1 (completely wrong) to +1 (perfect). |

## Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Cargo | 1.0000 | 0.2000 | 0.3333 | 10 |
| Passenger | 0.4000 | 0.8000 | 0.5333 | 10 |
| Tanker | 0.5385 | 0.7000 | 0.6087 | 10 |
| Tug | 1.0000 | 0.5000 | 0.6667 | 10 |

## Confusion Matrix

![Confusion matrix](model_card_hydro_precise_confusion.png)

Rows = ground truth, columns = predicted.  Diagonal = correct.

| True \ Pred | Cargo | Passenger | Tanker | Tug |
|---|---|---|---|---|
| Cargo | 2 | 4 | 4 | 0 |
| Passenger | 0 | 8 | 2 | 0 |
| Tanker | 0 | 3 | 7 | 0 |
| Tug | 0 | 5 | 0 | 5 |

## Training Metrics (best epoch)

Selected by `val/macro_precision` from `lightning_logs/grid_precise_verify/verify_B_m0.30_g2.0_s0.05_gw0.0_seed2026/version_0/metrics.csv` (epoch 13).

| Metric | Value |
|--------|-------|
| `val/acc` | 0.7010 |
| `val/auroc` | 0.9162 |
| `val/f1` | 0.7081 |
| `val/loss` | 0.9720 |
| `val/macro_precision` | 0.7231 |
| `val/mcc` | 0.6439 |
| `val/micro_precision` | 0.7444 |
| `val/precision_c0` | 0.7243 |
| `val/precision_c1` | 0.7291 |
| `val/precision_c2` | 0.7985 |
| `val/precision_c3` | 0.6407 |
| `val/recall` | 0.7010 |

### Test Metrics (last logged)

| Metric | Value |
|--------|-------|
| `test/acc` | 0.6195 |
| `test/auroc` | 0.8457 |
| `test/f1` | 0.6190 |
| `test/loss` | 1.4533 |
| `test/macro_precision` | 0.6282 |
| `test/mcc` | 0.4912 |
| `test/micro_precision` | 0.6322 |
| `test/recall` | 0.6195 |

## Insights

**Best at:** Most reliable at identifying Tug (per-class F1 = 0.67).

**False-positive bias:** Shows a bias toward false positives for Passenger (FPR = 40.0%).

**Most confused pair:** Tug → Passenger (5 misclassifications)

### Pros
- Moderate macro-F1 (0.54) — a viable baseline.

### Cons
- Poor detection of Cargo (F1 = 0.33) — consider more training data or class weighting for this category.
- High false-positive rate for Passenger (40.0%) — the model over-predicts this class.
- Most common confusion: Tug misclassified as Passenger (5 times) — acoustic overlap between these classes needs addressing.

## Metric Definitions

**Accuracy:** What fraction of all predictions were correct?  e.g. 0.75 means the model guessed right 75% of the time.  Can be misleading when classes are imbalanced.

**Macro F1:** Average F1 score across all classes, weighted equally.  F1 balances how often the model catches a class (recall) against how often it raises a false alarm for that class (precision).  A score of 1.0 is perfect; 0.0 is completely wrong.

**Macro Precision:** When the model predicts a class, how often is it correct?  High precision means few false alarms.  Averaged equally across all classes.

**Macro Recall:** Of all samples that truly belong to a class, how many did the model find?  High recall means the model misses few real events.  Averaged equally across all classes.

**Mcc:** Matthews Correlation Coefficient — a balanced single-number summary ranging from -1 (completely wrong) to +1 (perfect).  Unlike accuracy, MCC stays reliable even when class sizes are very unequal.

**Per Class F1:** F1 score for each vessel type individually.  Shows which classes the model handles well and which it struggles with.

**Per Class Precision:** For each class: when the model predicts it, how often is it right?

**Per Class Recall:** For each class: of all true examples, how many did the model detect?
