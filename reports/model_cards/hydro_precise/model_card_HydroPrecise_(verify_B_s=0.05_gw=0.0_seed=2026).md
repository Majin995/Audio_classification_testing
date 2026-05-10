# HydroVision Model Evaluation Card

**Model:** HydroPrecise (verify_B s=0.05 gw=0.0 seed=2026)  
**Generated:** 2026-05-01T14:39:40  
**Dataset:** `reports/model_cards/hydro_precise/manifest.csv` (1200 samples)  
**Deployment ready:** Yes

## Performance Summary (holdout)

| Metric | Score | Plain English |
|--------|-------|---------------|
| Accuracy | 0.7558 | What fraction of all predictions were correct?  e. |
| Macro F1 | 0.7580 | Average F1 score across all classes, weighted equally. |
| Macro Precision | 0.7765 | When the model predicts a class, how often is it correct?  High precision means few false alarms. |
| Macro Recall | 0.7558 | Of all samples that truly belong to a class, how many did the model find?  High recall means the model misses few real events. |
| Mcc | 0.6804 | Matthews Correlation Coefficient — a balanced single-number summary ranging from -1 (completely wrong) to +1 (perfect). |

## Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Cargo | 0.7529 | 0.6400 | 0.6919 | 300 |
| Passenger | 0.6220 | 0.8667 | 0.7242 | 300 |
| Tanker | 0.8182 | 0.7500 | 0.7826 | 300 |
| Tug | 0.9127 | 0.7667 | 0.8333 | 300 |

## Confusion Matrix

![Confusion matrix](model_card_HydroPrecise_(verify_B_s=0.05_gw=0.0_seed=2026)_confusion.png)

Rows = ground truth, columns = predicted.  Diagonal = correct.

| True \ Pred | Cargo | Passenger | Tanker | Tug |
|---|---|---|---|---|
| Cargo | 192 | 66 | 36 | 6 |
| Passenger | 14 | 260 | 13 | 13 |
| Tanker | 46 | 26 | 225 | 3 |
| Tug | 3 | 66 | 1 | 230 |

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

**Best at:** Most reliable at identifying Tug (per-class F1 = 0.83).

**False-positive bias:** Shows a bias toward false positives for Passenger (FPR = 17.6%).

**Most confused pair:** Cargo → Passenger (66 misclassifications)

### Pros
- Strong overall macro-F1 of 0.76.
- High MCC (0.68) confirms reliability on imbalanced classes.
- Excellent detection of Tug (F1 = 0.83).
- High macro-recall: the model misses few real targets.

### Cons
- High false-positive rate for Passenger (17.6%) — the model over-predicts this class.
- Most common confusion: Cargo misclassified as Passenger (66 times) — acoustic overlap between these classes needs addressing.

## Metric Definitions

**Accuracy:** What fraction of all predictions were correct?  e.g. 0.75 means the model guessed right 75% of the time.  Can be misleading when classes are imbalanced.

**Macro F1:** Average F1 score across all classes, weighted equally.  F1 balances how often the model catches a class (recall) against how often it raises a false alarm for that class (precision).  A score of 1.0 is perfect; 0.0 is completely wrong.

**Macro Precision:** When the model predicts a class, how often is it correct?  High precision means few false alarms.  Averaged equally across all classes.

**Macro Recall:** Of all samples that truly belong to a class, how many did the model find?  High recall means the model misses few real events.  Averaged equally across all classes.

**Mcc:** Matthews Correlation Coefficient — a balanced single-number summary ranging from -1 (completely wrong) to +1 (perfect).  Unlike accuracy, MCC stays reliable even when class sizes are very unequal.

**Per Class F1:** F1 score for each vessel type individually.  Shows which classes the model handles well and which it struggles with.

**Per Class Precision:** For each class: when the model predicts it, how often is it right?

**Per Class Recall:** For each class: of all true examples, how many did the model detect?
