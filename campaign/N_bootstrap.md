# N. Bootstrap robustness of Dirichlet stacker

- B = 200 bootstrap resamples of rapid_1s val (n=64, with replacement)
- 200 resamples retained (others lost a class)
- Stacker: LogisticRegression(C=0.1) on log-probs features (5x4=20 dim)
- Eval: full Split1s test (n=7872)

| metric | mean | std | p05 | p50 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| f1 | 0.8441 | 0.0181 | 0.8147 | 0.8476 | 0.8655 | 0.7541 | 0.8729 |
| macroP | 0.8502 | 0.0141 | 0.8264 | 0.8534 | 0.8677 | 0.7854 | 0.8745 |
| recall | 0.8457 | 0.0170 | 0.8158 | 0.8484 | 0.8653 | 0.7791 | 0.8724 |
| acc | 0.8443 | 0.0188 | 0.8111 | 0.8483 | 0.8660 | 0.7598 | 0.8739 |

## Interpretation
- Original (no bootstrap): full F1 = 0.8660, macroP = 0.8660
- Bootstrap mean F1 = 0.8441 ± 0.0181
- Bootstrap mean macroP = 0.8502 ± 0.0141
- Probability of beating N=5 arith baseline (full F1=0.7432): 100.0%
- Probability of beating geom-2+thr (full F1=0.8098): 96.0%