# T. Confusion-matrix delta — Stacker vs N=5 baseline

Eval: full Split1s test (n=7872)

## Baseline N=5 arith confusion matrix
(rows = true, cols = predicted)
```
            Cargo  Passenger     Tanker        Tug
 Cargo:       1158        628        408          1
Passenger:         18       1471          1          2
Tanker:         34        184       1705          3
   Tug:        154        585          0       1520
```

## Stacker (MLP(64) K=10) confusion matrix
```
            Cargo  Passenger     Tanker        Tug
 Cargo:       1706        111        261        117
Passenger:         50       1387          4         51
Tanker:        200         17       1686         23
   Tug:         58         43          0       2158
```

## Δ = Stacker - Baseline
Negative on off-diagonal = stacker FIXED mistakes; positive on diagonal = stacker RECOVERED samples.
```
            Cargo  Passenger     Tanker        Tug
 Cargo:       +548       -517       -147       +116
Passenger:        +32        -84         +3        +49
Tanker:       +166       -167        -19        +20
   Tug:        -96       -542         +0       +638
```

## Sample-level deltas
- Total samples baseline got wrong but stacker got right: 1313
- Total samples baseline got right but stacker got wrong: 230
- **Net improvement**: 1083 samples

Per-class breakdown of fixed/broken (by TRUE class):

| class | fixed (baseline wrong → stacker right) | broken (baseline right → stacker wrong) | net |
|---|---|---|---|
| Cargo | 599 | 51 | +548 |
| Passenger | 7 | 91 | -84 |
| Tanker | 69 | 88 | -19 |
| Tug | 638 | 0 | +638 |

## Where the stacker eats baseline mistakes

| true | base_pred | base_count | stacker also wrong here | fixed |
|---|---|---|---|---|
| Cargo | Passenger | 628 | 205 | 423 |
| Cargo | Tanker | 408 | 233 | 175 |
| Cargo | Tug | 1 | 0 | 1 |
| Passenger | Cargo | 18 | 11 | 7 |
| Passenger | Tanker | 1 | 1 | 0 |
| Passenger | Tug | 2 | 2 | 0 |
| Tanker | Cargo | 34 | 32 | 2 |
| Tanker | Passenger | 184 | 118 | 66 |
| Tanker | Tug | 3 | 2 | 1 |
| Tug | Cargo | 154 | 56 | 98 |
| Tug | Passenger | 585 | 45 | 540 |