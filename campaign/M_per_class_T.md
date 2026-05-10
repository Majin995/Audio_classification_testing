# M. Per-class (vector) temperature

## Geom-2 ensemble base
- Scalar T = 2.750
- Vector T = ['2.922', '0.010', '0.097', '3.278']

### Scalar T
- thresholds = [0.0, 0.87, 0.57, 0.0]
- raw full F1 = 0.7840, raw full macroP = 0.8057
- cal full F1 = 0.8098, cal full macroP = 0.8651, cov = 0.883

### Vector T
- thresholds = [0.0, 0.81, 0.0, 0.0]
- raw full F1 = 0.8168, raw full macroP = 0.8146
- cal full F1 = 0.8200, cal full macroP = 0.8275, cov = 0.975

## Stacker base
- Scalar T = 0.500
- Vector T = ['0.266', '0.392', '0.159', '0.321']

### Stacker + scalar T
- thresholds = [0.0, 0.0, 0.0, 0.0]
- raw full F1 = 0.8660, raw full macroP = 0.8660
- cal full F1 = 0.8660, cal full macroP = 0.8660, cov = 1.000
```
  acc=0.8673  f1=0.8660  macroP=0.8660  microP=0.8673  recall=0.8666  mcc=0.8218  cov=1.000
    P0=0.815  P1=0.865  P2=0.875  P3=0.910
    R0=0.790  R1=0.881  R2=0.834  R3=0.962
    F1_0=0.802  F1_1=0.873  F1_2=0.854  F1_3=0.935
```

### Stacker + vector T
- thresholds = [0.0, 0.0, 0.0, 0.0]
- raw full F1 = 0.8610, raw full macroP = 0.8621
- cal full F1 = 0.8610, cal full macroP = 0.8621, cov = 1.000
```
  acc=0.8620  f1=0.8610  macroP=0.8621  microP=0.8620  recall=0.8620  mcc=0.8152  cov=1.000
    P0=0.800  P1=0.853  P2=0.895  P3=0.900
    R0=0.793  R1=0.894  R2=0.800  R3=0.961
    F1_0=0.797  F1_1=0.873  F1_2=0.845  F1_3=0.929
```
