# Z. Production stacker run on full Split1s_eval

Final ship-grade run of `inference/ensemble_dirichlet.py --classifier
mlp64_k10` against `data/Split1s_eval` — a stratified split of
`data/Split1s/test` with 500/cls held out as val and the remainder as
test (carved with seed=42 by `data/_make_split1s_eval.py`).

## Setup

| | val | test |
|---|---|---|
| Cargo | 500 | 1695 |
| Passenger | 500 | 992 |
| Tanker | 500 | 1426 |
| Tug | 500 | 1785 |
| **total** | **2000 (1984 after DALI batching)** | **5898 (5856 after batching)** |

Note: DALI uses LastBatchPolicy.DROP (drops the partial last batch), so
n_val=1984 instead of 2000 and n_test=5856 instead of 5898 — minor
artifact of fixed batch_size=64 and not bias.

## Per-model metrics (raw, no stacker)

| model | val/F1 | val/macroP | test/F1 | test/macroP |
|---|---|---|---|---|
| hydra-026 (s1337) | 0.7854 | 0.8125 | 0.7550 | 0.7718 |
| hydra-012 (s42)   | 0.5587 | 0.7346 | 0.5409 | 0.7115 |
| hydra-010 (s2026) | 0.6513 | 0.7448 | 0.6188 | 0.7154 |
| hydra-032 (s7)    | 0.7419 | 0.7765 | 0.7214 | 0.7512 |
| hydra-031 (s12345)| 0.7950 | 0.8294 | 0.7629 | 0.7900 |

## Stacker (mlp64_k10) test metrics

```
F1:       0.9559
macro_P:  0.9549
recall:   0.9572
MCC:      0.9404
AUROC:    0.9957
```

| class | P | R | F1 |
|---|---|---|---|
| Cargo | 0.953 | 0.913 | **0.932** |
| Passenger | 0.961 | 0.977 | **0.969** |
| Tanker | 0.924 | 0.947 | **0.935** |
| Tug | 0.981 | 0.993 | **0.987** |

## Confusion matrix (test, n=5856)

```
                Cargo  Passenger     Tanker        Tug
       Cargo:    1547         19        107         22
   Passenger:       7        969          4         12
      Tanker:      60         16       1350          0
         Tug:       9          4          0       1762
```

Diagonal accuracy: Cargo 91.3%, Passenger 97.7%, Tanker 94.7%, Tug 99.3%.

## Lift summary

| stacker val source | val n | full-test F1 | full-test macroP |
|---|---|---|---|
| (none — N=5 arith baseline) | — | 0.7432 | 0.7911 |
| rapid_1s/val (campaign Phase Q) | 64 | 0.8820 | 0.8808 |
| **Split1s_eval/val (production run)** | **1984** | **0.9559** | **0.9549** |

**Headline:** going from val=64 → val=1984 lifts F1 by **+7.4 pp** on top of
the existing +13.9 pp lift from the rapid_1s fit. **Total lift vs N=5 arith
baseline: +21.3 pp F1, +16.4 pp macroP.**

The stacker scaled with val size. With 31× more val samples it found a
much better solution — clearly fitting at val=64 was leaving meaningful
performance on the table.

## Artifacts

- `lightning_logs/dirichlet_stacker_full/stacker.joblib` — fitted MLP K=10
- `lightning_logs/dirichlet_stacker_full/stacker_meta.json` — full per-model + stacker metrics
- `lightning_logs/dirichlet_stacker_full/stacker_eval.md` — markdown report
- `lightning_logs/dirichlet_stacker_full/final_test_probs.npy` — softmax probs (n=5856, 4)

## Reproduction

```bash
# 1) Carve val from Split1s/test (one-time)
python data/_make_split1s_eval.py
ln -sfn ../Split1s/train data/Split1s_eval/train  # DALI needs train for class weights

# 2) Fit + apply stacker on full Split1s_eval (~5 min on RTX 5090)
python -m inference.ensemble_dirichlet \
  --ckpts \
    lightning_logs/phaseG_R1_alpha/version_0/checkpoints/hydra-026-p0.6908.ckpt \
    lightning_logs/phaseI_R1_s42/version_0/checkpoints/hydra-012-p0.6929.ckpt \
    lightning_logs/phaseI_R1_s2026/version_0/checkpoints/hydra-010-p0.6943.ckpt \
    lightning_logs/phaseI_R1_s7/version_1/checkpoints/hydra-032-p0.7010.ckpt \
    lightning_logs/phaseI_R1_s12345/version_0/checkpoints/hydra-031-p0.7042.ckpt \
  --data_dir data/Split1s_eval \
  --out_dir lightning_logs/dirichlet_stacker_full \
  --classifier mlp64_k10
```

## Application to new data (e.g. real-time inference)

```bash
python -m inference.ensemble_dirichlet \
  --ckpts <5 ckpts> \
  --data_dir <NEW_DATASET> \
  --out_dir <OUT> \
  --stacker_path lightning_logs/dirichlet_stacker_full/stacker.joblib
```

The stacker is portable (joblib, ~few KB). Apply to any dataset with the
same class names and the same 5 ckpts.
