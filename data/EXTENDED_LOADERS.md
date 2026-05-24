# Extended audio loaders + stacker training

Two Lightning DataModules that take long-form audio files and split them
into fixed-length windows at load time, with the same per-window output as
`DALIAudioDataModule`. Drop-in swappable into any Lightning model.

| File | Class | Backend | When to use |
|---|---|---|---|
| `data/extended_audio_dali_loader.py` | `ExtendedDALIAudioDataModule` | NVIDIA DALI + `fn.external_source` | Default. GPU-side preprocessing, best throughput. |
| `data/extended_audio_threaded_loader.py` | `ExtendedThreadedAudioDataModule` | `soundfile` + `ThreadPoolExecutor` + `torchaudio.functional` | Backup. Use when DALI env is broken, unavailable, or you want a portable loader with zero CUDA dependencies. |

Per-window output of both is bit-identical to within ~5e-8 (verified by
`data/test_extended_loaders.py`). Both match `DALIAudioDataModule` on
matching 1 s files to within ~1.5e-3 max-abs (DALI's no-op `audio_resample`
filter adds ~0.04% drift — implementation noise, not a content difference).

---

## Common constructor args

Identical between the two loaders unless noted.

| Arg | Default | Meaning |
|---|---:|---|
| `data_dir` | — (required) | Dataset root containing `train/` `val/` `test/` subdirs, each holding `<class>/*.wav` files of arbitrary length. |
| `batch_size` | `64` | Windows per batch. |
| `target_sr` | `5120` | Sample rate. Files must already be at this rate (matches `DALIAudioDataModule` contract on the curated 1 s trees). |
| `window_sec` | `1.0` | Window length in seconds. `fixed_len = round(window_sec * target_sr)` samples. |
| `hop_sec` | `None` | Hop in seconds between consecutive windows. `None` → `window_sec` (non-overlapping). Use `< window_sec` for overlap, e.g. `0.5` for 50%. |
| `rms_normalize` | `False` | If `True`, each window is zero-mean / unit-std then scaled to `target_rms`. Matches `fn.normalize(axes=[0], epsilon=1e-9) * target_rms`. |
| `target_rms` | `0.1` | Post-normalize RMS magnitude. |
| `oversample_train` | `True` | If `True`, minority-class file lists are repeated until each class has the same file count. Window enumeration runs on the oversampled list. |
| `merge_classes` | `None` | `{src: tgt}` dict to fold one class into another (e.g. `{"Passenger": "Cargo"}`). |
| `shuffle_train` | `True` | Shuffle train windows per epoch. |

**Threaded-only:**

| Arg | Default | Meaning |
|---|---:|---|
| `num_workers` | `8` | PyTorch DataLoader worker processes. Use `0` for in-thread loading. |
| `pin_memory` | `True` | Pin CPU tensors for faster GPU transfer. **Set `False`** if you're driving threaded + DALI iterators concurrently (their CUDA-context init races the pin worker). |

**DALI-only:**

| Arg | Default | Meaning |
|---|---:|---|
| `num_threads` | `8` | DALI pipeline worker threads. |
| `device_id` | `0` | CUDA device index. |
| `seed` | `42` | RNG seed for train shuffling (val/test ignore this — order is deterministic). |

---

## Public attributes after `setup()`

Both loaders expose the same Lightning-model-facing attributes as
`DALIAudioDataModule`:

| Attribute | Type | Meaning |
|---|---|---|
| `class_to_idx` | `dict[str, int]` | e.g. `{"Cargo": 0, "Passenger": 1, ...}` (alphabetical) |
| `idx_to_class` | `dict[int, str]` | inverse mapping |
| `num_classes` | `int` | |
| `class_weights` | `list[float]` | inverse-frequency weights for the loss |

Plus internal state useful for inspection:

| Attribute | Type | Meaning |
|---|---|---|
| `_splits[split]` | `dict[str, list[str]]` | files per class per split (post `merge_classes`) |
| `_windows[split]` | `list[(path, label, start_sample, stop_sample)]` | full enumerated window manifest |

---

## Dataset layout

```
<data_dir>/
  train/
    Cargo/      *.wav         ← long files of arbitrary duration
    Passenger/  *.wav
    Tanker/     *.wav
    Tug/        *.wav
  val/
    <classes>/  *.wav
  test/
    <classes>/  *.wav
```

Windowing per file (window=1 s, hop=1 s on a 7.5 s file):

```
file (7.5 s) ─┬─ window 0 [0.0, 1.0]
              ├─ window 1 [1.0, 2.0]
              ├─ ...
              ├─ window 6 [6.0, 7.0]
              └─ window 7 [6.5, 7.5]      ← trailing partial, anchored to file tail
```

Files shorter than `window_sec` emit exactly one window spanning the whole
file, zero-padded to `fixed_len`.

---

## Plug-and-play swap

Switch a Lightning model from the original loader to the extended one by
replacing the import and constructor. Nothing in the model changes.

```python
# Before — original DALI loader, one window per file (file must equal window length)
from data.audio_lightning_loader import DALIAudioDataModule
dm = DALIAudioDataModule(
    data_dir="/data/Classifier_Dataset_1s_only",
    batch_size=64, target_sr=5120, fixed_len=5120,
    rms_normalize=True, target_rms=0.1,
)

# After — extended DALI loader, long files chunked at load time
from data.extended_audio_dali_loader import ExtendedDALIAudioDataModule
dm = ExtendedDALIAudioDataModule(
    data_dir="/data/LongFiles_root",
    batch_size=64, target_sr=5120,
    window_sec=1.0, hop_sec=1.0,
    rms_normalize=True, target_rms=0.1,
)

# Or, when DALI env is unavailable
from data.extended_audio_threaded_loader import ExtendedThreadedAudioDataModule
dm = ExtendedThreadedAudioDataModule(
    data_dir="/data/LongFiles_root",
    batch_size=64, target_sr=5120,
    window_sec=1.0, hop_sec=1.0,
    rms_normalize=True, target_rms=0.1,
)

trainer.fit(model, dm)   # unchanged
```

---

## Workflow with the stacker training script

`campaign/train_all_stackers.py` does **not** consume the Lightning
DataModules above directly — it reads pre-chunked 1 s WAVs from a
`Train/Val/Test/<class>/*.wav` tree and looks up frozen per-clip ensemble
probabilities from an NPZ cache. The extended loaders sit one step
upstream, on the **base** model training and the **dump** step.

```
long audio files (any length)
        │
        ▼
┌─────────────────────────────────────────────┐
│ Base model training (HydroHydra / Complete) │   ← use ExtendedDALI/Threaded
│  consumes (audio, label) per window         │     here, window_sec=1.0
└────────────────────┬────────────────────────┘
                     │  best.pt checkpoints
                     ▼
┌─────────────────────────────────────────────┐
│ Per-clip ensemble dump                      │   ← also reads windowed
│ (dump_ensemble_combined.py)                 │     audio. Either materialise
│  writes probs_*.npz keyed by clip path      │     1 s WAVs offline (current
└────────────────────┬────────────────────────┘     repo) or call the loader.
                     │  probs_*.npz
                     ▼
┌─────────────────────────────────────────────┐
│ Stacker training                            │   ← train_all_stackers.py.
│ (campaign/train_all_stackers.py)            │     Uses
│  HydroGraphProto + future registry entries  │     data/threaded_audio_loader
└─────────────────────────────────────────────┘     (the plain non-DALI one)
```

The stacker trainer requires the dataset to already be chunked into 1 s
files on disk (one window per WAV). To go from long files to that layout
there are two paths:

**Path A — materialise 1 s WAVs offline (matches the current repo's
`Combined_IARA_Deepship_1s` build):** run a one-shot script that uses the
threaded loader's `enumerate_windows` + `sf.write` to dump each window to
`<out>/Train/<class>/<src_id>_seg_<n>.wav`. Then point the stacker trainer
at `<out>`.

```python
# scripts/materialise_windows.py
from pathlib import Path
import soundfile as sf
from data.extended_audio_threaded_loader import (
    _scan_split, enumerate_windows,
)

SRC = Path("/data/LongFiles_root")
DST = Path("/data/LongFiles_1s")
TARGET_SR, WIN, HOP = 5120, 5120, 5120

for split in ("train", "val", "test"):
    fbc = _scan_split(SRC, split)
    classes = sorted(fbc.keys())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    windows = enumerate_windows(fbc, cls_to_idx, WIN, HOP)
    for path, label, start, stop in windows:
        cls = classes[label]
        src_id = Path(path).stem
        seg_idx = start // HOP
        out = DST / split.capitalize() / cls / f"{src_id}_seg_{seg_idx}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        a, sr = sf.read(path, start=start, stop=stop, dtype="float32")
        if len(a) < WIN:
            import numpy as np
            a = np.pad(a, (0, WIN - len(a)))
        sf.write(out, a, TARGET_SR, subtype="FLOAT")
```

Then dump per-clip probs and train the stacker the usual way:

```bash
PYTHONPATH=. python campaign/dump_ensemble_combined.py \
  --data_dir /data/LongFiles_1s \
  --out campaign/probs_longfiles_7ckpt.npz

PYTHONPATH=. python campaign/train_all_stackers.py \
  --data_dir /data/LongFiles_1s \
  --ens_npz campaign/probs_longfiles_7ckpt.npz \
  --out_root lightning_logs/longfiles_stackers
```

**Path B — keep long files and chunk on-the-fly during base training
only.** Use `ExtendedThreadedAudioDataModule` or `ExtendedDALIAudioDataModule`
in your HydroHydra/HydroComplete trainer. After training, materialise 1 s
WAVs once (same script as Path A) so the dump + stacker trainer have the
fixed-window layout they expect. The extended loaders are not currently
plumbed into `dump_ensemble_combined.py` or `train_all_stackers.py`
themselves — those scripts assume the curated 1 s tree convention used
throughout the repo.

---

## Stacker trainer args refresher (`campaign/train_all_stackers.py`)

Once your dataset is in the chunked layout, training is a one-liner. The
critical knobs:

| Arg | Default | Meaning |
|---|---:|---|
| `--data_dir` | required | Chunked tree, `Train/Val/Test/<class>/*.wav` |
| `--ens_npz` | required | Per-clip ensemble cache for this dataset |
| `--out_root` | required | Each model writes to `<out_root>/<model_name>/` |
| `--models` | `hydro_graph_proto` | Comma-separated registry entries to train sequentially |
| `--finetune` | off | Warm-start each model from `<finetune_from>/<model_name>/best.pt` |
| `--finetune_from` | `None` | Required when `--finetune` is set |
| `--finetune_lr` | `5e-4` | LR override for fine-tune mode (cold-start uses `--lr`, default `2e-3`) |
| `--K_train` / `--K_eval` | `30` / `60` | Clips per source per training/eval batch |
| `--per_class` | `4` | Sources per class per training batch |
| `--steps` | `8000` | Optimisation steps; cosine LR schedule, patience-15 early stop |

Cold start:

```bash
PYTHONPATH=. python campaign/train_all_stackers.py \
  --data_dir /data/MyDataset_1s \
  --ens_npz campaign/probs_mydataset_6ckpt.npz \
  --out_root lightning_logs/mydataset_stackers
```

Fine-tune from a previous run on the same dataset:

```bash
PYTHONPATH=. python campaign/train_all_stackers.py \
  --data_dir /data/MyDataset_1s \
  --ens_npz campaign/probs_mydataset_6ckpt.npz \
  --out_root lightning_logs/mydataset_stackers_ft \
  --finetune --finetune_from lightning_logs/mydataset_stackers
```

The `--finetune` flag applies to **every** model listed in `--models`
in one go. GraphProto skips its k-means prototype init automatically on
warm start (the loaded state dict already contains learned prototypes).
