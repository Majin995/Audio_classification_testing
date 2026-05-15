"""Import 5s/15s/30s HydroPrecise variants into Tiamat models_registry.

Creates Tiamat/models_registry/HydroPrecise_<L>s/ for L in (5, 15, 30) with:
  best.ckpt, temperature.pt, thresholds.json, handler.py (FIXED_LEN patched +
  post-load inferencer.fixed_len override), inferencer.py (verbatim clone of
  the 1 s entry), manifest.yaml, README.md.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

PROJECT  = Path("/var/home/damo/Documents/Git/Audio_classification_testing")
TIAMAT   = Path("/var/home/damo/Documents/Git/Tiamat")
SRC_REG  = TIAMAT / "models_registry/HydroPrecise"
DST_REG  = TIAMAT / "models_registry"
TARGET_SR = 5_120

VARIANTS = [
    {
        "L": 5, "fixed_len": 5 * TARGET_SR,
        "src_run": "lightning_logs/precise_deepship_5s/version_0/checkpoints",
        "ckpt": "precise-001-p0.6038.ckpt",
        "best_val": 0.6038, "test_macroP": 0.5900, "test_microP": 0.6030,
        "test_f1": 0.5480, "test_recall": 0.5612, "mp_cov_85": 0.6948,
    },
    {
        "L": 15, "fixed_len": 15 * TARGET_SR,
        "src_run": "lightning_logs/precise_deepship_15s/version_0/checkpoints",
        "ckpt": "precise-002-p0.6141.ckpt",
        "best_val": 0.6141, "test_macroP": 0.5975, "test_microP": 0.5828,
        "test_f1": 0.5324, "test_recall": 0.5468, "mp_cov_85": 0.7328,
    },
    {
        "L": 30, "fixed_len": 30 * TARGET_SR,
        "src_run": "lightning_logs/precise_deepship_30s/version_0/checkpoints",
        "ckpt": "precise-007-p0.6220.ckpt",
        "best_val": 0.6220, "test_macroP": 0.6174, "test_microP": 0.6112,
        "test_f1": 0.5567, "test_recall": 0.5509, "mp_cov_85": 0.6745,
    },
]


HANDLER_TEMPLATE = '''"""HydroPrecise{L_TAG} — Multi-stream high-precision UATR classifier ({SR} Hz, {L}s).

Trained on the DeepShip raw corpus, resampled 32k->{SR}, split 60/15/25 at the
source-file level, chunked into non-overlapping {L}s windows (no padding).

Wraps the deploy bundle's ``HydroPreciseInferencer`` (sibling ``inferencer.py``)
so it plugs into the standard BaseModel interface.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from app.model_registry_base import BaseModel

LIGHTNING_LOG_DIRS: tuple[str, ...] = ()


def _load_inferencer_class():
    here = Path(__file__).resolve().parent
    mod_name = "hv_models.HydroPrecise{L_TAG}._inferencer"
    if mod_name in sys.modules:
        return sys.modules[mod_name].HydroPreciseInferencer
    spec = importlib.util.spec_from_file_location(mod_name, here / "inferencer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module.HydroPreciseInferencer


class ModelHandler(BaseModel):
    NAME        = "HydroPrecise{L_TAG}"
    DESCRIPTION = (
        "HydroPrecise on DeepShip raw, {L}s clips at {SR} Hz. "
        "Gabor+PCEN, CQT, DEMON branches -> cross-attention -> margin head. "
        "Held-out test macro-P {TEST_MACROP:.3f}, gated MP@cov 0.85 = {MP85:.3f}."
    )
    CLASSES = ["Cargo", "Passenger", "Tanker", "Tug"]

    TARGET_SR  = {SR}
    FIXED_LEN  = {FIXED_LEN}      # {L} s x {SR} Hz

    def __init__(self) -> None:
        self._inf    = None
        self._device = "cpu"

    def load(self, ckpt_path: str, device: str = "cuda") -> None:
        Inferencer = _load_inferencer_class()
        run_dir = Path(ckpt_path).parent
        self._inf = Inferencer(
            run_dir=str(run_dir),
            class_names=self.CLASSES,
            device=device,
        )
        # The shared inferencer assumes 1 s clips; override for this variant.
        self._inf.fixed_len = self.FIXED_LEN
        self._device = device
        if self._inf.num_classes != len(self.CLASSES):
            self.CLASSES = [f"class_{{i}}" for i in range(self._inf.num_classes)]

    def _prep_numpy(self, waveform: torch.Tensor, sample_rate: int):
        if sample_rate != self.TARGET_SR:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.TARGET_SR,
            )
        wav = waveform.mean(dim=0)
        L = wav.shape[-1]
        if L < self.FIXED_LEN:
            wav = F.pad(wav, (0, self.FIXED_LEN - L))
        else:
            wav = wav[..., : self.FIXED_LEN]
        return wav.detach().cpu().numpy()

    def predict(self, waveform: torch.Tensor, sample_rate: int) -> dict[str, float]:
        wav = self._prep_numpy(waveform, sample_rate)
        result = self._inf.predict(wav)
        return {{cls: float(p) for cls, p in zip(self.CLASSES, result.probs)}}

    def explain(self, waveform: torch.Tensor, sample_rate: int, target_class: Optional[str] = None):
        if self._inf is None:
            return None
        wav = self._prep_numpy(waveform, sample_rate)
        tgt = None
        if target_class is not None and target_class in self.CLASSES:
            tgt = self.CLASSES.index(target_class)
        sal = self._inf.saliency(wav, target=tgt, smooth_window=64)
        return sal

    @property
    def is_loaded(self) -> bool:
        return self._inf is not None
'''

MANIFEST_TEMPLATE = '''runtime: container
image: tiamat/model-runner:1.0
cpu_cores: 2
mem_limit_gb: 4
gpu_ids: null          # CPU-only
idle_timeout_s: 300
'''

README_TEMPLATE = '''# HydroPrecise{L_TAG} - DeepShip {L}s deploy bundle

HydroPrecise trained on **DeepShip raw** corpus (32 kHz native, resampled to
**{SR} Hz**), split 60% Train / 15% Test / 25% Holdout at the source-file
level (no source leakage), chunked into non-overlapping **{L} s** windows
(no padding - tail audio shorter than {L} s discarded).

Seed 42 single-seed run; LMF loss (margin=0.5, gamma=2.0, label-smoothing=0.05),
gambler weight 0.1, batch size {BATCH}, {EPOCHS} epoch budget with
warmup=2 / patience=20.

## Headline test metrics

| metric | value |
|---|---|
| Best val/micro_precision | {BEST_VAL:.4f} (epoch {BEST_EPOCH}) |
| test/macro_precision | {TEST_MACROP:.4f} |
| test/micro_precision | {TEST_MICROP:.4f} |
| test/F1              | {TEST_F1:.4f} |
| test/recall          | {TEST_RECALL:.4f} |
| Post-cal MP @ cov 0.85 | **{MP85:.4f}** |

## Bundle contents

```
best.ckpt         Lightning checkpoint
temperature.pt    Calibration scalar T (for softmax/T)
thresholds.json   Per-class softmax gates (target_coverage = 0.85)
handler.py        BaseModel adapter (FIXED_LEN = {FIXED_LEN})
inferencer.py     Standalone inference + saliency
manifest.yaml     Container runtime spec
```

## Inputs
- 1-D mono waveform at any sample rate; resampled to {SR} Hz inside `_prep_numpy`.
- Length: zero-padded to {FIXED_LEN} samples ({L} s) if shorter, truncated if longer.

## Output

`ModelHandler.predict(waveform, sample_rate)` -> `{{class_name: probability}}` dict
over Cargo/Passenger/Tanker/Tug, calibrated by temperature scaling.

`ModelHandler.explain(waveform, sample_rate, target_class)` -> `(T,)` numpy
saliency map (gradient x input) aligned to the input waveform.

## Reproduce

```bash
DATA_DIR=/var/mnt/5A009BF8009BD8F9/Data/Deepship_{L}s python training/train_precise.py \\
  --run_name precise_deepship_{L}s \\
  --sample_rate {SR} --fixed_len {FIXED_LEN} \\
  --batch_size {BATCH} --max_epochs {EPOCHS} \\
  --warmup_epochs 2 --patience 20 --seed 42
```
'''


def _ckpt_epoch(name: str) -> int:
    m = re.search(r"precise-(\d+)", name)
    return int(m.group(1)) if m else -1


def import_variant(v: dict) -> Path:
    L = v["L"]
    L_TAG = f"_{L}s"
    out = DST_REG / f"HydroPrecise{L_TAG}"
    out.mkdir(parents=True, exist_ok=True)

    src = PROJECT / v["src_run"]
    shutil.copy2(src / v["ckpt"],            out / "best.ckpt")
    shutil.copy2(src / "temperature.pt",     out / "temperature.pt")
    shutil.copy2(src / "thresholds.json",    out / "thresholds.json")
    shutil.copy2(SRC_REG / "inferencer.py",  out / "inferencer.py")

    batch  = {5: 64, 15: 32, 30: 16}[L]
    epochs = {5: 60, 15: 80, 30: 80}[L]

    (out / "handler.py").write_text(HANDLER_TEMPLATE.format(
        L=L, L_TAG=L_TAG, SR=TARGET_SR,
        FIXED_LEN=v["fixed_len"],
        TEST_MACROP=v["test_macroP"], MP85=v["mp_cov_85"],
    ))
    (out / "manifest.yaml").write_text(MANIFEST_TEMPLATE)
    (out / "README.md").write_text(README_TEMPLATE.format(
        L=L, L_TAG=L_TAG, SR=TARGET_SR,
        FIXED_LEN=v["fixed_len"], BATCH=batch, EPOCHS=epochs,
        BEST_VAL=v["best_val"], BEST_EPOCH=_ckpt_epoch(v["ckpt"]),
        TEST_MACROP=v["test_macroP"], TEST_MICROP=v["test_microP"],
        TEST_F1=v["test_f1"], TEST_RECALL=v["test_recall"],
        MP85=v["mp_cov_85"],
    ))
    print(f"  -> {out}")
    return out


def main():
    if not SRC_REG.exists():
        raise SystemExit(f"missing source registry: {SRC_REG}")
    print(f"Importing into {DST_REG}")
    paths = [import_variant(v) for v in VARIANTS]
    print(f"\nDone. {len(paths)} entries written.")
    p = paths[0]
    print(f"\nSanity check ({p.name}):")
    for f in ("best.ckpt", "temperature.pt", "thresholds.json", "handler.py",
              "inferencer.py", "manifest.yaml", "README.md"):
        full = p / f
        size = full.stat().st_size if full.exists() else -1
        print(f"  {f:<20} {size:>12} B")


if __name__ == "__main__":
    main()
