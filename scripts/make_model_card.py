"""
Run the architecture-agnostic ModelCardGenerator on an exported ONNX model.

Usage
-----
    python scripts/make_model_card.py \
        --onnx exports/hydro_precise.onnx \
        --data-dir data/Split1s/test \
        --run-dir lightning_logs/grid_precise_verify/verify_B_m0.30_g2.0_s0.05_gw0.0_seed2026 \
        --output-dir reports/model_cards/hydro_precise \
        --max-per-class 200

The ``--data-dir`` is treated as ``<root>/<class_name>/*.wav``. A two-column
CSV manifest is built on the fly. Pass ``--csv`` to use an existing one.

If ``--run-dir`` is given:
  * `temperature.pt` is used to calibrate ONNX softmax,
  * the latest `version_*/metrics.csv` populates the "Training metrics"
    section in the card.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation import ModelCardGenerator  # noqa: E402


# ─── ONNX-backed predictor ────────────────────────────────────────────────────

class ONNXPredictor:
    """Thin wrapper exposing predict(wav, sr) -> {class_name: probability}.

    The model is loaded once; class names come from the YAML config; an
    optional temperature calibration is applied before softmax.
    """

    def __init__(
        self,
        onnx_path: str | os.PathLike,
        class_names: Sequence[str],
        temperature: float = 1.0,
        target_len: int = 5_120,
        device: str = "cpu",
        model_name: Optional[str] = None,
    ) -> None:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if device == "cuda" else ["CPUExecutionProvider"]
        # Filter out missing providers
        providers = [p for p in providers if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.class_names = list(class_names)
        self.num_classes = len(class_names)
        self.temperature = float(temperature)
        self.target_len = int(target_len)
        self.model_name = model_name or Path(onnx_path).stem
        self.NAME = self.model_name

    def _prep(self, wav: torch.Tensor) -> np.ndarray:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim == 2 and wav.shape[0] != 1:
            # batch in — fine, just take all rows
            arr = wav.detach().cpu().numpy().astype(np.float32)
        else:
            arr = wav.detach().cpu().numpy().astype(np.float32)
        # Pad / truncate to target_len
        T = arr.shape[-1]
        if T < self.target_len:
            arr = np.pad(arr, ((0, 0), (0, self.target_len - T)))
        elif T > self.target_len:
            arr = arr[:, : self.target_len]
        return arr

    def predict(self, wav: torch.Tensor, sample_rate: int) -> Dict[str, float]:
        del sample_rate  # caller is responsible for resampling
        x = self._prep(wav)
        logits = self.sess.run(None, {self.input_name: x})[0]
        # Drop trailing abstention logit if present (HydroPrecise convention)
        if logits.shape[-1] == self.num_classes + 1:
            logits = logits[..., : self.num_classes]
        scaled = logits / max(self.temperature, 1e-6)
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        exp = np.exp(scaled)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        # Single sample → flat dict
        p = probs[0]
        return {cls: float(p[i]) for i, cls in enumerate(self.class_names)}


# ─── Manifest builders ────────────────────────────────────────────────────────

def build_manifest_from_dir(
    root: Path,
    class_names: Sequence[str],
    max_per_class: Optional[int],
    seed: int,
    extensions: Sequence[str] = (".wav", ".flac", ".ogg"),
) -> List[tuple[str, str]]:
    """Treat root/<class>/*.<ext> as labelled clips."""
    rng = random.Random(seed)
    rows: List[tuple[str, str]] = []
    for cls in class_names:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            print(f"[manifest] no dir for class '{cls}' at {cls_dir} — skipping")
            continue
        files = [p for p in cls_dir.iterdir() if p.suffix.lower() in extensions]
        files.sort()
        if max_per_class is not None and len(files) > max_per_class:
            files = rng.sample(files, max_per_class)
        for f in files:
            rows.append((str(f.resolve()), cls))
    return rows


def write_manifest_csv(rows: List[tuple[str, str]], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as f:
        writer = csv.writer(f)
        for path, label in rows:
            writer.writerow([path, label])
    return dest


# ─── Run-dir helpers ──────────────────────────────────────────────────────────

def resolve_temperature(run_dir: Optional[Path]) -> float:
    if run_dir is None:
        return 1.0
    candidates = list(run_dir.glob("**/temperature.pt"))
    if not candidates:
        return 1.0
    blob = torch.load(str(candidates[0]), map_location="cpu", weights_only=False)
    return float(blob.get("temperature", 1.0))


def resolve_metrics_csv(run_dir: Optional[Path]) -> Optional[Path]:
    if run_dir is None:
        return None
    candidates = sorted(run_dir.glob("version_*/metrics.csv"))
    return candidates[-1] if candidates else None


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="Path to the .onnx model.")
    ap.add_argument("--config", default="evaluation/model_config.yaml",
                    help="YAML/JSON config defining class names + report metadata.")
    ap.add_argument("--csv", default=None,
                    help="Pre-built two-column CSV (path,class). Skips --data-dir.")
    ap.add_argument("--data-dir", default=None,
                    help="Root with one subfolder per class containing audio.")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="Cap clips per class when building from --data-dir.")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    ap.add_argument("--run-dir", default=None,
                    help="Lightning run dir — pulls temperature.pt + metrics.csv.")
    ap.add_argument("--selection-metric", default="val/macro_precision",
                    help="Which val/* column picks the 'best epoch' for the report.")
    ap.add_argument("--sample-rate", type=int, default=5_120)
    ap.add_argument("--target-len", type=int, default=5_120)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--model-name", default=None,
                    help="Display name in the report (defaults to ONNX stem).")
    ap.add_argument("--output-dir", default="reports/model_cards/onnx")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    class_names = [c["name"] for c in cfg["classes"]]
    print(f"[card] classes ({len(class_names)}): {class_names}")

    run_dir = Path(args.run_dir) if args.run_dir else None
    temperature = resolve_temperature(run_dir)
    metrics_csv = resolve_metrics_csv(run_dir)
    print(f"[card] temperature: {temperature:.3f}")
    print(f"[card] training metrics CSV: {metrics_csv}")

    # Manifest
    if args.csv:
        manifest_csv = Path(args.csv)
        print(f"[card] using manifest: {manifest_csv}")
    else:
        if args.data_dir is None:
            raise SystemExit("--csv or --data-dir is required.")
        rows = build_manifest_from_dir(
            Path(args.data_dir), class_names,
            max_per_class=args.max_per_class, seed=args.seed,
        )
        print(f"[card] sampled {len(rows)} clips from {args.data_dir} "
              f"(max-per-class={args.max_per_class})")
        manifest_csv = Path(args.output_dir) / "manifest.csv"
        write_manifest_csv(rows, manifest_csv)
        print(f"[card] wrote manifest → {manifest_csv}")

    predictor = ONNXPredictor(
        onnx_path=args.onnx,
        class_names=class_names,
        temperature=temperature,
        target_len=args.target_len,
        device=args.device,
        model_name=args.model_name,
    )
    print(f"[card] loaded ONNX: {args.onnx}  "
          f"providers={predictor.sess.get_providers()}")

    gen = ModelCardGenerator(
        model=predictor,
        dataset_csv=manifest_csv,
        config_path=args.config,
        device=args.device,
        sample_rate=args.sample_rate,
        target_len=args.target_len,
        training_metrics_csv=metrics_csv,
        selection_metric=args.selection_metric,
    )
    report = gen.generate()
    written = gen.save_report(report, output_dir=args.output_dir,
                              formats=("json", "md", "png"))
    print("\n[card] artifacts:")
    for fmt, p in written.items():
        print(f"  {fmt:>4}: {p}")
    print(f"\n[card] holdout macro-F1   = {report['metrics']['macro_f1']:.4f}")
    print(f"[card] holdout accuracy   = {report['metrics']['accuracy']:.4f}")
    print(f"[card] holdout MCC        = {report['metrics']['mcc']:.4f}")
    return 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
