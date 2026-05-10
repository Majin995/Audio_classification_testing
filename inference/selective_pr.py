"""Selective-prediction precision sweep.

For a list of coverage targets in (0, 1], pick a single global confidence
threshold so that the kept fraction ≈ target coverage, then report:

  • coverage (actual fraction of samples kept)
  • micro_precision (fraction correct over kept samples)
  • macro_precision (mean of per-class precisions over kept predictions)
  • per-class precision (over kept predictions)

Output is a markdown table written next to the trained checkpoint.

This is meant to formalize the "MP @ coverage=0.85" callout the trainer's
post-cal block has been printing — and to expose the precision/coverage
trade-off so we can pick a deployment threshold by precision target rather
than by coverage target.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def selective_pr_curve(
    probs:        np.ndarray,
    targets:      np.ndarray,
    num_classes:  int,
    coverages:    Optional[Sequence[float]] = None,
) -> List[Dict]:
    coverages = list(coverages) if coverages is not None else \
        [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    p_max  = probs.max(axis=1)
    argmax = probs.argmax(axis=1)

    rows: List[Dict] = []
    for cov in coverages:
        # quantile on p_max: drop the lowest (1-cov) fraction.
        q = float(np.quantile(p_max, 1.0 - cov))
        keep = p_max >= q
        actual_cov = float(keep.mean())
        if keep.sum() == 0:
            rows.append({
                "target_coverage": cov, "coverage": 0.0,
                "threshold": q, "micro_precision": 0.0,
                "macro_precision": 0.0,
                "per_class_precision": [0.0] * num_classes,
            })
            continue
        preds, gts = argmax[keep], targets[keep]
        per_class: List[float] = []
        for c in range(num_classes):
            m = preds == c
            if m.sum() == 0:
                per_class.append(float("nan"))
                continue
            per_class.append(float((gts[m] == c).mean()))
        valid = [x for x in per_class if not np.isnan(x)]
        macro_p = float(np.mean(valid)) if valid else 0.0
        micro_p = float((preds == gts).mean())
        rows.append({
            "target_coverage": cov, "coverage": actual_cov, "threshold": q,
            "micro_precision": micro_p, "macro_precision": macro_p,
            "per_class_precision": per_class,
        })
    return rows


def render_markdown(rows: List[Dict], num_classes: int,
                    class_names: Optional[Sequence[str]] = None) -> str:
    cls_hdr = list(class_names) if class_names is not None else \
        [f"c{i}" for i in range(num_classes)]
    md = ["# Selective-prediction precision curve", ""]
    md.append("| target_cov | actual_cov | threshold | micro_P | macro_P | "
              + " | ".join(f"P[{n}]" for n in cls_hdr) + " |")
    md.append("|---|---|---|---|---|" + "---|" * num_classes)
    for r in rows:
        per_cls = " | ".join(
            f"{p:.4f}" if not np.isnan(p) else "—"
            for p in r["per_class_precision"]
        )
        md.append(
            f"| {r['target_coverage']:.2f} | {r['coverage']:.4f} | "
            f"{r['threshold']:.4f} | {r['micro_precision']:.4f} | "
            f"{r['macro_precision']:.4f} | {per_cls} |"
        )
    return "\n".join(md) + "\n"


def write_markdown(out_path: Path, rows: List[Dict], num_classes: int,
                   class_names: Optional[Sequence[str]] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(rows, num_classes, class_names))
