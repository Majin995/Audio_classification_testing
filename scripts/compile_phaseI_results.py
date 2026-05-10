"""Compile Phase I results into a markdown summary.

Walks lightning_logs/phaseI_*/version_0/, pulls best val/macro_precision from
the checkpoint filename + final test metrics from the log, and emits a
ranked table comparing each rendition to Phase G R1 baseline.
"""
from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path
from typing import Optional

LOG_ROOT = Path("lightning_logs")
PHASE_G_R1 = {"name": "phaseG_R1 (baseline)", "val_mp": 0.6908, "test_mp": 0.6620}


def parse_csv_metrics(metrics_csv: Path) -> dict:
    """Best epoch-level val/macro_precision and final test metrics from metrics.csv.

    Per-class precisions are pulled from the same row as the best macro_P
    (so they reflect the *best-checkpoint* state, not whatever the last
    logged epoch was).
    """
    if not metrics_csv.exists():
        return {}
    rows = list(csv.DictReader(metrics_csv.open()))
    val_rows  = [r for r in rows if r.get("val/macro_precision")]
    test_rows = [r for r in rows if r.get("test/macro_precision")]
    out = {}
    if val_rows:
        best = max(val_rows, key=lambda r: float(r["val/macro_precision"]))
        out["val_mp"]    = float(best["val/macro_precision"])
        out["val_epoch"] = int(float(best.get("epoch", -1)))
        for i in range(8):
            v = best.get(f"val/precision_c{i}")
            if v not in (None, ""):
                try:
                    out[f"val_p_c{i}"] = float(v)
                except ValueError:
                    pass
    if test_rows:
        last = test_rows[-1]
        out["test_mp"]     = float(last.get("test/macro_precision", "nan"))
        out["test_acc"]    = float(last.get("test/acc", "nan"))
        out["test_recall"] = float(last.get("test/recall", "nan"))
        out["test_mcc"]    = float(last.get("test/mcc", "nan"))
    return out


def parse_per_class_precision(log_path: Path) -> dict:
    """Reserved — per-class precision now read from metrics.csv (above)."""
    return {}


def parse_postcal(version_dir: Path) -> dict:
    """temperature.pt + thresholds.json + selective_pr.md numbers."""
    out = {}
    cp = version_dir / "checkpoints"
    if not cp.exists():
        return out
    th = next(cp.glob("thresholds.json"), None)
    if th and th.exists():
        try:
            j = json.loads(th.read_text())
            out["macroP_at_cov085"] = j.get("macroP_coverage", {}).get("macro_precision")
            out["recall_floor_micP"] = j.get("recall_floor", {}).get("micro_precision")
        except Exception:
            pass
    return out


def collect() -> list[dict]:
    rows = []
    for run_dir in sorted(LOG_ROOT.glob("phaseI_*")):
        v0 = run_dir / "version_0"
        if not v0.exists():
            continue
        metrics_csv = v0 / "metrics.csv"
        log_path    = LOG_ROOT / "phaseI_console" / f"{run_dir.name}.log"
        rec = {"name": run_dir.name}
        rec.update(parse_csv_metrics(metrics_csv))
        rec.update(parse_per_class_precision(log_path))
        rec.update(parse_postcal(v0))
        rows.append(rec)
    return rows


def fmt(x: Optional[float], width: int = 6) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—".rjust(width)
    return f"{x:.4f}".rjust(width)


def write_report(rows: list[dict], out_path: Path) -> None:
    base = PHASE_G_R1
    lines = [
        "# Phase I — Results Summary",
        "",
        f"Baseline: **{base['name']}** val/μP={base['val_mp']:.4f} "
        f"test/μP={base['test_mp']:.4f}  MP@0.85=0.7518  gap=−0.0288",
        "",
        "Phase I primary metric: **closing the val/test gap**. The gap "
        "column shows `val − test` — smaller (or less negative) is better.",
        "Sorted by **test/μP** (the production-relevant metric).",
        "",
        "| Run | val/μP | Δval | test/μP | Δtest | gap | c2 Tanker | c3 Tug | MP@0.85 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -(r.get("test_mp") or 0)):
        v = r.get("val_mp"); t = r.get("test_mp")
        dv = (v - base["val_mp"]) if v is not None else None
        dt = (t - base["test_mp"]) if t is not None else None
        gap = (v - t) if (v is not None and t is not None) else None
        c2 = r.get("val_p_c2"); c3 = r.get("val_p_c3")
        mp85 = r.get("macroP_at_cov085")
        lines.append(
            f"| {r['name']} | {fmt(v)} | {fmt(dv)} | {fmt(t)} | {fmt(dt)} | "
            f"{fmt(gap)} | {fmt(c2)} | {fmt(c3)} | {fmt(mp85)} |"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    rows = collect()
    out = LOG_ROOT / "phaseI_summary.md"
    write_report(rows, out)
    print(f"\n→ saved to {out}")
