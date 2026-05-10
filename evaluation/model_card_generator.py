"""
HydroVision — Architecture-Agnostic Model Card Generator.

Updates vs. the v5 archive copy
-------------------------------
* Renders a confusion-matrix PNG (matplotlib heatmap) alongside the
  Markdown / JSON outputs and embeds it in the report.
* Optionally pulls best-epoch validation metrics from a Lightning
  ``metrics.csv`` and surfaces them in a "Training metrics" section so the
  card reflects the actual training run, not just the holdout evaluation.

Public surface is unchanged: ``ModelCardGenerator(...).generate()`` returns
a dict and ``save_report(...)`` writes the artifacts.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import yaml

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Plain-English metric tooltips
# ═══════════════════════════════════════════════════════════════════════════

METRIC_TOOLTIPS: Dict[str, str] = {
    "accuracy": (
        "What fraction of all predictions were correct?  "
        "e.g. 0.75 means the model guessed right 75% of the time.  "
        "Can be misleading when classes are imbalanced."
    ),
    "macro_f1": (
        "Average F1 score across all classes, weighted equally.  "
        "F1 balances how often the model catches a class (recall) against "
        "how often it raises a false alarm for that class (precision).  "
        "A score of 1.0 is perfect; 0.0 is completely wrong."
    ),
    "macro_precision": (
        "When the model predicts a class, how often is it correct?  "
        "High precision means few false alarms.  Averaged equally across all classes."
    ),
    "macro_recall": (
        "Of all samples that truly belong to a class, how many did the model "
        "find?  High recall means the model misses few real events.  "
        "Averaged equally across all classes."
    ),
    "mcc": (
        "Matthews Correlation Coefficient — a balanced single-number summary "
        "ranging from -1 (completely wrong) to +1 (perfect).  "
        "Unlike accuracy, MCC stays reliable even when class sizes are very unequal."
    ),
    "per_class_f1": (
        "F1 score for each vessel type individually.  "
        "Shows which classes the model handles well and which it struggles with."
    ),
    "per_class_precision": (
        "For each class: when the model predicts it, how often is it right?"
    ),
    "per_class_recall": (
        "For each class: of all true examples, how many did the model detect?"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Training-metrics extractor (Lightning CSV logger)
# ═══════════════════════════════════════════════════════════════════════════

def load_training_metrics(
    metrics_csv: str | os.PathLike,
    selection_metric: str = "val/macro_precision",
) -> Optional[Dict[str, Any]]:
    """Pick the best epoch's val/* metrics from a Lightning metrics.csv.

    Returns None if the file does not exist or the selection metric is absent.
    """
    p = Path(metrics_csv)
    if not p.is_file():
        return None
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not installed — skipping training metrics import")
        return None

    df = pd.read_csv(p)
    if selection_metric not in df.columns:
        log.warning("selection metric '%s' missing from %s", selection_metric, p)
        return None

    df_sel = df.dropna(subset=[selection_metric])
    if df_sel.empty:
        return None
    best_idx = df_sel[selection_metric].idxmax()
    best_row = df.loc[best_idx]

    val_metrics = {}
    for col in df.columns:
        if not col.startswith("val/"):
            continue
        v = best_row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        val_metrics[col] = float(v)

    out = {
        "source_csv": str(p),
        "selection_metric": selection_metric,
        "best_epoch": int(best_row.get("epoch", -1)) if "epoch" in df.columns else None,
        "metrics": val_metrics,
    }
    # Test/* metrics if a test loop was run
    test_metrics = {}
    for col in df.columns:
        if not col.startswith("test/"):
            continue
        non_null = df[col].dropna()
        if not non_null.empty:
            test_metrics[col] = float(non_null.iloc[-1])
    if test_metrics:
        out["test_metrics"] = test_metrics
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Confusion matrix renderer (matplotlib heatmap)
# ═══════════════════════════════════════════════════════════════════════════

def render_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_path: str | os.PathLike,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    class_colors: Optional[Sequence[str]] = None,
) -> Path:
    """Save a heatmap PNG of the confusion matrix.

    With ``normalize=True`` each row sums to 1 (recall view).  Counts are
    annotated regardless.  ``class_colors`` is currently unused but kept in
    the signature so future per-class tinting can land without API churn.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    counts = cm.astype(np.int64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    n = len(class_names)
    fig_w = max(4.5, 0.85 * n + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85), dpi=150)
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else cm.max())
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Recall (row-normalised)" if normalize else "Count")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = im.norm(0.5) if normalize else cm.max() * 0.5
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            text_color = "white" if val > threshold else "black"
            if normalize:
                ax.text(j, i, f"{val:.2f}\n({counts[i, j]})",
                        ha="center", va="center", color=text_color, fontsize=9)
            else:
                ax.text(j, i, str(counts[i, j]),
                        ha="center", va="center", color=text_color, fontsize=10)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
#  ModelCardGenerator
# ═══════════════════════════════════════════════════════════════════════════

class ModelCardGenerator:
    """
    Architecture-agnostic model evaluation and card generation.

    Parameters
    ----------
    model :
        Any object exposing
        ``predict(waveform: torch.Tensor, sample_rate: int) -> dict[str, float]``.
    dataset_csv :
        Path to a two-column CSV file ``<audio_path>,<class_name>``.  An
        optional header row is detected and skipped automatically.
    config_path :
        Path to ``model_config.yaml`` (or compatible JSON).
    device :
        Torch device string used when constructing input tensors.
    sample_rate, target_len :
        Audio resampling / length normalisation applied before predict().
    training_metrics_csv :
        Optional path to a Lightning CSV logger ``metrics.csv``.  When set,
        the best-epoch val/* metrics are surfaced in the report.
    selection_metric :
        Which val/* column drives "best epoch" selection.
    """

    def __init__(
        self,
        model:       Any,
        dataset_csv: str | os.PathLike,
        config_path: str | os.PathLike = "evaluation/model_config.yaml",
        device:      str = "cpu",
        sample_rate: int = 5_120,
        target_len:  int = 5_120,
        training_metrics_csv: Optional[str | os.PathLike] = None,
        selection_metric: str = "val/macro_precision",
    ) -> None:
        self.model       = model
        self.dataset_csv = Path(dataset_csv)
        self.device      = device
        self.sample_rate = sample_rate
        self.target_len  = target_len
        self.training_metrics_csv = (
            Path(training_metrics_csv) if training_metrics_csv else None
        )
        self.selection_metric = selection_metric

        cfg_path = Path(config_path)
        if cfg_path.suffix in (".yaml", ".yml"):
            with open(cfg_path) as f:
                self._cfg = yaml.safe_load(f)
        else:
            with open(cfg_path) as f:
                self._cfg = json.load(f)

        self._class_cfg   = self._cfg.get("classes", [])
        self._class_names = [c["name"] for c in self._class_cfg]
        self._class_colors = [c.get("color", "#888888") for c in self._class_cfg]
        self._report_cfg  = self._cfg.get("report", {})

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> Dict[str, Any]:
        log.info("ModelCardGenerator: loading dataset from %s", self.dataset_csv)
        samples = self._load_dataset()
        log.info("ModelCardGenerator: %d samples — running inference …", len(samples))

        y_true, y_pred = self._run_inference(samples)

        log.info("ModelCardGenerator: computing metrics …")
        metrics = self._compute_metrics(y_true, y_pred)
        cm      = self._confusion_matrix(y_true, y_pred)
        insights = self._derive_insights(metrics, cm)

        model_name = getattr(self.model, "model_name", None) \
                  or getattr(self.model, "NAME",       None) \
                  or type(self.model).__name__

        report: Dict[str, Any] = {
            "title":          self._report_cfg.get("title", "Model Evaluation Card"),
            "model_name":     model_name,
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_csv":    str(self.dataset_csv),
            "sample_count":   len(samples),
            "class_names":    self._class_names,
            "metrics":        metrics,
            "confusion_matrix": cm.tolist(),
            "metric_tooltips":  METRIC_TOOLTIPS,
            "insights":       insights,
        }

        if self.training_metrics_csv is not None:
            tm = load_training_metrics(self.training_metrics_csv,
                                       selection_metric=self.selection_metric)
            if tm is not None:
                report["training_metrics"] = tm

        log.info(
            "ModelCardGenerator: done  macro-F1=%.4f  accuracy=%.4f",
            metrics["macro_f1"], metrics["accuracy"],
        )
        return report

    def save_report(
        self,
        report:     Dict[str, Any],
        output_dir: str | os.PathLike = "training_results/model_cards",
        formats:    Sequence[str] = ("json", "md", "png"),
    ) -> Dict[str, Path]:
        """Persist report. Formats: 'json', 'md', 'png' (confusion-matrix heatmap)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stem = f"model_card_{report['model_name'].replace(' ', '_')}"
        written: Dict[str, Path] = {}

        if "png" in formats:
            cm_path = out / f"{stem}_confusion.png"
            render_confusion_matrix(
                np.array(report["confusion_matrix"]),
                report["class_names"],
                cm_path,
                title=f"{report['model_name']} — Confusion Matrix "
                      f"(n={report['sample_count']})",
                normalize=True,
                class_colors=self._class_colors,
            )
            report["_confusion_png"] = cm_path.name
            written["png"] = cm_path
            log.info("ModelCardGenerator: saved PNG  → %s", cm_path)

        if "json" in formats:
            p = out / f"{stem}.json"
            with open(p, "w") as f:
                json.dump({k: v for k, v in report.items() if not k.startswith("_")},
                          f, indent=2)
            written["json"] = p
            log.info("ModelCardGenerator: saved JSON → %s", p)

        if "md" in formats:
            p = out / f"{stem}.md"
            with open(p, "w") as f:
                f.write(self._render_markdown(report))
            written["md"] = p
            log.info("ModelCardGenerator: saved Markdown → %s", p)

        return written

    # ── Dataset loading ───────────────────────────────────────────────────────

    def _load_dataset(self) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        with open(self.dataset_csv, newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if len(row) < 2:
                    continue
                path, label = row[0].strip(), row[1].strip()
                if i == 0 and not os.path.exists(path):
                    continue
                if label not in self._class_names:
                    log.warning("Unknown label '%s' — skipping %s", label, path)
                    continue
                rows.append((path, label))
        if not rows:
            raise ValueError(
                f"No valid samples in {self.dataset_csv}. Check that paths "
                "exist and labels match model_config.yaml."
            )
        return rows

    # ── Inference ─────────────────────────────────────────────────────────────

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            from math import gcd
            import scipy.signal
            g = gcd(sr, self.sample_rate)
            wav = scipy.signal.resample_poly(
                wav, self.sample_rate // g, sr // g
            ).astype(np.float32)
        n = len(wav)
        if n < self.target_len:
            wav = np.concatenate([wav, np.zeros(self.target_len - n, dtype=np.float32)])
        else:
            wav = wav[: self.target_len]
        return torch.from_numpy(wav).unsqueeze(0)

    def _run_inference(
        self, samples: List[Tuple[str, str]],
    ) -> Tuple[List[str], List[str]]:
        y_true: List[str] = []
        y_pred: List[str] = []
        n = len(samples)
        for i, (audio_path, true_label) in enumerate(samples):
            if i % 200 == 0 and i:
                log.info("  inference %d / %d …", i, n)
            try:
                wav = self._load_audio(audio_path).to(self.device)
                preds = self.model.predict(wav, self.sample_rate)
                predicted = max(preds, key=preds.get)
            except Exception as exc:
                log.warning("  inference failed for %s: %s — skipping",
                            audio_path, exc)
                continue
            y_true.append(true_label)
            y_pred.append(predicted)
        return y_true, y_pred

    # ── Metric computation ────────────────────────────────────────────────────

    def _confusion_matrix(self, y_true, y_pred) -> np.ndarray:
        n   = len(self._class_names)
        idx = {c: i for i, c in enumerate(self._class_names)}
        cm  = np.zeros((n, n), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if t in idx and p in idx:
                cm[idx[t], idx[p]] += 1
        return cm

    def _compute_metrics(self, y_true, y_pred) -> Dict[str, Any]:
        cm     = self._confusion_matrix(y_true, y_pred)
        total  = cm.sum()
        correct = cm.trace()
        accuracy = float(correct / total) if total else 0.0

        per_class: Dict[str, Dict[str, float]] = {}
        precisions, recalls, f1s = [], [], []
        for i, cls in enumerate(self._class_names):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            per_class[cls] = {
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "f1":        round(f1,   4),
                "support":   int(cm[i, :].sum()),
            }
            precisions.append(prec); recalls.append(rec); f1s.append(f1)

        return {
            "accuracy":         round(accuracy,                4),
            "macro_f1":         round(float(np.mean(f1s)),     4),
            "macro_precision":  round(float(np.mean(precisions)), 4),
            "macro_recall":     round(float(np.mean(recalls)), 4),
            "mcc":              round(self._mcc(cm),           4),
            "per_class":        per_class,
        }

    @staticmethod
    def _mcc(cm: np.ndarray) -> float:
        n     = cm.sum()
        s     = cm.sum(axis=1)
        p     = cm.sum(axis=0)
        t     = cm.trace()
        num   = n * t - float(np.dot(s, p))
        denom = np.sqrt(
            (n ** 2 - float(np.dot(p, p))) *
            (n ** 2 - float(np.dot(s, s)))
        )
        return float(num / denom) if denom > 0 else 0.0

    # ── Insights ──────────────────────────────────────────────────────────────

    def _derive_insights(self, metrics, cm) -> Dict[str, Any]:
        per_class  = metrics["per_class"]
        class_names = self._class_names

        best_cls = max(per_class, key=lambda c: per_class[c]["f1"])
        best_f1  = per_class[best_cls]["f1"]
        worst_cls = min(per_class, key=lambda c: per_class[c]["f1"])
        worst_f1  = per_class[worst_cls]["f1"]

        idx      = {c: i for i, c in enumerate(class_names)}
        total    = cm.sum()
        fp_rates: Dict[str, float] = {}
        for cls in class_names:
            i  = idx[cls]
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            tn = int(total - tp - fp - fn)
            fp_rates[cls] = fp / (fp + tn) if (fp + tn) else 0.0

        biased_fp_cls  = max(fp_rates, key=fp_rates.get)
        biased_fp_rate = fp_rates[biased_fp_cls]

        cm_copy = cm.copy().astype(float)
        np.fill_diagonal(cm_copy, 0)
        max_idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
        confused_true  = class_names[max_idx[0]]
        confused_pred  = class_names[max_idx[1]]
        confused_count = int(cm_copy[max_idx])

        pros: List[str] = []
        cons: List[str] = []
        if metrics["macro_f1"]      >= 0.65: pros.append(f"Strong overall macro-F1 of {metrics['macro_f1']:.2f}.")
        if metrics["mcc"]           >= 0.50: pros.append(f"High MCC ({metrics['mcc']:.2f}) confirms reliability on imbalanced classes.")
        if best_f1                  >= 0.75: pros.append(f"Excellent detection of {best_cls} (F1 = {best_f1:.2f}).")
        if metrics["macro_recall"]  >= 0.70: pros.append("High macro-recall: the model misses few real targets.")
        if metrics["macro_f1"]      <  0.50: cons.append(f"Macro-F1 of {metrics['macro_f1']:.2f} is below the 0.50 threshold — not yet deployment-ready.")
        if worst_f1                 <  0.40: cons.append(f"Poor detection of {worst_cls} (F1 = {worst_f1:.2f}) — consider more training data or class weighting for this category.")
        if biased_fp_rate           >  0.15: cons.append(f"High false-positive rate for {biased_fp_cls} ({biased_fp_rate:.1%}) — the model over-predicts this class.")
        if confused_count           >  0:    cons.append(f"Most common confusion: {confused_true} misclassified as {confused_pred} ({confused_count} times) — acoustic overlap between these classes needs addressing.")

        if not pros and metrics["macro_f1"] >= 0.45:
            pros.append(f"Moderate macro-F1 ({metrics['macro_f1']:.2f}) — a viable baseline.")
        if not cons:
            cons.append("No critical failure modes identified in this evaluation.")

        min_f1 = self._report_cfg.get("min_acceptable_f1", 0.50)
        return {
            "best_at": f"Most reliable at identifying {best_cls} (per-class F1 = {best_f1:.2f}).",
            "biased_fp": (
                f"Shows a bias toward false positives for {biased_fp_cls} (FPR = {biased_fp_rate:.1%})."
                if biased_fp_rate > 0.05 else "No significant false-positive bias detected."
            ),
            "most_confused_pair": (
                f"{confused_true} → {confused_pred} ({confused_count} misclassifications)"
                if confused_count > 0 else "No dominant confusion pair."
            ),
            "pros": pros,
            "cons": cons,
            "deployment_ready": metrics["macro_f1"] >= min_f1 and metrics["mcc"] >= 0.3,
        }

    # ── Markdown rendering ────────────────────────────────────────────────────

    def _render_markdown(self, report: Dict[str, Any]) -> str:
        m       = report["metrics"]
        ins     = report["insights"]
        classes = report["class_names"]
        cm      = np.array(report["confusion_matrix"])

        lines: List[str] = []
        app = lines.append

        app(f"# {report['title']}")
        app("")
        app(f"**Model:** {report['model_name']}  ")
        app(f"**Generated:** {report['timestamp']}  ")
        app(f"**Dataset:** `{report['dataset_csv']}` ({report['sample_count']} samples)  ")
        app(f"**Deployment ready:** {'Yes' if ins['deployment_ready'] else 'No'}")
        app("")

        # ── Holdout summary ─────────────────────────────────────────────
        app("## Performance Summary (holdout)")
        app("")
        app("| Metric | Score | Plain English |")
        app("|--------|-------|---------------|")
        for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "mcc"):
            short = METRIC_TOOLTIPS[key].split(".")[0] + "."
            app(f"| {key.replace('_', ' ').title()} | {m[key]:.4f} | {short} |")
        app("")

        # ── Per-class ───────────────────────────────────────────────────
        app("## Per-Class Performance")
        app("")
        app("| Class | Precision | Recall | F1 | Support |")
        app("|-------|-----------|--------|----|---------|")
        for cls in classes:
            pc = m["per_class"][cls]
            app(f"| {cls} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
                f"{pc['f1']:.4f} | {pc['support']} |")
        app("")

        # ── Confusion matrix (image + table) ────────────────────────────
        app("## Confusion Matrix")
        app("")
        if "_confusion_png" in report:
            app(f"![Confusion matrix]({report['_confusion_png']})")
            app("")
        app("Rows = ground truth, columns = predicted.  Diagonal = correct.")
        app("")
        app("| True \\ Pred | " + " | ".join(classes) + " |")
        app("|" + "---|" * (len(classes) + 1))
        for i, cls in enumerate(classes):
            app("| " + cls + " | "
                + " | ".join(str(cm[i, j]) for j in range(len(classes))) + " |")
        app("")

        # ── Training metrics (best epoch) ───────────────────────────────
        if "training_metrics" in report:
            tm = report["training_metrics"]
            app("## Training Metrics (best epoch)")
            app("")
            ep = tm.get("best_epoch")
            sel = tm.get("selection_metric")
            app(f"Selected by `{sel}` from `{tm['source_csv']}` "
                f"(epoch {ep}).")
            app("")
            app("| Metric | Value |")
            app("|--------|-------|")
            for k in sorted(tm["metrics"]):
                app(f"| `{k}` | {tm['metrics'][k]:.4f} |")
            app("")
            if "test_metrics" in tm:
                app("### Test Metrics (last logged)")
                app("")
                app("| Metric | Value |")
                app("|--------|-------|")
                for k in sorted(tm["test_metrics"]):
                    app(f"| `{k}` | {tm['test_metrics'][k]:.4f} |")
                app("")

        # ── Insights ────────────────────────────────────────────────────
        app("## Insights")
        app("")
        app(f"**Best at:** {ins['best_at']}")
        app("")
        app(f"**False-positive bias:** {ins['biased_fp']}")
        app("")
        app(f"**Most confused pair:** {ins['most_confused_pair']}")
        app("")
        app("### Pros")
        for pro in ins["pros"]:
            app(f"- {pro}")
        app("")
        app("### Cons")
        for con in ins["cons"]:
            app(f"- {con}")
        app("")

        # ── Definitions ─────────────────────────────────────────────────
        app("## Metric Definitions")
        app("")
        for key, tip in METRIC_TOOLTIPS.items():
            app(f"**{key.replace('_', ' ').title()}:** {tip}")
            app("")

        return "\n".join(lines)
