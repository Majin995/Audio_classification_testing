#!/usr/bin/env python3
"""
purge_checkpoints.py — Keep only the single best checkpoint per model.

"Model" is defined as the top-level directory name under lightning_logs/,
with Optuna trial suffixes (_tNNNN) stripped so that all trials for the
same architecture are treated as one group.

The F1 score is read from the checkpoint filename:
  e.g.  catfish-epoch=003-f1val/f1=0.2984.ckpt
  or    hydro-epoch=027-f1val/f1=0.3550.ckpt

Usage:
    # Dry-run (default) — print what WOULD be deleted
    python scripts/purge_checkpoints.py

    # Actually delete
    python scripts/purge_checkpoints.py --delete

    # Custom log root
    python scripts/purge_checkpoints.py --log_dir /path/to/lightning_logs --delete
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = ROOT / "lightning_logs"

# Strips Optuna trial suffix: optuna_catfish_t0023 → optuna_catfish
_TRIAL_RE = re.compile(r"_t\d+$")
# Extracts F1 from checkpoint filename.  Handles two conventions:
#   new:  ...-f1val/f1=0.2984.ckpt
#   old:  ...-val_f1_score=0.598.ckpt
_F1_RE = re.compile(r"(?:f1=|val_f1_score=)([0-9]+\.?[0-9]*)\.ckpt$")


def model_group(top_dir_name: str) -> str:
    """Return the logical model name for a lightning_logs sub-directory."""
    return _TRIAL_RE.sub("", top_dir_name)


def f1_from_path(ckpt: Path) -> float:
    """Parse the F1 score encoded in a checkpoint filename.
    Returns -1 if no F1 is found (caller falls back to mtime)."""
    m = _F1_RE.search(ckpt.name)
    return float(m.group(1)) if m else -1.0


def collect(log_dir: Path) -> dict[str, list[Path]]:
    """Return {model_group: [ckpt_path, ...]} for every .ckpt under log_dir."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for ckpt in sorted(log_dir.rglob("*.ckpt")):
        # First component relative to log_dir is the top-level run dir
        rel = ckpt.relative_to(log_dir)
        top = rel.parts[0]
        groups[model_group(top)].append(ckpt)
    return dict(groups)


def purge(log_dir: Path, *, delete: bool) -> None:
    groups = collect(log_dir)
    if not groups:
        print("No checkpoints found.")
        return

    total_kept = 0
    total_deleted = 0
    freed_bytes = 0

    for group, ckpts in sorted(groups.items()):
        # Sort key: (has_f1, f1_score, mtime) — all descending.
        # Checkpoints without an F1 in their name fall back to newest-by-mtime.
        def sort_key(p: Path):
            f1 = f1_from_path(p)
            has_f1 = 1 if f1 >= 0 else 0
            mtime = p.stat().st_mtime if p.exists() else 0.0
            return (has_f1, f1, mtime)

        ranked = sorted(ckpts, key=sort_key, reverse=True)
        best = ranked[0]
        rest = ranked[1:]

        group_freed = sum(p.stat().st_size for p in rest if p.exists())
        freed_bytes += group_freed

        print(f"\n{'─'*70}")
        print(f"  MODEL : {group}")
        print(f"  KEEP  : {best.relative_to(log_dir)}  (f1={f1_from_path(best):.4f})")
        if rest:
            print(f"  DELETE: {len(rest)} checkpoint(s)  ({group_freed / 1e6:.1f} MB)")
            for p in rest:
                print(f"    - {p.relative_to(log_dir)}  (f1={f1_from_path(p):.4f})")
        else:
            print("  DELETE: (none — only one checkpoint)")

        total_kept += 1
        total_deleted += len(rest)

        if delete:
            for p in rest:
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            # Remove empty checkpoint dirs and empty version dirs
            _prune_empty_dirs(log_dir)

    print(f"\n{'='*70}")
    action = "Deleted" if delete else "Would delete"
    print(f"  {action} {total_deleted} checkpoint(s), kept {total_kept}.")
    print(f"  Space {'freed' if delete else 'to free'}: {freed_bytes / 1e9:.2f} GB")
    if not delete:
        print("\n  Re-run with --delete to apply.")
    print(f"{'='*70}")


def _prune_empty_dirs(log_dir: Path) -> None:
    """Remove directories that contain no files (bottom-up)."""
    for d in sorted(log_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir():
            try:
                # Only remove if truly empty (no files of any kind)
                if not any(d.iterdir()):
                    d.rmdir()
            except (OSError, StopIteration):
                pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_dir", default=str(DEFAULT_LOG_DIR),
                   help="Root of Lightning log directories (default: %(default)s)")
    p.add_argument("--delete", action="store_true",
                   help="Actually delete files (default is dry-run only)")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"ERROR: {log_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    mode = "DELETE" if args.delete else "DRY-RUN"
    print(f"purge_checkpoints.py  [{mode}]")
    print(f"Log dir: {log_dir}")

    purge(log_dir, delete=args.delete)


if __name__ == "__main__":
    main()
