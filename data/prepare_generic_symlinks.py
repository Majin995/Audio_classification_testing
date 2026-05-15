"""Flatten any audio dataset into <out_dir>/<split>/<class>/<file>.wav symlinks.

Why: HydroHydra's DALI loader expects <data_dir>/<split>/<class>/<file>. If
your dataset organises audio files at a different depth (e.g.
<root>/<split>/<sub>/<class>/<file>), set --label_depth to point at the
folder that names the class — the immediate parent of the audio file is
label_depth=1, its grandparent is 2, and so on. We symlink every audio
file into a canonical layout the loader understands.

Usage:
    python data/prepare_generic_symlinks.py \\
        --src_dir /path/to/my_dataset \\
        --out_dir /tmp/my_dataset_flat \\
        --label_depth 1 \\
        --splits train val test
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label_depth", type=int, default=1,
                    help="Class folder is N levels up from each audio file "
                         "(1 = the file's immediate parent dir).")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Split subdirectory names under --src_dir.")
    ap.add_argument("--clean", action="store_true",
                    help="Delete --out_dir before populating.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the plan but don't create symlinks.")
    args = ap.parse_args()

    if args.label_depth < 1:
        print("[error] --label_depth must be >= 1", file=sys.stderr)
        sys.exit(2)

    src = Path(args.src_dir).resolve()
    out = Path(args.out_dir).resolve()
    if not src.is_dir():
        print(f"[error] --src_dir not a directory: {src}", file=sys.stderr)
        sys.exit(2)

    if args.clean and out.exists() and not args.dry_run:
        shutil.rmtree(out)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    all_classes = set()
    for split in args.splits:
        sd = src / split
        if not sd.is_dir():
            print(f"[warn] missing split dir: {sd}", file=sys.stderr)
            continue

        n_files = 0
        n_skipped = 0
        per_class: dict[str, int] = {}
        for f in sorted(sd.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() not in AUDIO_EXTS:
                continue
            try:
                cls_dir = f.parents[args.label_depth - 1]
                # Ensure the class folder lives under the split dir, not above.
                cls_dir.relative_to(sd)
            except (IndexError, ValueError):
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"[warn] file too shallow for label_depth="
                          f"{args.label_depth}: {f}", file=sys.stderr)
                continue
            cls_name = cls_dir.name
            per_class[cls_name] = per_class.get(cls_name, 0) + 1
            all_classes.add(cls_name)

            # Flatten relative path to a unique filename so two files at
            # different sub-paths with the same basename don't collide.
            rel = f.relative_to(sd)
            flat_name = "__".join(rel.parts)
            dst_dir = out / split / cls_name
            dst = dst_dir / flat_name
            if args.dry_run:
                n_files += 1
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(f.resolve())
            n_files += 1

        print(f"[ok] {split}: {n_files} files into {len(per_class)} classes"
              + (f"  (skipped {n_skipped})" if n_skipped else ""))
        for c in sorted(per_class):
            print(f"     {c}: {per_class[c]}")
        grand_total += n_files

    print()
    print(f"[done] {grand_total} {'planned' if args.dry_run else 'symlinks created'}"
          f" → {out}")
    print(f"[done] classes seen: {sorted(all_classes)}")


if __name__ == "__main__":
    main()
