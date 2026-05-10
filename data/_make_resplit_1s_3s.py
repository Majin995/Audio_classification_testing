"""Re-split the rapid-testing subset into matched 1s and 3s datasets,
hard-copied to /var/mnt/5A009BF8009BD8F9/Data/.

Source layout (symlinks → Split1s):
    data/Classification_rapid_testing/{train,val,test}/<class>/<RECID>_segment_<N>.wav

Each subset file is a 1.378s clip whose temporal location in the original
recording starts at N seconds (verified: byte-accurate at offset N*32000
samples @ 32 kHz). Originals live at:
    /run/media/damo/LaCie/Ubuntu BackUp/Deepship/Raw/<class>/<RECID>.wav

Outputs (hard copies, 32 kHz mono PCM_16 WAV):
    /var/mnt/5A009BF8009BD8F9/Data/
        Classification_rapid_testing_1s/{train,val,test}/<class>/<RECID>_seg<N>_1s.wav
        Classification_rapid_testing_3s/{train,val,test}/<class>/<RECID>_seg<N>_3s.wav

The 1s window = [N s, N+1 s].
The 3s window = [N-1 s, N+2 s] centered on the same segment, clamped to the
recording bounds (so end-of-recording segments shift the window inward
rather than zero-padding).

If a 3s window can't fit (recording shorter than 3s, very rare), the file
is skipped and reported. 1s windows that fall past the original's end are
also skipped.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SRC_SUBSET = Path(
    "/var/home/damo/Documents/Git/Audio_classification_testing/data/Classification_rapid_testing"
)
ORIG_ROOT = Path("/run/media/damo/LaCie/Ubuntu BackUp/Deepship/Raw")
DST_ROOT = Path("/var/mnt/5A009BF8009BD8F9/Data")
DST_1S = DST_ROOT / "Classification_rapid_testing_1s"
DST_3S = DST_ROOT / "Classification_rapid_testing_3s"

SR = 32000
LEN_1S = 1 * SR
LEN_3S = 3 * SR
SPLITS = ["train", "val", "test"]
NAME_RE = re.compile(r"^(?P<rec>\d+)_segment_(?P<n>\d+)$")


def parse_name(stem: str) -> tuple[str, int] | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    return m.group("rec"), int(m.group("n"))


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SR, subtype="PCM_16")


def main() -> int:
    if not ORIG_ROOT.exists():
        print(f"ERR originals not mounted: {ORIG_ROOT}", file=sys.stderr)
        return 1

    # Cache full-length originals once per (class, RECID) since multiple
    # subset segments can share an original.
    orig_cache: dict[tuple[str, str], np.ndarray | None] = {}

    def load_orig(cls: str, rec: str) -> np.ndarray | None:
        key = (cls, rec)
        if key not in orig_cache:
            p = ORIG_ROOT / cls / f"{rec}.wav"
            if not p.exists():
                orig_cache[key] = None
            else:
                audio, sr = sf.read(str(p), always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != SR:
                    print(f"WARN sr mismatch {sr} for {p.name} — skipping", file=sys.stderr)
                    orig_cache[key] = None
                else:
                    orig_cache[key] = audio.astype(np.float32, copy=False)
        return orig_cache[key]

    counts = {"1s_ok": 0, "3s_ok": 0, "missing_orig": 0, "out_of_range_1s": 0, "too_short_3s": 0, "bad_name": 0}

    for split in SPLITS:
        for cls_dir in sorted((SRC_SUBSET / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() != ".wav":
                    continue
                parsed = parse_name(f.stem)
                if parsed is None:
                    counts["bad_name"] += 1
                    continue
                rec, n = parsed
                orig = load_orig(cls, rec)
                if orig is None:
                    counts["missing_orig"] += 1
                    continue

                # 1s window: [n s, n+1 s]
                start_1s = n * SR
                if start_1s + LEN_1S > len(orig):
                    counts["out_of_range_1s"] += 1
                else:
                    clip_1s = orig[start_1s:start_1s + LEN_1S]
                    out_1s = DST_1S / split / cls / f"{rec}_seg{n}_1s.wav"
                    write_wav(out_1s, clip_1s)
                    counts["1s_ok"] += 1

                # 3s window: centered on segment, clamped
                if len(orig) < LEN_3S:
                    counts["too_short_3s"] += 1
                else:
                    seg_center = n * SR + SR // 2
                    start_3s = seg_center - LEN_3S // 2
                    start_3s = max(0, min(start_3s, len(orig) - LEN_3S))
                    clip_3s = orig[start_3s:start_3s + LEN_3S]
                    out_3s = DST_3S / split / cls / f"{rec}_seg{n}_3s.wav"
                    write_wav(out_3s, clip_3s)
                    counts["3s_ok"] += 1

    print("\nDone.")
    print(f"  Wrote 1s: {DST_1S}  ({counts['1s_ok']} files)")
    print(f"  Wrote 3s: {DST_3S}  ({counts['3s_ok']} files)")
    print(f"  Missing originals: {counts['missing_orig']}")
    print(f"  1s out-of-range:   {counts['out_of_range_1s']}")
    print(f"  Recording <3s:     {counts['too_short_3s']}")
    print(f"  Bad filenames:     {counts['bad_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
