"""Lazy in-memory pool of low-energy 1 s crops for SNR-aware noise injection.

Used by ``_WaveformAug`` (models/hydro_precise_v2.py) to mix corpus-derived
"ambient" noise into training waveforms at a controlled SNR. The pool is
populated from a list of training file paths the first time ``sample()`` is
invoked on a CUDA stream — kept lazy so it doesn't slow down model
construction or block CPU-only smoke tests.

Selection: read the first ~``max_clips * 4`` files at random, take 4 random
1 s crops per file, keep the lowest-``energy_quantile``-energy crops up to
``max_clips``. Cargo dominates the train pool by a wide margin so the
quietest quartile is overwhelmingly background.
"""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch


class OceanNoisePool:
    """In-memory pool of low-energy crops, sampled by index per training step."""

    def __init__(
        self,
        files: List[str],
        target_sr: int,
        fixed_len: int,
        max_clips: int = 4096,
        energy_quantile: float = 0.25,
        crops_per_file: int = 4,
        seed: int = 0,
    ):
        self.target_sr       = int(target_sr)
        self.fixed_len       = int(fixed_len)
        self._max            = int(max_clips)
        self._q              = float(energy_quantile)
        self._crops_per_file = int(crops_per_file)
        self._seed           = int(seed)
        # Owning a *copy* — caller's list isn't mutated by our shuffle.
        self._files: List[str] = list(files)
        self._buf:   Optional[torch.Tensor] = None

    # ── construction ───────────────────────────────────────────────────

    @torch.no_grad()
    def _lazy_build(self, device: torch.device) -> None:
        if self._buf is not None:
            return
        try:
            import soundfile as sf
        except ImportError as exc:                          # pragma: no cover
            raise RuntimeError(
                "OceanNoisePool requires `soundfile` (pip install soundfile)"
            ) from exc

        rng = random.Random(self._seed)
        files = list(self._files)
        rng.shuffle(files)

        crops:    List[np.ndarray] = []
        energies: List[float]      = []
        cap = self._max * 4

        for path in files:
            try:
                wav, sr = sf.read(path, dtype="float32", always_2d=False)
            except Exception:
                continue
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != self.target_sr:
                continue                                    # cheap nearest skip
            if len(wav) < self.fixed_len:
                continue
            for _ in range(self._crops_per_file):
                start = rng.randint(0, len(wav) - self.fixed_len)
                seg = wav[start:start + self.fixed_len]
                crops.append(seg)
                energies.append(float((seg ** 2).mean()))
            if len(crops) >= cap:
                break

        if not crops:
            # Degenerate fallback: use a single small white-noise segment so
            # downstream code never has to special-case an empty pool.
            self._buf = torch.zeros(1, self.fixed_len, device=device)
            return

        e   = np.asarray(energies)
        thr = float(np.quantile(e, self._q))
        keep = [c for c, ev in zip(crops, energies) if ev <= thr]
        keep = keep[: self._max]
        self._buf = torch.from_numpy(np.stack(keep)).to(device)

    # ── public API ─────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return ``(batch_size, fixed_len)`` random crops from the pool."""
        self._lazy_build(device)
        assert self._buf is not None
        if self._buf.device != device:
            self._buf = self._buf.to(device)
        idx = torch.randint(0, self._buf.size(0), (batch_size,), device=device)
        return self._buf.index_select(0, idx)

    @property
    def is_built(self) -> bool:
        return self._buf is not None

    def __len__(self) -> int:
        return 0 if self._buf is None else int(self._buf.size(0))
