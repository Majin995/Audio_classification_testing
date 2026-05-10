"""
Blind Source Separation — NMF + FastICA
=========================================

Single-channel target isolation for underwater acoustics.

Pipeline
--------
1. STFT → magnitude matrix |V|  (shape: freq_bins × time_frames).
2. Non-negative Matrix Factorization (sklearn NMF, KL divergence, NNDSVD init)
   → W (freq_bins × n_components), H (n_components × time_frames).
3. Cluster component columns of W by cosine similarity into n_sources groups.
   Build per-source Wiener masks from the reconstructed soft assignments.
4. Reconstruct n_sources time-domain signals via iSTFT with original phase.
5. Optional FastICA pass on the n_sources signals for residual independence.
6. Return the energy-dominant component (the target vessel signal).

Reference
---------
  Paris Smaragdis, "Blind Source Separation by Non-Negative Matrix Factorization",
  IEEE Signal Processing Letters, 2004.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

try:
    from sklearn.decomposition import NMF, FastICA
    from sklearn.preprocessing import normalize
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def nmf_ica_separate(
    x:            np.ndarray,
    fs:           int  = 5_120,
    n_sources:    int  = 2,
    n_components: int  = 40,
    n_fft:        int  = 256,
    hop_length:   int  = 64,
    method:       Literal["nmf", "ica", "nmf_ica"] = "nmf_ica",
    max_iter:     int  = 500,
) -> np.ndarray:
    """
    Isolate the dominant source from a single-channel mixture.

    Args:
        x            : Input waveform (T,), float32.
        fs           : Sample rate.
        n_sources    : Number of sources to model (2 = target + background).
        n_components : NMF dictionary size (number of basis vectors).
        n_fft        : STFT window size.
        hop_length   : STFT hop length.
        method       : ``'nmf'`` | ``'ica'`` | ``'nmf_ica'`` (default).
        max_iter     : NMF iteration budget.

    Returns:
        Dominant source waveform (T,), same dtype as input.
    """
    if not _SKLEARN_AVAILABLE or not _LIBROSA_AVAILABLE:
        return x   # pass-through if deps missing

    x = np.asarray(x, dtype=np.float32)
    original_len = len(x)

    # ── STFT ─────────────────────────────────────────────────────────────
    D  = librosa.stft(x.astype(np.float64), n_fft=n_fft, hop_length=hop_length)
    V  = np.abs(D)          # (freq_bins, time_frames)
    P  = np.angle(D)        # phase  (freq_bins, time_frames)

    if method == "ica":
        sources = _ica_separate(V, P, n_sources, n_fft, hop_length, original_len)
    elif method == "nmf":
        sources = _nmf_separate(V, P, n_sources, n_components, max_iter,
                                 n_fft, hop_length, original_len)
    else:   # nmf_ica
        sources = _nmf_separate(V, P, n_sources, n_components, max_iter,
                                 n_fft, hop_length, original_len)
        if len(sources) >= 2:
            sources = _ica_refine(sources)

    # Return the energy-dominant component
    dominant = max(sources, key=lambda s: np.mean(s ** 2))
    return dominant.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nmf_separate(
    V: np.ndarray, P: np.ndarray,
    n_sources: int, n_components: int, max_iter: int,
    n_fft: int, hop_length: int, orig_len: int,
) -> list[np.ndarray]:
    """NMF-based separation using soft Wiener masks."""
    F, T = V.shape
    V_safe = np.maximum(V, 1e-9)

    model = NMF(
        n_components=n_components,
        beta_loss="kullback-leibler",
        solver="mu",
        init="nndsvd",
        max_iter=max_iter,
        random_state=0,
    )
    W = model.fit_transform(V_safe.T)   # (T, n_components) — NMF on time-major
    H = model.components_               # (n_components, F)

    # W now = time activations, H = spectral bases — transpose back:
    W_freq = H.T   # (F, n_components) spectral bases
    H_time = W.T   # (n_components, T) time activations

    # Cluster spectral bases into n_sources groups
    labels = _cosine_cluster(W_freq.T, n_sources)   # cluster on (n_components, F)

    sources = []
    for src_id in range(n_sources):
        mask_idx  = labels == src_id
        if not mask_idx.any():
            continue
        V_src = (W_freq[:, mask_idx] @ H_time[mask_idx, :])   # (F, T)
        V_src = np.maximum(V_src, 0.0)
        # Wiener mask
        mask  = V_src / (V_safe + 1e-9)
        D_src = mask * V * np.exp(1j * P)
        wav   = librosa.istft(D_src, n_fft=n_fft, hop_length=hop_length,
                               length=orig_len)
        sources.append(wav.astype(np.float32))

    return sources or [np.zeros(orig_len, dtype=np.float32)]


def _ica_separate(
    V: np.ndarray, P: np.ndarray,
    n_sources: int, n_fft: int, hop_length: int, orig_len: int,
) -> list[np.ndarray]:
    """
    FastICA on spectrogram rows (frequency bands) — lightweight alternative
    when NMF fails or is too slow.
    """
    ica = FastICA(n_components=n_sources, random_state=0, max_iter=500, tol=0.01)
    try:
        S = ica.fit_transform(V.T)   # (T, n_sources) — independent components
    except Exception:
        return [librosa.istft(V * np.exp(1j * P), n_fft=n_fft,
                               hop_length=hop_length, length=orig_len).astype(np.float32)]

    sources = []
    for i in range(n_sources):
        mask = np.abs(S[:, i])[:, np.newaxis]  # (T, 1) → broadcast to (T, F)
        D_src = mask.T * V / (V.sum(axis=1, keepdims=True) + 1e-9) * V * np.exp(1j * P)
        wav   = librosa.istft(D_src, n_fft=n_fft, hop_length=hop_length, length=orig_len)
        sources.append(wav.astype(np.float32))
    return sources


def _ica_refine(sources: list[np.ndarray]) -> list[np.ndarray]:
    """Run a quick FastICA pass on stacked time-domain sources for residual decorrelation."""
    try:
        S = np.stack(sources, axis=1)   # (T, n_sources)
        ica = FastICA(n_components=S.shape[1], random_state=0, max_iter=300, tol=0.01)
        S_ica = ica.fit_transform(S)    # (T, n_sources)
        return [S_ica[:, i].astype(np.float32) for i in range(S_ica.shape[1])]
    except Exception:
        return sources


def _cosine_cluster(X: np.ndarray, k: int) -> np.ndarray:
    """
    Greedy cosine-similarity clustering of rows in X into k clusters.
    Cheap alternative to K-means for small n_components (<= 64).
    """
    X_norm  = normalize(X, norm="l2", axis=1)   # (n, F_norm)
    n       = len(X_norm)
    labels  = np.zeros(n, dtype=int)

    # Initialise k centroids by spreading coverage
    centroids = [X_norm[0]]
    for _ in range(1, k):
        sims = np.array([X_norm @ c for c in centroids])   # (k_so_far, n)
        max_sim = sims.max(axis=0)                          # (n,) best similarity
        labels[max_sim.argmin()] = len(centroids)
        centroids.append(X_norm[max_sim.argmin()])

    # One assignment pass
    sim_matrix = np.stack([X_norm @ c for c in centroids], axis=1)  # (n, k)
    labels     = sim_matrix.argmax(axis=1)
    return labels
