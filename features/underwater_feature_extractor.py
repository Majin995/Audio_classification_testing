"""
UnderwaterFeatureExtractor — Comprehensive Acoustic Feature Pipeline
=====================================================================

Extracts an exhaustive suite of time-domain, frequency-domain, and
spectro-temporal representations from a raw mono waveform. Designed
for passive sonar classification of surface vessels (Cargo, Passenger,
Tanker, Tug) in the LOFAR acoustic band (0–2560 Hz at 5120 Hz SR).

Output dictionary schema
------------------------
  scalars     (6,)         ZCR, RMS, Crest Factor, Kurtosis, Skewness, Higuchi FD
  psd         (129,)       Welch power spectral density
  spectral    (3,)         Centroid, Roll-off, Flatness (frame-averaged scalars)
  stft_lofar  (H, W)       Log-power LOFAR-style STFT
  mel         (H, W)       Log-mel spectrogram
  gammatone   (H, W)       ERB-spaced Gaussian filterbank spectrogram
  demon       (H, W)       DEMON envelope modulation spectrogram
  cqt         (H, W)       Constant-Q transform spectrogram
  bispectrum  (H, W)       Bispectrum magnitude (quadratic phase coupling)
  scalogram   (H, W)       Complex Morlet CWT scalogram
  wvd         (H, W)       Pseudo Wigner-Ville distribution
  hht         (H, W)       Hilbert-Huang marginal energy spectrum

All 2-D representations are independently instance-normalised and
anti-alias-resized to ``(target_h, target_w)``.

Dependencies
------------
  numpy, scipy, librosa, pywt (PyWavelets), PyEMD — install all before use.
  PyEMD is optional; if unavailable the HHT channel is filled with zeros
  and a one-time warning is emitted.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Tuple

import numpy as np
import scipy.signal
import scipy.stats
import scipy.ndimage
import librosa
import pywt
import torch

try:
    from PyEMD import EMD as _EMD
    _PYEMD_AVAILABLE = True
except ImportError:
    _PYEMD_AVAILABLE = False
    warnings.warn(
        "PyEMD not installed — HHT gram will be zero-filled. "
        "Install with:  pip install PyEMD",
        ImportWarning,
        stacklevel=1,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Module-level constants
# ═══════════════════════════════════════════════════════════════════════

#: 2D spectrogram channel ordering (matches ``AcousticOmniResNet`` channel 0–8)
GRAM_KEYS: Tuple[str, ...] = (
    "stft_lofar", "mel", "gammatone", "demon",
    "cqt", "bispectrum", "scalogram", "wvd", "hht",
)

#: 1D feature keys concatenated → scalar branch input
SCALAR_KEYS: Tuple[str, ...] = ("scalars", "psd", "spectral")

#: Concatenated 1D feature dimension: 6 + 129 + 3 = 138
INPUT_DIM_1D: int = 138


# ═══════════════════════════════════════════════════════════════════════
#  Main extractor class
# ═══════════════════════════════════════════════════════════════════════

class UnderwaterFeatureExtractor:
    r"""
    Exhaustive underwater acoustic feature extractor.

    Operates entirely on CPU using numpy/scipy/librosa — designed to
    run once per clip during dataset pre-computation and cache results
    to disk, not to be called at training time.

    Parameters
    ----------
    sample_rate : int
        Native sample rate after DALI resampling (default 5 120 Hz).
    n_fft : int
        FFT size for STFT / LOFAR / Gammatone frontends.
        At 5 120 Hz, ``n_fft=4096`` gives ~0.8 s windows with
        frequency resolution :math:`\Delta f = f_s / N = 1.25` Hz/bin.
    hop_length : int
        STFT hop in samples (~31 ms at 5 120 Hz).
    n_mels : int
        Mel filterbank bands for the mel and DEMON channels.
    n_gammatone : int
        ERB-spaced Gammatone filterbank bands.
    n_cqt_bins : int
        CQT bins spanning 7 octaves from ``f_min=20`` Hz to Nyquist.
    target_h : int
        Output height (frequency axis) for all 2-D grams.
    target_w : int
        Output width (time axis) for all 2-D grams.
    k_max : int
        Maximum lag for Higuchi fractal dimension estimation.
    welch_nperseg : int
        Welch PSD segment length; determines PSD vector length
        :math:`(N_{\rm seg}/2 + 1)`.
    wvd_window : int
        Lag-domain Hann window length for pseudo-WVD cross-term
        suppression. Must be even.
    wvd_n_time : int
        Number of sub-sampled time frames for WVD computation.
    hht_max_imf : int
        Maximum number of IMFs to extract for the HHT gram.
    bispectrum_n_freq : int
        Number of sub-sampled FFT bins for bispectrum computation.
        Full bispectrum would be :math:`O(N^2)` — subsampling keeps it
        tractable while retaining the quadratic coupling structure.
    """

    def __init__(
        self,
        sample_rate:        int   = 5_120,
        n_fft:              int   = 4_096,
        hop_length:         int   = 160,
        n_mels:             int   = 64,
        n_gammatone:        int   = 64,
        n_cqt_bins:         int   = 84,
        target_h:           int   = 64,
        target_w:           int   = 128,
        k_max:              int   = 10,
        welch_nperseg:      int   = 256,
        wvd_window:         int   = 64,
        wvd_n_time:         int   = 128,
        hht_max_imf:        int   = 8,
        bispectrum_n_freq:  int   = 128,
    ):
        self.sample_rate       = sample_rate
        self.n_fft             = n_fft
        self.hop_length        = hop_length
        self.n_mels            = n_mels
        self.n_gammatone       = n_gammatone
        self.n_cqt_bins        = n_cqt_bins
        self.target_h          = target_h
        self.target_w          = target_w
        self.k_max             = k_max
        self.welch_nperseg     = welch_nperseg
        self.wvd_window        = wvd_window
        self.wvd_n_time        = wvd_n_time
        self.hht_max_imf       = hht_max_imf
        self.bispectrum_n_freq = bispectrum_n_freq

        # Pre-build the Gammatone filterbank matrix (reused per clip)
        self._gammatone_fb = self._build_gammatone_filterbank()

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def extract(self, x: np.ndarray) -> Dict[str, torch.Tensor]:
        r"""
        Extract all features from a raw mono waveform.

        Parameters
        ----------
        x : np.ndarray, shape (N,)
            Float32 or float64 mono waveform at ``self.sample_rate`` Hz.

        Returns
        -------
        dict[str, torch.Tensor]
            All feature tensors as float32 CPU tensors.
        """
        x = self._validate_signal(x)

        # ── Time-domain scalars ──────────────────────────────────────
        scalars = np.array([
            self._zcr(x),
            self._rms(x),
            self._crest_factor(x),
            self._kurtosis(x),
            self._skewness(x),
            self._higuchi_fd(x),
        ], dtype=np.float32)

        # ── Frequency-domain ─────────────────────────────────────────
        psd      = self._welch_psd(x)
        spectral = self._spectral_features(x)

        # ── 2-D spectro-temporal representations ─────────────────────
        grams = {
            "stft_lofar": self._stft_lofar(x),
            "mel":        self._mel(x),
            "gammatone":  self._gammatone(x),
            "demon":      self._demon(x),
            "cqt":        self._cqt(x),
            "bispectrum": self._bispectrum(x),
            "scalogram":  self._scalogram(x),
            "wvd":        self._wvd(x),
            "hht":        self._hht(x),
        }

        result: Dict[str, torch.Tensor] = {
            "scalars":  torch.from_numpy(scalars),
            "psd":      torch.from_numpy(psd),
            "spectral": torch.from_numpy(spectral),
        }
        for key, gram in grams.items():
            result[key] = torch.from_numpy(gram)

        return result

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"UnderwaterFeatureExtractor("
            f"sr={self.sample_rate}, n_fft={self.n_fft}, "
            f"hop={self.hop_length}, n_mels={self.n_mels}, "
            f"n_gammatone={self.n_gammatone}, n_cqt={self.n_cqt_bins}, "
            f"out=({self.target_h},{self.target_w}))"
        )

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _validate_signal(self, x: np.ndarray) -> np.ndarray:
        """Cast to float32, ensure 1-D, pad or trim to exactly sample_rate samples."""
        x = np.asarray(x, dtype=np.float32).ravel()
        target = self.sample_rate           # 1 second
        if len(x) < target:
            x = np.pad(x, (0, target - len(x)))
        elif len(x) > target:
            x = x[:target]
        return x

    def _resize_gram(self, arr: np.ndarray) -> np.ndarray:
        r"""
        Instance-normalise and anti-alias resize to ``(target_h, target_w)``.

        Instance normalisation:
        :math:`\hat{a} = (a - \mu_a) / (\sigma_a + \epsilon)`

        where :math:`\mu_a` and :math:`\sigma_a` are the sample mean and
        standard deviation of the 2-D array.  This equalises the dynamic
        range across all channels regardless of their physical units.

        For downsampling ratios > 4:1, a box-average is applied first to
        suppress aliasing before bilinear interpolation.
        """
        arr = arr.astype(np.float32)

        # Anti-aliasing pre-average along width (time axis)
        h, w = arr.shape
        if w > 4 * self.target_w:
            factor = w // self.target_w
            n_trim = (w // factor) * factor
            arr = arr[:, :n_trim].reshape(h, n_trim // factor, factor).mean(axis=-1)
            w = arr.shape[1]

        # Anti-aliasing pre-average along height (freq axis)
        if h > 4 * self.target_h:
            factor = h // self.target_h
            n_trim = (h // factor) * factor
            arr = arr[:n_trim, :].reshape(n_trim // factor, factor, w).mean(axis=1)
            h = arr.shape[0]

        # Instance normalise
        mu, sigma = arr.mean(), arr.std()
        arr = (arr - mu) / (sigma + 1e-8)

        # Bilinear resize
        if arr.shape != (self.target_h, self.target_w):
            zh = self.target_h / arr.shape[0]
            zw = self.target_w / arr.shape[1]
            arr = scipy.ndimage.zoom(arr, (zh, zw), order=1)

        return arr.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────
    # Time-domain scalar features
    # ──────────────────────────────────────────────────────────────────

    def _zcr(self, x: np.ndarray) -> float:
        r"""
        Zero Crossing Rate.

        :math:`\text{ZCR} = \frac{1}{2N} \sum_{n=1}^{N-1} |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|`

        ZCR correlates inversely with the signal's dominant frequency and is
        elevated for broadband cavitation noise (Tug) and low for tonal
        machinery noise (Cargo/Tanker).
        """
        return float(np.mean(np.abs(np.diff(np.sign(x)))) / 2.0)

    def _rms(self, x: np.ndarray) -> float:
        r"""
        RMS Energy.

        :math:`E_{\rm RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n]^2}`

        Proportional to the square root of signal power.  Higher for vessels
        closer to the hydrophone and for high-cavitation vessels (Tug).
        """
        return float(np.sqrt(np.mean(x ** 2)))

    def _crest_factor(self, x: np.ndarray) -> float:
        r"""
        Crest Factor (peak-to-RMS ratio).

        :math:`CF = \frac{\max_n |x[n]|}{E_{\rm RMS} + \varepsilon}`

        High CF indicates sparse impulsive events (cavitation pops).
        Sinusoidal signals have :math:`CF = \sqrt{2} \approx 1.41`; white
        noise has :math:`CF \approx 3`.  Cavitation-heavy vessels typically
        exhibit :math:`CF > 4`.
        """
        rms = float(np.sqrt(np.mean(x ** 2)))
        return float(np.max(np.abs(x))) / (rms + 1e-12)

    def _kurtosis(self, x: np.ndarray) -> float:
        r"""
        Excess Kurtosis (Fisher definition).

        :math:`\kappa = \frac{\mu_4}{\sigma^4} - 3`

        where :math:`\mu_4` is the 4th central moment.  Gaussian signals
        have :math:`\kappa = 0`.  Impulsive (tonal + transient) ship noise
        typically has :math:`\kappa > 3`, while broadband Gaussian-like
        cavitation noise has :math:`\kappa \approx 0`.
        """
        return float(scipy.stats.kurtosis(x, fisher=True))

    def _skewness(self, x: np.ndarray) -> float:
        r"""
        Skewness (third standardised moment).

        :math:`\gamma_1 = \frac{\mu_3}{\sigma^3}`

        Non-zero skewness in hydrophone recordings indicates nonlinear
        acoustic propagation effects, which differ across vessel types due
        to different power levels and distances.
        """
        return float(scipy.stats.skew(x))

    def _higuchi_fd(self, x: np.ndarray) -> float:
        r"""
        Higuchi Fractal Dimension.

        For each time scale :math:`k = 1, \ldots, k_{\max}`, the curve
        length is estimated as:

        :math:`L(k) = \frac{1}{k} \sum_{m=1}^{k} \left[
            \frac{N-1}{\lfloor(N-m)/k\rfloor \cdot k^2}
            \sum_{i=1}^{\lfloor(N-m)/k\rfloor}
            |x[m + ik] - x[m + (i-1)k]|
        \right]`

        The fractal dimension :math:`D` is the slope of the log-log regression:

        :math:`\log L(k) \approx D \cdot \log(1/k) + \text{const}`

        High :math:`D` (approaching 2) indicates broadband noise (Tug
        cavitation); low :math:`D` (approaching 1) indicates tonal/harmonic
        structure (Cargo machinery lines).
        """
        N = len(x)
        L_arr, k_arr = [], []
        for k in range(1, self.k_max + 1):
            Lm_sum = 0.0
            for m in range(1, k + 1):
                n_seg = (N - m) // k
                if n_seg < 1:
                    continue
                idx = m - 1 + np.arange(n_seg + 1) * k
                idx = idx[idx < N]
                Lm_sum += np.sum(np.abs(np.diff(x[idx]))) * (N - 1) / (n_seg * k ** 2)
            if Lm_sum > 0:
                L_arr.append(Lm_sum / k)
                k_arr.append(k)

        if len(k_arr) < 2:
            return 1.0

        log_k_inv = np.log(1.0 / np.array(k_arr, dtype=np.float64))
        log_L     = np.log(np.array(L_arr,   dtype=np.float64))
        slope, _  = np.polyfit(log_k_inv, log_L, 1)
        return float(np.clip(slope, 1.0, 2.0))

    # ──────────────────────────────────────────────────────────────────
    # Frequency-domain vector features
    # ──────────────────────────────────────────────────────────────────

    def _welch_psd(self, x: np.ndarray) -> np.ndarray:
        r"""
        Welch Power Spectral Density estimate.

        :math:`\hat{S}(f) = \frac{1}{K} \sum_{k=0}^{K-1}
        \left| \sum_{n=0}^{M-1} x_k[n] \, w[n] \, e^{-j2\pi fn/M} \right|^2`

        where :math:`x_k[n]` is the :math:`k`-th overlapping segment of
        length :math:`M` and :math:`w[n]` is the Hann window.

        Returns
        -------
        np.ndarray, shape (nperseg//2 + 1,)
            Log-compressed, normalised PSD vector.
        """
        _, psd = scipy.signal.welch(
            x,
            fs=self.sample_rate,
            nperseg=self.welch_nperseg,
            noverlap=self.welch_nperseg // 2,
            window="hann",
            scaling="density",
        )
        psd = np.log1p(psd).astype(np.float32)
        # Normalise to zero mean unit std
        mu, sigma = psd.mean(), psd.std()
        return ((psd - mu) / (sigma + 1e-8)).astype(np.float32)

    def _spectral_features(self, x: np.ndarray) -> np.ndarray:
        r"""
        Compact spectral shape descriptors (frame-averaged).

        **Spectral Centroid**
        :math:`SC = \frac{\sum_k f_k \cdot S(k)}{\sum_k S(k)}`

        Indicates the "centre of mass" of the spectrum.  Low for tonal
        machinery noise (energy concentrated in low harmonics) and high
        for broadband cavitation.

        **Spectral Roll-off** (:math:`p = 0.85`)
        The frequency :math:`f_{\rm ro}` below which 85% of total spectral
        energy is contained:
        :math:`\sum_{k: f_k \le f_{\rm ro}} S(k) = 0.85 \sum_k S(k)`

        **Spectral Flatness** (Wiener entropy)
        :math:`F = \frac{\exp\bigl(\frac{1}{K}\sum_k \log S(k)\bigr)}
        {\frac{1}{K}\sum_k S(k)}`

        Ranges in :math:`[0, 1]`.  :math:`F \to 1` for white noise;
        :math:`F \to 0` for pure tones.

        Returns
        -------
        np.ndarray, shape (3,)
            [centroid_Hz, rolloff_Hz, flatness], all normalised to [0, 1].
        """
        sc   = float(np.mean(librosa.feature.spectral_centroid(
            y=x, sr=self.sample_rate)))
        sro  = float(np.mean(librosa.feature.spectral_rolloff(
            y=x, sr=self.sample_rate, roll_percent=0.85)))
        sfl  = float(np.mean(librosa.feature.spectral_flatness(y=x)))

        # Normalise centroid and rolloff by Nyquist; flatness is in [0,1]
        nyquist = self.sample_rate / 2.0
        return np.array([sc / nyquist, sro / nyquist, sfl], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────
    # 2-D spectro-temporal representations
    # ──────────────────────────────────────────────────────────────────

    def _stft_lofar(self, x: np.ndarray) -> np.ndarray:
        r"""
        LOFAR-style log-power STFT spectrogram.

        Uses a large analysis window (:math:`N_{\rm FFT} = 4096` samples
        :math:`\approx 800` ms at 5 120 Hz) to achieve fine frequency
        resolution :math:`\Delta f = 1.25` Hz/bin, sufficient to resolve
        individual propeller shaft harmonics separated by :math:`<5` Hz.

        Processing pipeline:

        1. Power spectrogram:
           :math:`S(k, t) = |X(k, t)|^2` where :math:`X = \text{STFT}(x)`
        2. Bandlimit to :math:`f \le f_{\rm max} = 2560` Hz
        3. Log compression:
           :math:`\hat{S} = \log(S + \varepsilon)`
        4. Per-frame instance normalisation
        """
        D = librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                         window="hann", center=True)
        S = np.abs(D) ** 2                                # power

        # Bandlimit to Nyquist
        freqs  = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        n_keep = int(np.searchsorted(freqs, self.sample_rate / 2.0)) + 1
        S      = S[:n_keep, :]

        S = np.log(S + 1e-9)
        return self._resize_gram(S)

    def _mel(self, x: np.ndarray) -> np.ndarray:
        r"""
        Log-Mel spectrogram.

        The Mel filterbank applies a set of triangular filters at Mel-spaced
        centre frequencies :math:`m = 2595 \log_{10}(1 + f / 700)`, giving
        finer resolution in the perceptually important low-frequency range
        where ship harmonics reside.

        :math:`M(\ell, t) = \sum_k H_\ell(k) \cdot S(k, t)`

        where :math:`H_\ell(k)` is the :math:`\ell`-th triangular Mel filter
        and :math:`S(k, t)` is the STFT power spectrogram.
        """
        S = librosa.feature.melspectrogram(
            y=x, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=20.0, fmax=self.sample_rate / 2.0,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        return self._resize_gram(S_db)

    def _build_gammatone_filterbank(self) -> np.ndarray:
        r"""
        Build the ERB-spaced Gaussian Gammatone filterbank matrix.

        Equivalent Rectangular Bandwidth (ERB) of a human auditory filter:

        :math:`\text{ERB}(f) = 24.7 \,(4.37\,f/1000 + 1)` [Hz]

        ERB-rate scale (Glasberg & Moore, 1990):

        :math:`\text{ERBS}(f) = 21.4 \log_{10}(4.37\,f/1000 + 1)`

        Centre frequencies :math:`f_c` are linearly spaced on the ERB-rate
        scale from :math:`f_{\min}` to Nyquist, then converted back to Hz.

        Each filter is a Gaussian with :math:`\sigma = \text{ERB}(f_c) / 2`:

        :math:`H_i(f) = \exp\!\left(-\!\left(\frac{f - f_{c,i}}{\text{ERB}(f_{c,i})/2}\right)^2\right)`

        Returns
        -------
        np.ndarray, shape (n_gammatone, n_freqs)
            Row-normalised filterbank matrix (rows sum to 1).
        """
        f_min   = 20.0
        f_max   = self.sample_rate / 2.0
        n_freqs = self.n_fft // 2 + 1
        freqs   = np.linspace(0.0, f_max, n_freqs)

        erbs_min = 21.4 * np.log10(max(4.37 * f_min / 1000.0 + 1.0, 1e-9))
        erbs_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
        erbs     = np.linspace(erbs_min, erbs_max, self.n_gammatone)
        fc       = (10.0 ** (erbs / 21.4) - 1.0) * 1000.0 / 4.37  # [Hz]

        erb_bw = 24.7 * (4.37 * fc / 1000.0 + 1.0)      # (n_bands,)
        # Gaussian weights: (n_bands, n_freqs)
        weights = np.exp(-((freqs[None, :] - fc[:, None]) / (erb_bw[:, None] / 2.0)) ** 2)
        norm    = weights.sum(axis=1, keepdims=True).clip(min=1e-9)
        return (weights / norm).astype(np.float32)        # (n_gammatone, n_freqs)

    def _gammatone(self, x: np.ndarray) -> np.ndarray:
        r"""
        ERB-spaced Gammatone filterbank spectrogram.

        Applies the pre-built filterbank matrix :math:`\mathbf{H} \in
        \mathbb{R}^{B \times K}` (ERB bands × FFT bins) to the STFT
        magnitude spectrogram :math:`\mathbf{S} \in \mathbb{R}^{K \times T}`:

        :math:`\mathbf{G} = \mathbf{H} \cdot \mathbf{S} \in \mathbb{R}^{B \times T}`

        ERB spacing provides finer resolution than Mel below ~500 Hz, better
        resolving the closely-spaced shaft and blade-rate harmonics that
        distinguish vessel classes.
        """
        D = librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                         window="hann", center=True)
        mag = np.abs(D)                                   # (n_freqs, T)

        # The filterbank has n_fft//2+1 columns; mag has n_fft//2+1 rows
        n_freqs = self.n_fft // 2 + 1
        mag_trim = mag[:n_freqs, :]                       # guard against edge case

        G = self._gammatone_fb @ mag_trim                 # (n_gammatone, T)
        return self._resize_gram(np.log1p(G))

    def _demon(self, x: np.ndarray) -> np.ndarray:
        r"""
        DEMON (Detection of Envelope Modulation ON Noise) spectrogram.

        Ships' propellers produce cavitation noise amplitude-modulated at
        the blade-passage frequency:

        :math:`f_{\rm BPF} = \frac{N_{\rm shafts} \times Z}{60}` [Hz]

        where :math:`Z` is the number of blades.  DEMON isolates this
        modulation:

        1. **Bandpass** to the cavitation band :math:`[f_{\rm cav}, f_s/2]`
           via FFT zero-masking:
           :math:`\tilde{X}(k) = X(k) \cdot \mathbf{1}[f_k \ge f_{\rm cav}]`

        2. **Envelope detection** via squaring (self-demodulation):
           :math:`e[n] = \tilde{x}[n]^2`

        3. **Mel spectrogram** of :math:`e[n]`:
           Low Mel bands capture slow blade-rate modulations; higher bands
           capture mechanical harmonics.
        """
        f_cav = 800.0
        X     = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=1.0 / self.sample_rate)
        X[freqs < f_cav] = 0.0
        x_bp  = np.fft.irfft(X, n=len(x)).astype(np.float32)

        x_env = x_bp ** 2
        S = librosa.feature.melspectrogram(
            y=x_env, sr=self.sample_rate,
            n_fft=1024, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=0.0, fmax=self.sample_rate / 2.0,
            power=1.0,
        )
        return self._resize_gram(np.log1p(S))

    def _cqt(self, x: np.ndarray) -> np.ndarray:
        r"""
        Constant-Q Transform (CQT) spectrogram.

        Unlike the STFT (linear frequency spacing), the CQT uses a
        geometrically-spaced frequency axis, so each octave occupies
        the same number of bins :math:`B_{\rm oct}`:

        :math:`f_k = f_{\min} \cdot 2^{k / B_{\rm oct}}, \quad k = 0, \ldots, K-1`

        The Q-factor (frequency / bandwidth) is constant across all bins:

        :math:`Q = \frac{1}{2^{1/B_{\rm oct}} - 1} \approx \frac{B_{\rm oct}}{\ln 2}`

        This is ideal for analysing harmonic series with fixed frequency
        ratios (propeller harmonics, engine overtones), where all harmonics
        of a fundamental are separated by one octave / :math:`B_{\rm oct}` bins.

        Coverage: :math:`f_{\min}=20` Hz to Nyquist (7 octaves at
        :math:`B_{\rm oct}=12` bins/octave).
        """
        C = librosa.cqt(
            x,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.n_cqt_bins,
            bins_per_octave=12,
            fmin=20.0,
            window="hann",
        )
        S = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        return self._resize_gram(S)

    def _bispectrum(self, x: np.ndarray) -> np.ndarray:
        r"""
        Bispectrum magnitude.

        The bispectrum detects **quadratic phase coupling (QPC)** — a
        third-order statistics measure of nonlinear interactions between
        frequency components.  For a stationary process :math:`X(f)`:

        :math:`B(f_1, f_2) = \mathbb{E}[X(f_1)\,X(f_2)\,X^*(f_1 + f_2)]`

        Non-zero bispectrum at :math:`(f_1, f_2)` indicates that energy
        at frequency :math:`f_1 + f_2` is generated by a nonlinear
        interaction of components at :math:`f_1` and :math:`f_2`.  This
        is characteristic of propeller cavitation (blade slap) and is
        largely absent in hull flow noise.

        Single-realisation estimate (for one 1-second clip):
        :math:`\hat{B}(f_1, f_2) = X(f_1)\,X(f_2)\,X^*(f_1 + f_2)`

        Subsampled to :math:`N_b = 128` frequency bins for tractability
        (:math:`O(N_b^2) = 16\,384` evaluations vs. :math:`O(N^2)` for
        the full spectrum).

        Principal domain: :math:`f_1 \ge 0,\; f_2 \ge 0,\; f_1 + f_2 < f_s/2`.
        """
        X      = np.fft.rfft(x)           # (N//2+1,) complex
        N_full = len(X)
        nb     = self.bispectrum_n_freq

        # Subsampled frequency indices
        stride = max(1, N_full // nb)
        idx    = np.arange(0, N_full, stride)[:nb]    # (nb,)
        X_sub  = X[idx]                                # (nb,) complex

        # Vectorised bispectrum evaluation
        f12_mat  = idx[:, None] + idx[None, :]         # (nb, nb) original indices
        valid    = f12_mat < N_full

        f12_clip = np.clip(f12_mat, 0, N_full - 1)
        B_cmplx  = X_sub[:, None] * X_sub[None, :] * np.conj(X[f12_clip])
        B_cmplx[~valid] = 0.0

        B = np.abs(B_cmplx).astype(np.float32)
        return self._resize_gram(np.log1p(B))

    def _scalogram(self, x: np.ndarray) -> np.ndarray:
        r"""
        Continuous Wavelet Transform (CWT) scalogram.

        Uses the complex Morlet wavelet :math:`\psi_{b,c}`:

        :math:`\psi_{b,c}(t) = \frac{1}{\sqrt{b}}\,\psi\!\left(\frac{t-c}{b}\right),
        \quad \psi(t) = \pi^{-1/4} e^{j\omega_0 t} e^{-t^2/2}`

        where :math:`\omega_0` is the centre frequency and :math:`b` is the
        scale (inversely proportional to centre frequency).

        The scalogram :math:`|W_x(b, \tau)|^2` provides simultaneous
        time-frequency localisation with scale-adaptive resolution:
        high temporal resolution at small scales (high frequencies) and
        high frequency resolution at large scales (low frequencies).

        Scales chosen to cover :math:`[20, 2560]` Hz:
        :math:`b \in \{f_s / f_{\max}, \ldots, f_s / f_{\min}\}`
        geometrically spaced with 64 points.

        Wavelet: ``'cmor1.5-1.0'`` (bandwidth=1.5, centre freq=1 Hz).
        """
        sr = self.sample_rate
        # Scales that map to physical frequencies f = center_freq * sr / scale
        # cmor1.5-1.0 has center_frequency ≈ 1.0 Hz in the PyWavelets convention
        # f = sr / scale  →  scale = sr / f
        f_min_hz = 20.0
        f_max_hz = sr / 2.0
        scales   = np.geomspace(sr / f_max_hz, sr / f_min_hz, self.target_h)

        coeffs, _ = pywt.cwt(x, scales, "cmor1.5-1.0",
                              sampling_period=1.0 / sr)
        # coeffs: (n_scales, N)
        S = np.abs(coeffs).astype(np.float32)
        return self._resize_gram(np.log1p(S))

    def _wvd(self, x: np.ndarray) -> np.ndarray:
        r"""
        Pseudo Wigner-Ville Distribution (PWVD) with Hann window smoothing.

        The Wigner-Ville Distribution offers theoretically optimal joint
        time-frequency resolution but suffers from cross-term interference
        between multi-component signals:

        :math:`W_z(t, f) = \int_{-\infty}^{\infty}
        z(t + \tau/2)\,z^*(t - \tau/2)\,e^{-j2\pi f\tau}\,d\tau`

        The **Pseudo-WVD** suppresses cross-terms by applying a smoothing
        window :math:`h(\tau)` in the lag domain:

        :math:`PW_z(t, f) = \int_{-\infty}^{\infty}
        h(\tau)\,z(t+\tau)\,z^*(t-\tau)\,e^{-j4\pi f\tau}\,d\tau`

        Implementation:

        1. Analytic signal :math:`z[n] = x[n] + j\,\mathcal{H}\{x\}[n]`
           via scipy's Hilbert transform.
        2. For each sub-sampled time index :math:`t_i`, compute the
           Hann-windowed lag-domain auto-correlation kernel:
           :math:`\phi_i[\tau] = h[\tau] \cdot z[t_i + \tau] \cdot z^*[t_i - \tau]`
        3. DFT :math:`\phi_i \to W_i`: positive-frequency half is the
           PWVD slice at time :math:`t_i`.

        The full kernel matrix :math:`\boldsymbol{\Phi} \in
        \mathbb{C}^{T_{\rm sub} \times L}` is batch-FFT'd in one call.
        """
        z  = scipy.signal.hilbert(x).astype(np.complex64)
        N  = len(z)
        L  = self.wvd_window          # lag window length (even)
        hl = L // 2
        hann = np.hanning(L).astype(np.float32)  # (L,)

        # Sub-sampled time indices
        step      = max(1, N // self.wvd_n_time)
        t_indices = np.arange(0, N, step)[: self.wvd_n_time]   # (n_t,)
        n_t       = len(t_indices)

        tau      = np.arange(-hl, hl, dtype=np.int32)           # (L,)
        t_plus   = t_indices[:, None] + tau[None, :]            # (n_t, L)
        t_minus  = t_indices[:, None] - tau[None, :]            # (n_t, L)
        valid    = (
            (t_plus  >= 0) & (t_plus  < N) &
            (t_minus >= 0) & (t_minus < N)
        )
        t_plus_c  = np.clip(t_plus,  0, N - 1)
        t_minus_c = np.clip(t_minus, 0, N - 1)

        kernel = hann[None, :] * z[t_plus_c] * np.conj(z[t_minus_c])
        kernel[~valid] = 0.0                                    # (n_t, L)

        # Batch real FFT along lag axis
        wvd_full = np.fft.fft(kernel, axis=1)                  # (n_t, L)
        wvd_pos  = np.real(wvd_full[:, :hl]).T                 # (hl, n_t) = (freq, time)

        # Take absolute value (real part can be slightly negative due to windowing)
        wvd_pos  = np.abs(wvd_pos).astype(np.float32)
        return self._resize_gram(wvd_pos)

    def _hht(self, x: np.ndarray) -> np.ndarray:
        r"""
        Hilbert-Huang Transform (HHT) marginal energy spectrum.

        The HHT is a fully data-adaptive time-frequency representation
        designed for nonlinear, non-stationary signals — a natural fit for
        underwater acoustic recordings that evolve as vessels manoeuvre.

        **Empirical Mode Decomposition (EMD):**
        Decomposes :math:`x[n]` into a sum of Intrinsic Mode Functions
        (IMFs) :math:`c_i[n]` and a residue :math:`r[n]`:

        :math:`x[n] = \sum_{i=1}^{M} c_i[n] + r[n]`

        Each IMF satisfies: (1) the number of extrema and zero-crossings
        differ by at most one; (2) the mean of the upper and lower envelopes
        is zero at every point.  IMFs are ordered from high to low frequency.

        **Instantaneous frequency and amplitude** (per IMF, via Hilbert):

        :math:`c_i^+(t) = c_i(t) + j\,\mathcal{H}\{c_i\}(t)`

        :math:`A_i(t) = |c_i^+(t)|, \quad
        \varphi_i(t) = \arg c_i^+(t)`

        :math:`f_i(t) = \frac{1}{2\pi}\,\frac{d\varphi_i}{dt} \cdot f_s`

        **Marginal Hilbert spectrum:** amplitude-weighted 2-D histogram:

        :math:`H(f, t) = \sum_i A_i(t) \cdot \delta(f - f_i(t))`

        binned into a :math:`(N_f \times N_t)` grid.

        Falls back to a zero array if PyEMD is unavailable.
        """
        if not _PYEMD_AVAILABLE:
            return np.zeros((self.target_h, self.target_w), dtype=np.float32)

        sr = self.sample_rate
        N  = len(x)

        try:
            emd  = _EMD()
            emd.MAX_ITERATION = 200
            imfs = emd.emd(x.astype(np.float64), max_imf=self.hht_max_imf)
        except Exception:
            return np.zeros((self.target_h, self.target_w), dtype=np.float32)

        n_freq_bins = self.target_h
        n_time_bins = self.target_w
        freq_edges  = np.linspace(0.0, sr / 2.0, n_freq_bins + 1)
        time_edges  = np.linspace(0.0, N / sr,   n_time_bins + 1)

        hht_gram = np.zeros((n_freq_bins, n_time_bins), dtype=np.float32)
        t_arr    = np.arange(N) / sr                        # (N,)

        for imf in imfs:
            analytic  = scipy.signal.hilbert(imf)
            amplitude = np.abs(analytic)[:-1].astype(np.float32)
            phase     = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(phase) / (2.0 * np.pi) * sr  # (N-1,)
            inst_freq = np.clip(inst_freq, 0.0, sr / 2.0).astype(np.float32)
            t_mid     = ((t_arr[:-1] + t_arr[1:]) / 2.0).astype(np.float32)

            # amplitude-weighted 2D histogram: axes (time, freq)
            h, _, _ = np.histogram2d(
                t_mid, inst_freq,
                bins=[time_edges, freq_edges],
                weights=amplitude,
            )
            hht_gram += h.T.astype(np.float32)             # → (n_freq, n_time)

        return self._resize_gram(hht_gram) if hht_gram.max() > 0 else hht_gram
