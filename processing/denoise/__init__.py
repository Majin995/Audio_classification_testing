"""
processing.denoise — waveform-level denoising utilities.

Exports
-------
emd_wavelet_denoise  : EMD → IMF selection → SURE-adaptive wavelet thresholding.
nmf_ica_separate     : NMF + FastICA single-channel blind source separation.
DenoiseTransform     : nn.Module wrapper — CPU denoising on a batch of waveforms.
"""

from processing.denoise.emd_wavelet import emd_wavelet_denoise
from processing.denoise.bss import nmf_ica_separate
from processing.denoise.transform import DenoiseTransform

__all__ = ["emd_wavelet_denoise", "nmf_ica_separate", "DenoiseTransform"]
