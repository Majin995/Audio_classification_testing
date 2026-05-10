"""
HydroStudentFusion — lightweight 3-class fusion student for knowledge distillation
====================================================================================

A compact variant of HydroFusion intended as the student model in multi-teacher
knowledge distillation.  Ships with 3-class defaults (Passenger merged into Cargo)
and reduced capacity compared to the full HydroFusion teacher.

Default presets vs. full HydroFusion
-------------------------------------
  Hyperparameter    Student default    Full HydroFusion
  ─────────────────────────────────────────────────────
  num_classes       3                  4
  sample_rate       5 120 Hz           32 000 Hz
  n_mels            64                 128
  hop_length        51                 320
  d_model           64                 128
  dilation_rates    [2, 4]             [2, 4, 8]
  n_s4              1                  2
  s4_d_state        32                 64
  n_mamba           1                  2
  use_xlsr          False (fixed)      False
  cssd_alpha        1.0  (fixed)       1.0

Approximate parameter count with defaults: ~320 K  (vs. ~1.3 M for full HydroFusion)

Classes
-------
  Index 0 — Cargo   (Passenger training samples are merged into this class)
  Index 1 — Tanker
  Index 2 — Tug

Usage
-----
  from models.hydro_student_fusion import HydroStudentFusion

  model = HydroStudentFusion(num_classes=3, class_weights=weights)
  # All HydroFusion hyperparameters are accepted via **kwargs
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .hydro_fusion import HydroFusion


class HydroStudentFusion(HydroFusion):
    """
    Compact 3-class HydroFusion student for multi-teacher distillation.

    XLSR stream and CSSD are permanently disabled (``use_xlsr=False``,
    ``cssd_alpha=1.0``); any values supplied via ``**kwargs`` for these
    parameters are silently overridden to keep the student architecture clean.

    All other HydroFusion hyperparameters are forwarded unchanged, allowing
    the training script to override individual settings (e.g. a larger
    ``d_model`` for an ablation study) while keeping sensible student defaults.

    Parameters
    ----------
    num_classes    : Number of output classes. Default 3 (Cargo, Tanker, Tug).
    sample_rate    : Native sample rate. Default 5 120 Hz (Nyquist-optimal for
                     vessel acoustics, max useful energy ≤ 2 560 Hz).
    n_mels         : Mel filterbank bins. Default 64.
    hop_length     : STFT hop in samples. Default 51 (~100 frames/s at 5 120 Hz).
    d_model        : Feature channel width. Default 64.
    dilation_rates : Dilation schedule for SE-Res2 blocks. Default [2, 4].
    n_s4           : Number of S4D blocks. Default 1.
    s4_d_state     : S4D state size per channel. Default 32.
    n_mamba        : Number of BidirMamba blocks. Default 1.
    mamba_d_state  : Mamba SSM state size. Default 16.
    **kwargs       : Forwarded to HydroFusion (learning_rate, dropout, etc.).
    """

    def __init__(
        self,
        num_classes:    int                = 3,
        class_weights:  Optional[torch.Tensor] = None,
        sample_rate:    int                = 5_120,
        n_mels:         int                = 64,
        hop_length:     int                = 51,
        d_model:        int                = 64,
        dilation_rates: Optional[List[int]] = None,
        n_s4:           int                = 1,
        s4_d_state:     int                = 32,
        n_mamba:        int                = 1,
        mamba_d_state:  int                = 16,
        **kwargs,
    ) -> None:
        if dilation_rates is None:
            dilation_rates = [2, 4]

        # Enforce student constraints regardless of what kwargs say
        kwargs.pop("use_xlsr",    None)
        kwargs.pop("cssd_alpha",  None)

        super().__init__(
            num_classes    = num_classes,
            class_weights  = class_weights,
            sample_rate    = sample_rate,
            n_mels         = n_mels,
            hop_length     = hop_length,
            d_model        = d_model,
            dilation_rates = dilation_rates,
            n_s4           = n_s4,
            s4_d_state     = s4_d_state,
            n_mamba        = n_mamba,
            mamba_d_state  = mamba_d_state,
            use_xlsr       = False,   # always off for student
            cssd_alpha     = 1.0,     # CSSD off — distilled externally
            **kwargs,
        )
