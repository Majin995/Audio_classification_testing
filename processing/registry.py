"""
Lightweight model registry for the UATR suite.

Usage
-----
    from processing.registry import build_model, MODEL_REGISTRY

    model = build_model("catfish", num_classes=4, class_weights=[...])

Notes
-----
- Only new models (and a subset of existing ones) are registered here.
- Existing train_*.py scripts import models directly — no breaking change.
- The registry is consumed by the grid-search driver and smoke tests.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl


def _lazy_registry() -> dict[str, type[pl.LightningModule]]:
    """
    Deferred import to avoid circular imports at module load time.
    Called once on first access.
    """
    from models.hydro_catfish     import HydroCATFISH
    from models.hydro_alsi        import HydroALSI
    from models.hydro_dcn         import HydroDCN
    from models.hydro_bahtnet     import HydroBAHTNet
    from models.hydro_sscp_mobile import HydroSSCPMobile
    from models.hydro_dart_mt     import HydroDARTMT
    from models.hydro_conformer   import HydroConformer
    from models.hydro_resnet      import HydroResNet
    from models.hydro_i2hofi      import I2HOFI
    from models.hydro_precise         import HydroPrecise
    from models.hydro_hydra           import HydroHydra
    from models.hydro_wave1d          import HydroWave1D
    from models.hydro_wave_scattering import HydroWaveScattering
    from models.hydro_cnnlstm_qc       import HydroCNNLSTMQC
    from models.hydro_quantum_transfer import HydroQuantumTransfer
    from models.hydro_vqc_features     import HydroVQCFeatures

    return {
        # ── New UATR paradigms ──────────────────────────────────────────
        "catfish":         HydroCATFISH,
        "alsi":            HydroALSI,
        "dcn":             HydroDCN,
        "bahtnet":         HydroBAHTNet,
        "sscp_mobile":     HydroSSCPMobile,
        "precise":         HydroPrecise,
        "hydra":           HydroHydra,
        "wave1d":          HydroWave1D,
        "wave_scattering": HydroWaveScattering,
        # ── Existing models (for unified sweep access) ──────────────────
        "dart_mt":     HydroDARTMT,
        "conformer":   HydroConformer,
        "resnet":      HydroResNet,
        "i2hofi":      I2HOFI,
        # ── Quantum / Hybrid models ─────────────────────────────────────
        "cnnlstm_qc":       HydroCNNLSTMQC,
        "quantum_transfer": HydroQuantumTransfer,
        "vqc_features":     HydroVQCFeatures,
    }


_registry_cache: dict[str, type[pl.LightningModule]] | None = None


def _get_registry() -> dict[str, type[pl.LightningModule]]:
    global _registry_cache
    if _registry_cache is None:
        _registry_cache = _lazy_registry()
    return _registry_cache


# Public alias (evaluates lazily on attribute access via __getattr__ trick)
class _RegistryProxy(dict):
    """A dict-like proxy that populates lazily on first access."""
    def __missing__(self, key):
        reg = _get_registry()
        if key in reg:
            self.update(reg)
            return reg[key]
        raise KeyError(key)


MODEL_REGISTRY: dict[str, type[pl.LightningModule]] = _RegistryProxy()


def build_model(name: str, **kwargs: Any) -> pl.LightningModule:
    """
    Instantiate a registered model by name.

    Args:
        name   : Registry key (e.g. ``'catfish'``, ``'bahtnet'``).
        **kwargs: Passed to the model's ``__init__``.

    Returns:
        Configured ``pl.LightningModule`` instance.

    Raises:
        KeyError: If ``name`` is not in ``MODEL_REGISTRY``.
    """
    reg = _get_registry()
    if name not in reg:
        available = ", ".join(sorted(reg))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return reg[name](**kwargs)


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(_get_registry())
