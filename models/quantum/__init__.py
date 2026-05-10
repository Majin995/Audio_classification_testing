"""
models.quantum — shared parameterised-quantum-circuit primitives.

Used by:
  * models/hydro_cnnlstm_qc.py        (Model A)
  * models/hydro_quantum_transfer.py  (Model B)
  * models/hydro_vqc_features.py      (Model D)

Implementation backend: TorchQuantum (pure-PyTorch quantum simulator,
GPU-capable, autograd-native — no separate parameter-shift bridge required).
"""

# ── Compatibility shim ────────────────────────────────────────────────────
# torchquantum 0.1.x imports `qiskit.providers.aer.noise.device.parameters`
# and `qiskit_ibm_runtime` at module load time. Both come from older qiskit
# packaging that no longer builds on Python 3.13 (qiskit-aer ≥ 0.13 wheels
# are unavailable on 3.13, and the qiskit-aer ≥ 0.14 namespace dropped
# `qiskit.providers.aer` entirely). We only use the simulator (Pauli-Z
# measurements, RY/RZ/CNOT gates) — none of the IBM-runtime / Aer-noise
# features that the broken imports exist for. Stubbing these modules with
# minimal placeholders lets torchquantum's package init succeed without
# changing any of its pure-numerical code paths.
import sys
import types as _types

def _install_qiskit_compat_stubs() -> None:
    def _stub(name: str) -> _types.ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        mod = _types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    aer = _stub("qiskit.providers.aer")
    noise = _stub("qiskit.providers.aer.noise")
    device = _stub("qiskit.providers.aer.noise.device")
    params = _stub("qiskit.providers.aer.noise.device.parameters")
    params.gate_error_values = lambda *a, **kw: {}

    class _MissingAer:
        def __init__(self, *a, **kw):
            raise RuntimeError(
                "qiskit-aer is not installed. The TorchQuantum simulator "
                "still works for forward and backward passes; only IBM-Aer "
                "noise modelling and real-hardware execution are unavailable."
            )
    noise.NoiseModel = _MissingAer
    aer.AerSimulator   = _MissingAer
    aer.QasmSimulator  = _MissingAer
    aer.noise = noise; noise.device = device; device.parameters = params

    runtime = _stub("qiskit_ibm_runtime")
    class _MissingRuntimeService:
        def __init__(self, *a, **kw):
            raise RuntimeError(
                "qiskit_ibm_runtime is not installed in this environment. "
                "Real-hardware execution is not supported via this stub; "
                "TorchQuantum simulation still works."
            )
    runtime.QiskitRuntimeService = _MissingRuntimeService

try:
    import torchquantum  # noqa: F401  (real install — no shim needed)
except ImportError:
    _install_qiskit_compat_stubs()

from models.quantum._qcircuits import (   # noqa: E402
    AngleEncoder,
    VariationalAnsatz,
    QuantumHead,
)

__all__ = ["AngleEncoder", "VariationalAnsatz", "QuantumHead"]
