"""
Smoke tests for the 3 hybrid quantum-classical models.

Checks:
  1. QuantumHead forward shape + gradient flow through quantum parameters.
  2. HydroCNNLSTMQC : (B=2, 5120) → (2, 4) logits, all params trainable.
  3. HydroQuantumTransfer : teacher params frozen, dressed block trainable,
     end-to-end forward shape correct (uses random teacher — wiring only).
  4. HydroVQCFeatures : (B=2, 138) → (2, 4) logits.

Run:
    python -m pytest tests/smoke/test_quantum_models.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import torch

# Importing ``models.quantum`` installs a compatibility shim for environments
# where qiskit-aer / qiskit_ibm_runtime are missing (Python 3.13 has no
# qiskit-aer < 0.14 wheels). Do this BEFORE checking for torchquantum so the
# importorskip below sees the stubs and does not fail on a transitive import.
try:
    import models.quantum  # noqa: F401  (side-effect: install shim)
except Exception:
    pass
torchquantum = pytest.importorskip("torchquantum")

BATCH  = 2
T      = 5_120
N_CLS  = 4
DEVICE = "cpu"


# ─────────────────────────────────────────────────────────────────────────
#  Shared QuantumHead
# ─────────────────────────────────────────────────────────────────────────

def test_quantum_head_forward_shape():
    from models.quantum import QuantumHead
    head = QuantumHead(in_dim=32, n_qubits=4, n_layers=2, num_classes=N_CLS)
    x = torch.randn(BATCH, 32)
    out = head(x)
    assert out.shape == (BATCH, N_CLS), f"Expected (2,4), got {out.shape}"
    assert torch.all(torch.isfinite(out))


def test_quantum_head_gradient_through_circuit():
    """Backward pass must produce non-None grads on the variational ansatz params."""
    from models.quantum import QuantumHead
    head = QuantumHead(in_dim=16, n_qubits=4, n_layers=2, num_classes=N_CLS)
    x = torch.randn(BATCH, 16, requires_grad=False)
    out = head(x)
    out.sum().backward()
    ansatz_grads = [p.grad for n, p in head.named_parameters()
                    if "ansatz" in n and p.grad is not None]
    assert len(ansatz_grads) > 0, "No gradient flowed into the variational ansatz"
    assert all(torch.all(torch.isfinite(g)) for g in ansatz_grads)


# ─────────────────────────────────────────────────────────────────────────
#  Model A — HydroCNNLSTMQC
# ─────────────────────────────────────────────────────────────────────────

class TestHydroCNNLSTMQC:

    def setup_method(self):
        from models.hydro_cnnlstm_qc import HydroCNNLSTMQC
        # Smaller config for fast CPU smoke test.
        self.model = HydroCNNLSTMQC(
            num_classes=N_CLS,
            cnn_channels=(8, 16, 32, 64),
            lstm_hidden=32, lstm_layers=1,
            n_qubits=4, n_layers=2,
        ).to(DEVICE).eval()

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(torch.randn(BATCH, T))
        assert out.shape == (BATCH, N_CLS)
        assert torch.all(torch.isfinite(out))

    def test_all_params_trainable(self):
        non_trainable = [n for n, p in self.model.named_parameters() if not p.requires_grad]
        assert non_trainable == [], f"Unexpected frozen params: {non_trainable}"


# ─────────────────────────────────────────────────────────────────────────
#  Model B — HydroQuantumTransfer (random teacher; wiring only)
# ─────────────────────────────────────────────────────────────────────────

class TestHydroQuantumTransfer:

    def setup_method(self):
        from models.hydro_quantum_transfer import HydroQuantumTransfer
        # Small BAHTNet teacher to keep the smoke test fast on CPU.
        self.model = HydroQuantumTransfer(
            num_classes=N_CLS,
            teacher_arch="bahtnet",
            teacher_kwargs=dict(model_dim=64, n_heads=4, n_layers=2),
            teacher_ckpt=None,        # random weights — wiring smoke test only
            dressed_dim=16,
            n_qubits=4, n_layers=2,
        ).to(DEVICE).eval()

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(torch.randn(BATCH, T))
        assert out.shape == (BATCH, N_CLS)
        assert torch.all(torch.isfinite(out))

    def test_teacher_frozen(self):
        teacher_trainable = [n for n, p in self.model.teacher.named_parameters()
                             if p.requires_grad]
        assert teacher_trainable == [], (
            f"Teacher params should be frozen but {len(teacher_trainable)} are trainable"
        )

    def test_dressed_block_trainable(self):
        dressed = [n for n, p in self.model.named_parameters()
                   if ("dressed" in n or "qhead" in n) and p.requires_grad]
        assert len(dressed) > 0, "Dressed quantum block should have trainable params"


# ─────────────────────────────────────────────────────────────────────────
#  Model D — HydroVQCFeatures
# ─────────────────────────────────────────────────────────────────────────

class TestHydroVQCFeatures:

    def setup_method(self):
        from models.hydro_vqc_features import HydroVQCFeatures
        self.model = HydroVQCFeatures(
            num_classes=N_CLS,
            compress_dim=8,
            n_qubits=4, n_layers=2,
        ).to(DEVICE).eval()

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(torch.randn(BATCH, 138))
        assert out.shape == (BATCH, N_CLS)
        assert torch.all(torch.isfinite(out))

    def test_unpacks_3tuple_batch(self):
        """Must accept the (feat_1d, feat_2d, labels) tuple from omni_collate_fn."""
        feat_1d = torch.randn(BATCH, 138)
        feat_2d = torch.randn(BATCH, 9, 16, 16)   # placeholder, ignored
        labels  = torch.randint(0, N_CLS, (BATCH,))
        x, y = self.model._unpack((feat_1d, feat_2d, labels))
        assert x.shape == (BATCH, 138)
        assert y.shape == (BATCH,)


# ─────────────────────────────────────────────────────────────────────────
#  Registry round-trip
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,kwargs,input_shape", [
    ("cnnlstm_qc",
     dict(cnn_channels=(8, 16, 32, 64), lstm_hidden=32, lstm_layers=1,
          n_qubits=4, n_layers=2),
     (BATCH, T)),
    ("vqc_features",
     dict(compress_dim=8, n_qubits=4, n_layers=2),
     (BATCH, 138)),
])
def test_registry_round_trip(name, kwargs, input_shape):
    from processing.registry import build_model, list_models
    assert name in list_models(), f"{name} missing from registry"
    m = build_model(name, num_classes=N_CLS, **kwargs).eval()
    with torch.no_grad():
        out = m(torch.randn(*input_shape))
    assert out.shape == (BATCH, N_CLS)
