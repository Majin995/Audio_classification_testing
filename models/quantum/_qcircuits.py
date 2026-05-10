"""
Parameterised-quantum-circuit primitives shared by the three hybrid models.

Pipeline shape contract
-----------------------
    classical (B, in_dim)
        │  Linear → tanh × π   (compress to qubit-count rotation angles in (-π, π))
        ▼
    AngleEncoder                  data re-uploading: RY then RZ on every qubit
        ▼
    VariationalAnsatz             n_layers × [trainable RY+RZ + CNOT-ring]
        ▼
    MeasureAll(PauliZ)            (B, n_qubits) expectation values in [-1, 1]
        ▼
    Linear(n_qubits → num_classes)

All gradients flow end-to-end (TorchQuantum is pure-PyTorch).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import torchquantum as tq
import torchquantum.functional as tqf


# ─────────────────────────────────────────────────────────────────────────
#  AngleEncoder — classical-to-quantum data re-uploading layer
# ─────────────────────────────────────────────────────────────────────────

class AngleEncoder(tq.QuantumModule):
    """
    Encode a classical ``(B, n_qubits)`` tensor onto an ``n_qubits``-qubit
    register via per-qubit RY then RZ rotations, optionally repeated
    (``n_reuploads`` > 1) for richer expressivity.

    Args:
        n_qubits    : Width of the qubit register.
        n_reuploads : Number of times to re-encode the input (default 1).
    """

    def __init__(self, n_qubits: int, n_reuploads: int = 1):
        super().__init__()
        self.n_qubits    = n_qubits
        self.n_reuploads = n_reuploads

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        """
        Args:
            q_device : Active TorchQuantum device (batched).
            x        : (B, n_qubits) classical input — rotation angles in radians.
        """
        for _ in range(self.n_reuploads):
            for q in range(self.n_qubits):
                tqf.ry(q_device, wires=q, params=x[:, q])
            for q in range(self.n_qubits):
                tqf.rz(q_device, wires=q, params=x[:, q])


# ─────────────────────────────────────────────────────────────────────────
#  VariationalAnsatz — trainable RY+RZ + CNOT-ring entangler
# ─────────────────────────────────────────────────────────────────────────

class VariationalAnsatz(tq.QuantumModule):
    """
    Hardware-efficient ansatz: ``n_layers`` × [parameterised RY+RZ on every
    qubit, followed by a CNOT ring (q → q+1 mod n_qubits)].

    Initialisation uses a small std (0.01) on the rotation angles — this is
    a common mitigation for barren plateaus in deep variational circuits.

    Args:
        n_qubits : Width of the qubit register.
        n_layers : Number of variational layers (default 4).
    """

    def __init__(self, n_qubits: int, n_layers: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ry = nn.Parameter(torch.empty(n_layers, n_qubits))
        self.rz = nn.Parameter(torch.empty(n_layers, n_qubits))
        nn.init.normal_(self.ry, std=0.01)
        nn.init.normal_(self.rz, std=0.01)

    def forward(self, q_device: tq.QuantumDevice) -> None:
        for L in range(self.n_layers):
            for q in range(self.n_qubits):
                tqf.ry(q_device, wires=q, params=self.ry[L, q])
                tqf.rz(q_device, wires=q, params=self.rz[L, q])
            # CNOT ring entangler
            for q in range(self.n_qubits):
                tqf.cnot(q_device, wires=[q, (q + 1) % self.n_qubits])


# ─────────────────────────────────────────────────────────────────────────
#  QuantumHead — one block, plug into any classical backbone
# ─────────────────────────────────────────────────────────────────────────

class QuantumHead(nn.Module):
    """
    Classifier head that runs its input through a parameterised quantum
    circuit before producing logits.

    Forward shape: ``(B, in_dim) → (B, num_classes)``.

    Args:
        in_dim       : Width of the classical embedding fed in.
        n_qubits     : Qubit register width (default 8 — practical ceiling
                       on CPU TorchQuantum simulation).
        n_layers     : Variational ansatz depth (default 4).
        n_reuploads  : Data-reuploading repetitions in the encoder (default 1).
        num_classes  : Number of output classes (default 4).
    """

    def __init__(
        self,
        in_dim: int,
        n_qubits: int = 8,
        n_layers: int = 4,
        n_reuploads: int = 1,
        num_classes: int = 4,
    ):
        super().__init__()
        self.n_qubits   = n_qubits
        self.n_layers   = n_layers
        self.proj       = nn.Linear(in_dim, n_qubits)
        self.encoder    = AngleEncoder(n_qubits, n_reuploads)
        self.ansatz    = VariationalAnsatz(n_qubits, n_layers)
        self.measure    = tq.MeasureAll(tq.PauliZ)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz   = x.shape[0]
        # Map any classical embedding to rotation angles in (-π, π).
        angles = torch.tanh(self.proj(x)) * math.pi
        qdev  = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device)
        self.encoder(qdev, angles)
        self.ansatz(qdev)
        # (B, n_qubits) of PauliZ expectation values in [-1, 1].
        expvals = self.measure(qdev)
        return self.classifier(expvals)

    def extra_repr(self) -> str:
        return f"n_qubits={self.n_qubits}, n_layers={self.n_layers}"
