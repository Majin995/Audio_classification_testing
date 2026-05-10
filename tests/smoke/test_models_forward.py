"""
Smoke tests for the 5 new UATR model architectures.

Checks:
  1. Model instantiates with default args.
  2. forward(torch.randn(2, 5120)) returns (2, 4) logits.
  3. No NaN or Inf in outputs.
  4. HydroSSCPMobile params < 32k (< 128 kB fp32).
  5. HydroBAHTNet uses LargeMarginFocalLoss by default.

Run:
    python -m pytest tests/smoke/test_models_forward.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pytest

BATCH = 2
T     = 5_120        # 1 s at 5120 Hz
N_CLS = 4
DEVICE = "cpu"       # smoke tests run on CPU


def _make_batch(B=BATCH, T=T, device=DEVICE):
    return torch.randn(B, T, device=device)


# ─────────────────────────────────────────────────────────────────────────────
#  HydroCATFISH
# ─────────────────────────────────────────────────────────────────────────────

class TestHydroCATFISH:

    def setup_method(self):
        from models.hydro_catfish import HydroCATFISH
        self.model = HydroCATFISH(num_classes=N_CLS).to(DEVICE).eval()

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert out.shape == (BATCH, N_CLS), f"Expected (2,4), got {out.shape}"

    def test_no_nan(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert torch.all(torch.isfinite(out)), "Output has NaN/Inf"

    def test_gradient_flows(self):
        x = _make_batch().requires_grad_(False)
        out = self.model.train()(_make_batch())
        out.sum().backward()   # should not raise


# ─────────────────────────────────────────────────────────────────────────────
#  HydroALSI
# ─────────────────────────────────────────────────────────────────────────────

class TestHydroALSI:

    def setup_method(self):
        from models.hydro_alsi import HydroALSI
        # Use small fusion_dim for fast smoke test
        self.model = HydroALSI(
            num_classes=N_CLS,
            fusion_dim=64,
            fusion_heads=4,
        ).to(DEVICE).eval()

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert out.shape == (BATCH, N_CLS), f"Expected (2,4), got {out.shape}"

    def test_no_nan(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert torch.all(torch.isfinite(out))


# ─────────────────────────────────────────────────────────────────────────────
#  HydroDCN
# ─────────────────────────────────────────────────────────────────────────────

class TestHydroDCN:

    def setup_method(self):
        from models.hydro_dcn import HydroDCN
        self.model = HydroDCN(num_classes=N_CLS).to(DEVICE).eval()

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert out.shape == (BATCH, N_CLS), f"Expected (2,4), got {out.shape}"

    def test_no_nan(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert torch.all(torch.isfinite(out))

    def test_complex_layer_types(self):
        from models.hydro_dcn import ComplexConv2d, ComplexBatchNorm2d, ModReLU
        from models.hydro_dcn import HydroDCN
        model = HydroDCN(num_classes=N_CLS)
        has_complex = any(isinstance(m, (ComplexConv2d, ComplexBatchNorm2d, ModReLU))
                          for m in model.modules())
        assert has_complex, "Model should contain complex-valued layers"


# ─────────────────────────────────────────────────────────────────────────────
#  HydroBAHTNet
# ─────────────────────────────────────────────────────────────────────────────

class TestHydroBAHTNet:

    def setup_method(self):
        from models.hydro_bahtnet import HydroBAHTNet
        # Smaller model for fast test
        self.model = HydroBAHTNet(
            num_classes=N_CLS,
            model_dim=64, n_heads=4, n_layers=2,
        ).to(DEVICE).eval()

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert out.shape == (BATCH, N_CLS), f"Expected (2,4), got {out.shape}"

    def test_no_nan(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert torch.all(torch.isfinite(out))

    def test_uses_lmf(self):
        from models.hydro_bahtnet import HydroBAHTNet
        from processing.losses import LargeMarginFocalLoss
        model = HydroBAHTNet(num_classes=N_CLS)
        assert isinstance(model.criterion, LargeMarginFocalLoss), \
            "BAHTNet should use LargeMarginFocalLoss by default"

    def test_boundary_gate_learnable(self):
        from models.hydro_bahtnet import HydroBAHTNet, BoundaryAwareAttention
        model = HydroBAHTNet(num_classes=N_CLS)
        gates = [p for n, p in model.named_parameters() if 'boundary_gate' in n]
        assert len(gates) > 0, "Should have learnable boundary_gate parameters"


# ─────────────────────────────────────────────────────────────────────────────
#  HydroSSCPMobile
# ─────────────────────────────────────────────────────────────────────────────

class TestHydroSSCPMobile:

    def setup_method(self):
        from models.hydro_sscp_mobile import HydroSSCPMobile
        self.model = HydroSSCPMobile(num_classes=N_CLS).to(DEVICE).eval()

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert out.shape == (BATCH, N_CLS), f"Expected (2,4), got {out.shape}"

    def test_no_nan(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert torch.all(torch.isfinite(out))

    def test_param_budget(self):
        """Model must have fewer than 32k trainable float32 parameters (128 kB)."""
        from models.hydro_sscp_mobile import HydroSSCPMobile, _PARAM_BUDGET_BYTES
        model = HydroSSCPMobile(num_classes=N_CLS)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_bytes = n_params * 4
        assert param_bytes < _PARAM_BUDGET_BYTES, (
            f"Model exceeds 128 kB budget: {n_params} params = {param_bytes / 1024:.1f} kB"
        )

    def test_no_teacher_supervised_only(self):
        """Without a teacher ckpt, _teacher should be None."""
        from models.hydro_sscp_mobile import HydroSSCPMobile
        model = HydroSSCPMobile(num_classes=N_CLS, teacher_ckpt=None)
        assert model._teacher is None


# ─────────────────────────────────────────────────────────────────────────────
#  SuperModel1D
# ─────────────────────────────────────────────────────────────────────────────

class TestSuperModel1D:
    """Smoke tests for SuperModel1D using a reduced config that runs on CPU."""

    # Small config: stays well below 1 M params and runs in a few seconds on CPU
    _KWARGS = dict(
        num_classes     = N_CLS,
        gabor_n_filters = 32,
        gabor_kernel    = 129,
        stem_stride     = 8,
        g_ch            = 32,
        spec_n_mels     = 16,
        spec_hop        = 51,
        wb_n_fft        = 128,
        nb_n_fft        = 512,
        s_ch            = 32,
        lofar_n_fft     = 512,
        lofar_time_bins = 16,
        lofar_freq_bins = 64,
        l_ch            = 16,
        d_model         = 64,
        scale           = 4,
        dilation_rates  = [2, 4],
        n_dart          = 1,
        dart_heads      = 4,
        n_s4            = 1,
        s4_d_state      = 16,
        n_mamba         = 1,
        mamba_d_state   = 8,
        mamba_expand    = 2,
        mamba_d_conv    = 4,
        drop_path_rate  = 0.0,
        dropout         = 0.1,
    )

    def setup_method(self):
        from models.super_model import SuperModel1D
        self.model = SuperModel1D(**self._KWARGS).to(DEVICE).eval()

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert out.shape == (BATCH, N_CLS), f"Expected ({BATCH},{N_CLS}), got {out.shape}"

    def test_no_nan(self):
        with torch.no_grad():
            out = self.model(_make_batch())
        assert torch.all(torch.isfinite(out)), "Output has NaN/Inf"

    def test_gradient_flows(self):
        """Backward pass should produce non-None gradients on key parameters."""
        from models.super_model import SuperModel1D
        model = SuperModel1D(**self._KWARGS).to(DEVICE).train()
        out = model(_make_batch())
        out.sum().backward()
        # Check Gabor filterbank parameters
        gabor_params = [(n, p) for n, p in model.named_parameters()
                        if "gabor" in n and p.grad is not None]
        assert len(gabor_params) > 0, "Gabor parameters received no gradient"
        # Check SE-Res2 backbone parameters
        res_params = [(n, p) for n, p in model.named_parameters()
                      if "res_blocks" in n and p.grad is not None]
        assert len(res_params) > 0, "res_blocks parameters received no gradient"


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-model: all new models accept same raw-waveform input
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_cls,kwargs", [
    ("HydroCATFISH",   {}),
    ("HydroALSI",      {"fusion_dim": 64, "fusion_heads": 4}),
    ("HydroDCN",       {}),
    ("HydroBAHTNet",   {"model_dim": 64, "n_heads": 4, "n_layers": 2}),
    ("HydroSSCPMobile", {}),
])
def test_all_models_same_input_interface(model_cls, kwargs):
    """All 5 models accept (B, T) raw waveform and return (B, 4) logits."""
    import importlib
    mod = importlib.import_module("models")
    cls = getattr(mod, model_cls)
    model = cls(num_classes=N_CLS, **kwargs).eval()
    with torch.no_grad():
        out = model(_make_batch())
    assert out.shape == (BATCH, N_CLS), \
        f"{model_cls}: expected ({BATCH},{N_CLS}), got {out.shape}"
    assert torch.all(torch.isfinite(out)), f"{model_cls}: output has NaN/Inf"
