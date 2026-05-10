"""Sanity tests for ``models.hydro_audio_llm``.

Whisper-large-v3 (~1.5 GB) and Llama-3.2-1B (~2.4 GB) cannot be downloaded
in CI, so these tests monkey-patch ``WhisperFeatureExtractor``, ``WhisperModel``
and ``LlamaModel`` ``from_pretrained`` constructors with tiny stubs that
expose the same interface (``last_hidden_state``, ``config.d_model``,
``config.hidden_size``).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


B, T = 2, 5_120
STUB_WHISPER_DIM = 64
STUB_LLAMA_DIM = 32
STUB_AUDIO_FRAMES = 128


# ═══════════════════════════════════════════════════════════════════════
#  Stub backbones
# ═══════════════════════════════════════════════════════════════════════

class _StubFeatureExtractor:
    """Returns ``(B, 80, 3000)`` log-mel-shaped tensor regardless of input."""

    def __call__(self, audio_list, sampling_rate, return_tensors="pt"):
        bsz = len(audio_list)
        feat = torch.randn(bsz, 80, 3000)
        return SimpleNamespace(input_features=feat)


class _StubWhisperEncoder(nn.Module):
    def __init__(self, d_model: int = STUB_WHISPER_DIM,
                 t_out: int = STUB_AUDIO_FRAMES):
        super().__init__()
        self.proj = nn.Conv1d(80, d_model, kernel_size=1)
        self.t_out = t_out

    def forward(self, input_features: torch.Tensor):
        x = self.proj(input_features)                                  # (B, d, 3000)
        x = nn.functional.adaptive_avg_pool1d(x, self.t_out)           # (B, d, t_out)
        return SimpleNamespace(last_hidden_state=x.transpose(1, 2))    # (B, t_out, d)


class _StubWhisperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _StubWhisperEncoder()
        self.config = SimpleNamespace(d_model=STUB_WHISPER_DIM)


class _StubLlamaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=STUB_LLAMA_DIM)
        self.proj = nn.Linear(STUB_LLAMA_DIM, STUB_LLAMA_DIM)

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = self.proj(inputs_embeds)
        return SimpleNamespace(last_hidden_state=x)


# ═══════════════════════════════════════════════════════════════════════
#  Fixture: monkey-patch HF backbones before importing/instantiating model
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def stub_backbones(monkeypatch):
    import transformers

    def fake_fe_from_pretrained(*args, **kwargs):
        return _StubFeatureExtractor()

    def fake_whisper_from_pretrained(*args, **kwargs):
        return _StubWhisperModel()

    def fake_llama_from_pretrained(*args, **kwargs):
        return _StubLlamaModel()

    monkeypatch.setattr(
        transformers.WhisperFeatureExtractor, "from_pretrained",
        fake_fe_from_pretrained,
    )
    monkeypatch.setattr(
        transformers.WhisperModel, "from_pretrained",
        fake_whisper_from_pretrained,
    )
    monkeypatch.setattr(
        transformers.LlamaModel, "from_pretrained",
        fake_llama_from_pretrained,
    )
    yield


def _make_model(num_classes: int = 4, **kw):
    from models.hydro_audio_llm import HydroAudioLLM
    return HydroAudioLLM(num_classes=num_classes, max_epochs=10,
                         warmup_epochs=1, **kw)


# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════

def test_forward_shape(stub_backbones):
    torch.manual_seed(0)
    model = _make_model(num_classes=4).eval()
    x = torch.randn(B, T)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (B, 4), f"forward shape mismatch: {out.shape}"


def test_freeze_flags_default(stub_backbones):
    """Default ``freeze_encoder=True`` and ``freeze_llm=True`` should
    leave only adapter + head parameters trainable."""
    model = _make_model()

    enc_trainable = [n for n, p in model.audio_encoder.named_parameters()
                     if p.requires_grad]
    llm_trainable = [n for n, p in model.llm.named_parameters()
                     if p.requires_grad]
    adapter_trainable = [n for n, p in model.adapter.named_parameters()
                         if p.requires_grad]
    head_trainable = [n for n, p in model.head.named_parameters()
                      if p.requires_grad]

    assert enc_trainable == [], (
        f"encoder should be fully frozen, found {len(enc_trainable)} trainable"
    )
    assert llm_trainable == [], (
        f"llm should be fully frozen, found {len(llm_trainable)} trainable"
    )
    assert len(adapter_trainable) > 0
    assert len(head_trainable) > 0


def test_unfreeze_flags(stub_backbones):
    model = _make_model(freeze_encoder=False, freeze_llm=False)
    enc_trainable = [p for p in model.audio_encoder.parameters() if p.requires_grad]
    llm_trainable = [p for p in model.llm.parameters() if p.requires_grad]
    assert len(enc_trainable) > 0
    assert len(llm_trainable) > 0


def test_training_step_returns_scalar_loss(stub_backbones):
    torch.manual_seed(0)
    model = _make_model(num_classes=4).train()
    batch = (torch.randn(B, T), torch.randint(0, 4, (B,)))
    loss = model.training_step(batch, batch_idx=0)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()


def test_save_hyperparameters_present(stub_backbones):
    model = _make_model()
    for key in ("whisper_model_id", "llm_model_id", "freeze_encoder",
                "freeze_llm", "adapter_type", "num_classes"):
        assert key in model.hparams, f"missing hparam: {key}"
    assert model.hparams.adapter_type == "linear"


def test_unsupported_adapter_type_raises(stub_backbones):
    with pytest.raises(NotImplementedError):
        _make_model(adapter_type="moe")
