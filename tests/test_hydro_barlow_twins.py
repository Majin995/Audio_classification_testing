"""Sanity tests for ``models.hydro_barlow_twins``: forward shapes, loss
identity case, training-step contract, and two-view augmentation diversity."""

import torch

from models.hydro_barlow_twins import (
    HydroBarlowTwins,
    _TwoViewWaveformAug,
)


B, T = 4, 5_120
PROJ_DIM = 2048


def _make_model() -> HydroBarlowTwins:
    return HydroBarlowTwins(
        num_classes=4,
        sample_rate=5_120,
        projection_hidden=PROJ_DIM,
        projection_dim=PROJ_DIM,
        max_epochs=10,
        warmup_epochs=1,
    )


def test_encode_and_project_shapes():
    torch.manual_seed(0)
    model = _make_model().train()
    x = torch.randn(B, T)

    embed = model.encode(x)
    assert embed.shape == (B, model._embed_dim), (
        f"encoder shape mismatch: {embed.shape}"
    )

    z = model.project(embed)
    assert z.shape == (B, PROJ_DIM), f"projector shape mismatch: {z.shape}"

    out = model(x)
    assert out.shape == (B, PROJ_DIM), f"forward shape mismatch: {out.shape}"


def test_training_step_returns_scalar_finite_loss():
    torch.manual_seed(0)
    model = _make_model().train()
    batch = (torch.randn(B, T), torch.randint(0, 4, (B,)))
    loss = model.training_step(batch, batch_idx=0)
    assert loss.dim() == 0, f"loss must be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss).item(), f"loss must be finite, got {loss}"


def test_barlow_twins_loss_on_diag_zero_when_identical_views():
    """Cross-correlation diagonal should equal 1 (so on_diag → 0) when
    z1 == z2, because BatchNorm whitens each feature to unit variance."""
    torch.manual_seed(0)
    model = _make_model().train()

    N = 256
    z = torch.randn(N, PROJ_DIM)

    loss, on_diag, off_diag = model.barlow_twins_loss(z, z)

    assert on_diag.item() < 1e-3, (
        f"on_diag should be ~0 when z1==z2 after BN whitening, got {on_diag.item()}"
    )
    assert torch.isfinite(loss).item()
    assert off_diag.item() >= 0.0


def test_barlow_twins_loss_higher_when_views_decorrelated():
    """Loss should be larger for fully-decorrelated z1, z2 than for identical
    z1 == z2 (the latter has on_diag ≈ 0 by construction)."""
    torch.manual_seed(0)
    model = _make_model().train()

    N = 256
    z1 = torch.randn(N, PROJ_DIM)
    z2 = torch.randn(N, PROJ_DIM)

    loss_identical, _, _ = model.barlow_twins_loss(z1, z1)
    loss_decorrelated, on_diag_d, _ = model.barlow_twins_loss(z1, z2)

    assert loss_decorrelated.item() > loss_identical.item(), (
        f"decorrelated views should have higher loss: "
        f"identical={loss_identical.item()}, decorrelated={loss_decorrelated.item()}"
    )
    # Decorrelated views have C diag ≈ 0 → on_diag ≈ P (target = 1 each).
    assert on_diag_d.item() > 0.5 * PROJ_DIM


def test_two_view_aug_produces_different_views():
    torch.manual_seed(0)
    aug = _TwoViewWaveformAug(
        noise_prob=1.0, gain_prob=1.0, shift_prob=1.0,
    )
    x = torch.randn(2, T)
    v1, v2 = aug(x)
    assert v1.shape == x.shape and v2.shape == x.shape
    # Views should not be identical given prob=1.0 on all stochastic ops.
    assert not torch.allclose(v1, v2), "two views should differ under aug"


def test_save_hyperparameters_present():
    model = _make_model()
    for key in ("projection_dim", "lambda_param", "learning_rate",
               "warmup_epochs", "max_epochs"):
        assert key in model.hparams, f"missing hparam: {key}"
    assert model.hparams.projection_dim == PROJ_DIM
