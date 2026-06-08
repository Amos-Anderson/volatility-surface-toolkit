"""Unit tests for Generation-3 ConvSurfaceNet."""

import numpy as np
import pytest

try:
    import torch
except ImportError:
    pytest.skip("PyTorch not installed", allow_module_level=True)

from volsurf.models.gen3_dl import ConvSurfaceNet


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return ConvSurfaceNet(grid_shape=(20, 8), bottleneck_dim=64)


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

def test_forward_shape(model):
    """Output shape must match grid_shape."""
    x = torch.randn(2, 2, 20, 8)  # (batch=2, channels=2, n_K, n_T)
    out = model(x)
    assert out.shape == (2, 1, 20, 8)


def test_batched_consistency(model):
    """Batch dimension must not interact."""
    x1 = torch.randn(1, 2, 20, 8)
    x2 = torch.randn(1, 2, 20, 8)
    x_b = torch.cat([x1, x2], dim=0)

    out1 = model(x1)
    out2 = model(x2)
    out_b = model(x_b)

    assert torch.allclose(out_b[0:1], out1, atol=1e-5)
    assert torch.allclose(out_b[1:2], out2, atol=1e-5)


def test_loss_computation(model):
    """Loss dict must contain 'total', 'data', 'smooth'."""
    pred = torch.randn(1, 1, 20, 8)
    target = torch.randn(1, 1, 20, 8)
    mask = torch.ones(1, 1, 20, 8)
    mask[:, :, 5:15, 2:6] = 0  # hold out a block

    loss_dict = model.loss(pred, target, mask)
    assert "total" in loss_dict
    assert "data" in loss_dict
    assert "smooth" in loss_dict
    assert loss_dict["total"].item() >= 0
    assert loss_dict["data"].item() >= 0
    assert loss_dict["smooth"].item() >= 0


def test_loss_masked_positions_only(model):
    """Changing an unobserved pixel must not affect data loss."""
    pred = torch.zeros(1, 1, 20, 8)
    target = torch.zeros(1, 1, 20, 8)
    mask = torch.zeros(1, 1, 20, 8)
    mask[0, 0, 3, 3] = 1  # only one observed cell

    loss1 = model.loss(pred, target, mask)["data"]

    pred2 = pred.clone()
    pred2[0, 0, 10, 5] = 100.0  # change unobserved cell
    loss2 = model.loss(pred2, target, mask)["data"]

    assert torch.isclose(loss1, loss2, atol=1e-6)


# ---------------------------------------------------------------------------
# Grid helper tests
# ---------------------------------------------------------------------------

def test_prepare_grid_shapes(model):
    """Grid shapes must match requested bins."""
    n = 100
    strikes = np.linspace(280, 330, n)
    mats = np.random.uniform(0.02, 0.4, n)
    ivs = np.random.uniform(0.1, 0.3, n)

    iv_g, mask_g, k_c, t_c = model.prepare_grid(strikes, mats, ivs)
    assert iv_g.shape == model.grid_shape
    assert mask_g.shape == model.grid_shape
    assert k_c.shape[0] == model.grid_shape[0]
    assert t_c.shape[0] == model.grid_shape[1]


def test_prepare_grid_mask_sums_to_nunique(model):
    """Observed grid cells are capped by the requested grid resolution."""
    n = 50
    strikes = np.linspace(280, 330, n)
    mats = np.linspace(0.02, 0.4, 11)
    # Create one quote per (strike, maturity) pair
    s_all, m_all, iv_all = [], [], []
    for T in mats:
        for K in strikes:
            s_all.append(K)
            m_all.append(T)
            iv_all.append(0.15)

    iv_g, mask_g, _, _ = model.prepare_grid(
        np.array(s_all), np.array(m_all), np.array(iv_all)
    )
    n_unique_pairs = len(mats) * len(strikes)
    n_grid_cells = model.grid_shape[0] * model.grid_shape[1]
    assert mask_g.sum() <= n_unique_pairs
    assert mask_g.sum() == min(n_unique_pairs, n_grid_cells)


# ---------------------------------------------------------------------------
# End-to-end sanity
# ---------------------------------------------------------------------------

def test_overfit_single_surface(model):
    """Network should overfit a single smooth surface to low MSE."""
    torch.manual_seed(0)

    # Create synthetic surface: smooth 2D function
    n_k, n_t = model.grid_shape
    k = np.linspace(0, 1, n_k)
    t = np.linspace(0, 1, n_t)
    K, T = np.meshgrid(k, t, indexing="ij")
    surface = 0.15 + 0.05 * K + 0.03 * np.sin(2 * np.pi * T)

    mask = np.ones_like(surface)
    x = np.stack([surface, mask], axis=0)
    x_tensor = torch.from_numpy(x).unsqueeze(0).float()
    target = torch.from_numpy(surface).unsqueeze(0).unsqueeze(0).float()
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        pred = model(x_tensor)
        loss = model.loss(pred, target, mask_tensor)["total"]
        loss.backward()
        opt.step()

    final_pred = model(x_tensor).detach().numpy()
    mse = np.mean((final_pred - surface) ** 2)
    assert mse < 5e-4, f"MSE too high: {mse}"


def test_predict_surface_returns_numpy(model):
    """predict_surface must return a numpy array of correct shape."""
    n_k, n_t = model.grid_shape
    iv_grid = np.random.uniform(0.1, 0.3, (n_k, n_t))
    mask_grid = np.random.choice([0.0, 1.0], size=(n_k, n_t)).astype(np.float32)

    pred = model.predict_surface(iv_grid, mask_grid)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (n_k, n_t)
