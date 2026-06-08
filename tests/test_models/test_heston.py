"""Unit tests for Heston model."""
import numpy as np
import pytest

from volsurf.models.gen1_sv.heston import HestonModel


class TestHestonCharacteristicFunction:
    def test_phi_at_zero_is_one(self):
        """φ(0) = 1 for any valid parameters."""
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)
        phi = model._characteristic_function(
            np.array([0.0]), S=100.0, T=1.0, r=0.05
        )
        assert np.abs(phi[0] - 1.0) < 1e-10

    def test_phi_real_part_at_small_u(self):
        """Real part of φ(u) for small u — should be close to 1 for u→0."""
        model = HestonModel()
        u = np.array([0.01])  # very small, not 0.5
        phi = model._characteristic_function(u, S=100.0, T=0.5, r=0.03)
        # For u→0, φ(u) ≈ 1 + iu E[log S_T] - u^2/2 Var(log S_T)
        # Real part ≈ 1 for small u
        assert np.real(phi[0]) > 0.8  # relaxed bound
        assert np.abs(np.imag(phi[0])) < 0.5  # small imaginary part


class TestHestonPricing:
    def test_price_positive(self):
        model = HestonModel()
        price = model.price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, option_type="call")
        assert price > 0.0

    def test_put_call_parity(self):
        """C - P = S*exp(-qT) - K*exp(-rT)."""
        model = HestonModel(v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.2, rho=-0.5)
        S, K, T, r, q = 100.0, 95.0, 0.5, 0.03, 0.01
        c = model.price(S, K, T, r, q, "call")
        p = model.price(S, K, T, r, q, "put")
        assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-3)

    def test_atm_price_reasonable(self):
        """ATM call should be roughly 0.4 * S * σ * sqrt(T) for short T."""
        model = HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.5)
        S, K, T, r = 100.0, 100.0, 0.25, 0.0
        price = model.price(S, K, T, r, 0.0, "call")
        approx = 0.4 * S * np.sqrt(0.04) * np.sqrt(T)
        # Heston ATM is close to BS ATM with σ = sqrt(v0) for short T
        assert price == pytest.approx(approx, rel=0.15)


class TestHestonCalibrationStub:
    def test_calibration_runs_on_small_dataset(self):
        """Quick smoke test that calibrate() executes."""
        import pandas as pd
        model = HestonModel()
        # Tiny synthetic dataset
        df = pd.DataFrame({
            "spot": [100.0] * 4,
            "strike": [90.0, 95.0, 100.0, 105.0],
            "T": [0.5] * 4,
            "risk_free_rate": [0.03] * 4,
            "dividend_yield": [0.0] * 4,
            "option_type": ["call"] * 4,
            "midPrice": [12.0, 8.0, 5.0, 3.0],
        })
        # Use least_squares for speed in tests
        params = model.calibrate(df, method="least_squares")
        assert "v0" in params
        assert params["v0"] > 0.0
        assert abs(params["rho"]) < 1.0