"""Unit tests for Black-Scholes pricer and implied volatility solver."""
import numpy as np
import pytest

from volsurf.pricing import black_scholes as bs


class TestBlackScholesPrice:
    def test_call_price_atm(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.0, 0.2
        price = bs.price(S, K, T, r, sigma, option_type="call")
        approx = 0.4 * S * sigma * np.sqrt(T)
        assert price == pytest.approx(approx, rel=0.05)

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.03, 0.25
        c = bs.price(S, K, T, r, sigma, option_type="call")
        p = bs.price(S, K, T, r, sigma, option_type="put")
        assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-6)

    def test_intrinsic_value_bound(self):
        S, K, T, r, sigma = 100.0, 90.0, 0.25, 0.05, 0.15
        price = bs.price(S, K, T, r, sigma, option_type="call")
        assert price >= max(0.0, S - K * np.exp(-r * T)) - 1e-10

    def test_zero_volatility(self):
        S, K, T, r = 100.0, 95.0, 0.5, 0.03
        price = bs.price(S, K, T, r, 1e-8, option_type="call")
        intrinsic = max(0.0, S - K * np.exp(-r * T))
        assert price == pytest.approx(intrinsic, abs=1e-4)


class TestBlackScholesGreeks:
    def test_vega_positive(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        assert bs.vega(S, K, T, r, sigma) > 0.0

    def test_delta_call_range(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        assert 0.0 <= bs.delta(S, K, T, r, sigma, option_type="call") <= 1.0

    def test_delta_put_range(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        assert -1.0 <= bs.delta(S, K, T, r, sigma, option_type="put") <= 0.0

    def test_gamma_positive(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        assert bs.gamma(S, K, T, r, sigma) > 0.0

    def test_vega_finite_difference(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
        h = 1e-5
        fd = (bs.price(S, K, T, r, sigma + h) - bs.price(S, K, T, r, sigma - h)) / (2 * h)
        assert bs.vega(S, K, T, r, sigma) == pytest.approx(fd, rel=1e-4)


class TestImpliedVolatility:
    def test_round_trip_call(self):
        S, K, T, r, sig = 100.0, 100.0, 0.5, 0.03, 0.25
        p = bs.price(S, K, T, r, sig, option_type="call")
        assert bs.implied_volatility(S, K, T, r, p, option_type="call") == pytest.approx(sig, abs=1e-6)

    def test_round_trip_put(self):
        S, K, T, r, sig = 100.0, 105.0, 0.75, 0.04, 0.30
        p = bs.price(S, K, T, r, sig, option_type="put")
        assert bs.implied_volatility(S, K, T, r, p, option_type="put") == pytest.approx(sig, abs=1e-6)

    def test_invalid_price_raises(self):
        with pytest.raises(ValueError):
            bs.implied_volatility(100.0, 100.0, 1.0, 0.05, 200.0, option_type="call")

    def test_short_maturity(self):
        S, K, T, r, sig = 100.0, 100.0, 1/365, 0.03, 0.20
        p = bs.price(S, K, T, r, sig, option_type="call")
        assert bs.implied_volatility(S, K, T, r, p, option_type="call") == pytest.approx(sig, abs=1e-5)