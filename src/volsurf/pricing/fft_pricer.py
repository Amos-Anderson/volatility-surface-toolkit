"""Carr–Madan FFT European call pricer.

Works with any risk-neutral characteristic function of log(S_T/S_0).
"""

import numpy as np


class FFTPricer:
    """Fast Fourier Transform call pricer via Carr–Madan (1999)."""

    def price_calls(
        self,
        cf,
        S0,
        K,
        T,
        r,
        q=0.0,
        alpha=1.5,
        N=2**14,
        B=200.0,
    ):
        """Price European calls for an array of strikes.

        Parameters
        ----------
        cf : callable
            Risk-neutral characteristic function of log(S_T/S_0).
            Must accept a complex numpy array and return a complex array.
        S0 : float
            Spot price.
        K : array-like
            Strikes.
        T : float
            Time to maturity (years).
        r : float
            Risk-free rate (annualised).
        q : float, optional
            Dividend yield (annualised).
        alpha : float, optional
            Carr–Madan damping factor (must be > 0).
        N : int, optional
            Number of FFT points (power of 2).
        B : float, optional
            Upper truncation bound for the Fourier integral.

        Returns
        -------
        np.ndarray
            Call prices aligned with input strikes.
        """
        K = np.asarray(K, dtype=float)

        # --- Simpson quadrature on [0, B] ---
        eta = B / N
        v = np.arange(N) * eta

        w = np.ones(N)
        w[0] = 1.0
        w[1::2] = 4.0   # odd indices
        w[2::2] = 2.0   # even indices (except 0)
        w *= eta / 3.0

        # --- Damped characteristic function ---
        u = v - 1j * (alpha + 1)
        phi = cf(u)

        denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
        psi = np.exp(-r * T) * phi / denom

        # --- FFT input: (-1)^j factor comes from k_min = -pi/eta ---
        x = (-1) ** np.arange(N) * psi * w
        y = np.fft.fft(x)

        # --- Log-strike grid ---
        lambd = 2.0 * np.pi / (N * eta)
        k_grid = -N * lambd / 2.0 + lambd * np.arange(N)

        # --- Call prices on the grid ---
        prices_grid = S0 * np.exp(-alpha * k_grid) * np.clip(y.real, 0.0, None) / np.pi

        # --- Interpolate to target strikes ---
        k_target = np.log(K / S0)
        prices = np.interp(k_target, k_grid, prices_grid)
        prices = np.maximum(prices, 0.0)

        return prices


def fft_call_price(
    cf,
    S0,
    K,
    T,
    r,
    q=0.0,
    alpha=1.5,
    N=2**14,
    B=200.0,
):
    """Compatibility wrapper for pricing a single European call.

    ``FFTPricer.price_calls`` is the canonical API. Older model code imports
    ``fft_call_price`` directly, so this wrapper keeps that code path working.
    The characteristic function must be for log(S_T / S_0).
    """
    pricer = FFTPricer()
    price = pricer.price_calls(cf, S0, np.asarray([K], dtype=float), T, r, q, alpha, N, B)
    return float(price[0])
