"""Black-Scholes analytical pricing, Greeks, and implied volatility."""
from typing import Literal

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    sig_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sig_sqrt_t
    d2 = d1 - sig_sqrt_t
    return float(d1), float(d2)


def price(S, K, T, r, sigma, q=0.0, option_type="call"):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc = np.exp(-q * T)
    r_disc = np.exp(-r * T)
    if option_type == "call":
        return float(S * disc * norm.cdf(d1) - K * r_disc * norm.cdf(d2))
    return float(K * r_disc * norm.cdf(-d2) - S * disc * norm.cdf(-d1))


def vega(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def delta(S, K, T, r, sigma, q=0.0, option_type="call"):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return float(np.exp(-q * T) * norm.cdf(d1))
    return float(-np.exp(-q * T) * norm.cdf(-d1))


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def theta(S, K, T, r, sigma, q=0.0, option_type="call"):
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
    return float(term1 + term2 + term3) / 365.0


def implied_volatility(S, K, T, r, market_price, q=0.0, option_type="call",
                       tol=1e-8, max_iter=100, bounds=(1e-6, 5.0)):
    # Arbitrage bounds check
    if option_type == "call":
        lower = max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
        upper = S * np.exp(-q * T)
    else:
        lower = max(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))
        upper = K * np.exp(-r * T)

    if market_price < lower - 1e-10 or market_price > upper + 1e-10:
        raise ValueError(f"Price {market_price} outside bounds [{lower:.4f}, {upper:.4f}]")

    if market_price <= lower + 1e-12:
        return float(bounds[0])

    # Newton-Raphson
    sigma = 0.3
    for _ in range(max_iter):
        p = price(S, K, T, r, sigma, q, option_type)
        v = vega(S, K, T, r, sigma, q)
        diff = p - market_price
        if abs(diff) < tol:
            return float(sigma)
        if v < 1e-12:
            break
        sigma = max(bounds[0], min(bounds[1], sigma - diff / v))

    # Bisection fallback
    lo, hi = bounds
    p_lo = price(S, K, T, r, lo, q, option_type)
    p_hi = price(S, K, T, r, hi, q, option_type)
    if (p_lo - market_price) * (p_hi - market_price) > 0:
        while p_hi < market_price and hi < 10.0:
            hi *= 2.0
            p_hi = price(S, K, T, r, hi, q, option_type)

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        p_mid = price(S, K, T, r, mid, q, option_type)
        if abs(p_mid - market_price) < tol or (hi - lo) / 2.0 < tol:
            return float(mid)
        if (p_mid - market_price) * (p_lo - market_price) <= 0:
            hi, p_hi = mid, p_mid
        else:
            lo, p_lo = mid, p_mid

    raise ValueError(f"IV solver failed: K={K}, T={T}, price={market_price}")