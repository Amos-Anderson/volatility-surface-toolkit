"""Heston (1993) stochastic volatility model with Carr-Madan FFT pricing."""

try:
    from typing import override
except ImportError:  # pragma: no cover - Python < 3.12 compatibility
    def override(func):
        return func

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from volsurf.models.base import BaseModel
from volsurf.pricing import black_scholes as bs
from volsurf.pricing.fft_pricer import fft_call_price


class HestonModel(BaseModel):
    """Heston stochastic volatility: CIR variance + correlated Brownian motion.

    Dynamics (risk-neutral):
        dS_t = (r-q) S_t dt + sqrt(v_t) S_t dW_t^1
        dv_t = κ(θ - v_t) dt + σ sqrt(v_t) dW_t^2
        corr(dW^1, dW^2) = ρ

    Parameters: v0, κ, θ, σ, ρ
    """

    def __init__(self, v0: float = 0.04, kappa: float = 2.0,
                 theta: float = 0.04, sigma_v: float = 0.3, rho: float = -0.7) -> None:
        self._params = {
            "v0": max(v0, 1e-6),
            "kappa": max(kappa, 1e-6),
            "theta": max(theta, 1e-6),
            "sigma_v": max(sigma_v, 1e-6),
            "rho": max(-0.999, min(rho, 0.999)),
        }

    # ------------------------------------------------------------------
    # Characteristic function — Albrecher et al. (2007) "little trap"
    # ------------------------------------------------------------------
    def _characteristic_function(self, u: np.ndarray, S: float,
                                  T: float, r: float, q: float = 0.0) -> np.ndarray:
        """Characteristic function of log(S_T) under Heston — numerically stable."""
        v0 = self._params["v0"]
        kappa = self._params["kappa"]
        theta = self._params["theta"]
        sigma_v = self._params["sigma_v"]
        rho = self._params["rho"]

        # Complex frequency
        xi = u  # shape (n,)

        # d = sqrt((κ - ρσ i ξ)^2 + σ^2 (ξ^2 + i ξ))
        d_arg = (kappa - rho * sigma_v * 1j * xi)**2 + sigma_v**2 * (xi**2 + 1j * xi)
        d = np.sqrt(d_arg)

        # "Little trap": use g = (κ - ρσ i ξ + d) / (κ - ρσ i ξ - d)
        # This ensures |g| < 1, avoiding branch cut issues
        denom_g = kappa - rho * sigma_v * 1j * xi - d
        # Protect against exact zero
        denom_g = np.where(np.abs(denom_g) < 1e-12, 1e-12, denom_g)
        g = (kappa - rho * sigma_v * 1j * xi + d) / denom_g

        # G = (1 - g e^{d T}) / (1 - g)  — used in C
        exp_dT = np.exp(d * T)
        G = (1.0 - g * exp_dT) / (1.0 - g)

        # D term
        D = (kappa - rho * sigma_v * 1j * xi + d) / sigma_v**2 \
            * (1.0 - exp_dT) / (1.0 - g * exp_dT)

        # C term
        C = (r - q) * 1j * xi * T \
            + kappa * theta / sigma_v**2 \
            * ((kappa - rho * sigma_v * 1j * xi + d) * T - 2.0 * np.log(G))

        # log-characteristic function
        phi = np.exp(C + D * v0 + 1j * xi * np.log(S))
        return phi

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------
    @override
    def price(self, S: float, K: float, T: float, r: float,
              q: float = 0.0, option_type: str = "call") -> float:
        """Heston price via Carr-Madan FFT."""

        def char_fn(u: np.ndarray) -> np.ndarray:
            phi_log_price = self._characteristic_function(u, S, T, r, q)
            return phi_log_price * np.exp(-1j * u * np.log(S))

        call_price = fft_call_price(char_fn, S, K, T, r, q)

        if option_type == "put":
            call_price = max(call_price, 0.0)
            put_price = call_price - S * np.exp(-q * T) + K * np.exp(-r * T)
            return float(max(put_price, 0.0))
        return float(max(call_price, 0.0))

    @override
    def price_points(self, strikes, maturities, spot, r, q, params=None):
        """Price paired call options for array-like strikes and maturities."""
        old_params = dict(self._params)
        if params is not None:
            self._params.update(params)

        strikes = np.asarray(strikes, dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        prices = np.empty(len(strikes), dtype=float)

        try:
            for i, (K_i, T_i) in enumerate(zip(strikes, maturities)):
                prices[i] = self.price(spot, float(K_i), float(T_i), r, q, "call")
        finally:
            if params is not None:
                self._params = old_params

        return prices

    @override
    def implied_vol(self, S: float, K: float, T: float, r: float,
                    market_price: float, q: float = 0.0,
                    option_type: str = "call") -> float:
        """Back out BS implied vol from Heston price."""
        model_price = self.price(S, K, T, r, q, option_type)
        return bs.implied_volatility(S, K, T, r, model_price, q, option_type)

    # ------------------------------------------------------------------
    # Calibration — FAST VERSION with subsampling
    # ------------------------------------------------------------------
    @override
    def calibrate(self, market_data: pd.DataFrame,
                  method: str = "least_squares",
                  n_sample: int = 80) -> dict[str, float]:
        """Calibrate Heston parameters to market option prices.

        Parameters
        ----------
        market_data : pd.DataFrame
            Must have columns: spot, strike, T, risk_free_rate,
            dividend_yield, option_type, midPrice.
        method : str
            "least_squares" (fast, ~10-30 seconds) or
            "differential_evolution" (slow, ~2-5 minutes).
        n_sample : int
            Number of representative quotes to use for calibration.
            Full dataset is subsampled: ATM + 2 OTM puts + 2 OTM calls per maturity.
            Default 80 gives ~8 maturities × 10 strikes.

        Returns
        -------
        dict
            Fitted parameters.
        """
        # ---- Subsample for speed ----
        sampled = self._subsample_quotes(market_data, n_target=n_sample)
        print(f"Calibration subsample: {len(sampled)} quotes (from {len(market_data)} total)")

        # Extract arrays
        spots = sampled["spot"].values
        strikes = sampled["strike"].values
        Ts = sampled["T"].values
        rs = sampled["risk_free_rate"].values
        qs = sampled["dividend_yield"].values
        types = sampled["option_type"].values
        market_prices = sampled["midPrice"].values

        def objective(params: np.ndarray) -> float:
            v0, kappa, theta, sigma_v, rho = params

            # Hard constraints
            if v0 <= 1e-6 or theta <= 1e-6 or sigma_v <= 1e-6 or kappa <= 1e-6:
                return 1e8
            if abs(rho) >= 0.999:
                return 1e8

            # Feller condition soft penalty
            feller = 2 * kappa * theta - sigma_v**2
            feller_penalty = max(0.0, -feller) * 1e4

            self._params["v0"] = v0
            self._params["kappa"] = kappa
            self._params["theta"] = theta
            self._params["sigma_v"] = sigma_v
            self._params["rho"] = rho

            # Weighted SSE: ATM gets weight 1.0, OTM gets lower weight
            sse = 0.0
            for i in range(len(market_prices)):
                try:
                    p = self.price(spots[i], strikes[i], Ts[i], rs[i], qs[i], types[i])
                    # Weight: higher for ATM, lower for deep OTM
                    moneyness = strikes[i] / spots[i]
                    weight = np.exp(-2.0 * (moneyness - 1.0)**2)  # peaks at ATM
                    sse += weight * (p - market_prices[i])**2
                except Exception:
                    return 1e8
            return sse + feller_penalty

        if method == "differential_evolution":
            bounds = [
                (0.001, 0.25),   # v0
                (0.1, 10.0),     # kappa
                (0.001, 0.25),   # theta
                (0.01, 2.0),     # sigma_v
                (-0.99, 0.99),   # rho
            ]
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                tol=1e-5,
                workers=1,  # Windows-safe
                polish=True,
                seed=42,
            )
            opt_params = result.x
        else:
            # Least squares — FAST
            x0 = np.array([
                self._params["v0"],
                self._params["kappa"],
                self._params["theta"],
                self._params["sigma_v"],
                self._params["rho"],
            ])
            bounds = [
                (0.001, 0.25),
                (0.1, 10.0),
                (0.001, 0.25),
                (0.01, 2.0),
                (-0.99, 0.99),
            ]
            result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 200, "ftol": 1e-8})
            opt_params = result.x

        # Store optimal
        self._params["v0"] = float(opt_params[0])
        self._params["kappa"] = float(opt_params[1])
        self._params["theta"] = float(opt_params[2])
        self._params["sigma_v"] = float(opt_params[3])
        self._params["rho"] = float(opt_params[4])

        return dict(self._params)

    def _subsample_quotes(self, df: pd.DataFrame, n_target: int = 80) -> pd.DataFrame:
        """Select representative quotes: ATM + near-OTM per maturity."""
        rows = []
        for T_val, group in df.groupby("T"):
            group = group.copy()
            group["moneyness"] = group["strike"] / group["spot"]
            group = group.sort_values("moneyness")

            # Pick representative strikes
            atm = group.iloc[len(group)//2] if len(group) > 0 else None
            if atm is not None:
                rows.append(atm)

            # OTM puts: 2 deepest
            puts = group[group["option_type"] == "put"].nsmallest(2, "moneyness")
            rows.extend([puts.iloc[i] for i in range(min(2, len(puts))) if i < len(puts)])

            # OTM calls: 2 deepest
            calls = group[group["option_type"] == "call"].nlargest(2, "moneyness")
            rows.extend([calls.iloc[i] for i in range(min(2, len(calls))) if i < len(calls)])

        sampled = pd.DataFrame(rows).drop_duplicates()
        if len(sampled) > n_target:
            # Randomly downsample if too many
            sampled = sampled.sample(n=n_target, random_state=42)
        return sampled

    @property
    @override
    def params(self) -> dict[str, float]:
        return dict(self._params)
