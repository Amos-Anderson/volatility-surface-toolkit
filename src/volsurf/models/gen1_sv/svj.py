"""Bates (1996) Stochastic Volatility with Jumps (SVJ) — Heston + compound Poisson."""

try:
    from typing import override
except ImportError:  # pragma: no cover - Python < 3.12 compatibility
    def override(func):
        return func

import numpy as np
from scipy.optimize import minimize

from volsurf.models.gen1_sv.heston import HestonModel


class SVJModel(HestonModel):
    """Bates SVJ: Heston stochastic volatility + log-normal compound Poisson jumps.

    Additional dynamics:
        dS_t/S_t = ... + (e^J - 1) dN_t
    where N_t is Poisson with intensity λ, and J ~ N(μ_J, σ_J²).

    The risk-neutral drift adjustment includes the jump compensator.
    """

    def __init__(self,
                 v0: float = 0.04, kappa: float = 2.0,
                 theta: float = 0.04, sigma_v: float = 0.3, rho: float = -0.7,
                 lam: float = 0.5, mu_j: float = -0.1, sigma_j: float = 0.2) -> None:
        super().__init__(v0, kappa, theta, sigma_v, rho)
        self._params["lam"] = max(lam, 1e-6)      # jump arrival rate
        self._params["mu_j"] = mu_j                # mean log-jump size
        self._params["sigma_j"] = max(sigma_j, 1e-6)  # jump size vol

    @override
    def _characteristic_function(self, u: np.ndarray, S: float,
                                  T: float, r: float, q: float = 0.0) -> np.ndarray:
        """SVJ characteristic function = Heston CF × Merton jump CF."""

        # Heston part (inherited, already stable)
        phi_heston = super()._characteristic_function(u, S, T, r, q)

        # Jump parameters
        lam = self._params["lam"]
        mu_j = self._params["mu_j"]
        sigma_j = self._params["sigma_j"]

        # Merton jump characteristic function
        # φ_jump(u) = exp( λT [ (e^{μ_J + σ_J²/2})^{iu} e^{-iu(μ_J+σ_J²/2)} - 1 ] )
        # Simplified: φ_jump(u) = exp( λT [ exp(iuμ_J - u²σ_J²/2 + iuσ_J²/2) - 1 ] )
        # Actually the standard form is:
        # φ_jump(u) = exp( λT [ exp(iuμ_J - σ_J²u²/2) - 1 ] )
        # with drift compensator already in the risk-neutral rate
        
        # Correct risk-neutral form with drift adjustment:
        # The jump CF in the risk-neutral measure is:
        # exp( λT [ ψ(u) - 1 - iu ψ'(0) ] )
        # where ψ(u) = exp(iuμ_J - σ_J²u²/2)
        
        psi = np.exp(1j * u * mu_j - 0.5 * sigma_j**2 * u**2)
        psi_prime_0 = 1j * mu_j  # derivative at 0
        
        # With compensator (ensures martingale)
        phi_jump = np.exp(lam * T * (psi - 1.0 - 1j * u * (np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)))
        
        # Alternative simpler form (also valid with risk-neutral drift):
        # phi_jump = np.exp(lam * T * (np.exp(1j*u*mu_j - 0.5*sigma_j**2*u**2) - 1.0))
        
        return phi_heston * phi_jump

    def calibrate_jumps_only(self, market_data, heston_params=None):
        """Calibrate ONLY jump parameters, with Heston params frozen.
        
        Parameters
        ----------
        market_data : pd.DataFrame
            Same format as Heston calibrate.
        heston_params : dict or None
            If provided, freeze these Heston params. If None, use current self._params.
        """
        import pandas as pd
        
        if heston_params:
            for k in ["v0", "kappa", "theta", "sigma_v", "rho"]:
                self._params[k] = heston_params[k]
        
        # Subsample
        sampled = self._subsample_quotes(market_data, n_target=80)
        spots = sampled["spot"].values
        strikes = sampled["strike"].values
        Ts = sampled["T"].values
        rs = sampled["risk_free_rate"].values
        qs = sampled["dividend_yield"].values
        types = sampled["option_type"].values
        market_prices = sampled["midPrice"].values

        def objective(params):
            lam, mu_j, sigma_j = params
            if lam <= 0 or sigma_j <= 0:
                return 1e8
            
            self._params["lam"] = lam
            self._params["mu_j"] = mu_j
            self._params["sigma_j"] = sigma_j
            
            sse = 0.0
            for i in range(len(market_prices)):
                try:
                    p = self.price(spots[i], strikes[i], Ts[i], rs[i], qs[i], types[i])
                    moneyness = strikes[i] / spots[i]
                    # Weight deep OTM puts MORE heavily (that's what jumps fix)
                    weight = 1.0 + 3.0 * max(0, 0.98 - moneyness)  # boost puts
                    sse += weight * (p - market_prices[i])**2
                except Exception:
                    return 1e8
            return sse

        # Grid search for robustness
        best_sse = 1e10
        best_params = [0.5, -0.15, 0.3]
        
        # Coarse grid
        for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
            for mu_j in [-0.3, -0.2, -0.1, -0.05]:
                for sigma_j in [0.1, 0.2, 0.3, 0.5]:
                    sse = objective([lam, mu_j, sigma_j])
                    if sse < best_sse:
                        best_sse = sse
                        best_params = [lam, mu_j, sigma_j]
        
        # Fine polish with local optimizer
        result = minimize(objective, best_params, method="L-BFGS-B",
                          bounds=[(0.01, 10.0), (-0.5, 0.0), (0.01, 1.0)])
        
        self._params["lam"] = float(result.x[0])
        self._params["mu_j"] = float(result.x[1])
        self._params["sigma_j"] = float(result.x[2])
        
        return dict(self._params)
