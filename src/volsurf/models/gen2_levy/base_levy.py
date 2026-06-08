"""Base class for exponential Lévy models with Esscher transform.

Every subclass must implement:
    characteristic_function(u, t, params)
    cumulant_generating_function(z, params)
    esscher_bounds(params)          [optional]
    _validate_params(params)        [optional]
"""

import numpy as np
from scipy.optimize import brentq, differential_evolution, minimize, minimize_scalar

from volsurf.models.base import BaseModel
from volsurf.pricing.fft_pricer import FFTPricer


class BaseLevyModel(BaseModel):
    """Abstract base for exponential Lévy models.

    Risk-neutralisation is performed via the Esscher transform.
    Pricing is done via Carr-Madan FFT.
    """

    PARAM_NAMES: list[str] = []
    BOUNDS: list[tuple[float, float]] = []

    def __init__(self, params=None):
        super().__init__(params or {})

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def characteristic_function(self, u, t, params):
        """Characteristic function E_P[e^{iuX_t}] under the physical measure."""
        raise NotImplementedError

    def cumulant_generating_function(self, z, params):
        """CGF K(z) = log E_P[e^{zX_1}] for *one* unit of time."""
        raise NotImplementedError

    def esscher_bounds(self, params):
        """Return (lower, upper) for the Esscher parameter theta."""
        return (-50.0, 50.0)

    def _validate_params(self, params):
        """Return True if parameters are inside the model domain."""
        return True

    # ------------------------------------------------------------------
    # Esscher transform
    # ------------------------------------------------------------------

    def solve_esscher(self, r, q, params):
        """Solve K(theta+1) - K(theta) = r - q for the Esscher parameter theta.

        If an exact root exists in the domain, returns it via Brent's method.
        Otherwise minimises |K(theta+1)-K(theta)-(r-q)| on the domain and
        returns the best feasible theta.  The caller should check validity
        via ``_esscher_valid``.
        """
        target = r - q
        lower, upper = self.esscher_bounds(params)

        def g_diff(theta):
            return (
                self.cumulant_generating_function(theta + 1.0, params)
                - self.cumulant_generating_function(theta, params)
            )

        def f(theta):
            return g_diff(theta) - target

        # --- 1. Try Brent's method (exact root) ---
        for _ in range(6):
            if lower >= upper:
                break
            try:
                fl = float(f(lower))
                fu = float(f(upper))
                if np.isfinite(fl) and np.isfinite(fu) and fl * fu <= 0:
                    return float(brentq(f, lower, upper))
            except Exception:
                pass
            lower -= 10.0
            upper += 10.0

        # --- 2. No exact root – minimise |f(theta)| on the domain ---
        try:
            lo = max(lower, self.esscher_bounds(params)[0])
            hi = min(upper, self.esscher_bounds(params)[1])
            if lo < hi:
                res = minimize_scalar(
                    lambda t: abs(f(t)), bounds=(lo, hi), method="bounded"
                )
                if res.success and np.isfinite(res.fun):
                    return float(res.x)
        except Exception:
            pass

        # --- 3. Ultimate fallback: boundary closest to target ---
        try:
            bl = self.esscher_bounds(params)[0]
            bh = self.esscher_bounds(params)[1]
            if bl < bh:
                vl = abs(f(bl))
                vh = abs(f(bh))
                return float(bl if vl <= vh else bh)
        except Exception:
            pass

        return 0.0

    def _esscher_valid(self, r, q, params, tol=0.01):
        """Check whether the Esscher solution satisfies the martingale
        constraint to within *tol* (absolute tolerance on the log-ratio).

        A tolerance of 0.01 means the no-arbitrage condition is satisfied
        to within ~1%, which is sufficient for stable FFT pricing.
        """
        try:
            theta = self.solve_esscher(r, q, params)
            lhs = (
                self.cumulant_generating_function(theta + 1.0, params)
                - self.cumulant_generating_function(theta, params)
            )
            if not np.isfinite(lhs):
                return False
            return abs(lhs - (r - q)) <= tol
        except Exception:
            return False

    def characteristic_function_rn(self, u, t, r, q, params):
        """Risk-neutral characteristic function of log(S_T/S_0).

        Under Q:
            log(S_T/S_0) = (r-q)T + X_T^Q
        where X_T^Q is the Esscher-transformed Lévy process with
        E_Q[e^{X_1^Q}] = 1.
        """
        theta = self.solve_esscher(r, q, params)
        phi_P = self.characteristic_function

        denom = phi_P(-1j * theta, t, params)
        if np.ndim(denom) == 0:
            if abs(denom) < 1e-16:
                denom = 1e-16
        else:
            denom = np.where(np.abs(denom) < 1e-16, 1e-16, denom)

        numer = phi_P(u - 1j * theta, t, params)
        phi_Q = numer / denom

        # CF of log(S_T/S_0) — Esscher transform already embeds the RN drift,
        # so no additional exp(iu(r-q)T) factor is needed.
        return phi_Q

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price_points(
        self, strikes, maturities, spot, r, q, params=None, alpha_cm=1.5
    ):
        """Price European calls for paired (strike, maturity) rows."""
        params = params or self.params
        if not self._validate_params(params):
            return None
        if not self._esscher_valid(r, q, params):
            return None

        pricer = FFTPricer()
        strikes = np.asarray(strikes)
        maturities = np.asarray(maturities)
        prices = np.zeros(len(strikes))

        unique_T = np.unique(maturities)
        for T in unique_T:
            mask = maturities == T
            if not np.any(mask):
                continue
            K_arr = strikes[mask]

            def cf(u):
                return self.characteristic_function_rn(u, T, r, q, params)

            try:
                T_prices = pricer.price_calls(cf, spot, K_arr, T, r, q, alpha=alpha_cm)
                prices[mask] = T_prices
            except Exception:
                prices[mask] = np.nan

        return prices

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _params_to_x(self, params):
        return np.array([params[k] for k in self.PARAM_NAMES])

    def _x_to_params(self, x):
        return dict(zip(self.PARAM_NAMES, x))

    def calibrate(
        self,
        market_df,
        spot,
        r,
        q,
        strategy="de",
        subsample=None,
        de_popsize=10,
        de_maxiter=80,
        de_tol=0.01,
        polish=True,
        seed=None,
    ):
        """Calibrate model parameters to market option prices.

        Differential evolution is the default strategy because the Lévy
        calibration landscape is highly non-convex.  After DE finds a
        promising basin, an optional L-BFGS-B polish step refines the
        solution.

        Parameters
        ----------
        market_df : DataFrame
            Must contain columns: strike, maturity, mid_price
        spot : float
        r, q : float
            Risk-free rate and dividend yield.
        strategy : {"de", "lbfgsb"}
            "de"   = differential evolution (global, robust, slower)
            "lbfgsb" = L-BFGS-B local search from current params (fast,
                       may get stuck).
        subsample : int or None
            If given, randomly subsample this many quotes for speed.
        de_popsize : int
            DE population per parameter (total pop = popsize * n_params).
        de_maxiter : int
            Maximum DE generations.
        de_tol : float
            DE convergence tolerance (relative change in best objective).
        polish : bool
            If True, run L-BFGS-B from the DE best after DE finishes.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        OptimizeResult
        """
        strikes_all = market_df["strike"].values
        mats_all = market_df["maturity"].values
        prices_all = market_df["mid_price"].values

        # Subsample if requested
        if subsample is not None and len(market_df) > subsample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(market_df), subsample, replace=False)
            strikes = strikes_all[idx]
            maturities = mats_all[idx]
            market_prices = prices_all[idx]
        else:
            strikes, maturities, market_prices = strikes_all, mats_all, prices_all

        bounds = self.BOUNDS

        def objective(x):
            params = self._x_to_params(x)
            if not self._validate_params(params):
                return 1e10
            if not self._esscher_valid(r, q, params):
                return 1e10
            try:
                model_prices = self.price_points(
                    strikes, maturities, spot, r, q, params
                )
                if model_prices is None or np.any(np.isnan(model_prices)):
                    return 1e10
            except Exception:
                return 1e10
            eps = 1e-8
            rel_err = np.abs(model_prices - market_prices) / (market_prices + eps)
            return float(np.mean(rel_err) * 100.0)

        x0 = self._params_to_x(self.params)
        f0 = objective(x0)

        if strategy == "de":
            result = differential_evolution(
                objective,
                bounds,
                popsize=de_popsize,
                maxiter=de_maxiter,
                tol=de_tol,
                polish=False,           # we polish manually below
                seed=seed,
                workers=1,              # local function -> single thread
            )
            # Optional L-BFGS-B polish
            if polish:
                result_polish = minimize(
                    objective,
                    result.x,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 200, "ftol": 1e-10},
                )
                if result_polish.fun < result.fun:
                    result.x = result_polish.x
                    result.fun = result_polish.fun
            if result.fun < f0:
                self.params = self._x_to_params(result.x)
        else:
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 400, "ftol": 1e-8},
            )
            if result.success or result.fun < f0:
                self.params = self._x_to_params(result.x)

        # Store full-sample MAPE in result for convenience
        result.full_sample_mape = None
        if self.params:
            try:
                full_prices = self.price_points(strikes_all, mats_all, spot, r, q)
                if full_prices is not None and not np.any(np.isnan(full_prices)):
                    rel = np.abs(full_prices - prices_all) / (prices_all + 1e-8)
                    result.full_sample_mape = float(np.mean(rel) * 100.0)
            except Exception:
                pass

        return result