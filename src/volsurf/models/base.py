"""BaseModel with required interface."""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all option pricing models.

    Subclasses must implement:
        - price_points(strikes, maturities, spot, r, q, params=None)
        - calibrate(market_df, spot, r, q, ...)
    """

    PARAM_NAMES: list[str] = []
    BOUNDS: list[tuple[float, float]] = []

    def __init__(self, params=None):
        self.params = params or {}

    @abstractmethod
    def price_points(self, strikes, maturities, spot, r, q, params=None):
        """Return 1-D array of call prices for paired (strike, maturity) rows."""
        raise NotImplementedError

    def price(self, strikes, maturities, spot, r, q, params=None):
        """Grid pricing: returns 2-D array of shape (n_maturities, n_strikes)."""
        params = params or self.params
        prices = []
        for T in maturities:
            prices.append(
                self.price_points(strikes, [T] * len(strikes), spot, r, q, params)
            )
        return prices

    def implied_vol(self, market_prices, strikes, maturities, spot, r, q, params=None):
        """Invert Black-Scholes on model prices to get model-implied vols."""
        from volsurf.pricing import black_scholes as bs

        model_prices = self.price_points(strikes, maturities, spot, r, q, params)
        ivs = []
        for mp, K, T in zip(model_prices, strikes, maturities):
            try:
                iv = bs.implied_volatility(
                    spot, K, T, r, float(mp), q, option_type="call", bounds=(1e-6, 5.0)
                )
                ivs.append(iv)
            except Exception:
                ivs.append(float("nan"))
        return ivs

    @abstractmethod
    def calibrate(self, market_df, spot, r, q, **kwargs):
        """Fit model parameters to market option prices."""
        raise NotImplementedError