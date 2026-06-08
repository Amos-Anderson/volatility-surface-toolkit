"""Variance Gamma (VG) Lévy model."""

import numpy as np
from volsurf.models.gen2_levy.base_levy import BaseLevyModel


class VGModel(BaseLevyModel):
    r"""Variance Gamma model.

    Brownian motion with drift, time-changed by a Gamma subordinator
    with mean 1 and variance $\nu$ per unit time:

    .. math::
        X_t = \theta G_t + \sigma W_{G_t}, \qquad
        G_t \sim \Gamma\!\left(\frac{t}{\nu},\,\frac{1}{\nu}\right)

    Parameters
    ----------
    sigma : float > 0
        Volatility of the Brownian motion.
    nu : float > 0
        Variance of the Gamma time change (kurtosis controller).
    theta : float
        Drift of the Brownian motion (skewness controller).
    """

    PARAM_NAMES = ["sigma", "nu", "theta"]
    BOUNDS = [(0.05, 1.0), (0.01, 5.0), (-1.0, 1.0)]

    def __init__(self, params=None):
        defaults = {"sigma": 0.2, "nu": 0.5, "theta": -0.1}
        super().__init__(params or defaults)

    def _validate_params(self, params):
        return params["sigma"] > 0 and params["nu"] > 0

    def characteristic_function(self, u, t, params):
        sigma = params["sigma"]
        nu = params["nu"]
        theta = params["theta"]

        # phi(u) = (1 - i*u*theta*nu + 0.5*sigma^2*nu*u^2)^(-t/nu)
        denom = 1.0 - 1j * u * theta * nu + 0.5 * sigma ** 2 * nu * u ** 2
        return np.exp((-t / nu) * np.log(denom))

    def cumulant_generating_function(self, z, params):
        sigma = params["sigma"]
        nu = params["nu"]
        theta = params["theta"]

        val = 1.0 - theta * nu * z - 0.5 * sigma ** 2 * nu * z ** 2
        if np.isscalar(z):
            if val <= 0:
                return float("inf")
        else:
            if np.any(val <= 0):
                return float("inf")
        return -(1.0 / nu) * np.log(val)

    def esscher_bounds(self, params):
        sigma = params["sigma"]
        nu = params["nu"]
        theta = params["theta"]

        # Quadratic: 1 - theta*nu*z - 0.5*sigma^2*nu*z^2
        a = -0.5 * sigma ** 2 * nu
        b = -theta * nu
        c = 1.0
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return (-50.0, 50.0)

        sqrt_disc = np.sqrt(discriminant)
        z1 = (-b - sqrt_disc) / (2 * a)
        z2 = (-b + sqrt_disc) / (2 * a)
        lo = float(min(z1, z2)) + 1e-6
        hi = float(max(z1, z2)) - 1.0 - 1e-6
        return (lo, hi)


__all__ = ["VGModel"]