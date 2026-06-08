"""Normal Inverse Gaussian (NIG) Lévy model."""

import numpy as np
from volsurf.models.gen2_levy.base_levy import BaseLevyModel


class NIGModel(BaseLevyModel):
    r"""Normal Inverse Gaussian model.

    The NIG process is obtained by subordinating a Brownian motion
    with drift by an *Inverse Gaussian* subordinator $T_t$:

    .. math::
        L_t = \theta T_t + \sigma W_{T_t}

    where $T_t \sim \text{IG}(\delta t, \sqrt{\alpha^2-\beta^2})$.
    The resulting marginal distribution is NIG$(\alpha,\beta,\delta t,\mu t)$.

    Parameters
    ----------
    alpha : float > 0
        Steepness (tail decay rate).
    beta : float, |beta| + 0.5 < alpha
        Asymmetry (skewness).  Kept away from alpha for numerical stability.
    delta : float > 0
        Scale.
    mu : float
        Location (drift).
    """

    PARAM_NAMES = ["alpha", "beta", "delta", "mu"]
    BOUNDS = [(2.0, 25.0), (-8.0, 8.0), (0.01, 5.0), (-0.5, 0.5)]

    def __init__(self, params=None):
        defaults = {"alpha": 8.0, "beta": -2.0, "delta": 0.5, "mu": 0.03}
        super().__init__(params or defaults)

    def _validate_params(self, params):
        alpha = params["alpha"]
        beta = params["beta"]
        delta = params["delta"]
        # Keep |beta| well away from alpha to avoid numerical blow-up
        # in sqrt(alpha^2 - (beta+iu)^2) and to ensure a healthy Esscher domain.
        return (
            alpha > 0
            and delta > 0
            and abs(beta) + 0.5 < alpha
        )

    def characteristic_function(self, u, t, params):
        alpha = params["alpha"]
        beta = params["beta"]
        delta = params["delta"]
        mu = params["mu"]

        iu = 1j * u
        gamma = np.sqrt(alpha ** 2 - beta ** 2)

        z = alpha ** 2 - (beta + iu) ** 2
        sqrt_z = np.sqrt(z)
        sqrt_z = np.where(np.real(sqrt_z) < 0, -sqrt_z, sqrt_z)

        return np.exp(iu * mu * t + delta * t * (gamma - sqrt_z))

    def cumulant_generating_function(self, z, params):
        alpha = params["alpha"]
        beta = params["beta"]
        delta = params["delta"]
        mu = params["mu"]

        gamma = np.sqrt(alpha ** 2 - beta ** 2)
        val = alpha ** 2 - (beta + z) ** 2
        sqrt_val = np.sqrt(val)
        if np.isscalar(z):
            if np.real(sqrt_val) < 0:
                sqrt_val = -sqrt_val
        else:
            sqrt_val = np.where(np.real(sqrt_val) < 0, -sqrt_val, sqrt_val)

        return mu * z + delta * (gamma - sqrt_val)

    def esscher_bounds(self, params):
        alpha = params["alpha"]
        beta = params["beta"]
        # Domain for real z: |beta + z| < alpha
        lower = -alpha - beta + 1e-6
        upper = alpha - beta - 1 - 1e-6
        return (float(lower), float(upper))


__all__ = ["NIGModel"]