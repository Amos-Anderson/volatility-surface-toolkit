"""Bilateral Gamma Motion (BGM) model."""

import numpy as np
from volsurf.models.gen2_levy.base_levy import BaseLevyModel


class BGMModel(BaseLevyModel):
    r"""Bilateral Gamma Motion.

    A pure-jump Lévy process obtained as the difference of two independent
    Gamma processes:

    .. math::
        X_t = G^+_t - G^-_t

    where $G^+_t$ and $G^-_t$ are independent Gamma processes.  The Lévy
    measure has two (one-sided) Gamma components:

    .. math::
        \nu(dx) = \frac{\alpha_+}{x}e^{-\lambda_+ x}\mathbf{1}_{x>0}dx
                + \frac{\alpha_-}{|x|}e^{-\lambda_- |x|}\mathbf{1}_{x<0}dx .

    Parameters
    ----------
    alpha_plus : float > 0
        Shape parameter of the positive-jump Gamma process.
    lambda_plus : float > 0
        Rate (inverse scale) of the positive-jump Gamma process.
    alpha_minus : float > 0
        Shape parameter of the negative-jump Gamma process.
    lambda_minus : float > 0
        Rate of the negative-jump Gamma process.
    """

    PARAM_NAMES = ["alpha_plus", "lambda_plus", "alpha_minus", "lambda_minus"]
    BOUNDS = [(0.1, 20.0), (1.0, 200.0), (0.1, 20.0), (1.0, 200.0)]

    def __init__(self, params=None):
        defaults = {
            "alpha_plus": 2.0,
            "lambda_plus": 50.0,
            "alpha_minus": 3.0,
            "lambda_minus": 20.0,
        }
        super().__init__(params or defaults)

    def _validate_params(self, params):
        return all(params[k] > 0 for k in self.PARAM_NAMES)

    def characteristic_function(self, u, t, params):
        ap = params["alpha_plus"]
        lp = params["lambda_plus"]
        am = params["alpha_minus"]
        lm = params["lambda_minus"]

        # phi(u) = [ (lp/(lp - iu))^ap * (lm/(lm + iu))^am ]^t
        log_phi = t * (
            ap * np.log(lp / (lp - 1j * u))
            + am * np.log(lm / (lm + 1j * u))
        )
        return np.exp(log_phi)

    def cumulant_generating_function(self, z, params):
        ap = params["alpha_plus"]
        lp = params["lambda_plus"]
        am = params["alpha_minus"]
        lm = params["lambda_minus"]

        # Domain: z < lp and z > -lm
        if np.isscalar(z):
            if z >= lp or z <= -lm:
                return np.inf
        else:
            if np.any(z >= lp) or np.any(z <= -lm):
                return np.inf

        val = -ap * np.log(1.0 - z / lp) - am * np.log(1.0 + z / lm)
        return val

    def esscher_bounds(self, params):
        lp = params["lambda_plus"]
        lm = params["lambda_minus"]
        lower = float(-lm) + 1e-6
        upper = float(lp) - 1.0 - 1e-6
        return (lower, upper)


__all__ = ["BGMModel"]