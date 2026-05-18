"""Analytic pricing formulae for the W3 Monte Carlo pipeline.

The functions defined here provide the benchmark prices and the implied
volatility inversion used throughout the project:

* :func:`black_scholes_call` is the textbook Black--Scholes European call price;
* :func:`merton_exact_call` evaluates the Poisson-weighted Black--Scholes sum
  derived in W2;
* :func:`implied_volatility` inverts a call price to the Black--Scholes volatility
  with Brent's method, which is convenient for the implied-volatility smile.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm

from config import MarketParams


def black_scholes_call(s: float, k: float, t: float, r: float, sigma: float) -> float:
    """Return the Black--Scholes European call price."""
    if t <= 0.0:
        return max(s - k, 0.0)
    if sigma <= 0.0:
        forward_intrinsic = s - k * math.exp(-r * t)
        return max(forward_intrinsic, 0.0)

    vol_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2))


def merton_exact_call(
    params: MarketParams,
    strike: float | None = None,
    n_terms: int = 150,
) -> float:
    """Return Merton's Poisson-weighted Black--Scholes call price.

    C = sum_n Pois(n; lambda_prime T) * BS(S, K, T, r_n, sigma_n), where
    r_n and sigma_n follow directly from conditioning on the number of jumps.
    """
    k = params.strike if strike is None else strike
    t = params.maturity
    lam_t = params.lambda_prime * t
    price = 0.0

    for n in range(n_terms):
        if lam_t > 0.0:
            log_weight = -lam_t + n * math.log(lam_t) - math.lgamma(n + 1)
            weight = math.exp(log_weight)
        else:
            weight = 1.0 if n == 0 else 0.0

        if n > 5 and weight < 1e-16:
            break

        r_n = (
            params.rate
            - params.jump_intensity * params.kappa
            + n * params.jump_mean / t
            + n * params.jump_vol**2 / (2.0 * t)
        )
        sigma_n = math.sqrt(params.sigma**2 + n * params.jump_vol**2 / t)
        price += weight * black_scholes_call(params.s0, k, t, r_n, sigma_n)

    return float(price)


def implied_volatility(price: float, s: float, k: float, t: float, r: float) -> float:
    """Invert a call price to Black--Scholes implied volatility."""
    lower_bound = max(s - k * math.exp(-r * t), 0.0)
    if price < lower_bound - 1e-10:
        return float("nan")

    def objective(vol: float) -> float:
        return black_scholes_call(s, k, t, r, vol) - price

    try:
        return float(brentq(objective, 1e-6, 5.0, xtol=1e-10, maxiter=200))
    except ValueError:
        return float("nan")
