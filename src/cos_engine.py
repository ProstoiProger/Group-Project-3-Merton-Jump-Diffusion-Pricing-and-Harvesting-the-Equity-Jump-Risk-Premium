"""COS / Fourier pricing engine for Merton and Kou jump-diffusion models (W4).

Implements the COS method of Fang & Oosterlee (2008) with characteristic
functions for both the Merton (1976) normal-jump and Kou (2002) double-
exponential models.  All model parameters are injected via config objects;
no values are hard-coded.

Supported payoffs
-----------------
- European call / put (vanilla)
- Cash-or-nothing call / put  (digital)
- Asset-or-nothing call / put (digital)

Validation criteria produced by write_outputs()
-------------------------------------------------
  2a  Spectral convergence: |COS - Merton exact| < 10^-8 at N ≥ 256
  2b  COS within Monte Carlo ±2σ band
  3   Merton vs Kou implied-vol smile comparison
  4   All four digital option types across strikes

Entry point
-----------
    from src.cos_engine import write_outputs
    write_outputs(params, kou_params, run_config, output_dir)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from .config import KouParams, MertonParams, RunConfig

LOGGER = logging.getLogger(__name__)


# ── characteristic functions ─────────────────────────────────────────────────

def merton_cf(
    u: np.ndarray,
    s0: float, r: float, t: float,
    sigma: float, lam: float, mu_j: float, sigma_j: float,
) -> np.ndarray:
    """Characteristic function of log(S_T) under the Merton jump-diffusion.

    Jumps: log-jump sizes ~ Normal(mu_j, sigma_j^2).
    Martingale correction: mu_bar = exp(mu_j + 0.5*sigma_j^2) - 1.
    """
    x0 = np.log(s0)
    mu_bar = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    drift = 1j * u * (r - 0.5 * sigma ** 2 - lam * mu_bar)
    diffusion = -0.5 * sigma ** 2 * u ** 2
    jumps = lam * (np.exp(1j * u * mu_j - 0.5 * sigma_j ** 2 * u ** 2) - 1.0)
    return np.exp(1j * u * x0 + (drift + diffusion + jumps) * t)


def kou_cf(
    u: np.ndarray,
    s0: float, r: float, t: float,
    sigma: float, lam: float,
    p_up: float, eta1: float, eta2: float,
) -> np.ndarray:
    """Characteristic function of log(S_T) under the Kou double-exponential model.

    Up-jump (prob p_up):   Y ~ Exp(eta1),   mean 1/eta1
    Down-jump (1-p_up):    Y ~ Exp(eta2),   mean 1/eta2 (negative side)
    Martingale correction: kappa = lam*(p_up*eta1/(eta1-1) + (1-p_up)*eta2/(eta2+1) - 1)
    """
    x0 = np.log(s0)
    kappa = lam * (p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1)
    drift = 1j * u * (r - 0.5 * sigma ** 2 - kappa)
    diffusion = -0.5 * sigma ** 2 * u ** 2
    m_u = p_up * eta1 / (eta1 - 1j * u) + (1 - p_up) * eta2 / (eta2 + 1j * u)
    jumps = lam * (m_u - 1.0)
    return np.exp(1j * u * x0 + (drift + diffusion + jumps) * t)


# ── cumulants for domain truncation ──────────────────────────────────────────

def merton_cumulants(
    s0: float, r: float, t: float,
    sigma: float, lam: float, mu_j: float, sigma_j: float,
) -> tuple[float, float, float]:
    """First, second, and fourth cumulants of log(S_T) under Merton."""
    mu_bar = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    c1 = np.log(s0) + (r - 0.5 * sigma ** 2 - lam * mu_bar) * t + lam * t * mu_j
    c2 = sigma ** 2 * t + lam * t * (mu_j ** 2 + sigma_j ** 2)
    c4 = lam * t * (mu_j ** 4 + 6 * mu_j ** 2 * sigma_j ** 2 + 3 * sigma_j ** 4)
    return c1, c2, c4


def kou_cumulants(
    s0: float, r: float, t: float,
    sigma: float, lam: float,
    p_up: float, eta1: float, eta2: float,
) -> tuple[float, float, float]:
    """First, second, and fourth cumulants of log(S_T) under Kou."""
    kappa = lam * (p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1)
    c1 = np.log(s0) + (r - 0.5 * sigma ** 2 - kappa) * t + lam * t * (p_up / eta1 - (1 - p_up) / eta2)
    c2 = sigma ** 2 * t + lam * t * (2 * p_up / eta1 ** 2 + 2 * (1 - p_up) / eta2 ** 2)
    c4 = lam * t * (24 * p_up / eta1 ** 4 + 24 * (1 - p_up) / eta2 ** 4)
    return c1, c2, c4


def _cos_bounds(cumulants: tuple[float, float, float], L: int = 10) -> tuple[float, float]:
    c1, c2, c4 = cumulants
    a = c1 - L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    b = c1 + L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    return a, b


# ── COS payoff integrals ──────────────────────────────────────────────────────

def _chi_k(k: int, a: float, b: float, c: float, d: float) -> float:
    """Integral of exp(x)*cos(k*pi*(x-a)/(b-a)) over [c, d]."""
    k_pi = k * np.pi / (b - a)
    expr1 = np.cos(k_pi * (d - a)) * np.exp(d) - np.cos(k_pi * (c - a)) * np.exp(c)
    expr2 = k_pi * np.sin(k_pi * (d - a)) * np.exp(d) - k_pi * np.sin(k_pi * (c - a)) * np.exp(c)
    return (expr1 + expr2) / (1.0 + k_pi ** 2)


def _psi_k(k: int, a: float, b: float, c: float, d: float) -> float:
    """Integral of cos(k*pi*(x-a)/(b-a)) over [c, d]."""
    if k == 0:
        return d - c
    k_pi = k * np.pi / (b - a)
    return (np.sin(k_pi * (d - a)) - np.sin(k_pi * (c - a))) / k_pi


# ── generic COS pricers ───────────────────────────────────────────────────────

def cos_price_european(
    phi_func,
    s0: float, k: float, r: float, t: float,
    option_type: str = "call",
    n: int = 512,
    l: int = 10,
    cumulants: tuple | None = None,
) -> float:
    """COS method pricer for European calls and puts.

    Parameters
    ----------
    phi_func:
        Characteristic function phi(u) of log(S_T).
    option_type:
        ``'call'`` or ``'put'``.
    n:
        Number of COS terms.
    l:
        Domain truncation multiplier.
    cumulants:
        (c1, c2, c4) tuple; if None a wide fallback is used.
    """
    if cumulants is None:
        x0 = np.log(s0)
        a, b = x0 - l * 0.5, x0 + l * 0.5
    else:
        a, b = _cos_bounds(cumulants, l)

    c = np.log(k)
    ks = np.arange(0, n)
    u = ks * np.pi / (b - a)
    phi = phi_func(u)

    if option_type == "call":
        chi_vals = np.array([_chi_k(ki, a, b, c, b) for ki in ks])
        psi_vals = np.array([_psi_k(ki, a, b, c, b) for ki in ks])
        U_k = (2.0 / (b - a)) * (chi_vals - k * psi_vals)
    else:
        chi_vals = np.array([_chi_k(ki, a, b, a, c) for ki in ks])
        psi_vals = np.array([_psi_k(ki, a, b, a, c) for ki in ks])
        U_k = (2.0 / (b - a)) * (-chi_vals + k * psi_vals)

    inner = np.real(phi * np.exp(-1j * u * a)) * U_k
    inner[0] *= 0.5
    return max(0.0, float(np.exp(-r * t) * np.sum(inner)))


def cos_price_digital(
    phi_func,
    s0: float, k: float, r: float, t: float,
    digital_type: str = "cash_call",
    n: int = 512,
    l: int = 10,
    cumulants: tuple | None = None,
) -> float:
    """COS method pricer for digital (binary) options.

    digital_type
    ------------
    ``'cash_call'``  : pays $1 if S_T > K
    ``'cash_put'``   : pays $1 if S_T < K
    ``'asset_call'`` : pays S_T if S_T > K
    ``'asset_put'``  : pays S_T if S_T < K
    """
    if cumulants is None:
        x0 = np.log(s0)
        a, b = x0 - l * 0.5, x0 + l * 0.5
    else:
        a, b = _cos_bounds(cumulants, l)

    c = np.log(k)
    ks = np.arange(0, n)
    u = ks * np.pi / (b - a)
    phi = phi_func(u)

    if digital_type == "cash_call":
        coeff = np.array([_psi_k(ki, a, b, c, b) for ki in ks])
    elif digital_type == "cash_put":
        coeff = np.array([_psi_k(ki, a, b, a, c) for ki in ks])
    elif digital_type == "asset_call":
        coeff = np.array([_chi_k(ki, a, b, c, b) for ki in ks])
    elif digital_type == "asset_put":
        coeff = np.array([_chi_k(ki, a, b, a, c) for ki in ks])
    else:
        raise ValueError(f"Unknown digital_type: {digital_type!r}")

    U_k = (2.0 / (b - a)) * coeff
    inner = np.real(phi * np.exp(-1j * u * a)) * U_k
    inner[0] *= 0.5
    return max(0.0, float(np.exp(-r * t) * np.sum(inner)))


# ── model-specific convenience wrappers ──────────────────────────────────────

def merton_call(
    params: MertonParams, strike: float | None = None,
    n: int = 512, l: int = 10,
) -> float:
    """COS call price under the Merton model."""
    k = strike if strike is not None else params.strike
    cum = merton_cumulants(params.s0, params.rate, params.maturity,
                           params.sigma, params.lam, params.mu_j, params.sigma_j)
    phi = lambda u: merton_cf(u, params.s0, params.rate, params.maturity,
                               params.sigma, params.lam, params.mu_j, params.sigma_j)
    return cos_price_european(phi, params.s0, k, params.rate, params.maturity, "call", n, l, cum)


def merton_put(
    params: MertonParams, strike: float | None = None,
    n: int = 512, l: int = 10,
) -> float:
    """COS put price under the Merton model."""
    k = strike if strike is not None else params.strike
    cum = merton_cumulants(params.s0, params.rate, params.maturity,
                           params.sigma, params.lam, params.mu_j, params.sigma_j)
    phi = lambda u: merton_cf(u, params.s0, params.rate, params.maturity,
                               params.sigma, params.lam, params.mu_j, params.sigma_j)
    return cos_price_european(phi, params.s0, k, params.rate, params.maturity, "put", n, l, cum)


def merton_digital(
    params: MertonParams, strike: float | None = None,
    digital_type: str = "cash_call",
    n: int = 512, l: int = 10,
) -> float:
    """COS digital option price under the Merton model."""
    k = strike if strike is not None else params.strike
    cum = merton_cumulants(params.s0, params.rate, params.maturity,
                           params.sigma, params.lam, params.mu_j, params.sigma_j)
    phi = lambda u: merton_cf(u, params.s0, params.rate, params.maturity,
                               params.sigma, params.lam, params.mu_j, params.sigma_j)
    return cos_price_digital(phi, params.s0, k, params.rate, params.maturity, digital_type, n, l, cum)


def kou_call(
    params: MertonParams, kou: KouParams, strike: float | None = None,
    n: int = 512, l: int = 10,
) -> float:
    """COS call price under the Kou double-exponential model."""
    k = strike if strike is not None else params.strike
    cum = kou_cumulants(params.s0, params.rate, params.maturity,
                        params.sigma, params.lam, kou.p_up, kou.eta1, kou.eta2)
    phi = lambda u: kou_cf(u, params.s0, params.rate, params.maturity,
                            params.sigma, params.lam, kou.p_up, kou.eta1, kou.eta2)
    return cos_price_european(phi, params.s0, k, params.rate, params.maturity, "call", n, l, cum)


# ── analytic benchmark ────────────────────────────────────────────────────────

def merton_exact_call(params: MertonParams, strike: float | None = None, n_max: int = 50) -> float:
    """Merton (1976) analytic call price via Poisson-weighted Black-Scholes series."""
    k = strike if strike is not None else params.strike
    mu_bar = np.exp(params.mu_j + 0.5 * params.sigma_j ** 2) - 1.0
    lam_prime = params.lam * (1.0 + mu_bar)
    price = 0.0
    for n in range(n_max):
        weight = (np.exp(-lam_prime * params.maturity) * (lam_prime * params.maturity) ** n) / math.factorial(n)
        r_n = params.rate - params.lam * mu_bar + n * params.mu_j / params.maturity + n * params.sigma_j ** 2 / (2 * params.maturity)
        sigma_n = np.sqrt(params.sigma ** 2 + n * params.sigma_j ** 2 / params.maturity)
        d1 = (np.log(params.s0 / k) + (r_n + 0.5 * sigma_n ** 2) * params.maturity) / (sigma_n * np.sqrt(params.maturity))
        d2 = d1 - sigma_n * np.sqrt(params.maturity)
        bs = params.s0 * stats.norm.cdf(d1) - k * np.exp(-r_n * params.maturity) * stats.norm.cdf(d2)
        price += weight * bs
    return float(price)


def implied_vol(
    price: float, s: float, k: float, r: float, t: float,
    tol: float = 1e-8, max_iter: int = 200,
) -> float:
    """Newton-Raphson implied volatility from a call price."""
    sigma = 0.3
    for _ in range(max_iter):
        d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        f = s * stats.norm.cdf(d1) - k * np.exp(-r * t) * stats.norm.cdf(d2) - price
        vega = s * stats.norm.pdf(d1) * np.sqrt(t)
        if abs(vega) < 1e-12:
            break
        sigma -= f / vega
        sigma = max(sigma, 1e-6)
        if abs(f) < tol:
            break
    return float(sigma)


# ── Monte Carlo validation (Criterion 2b) ────────────────────────────────────

def mc_merton_call(
    params: MertonParams, strike: float | None = None,
    n_paths: int = 200_000, seed: int = 42,
) -> tuple[float, float]:
    """Monte Carlo Merton call price with standard error for band validation."""
    k = strike if strike is not None else params.strike
    rng = np.random.default_rng(seed)
    mu_bar = np.exp(params.mu_j + 0.5 * params.sigma_j ** 2) - 1.0
    z = rng.standard_normal(n_paths)
    n_jumps = rng.poisson(params.lam * params.maturity, n_paths)
    log_jumps = np.array([
        np.sum(rng.normal(params.mu_j, params.sigma_j, n)) if n > 0 else 0.0
        for n in n_jumps
    ])
    log_st = (
        np.log(params.s0)
        + (params.rate - 0.5 * params.sigma ** 2 - params.lam * mu_bar) * params.maturity
        + params.sigma * np.sqrt(params.maturity) * z
        + log_jumps
    )
    payoffs = np.maximum(np.exp(log_st) - k, 0.0)
    disc = np.exp(-params.rate * params.maturity)
    return float(disc * np.mean(payoffs)), float(disc * np.std(payoffs) / np.sqrt(n_paths))


# ── pipeline entry point ──────────────────────────────────────────────────────

def write_outputs(
    params: MertonParams,
    kou: KouParams,
    config: RunConfig,
    output_dir: Path | None = None,
) -> None:
    """Run W4 validation suite and save a 4-panel figure.

    Parameters
    ----------
    params:
        Merton model parameters (shared baseline with W3).
    kou:
        Kou model extra parameters.
    config:
        COS engine settings (cos_n, cos_l).
    output_dir:
        Directory for output; defaults to config.output_dir/w4.
    """
    out = (output_dir or config.output_dir) / "w4"
    out.mkdir(parents=True, exist_ok=True)

    N = config.cos_n
    L = config.cos_l
    strikes = np.linspace(80, 120, 25)
    N_vals = [16, 32, 64, 128, 256, 512, 1024]

    # Criterion 2a — spectral convergence
    exact = merton_exact_call(params)
    errors = [abs(merton_call(params, n=Nv, l=L) - exact) for Nv in N_vals]
    LOGGER.info("Exact Merton price: %.8f", exact)
    LOGGER.info("COS N=%d error: %.2e", N, errors[N_vals.index(N)] if N in N_vals else float("nan"))

    # Criterion 2b — COS vs Monte Carlo
    mc_price, mc_se = mc_merton_call(params, n_paths=200_000, seed=config.seed)
    cos_atm = merton_call(params, n=N, l=L)
    inside = (mc_price - 2 * mc_se) <= cos_atm <= (mc_price + 2 * mc_se)
    LOGGER.info("MC: %.6f ±%.6f  COS: %.6f  inside band: %s", mc_price, mc_se, cos_atm, inside)

    # Criterion 3 — smile comparison
    merton_ivols, kou_ivols, mc_prices, mc_ses, cos_smile = [], [], [], [], []
    for ks in strikes:
        mp = merton_call(params, strike=ks, n=N, l=L)
        kp = kou_call(params, kou, strike=ks, n=N, l=L)
        mc_p, mc_e = mc_merton_call(params, strike=ks, n_paths=100_000, seed=config.seed)
        merton_ivols.append(implied_vol(mp, params.s0, ks, params.rate, params.maturity) * 100)
        kou_ivols.append(implied_vol(kp, params.s0, ks, params.rate, params.maturity) * 100)
        mc_prices.append(mc_p)
        mc_ses.append(mc_e)
        cos_smile.append(mp)

    # Criterion 4 — digital options
    dig_types = ["cash_call", "cash_put", "asset_call", "asset_put"]
    dig_colors = ["blue", "red", "green", "purple"]
    dig_prices = {
        dt: [merton_digital(params, strike=ks, digital_type=dt, n=N, l=L) for ks in strikes]
        for dt in dig_types
    }

    # 4-panel figure
    mc_arr = np.array(mc_prices)
    se_arr = np.array(mc_ses)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("W4 — COS Pricing Engine Validation", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.semilogy(N_vals, errors, "o-", color="darkblue", linewidth=2)
    ax.axhline(1e-8, color="red", linestyle="--", label="Target $10^{-8}$")
    ax.set_title("Criterion 2a: Spectral Convergence")
    ax.set_xlabel("N (COS terms)")
    ax.set_ylabel("|COS − Exact| (log scale)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    ax = axes[0, 1]
    ax.fill_between(strikes, mc_arr - 2 * se_arr, mc_arr + 2 * se_arr,
                    alpha=0.3, color="orange", label="MC ±2σ band")
    ax.plot(strikes, mc_arr, "o", color="orange", markersize=3, label="MC price")
    ax.plot(strikes, cos_smile, "-", color="darkblue", linewidth=2, label=f"COS (N={N})")
    ax.set_title("Criterion 2b: COS vs Monte Carlo")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Call Price")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.5)

    ax = axes[1, 0]
    ax.plot(strikes, merton_ivols, "b-o", markersize=4, linewidth=2, label="Merton")
    ax.plot(strikes, kou_ivols, "r-s", markersize=4, linewidth=2, label="Kou")
    ax.set_title("Criterion 3: Merton vs Kou Implied-Vol Smile")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied Volatility (%)")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

    ax = axes[1, 1]
    for dt, col in zip(dig_types, dig_colors):
        ax.plot(strikes, dig_prices[dt], color=col, linewidth=2, label=dt)
    ax.set_title("Criterion 4: Digital Options via COS")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    out_path = out / "W4_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info("W4 complete — plot saved to %s", out_path)
