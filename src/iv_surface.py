"""W6 -> W7 hand-off: implied volatility surface helper.

Exposes the calibrated Merton and Kou parameters and provides one-line
functions to compute model implied volatilities on arbitrary (K, T) grids.

This is the contract between W6 (Calibration) and W7 (Vol Analyst).
W7 should not re-run the calibration or import internal calibration
helpers; everything needed for the volatility-surface analysis is here.

Example usage from W7 code
--------------------------
    from src.iv_surface import (
        get_merton_params,                 # unregularised calibrated params
        get_merton_params_regularised,     # L1-regularised (literature region)
        get_kou_params,
        merton_iv,                         # one option
        merton_iv_surface,                 # 2D grid
        kou_iv_surface,
    )

    p = get_merton_params()

    # Single option
    iv_atm = merton_iv(S0=735.60, K=735.60, T=0.25, r=0.05,
                       opt_type="call", params=p)

    # Full 3D surface
    import numpy as np
    K_grid = np.linspace(620, 850, 30)
    T_grid = np.linspace(0.08, 1.05, 20)
    IV = merton_iv_surface(S0=735.60, r=0.05,
                           K_grid=K_grid, T_grid=T_grid, params=p)
    # IV.shape == (len(T_grid), len(K_grid))
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .config import KouParams, MertonParams, RunConfig
from .cos_engine import (
    cos_price_european,
    kou_cf,
    kou_cumulants,
    merton_cf,
    merton_cumulants,
)

ROOT = Path(__file__).resolve().parent.parent
W6_OUT = ROOT / "outputs" / "w6"


# ── parameter loaders ────────────────────────────────────────────────────────

def get_merton_params() -> dict:
    """Return the unregularised Merton calibration result.

    Note: the unregularised optimum lies on the boundary mu_J = -0.50,
    consistent with the peso problem documented in W6 paper section.
    For literature-consistent parameters use get_merton_params_regularised().
    """
    df = pd.read_csv(W6_OUT / "merton_calibration.csv")
    row = df.iloc[0]
    sigma_J = float(row["sigma_J"])
    mu_J = float(row["mu_J"])
    return {
        "sigma": float(row["sigma"]),
        "lambda": float(row["lambda"]),
        "mu_J": mu_J,
        "sigma_J": sigma_J,
        "kappa": math.exp(mu_J + 0.5 * sigma_J ** 2) - 1.0,
        "rmse": float(row["rmse"]),
        "source": "calibration.py (unregularised, boundary optimum)",
    }


def get_merton_params_regularised(alpha: float = 0.01) -> dict:
    """Return the L1-regularised Merton parameters at the given alpha.

    alpha=0.01 lands at mu_J ~ -0.20, lambda ~ 0.49 -- between the boundary
    and the literature region. Use alpha=0.05 for the strongest pull
    (mu_J ~ -0.06, lambda ~ 2.5, the textbook 'moderate jumps' calibration).
    """
    df = pd.read_csv(W6_OUT / "regularisation_path.csv")
    row = df.iloc[(df["alpha"] - alpha).abs().argmin()]
    sigma_J = float(row["sigma_J"])
    mu_J = float(row["mu_J"])
    return {
        "sigma": float(row["sigma"]),
        "lambda": float(row["lambda"]),
        "mu_J": mu_J,
        "sigma_J": sigma_J,
        "kappa": math.exp(mu_J + 0.5 * sigma_J ** 2) - 1.0,
        "alpha": float(row["alpha"]),
        "rmse_unreg": float(row["rmse_unreg"]),
        "source": f"regularisation_path.csv (alpha={float(row['alpha'])})",
    }


def get_kou_params() -> dict:
    """Return the Kou calibration result."""
    df = pd.read_csv(W6_OUT / "kou_calibration.csv")
    row = df.iloc[0]
    return {
        "sigma": float(row["sigma"]),
        "lambda": float(row["lambda"]),
        "p_up": float(row["p_up"]),
        "eta1": float(row["eta1"]),
        "eta2": float(row["eta2"]),
        "rmse": float(row["rmse"]),
        "source": "calibrate_kou (calibration.py)",
    }


# ── Black-Scholes IV inversion ───────────────────────────────────────────────

def _bs_call(s, k, t, r, sigma):
    if t <= 0 or sigma <= 0:
        return max(s - k * math.exp(-r * t), 0.0)
    vt = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / vt
    d2 = d1 - vt
    return s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)


def _bs_iv(price, s, k, t, r, opt_type):
    if t <= 0 or price <= 0:
        return float("nan")
    if opt_type == "call":
        def f(sig):
            return _bs_call(s, k, t, r, sig) - price
    else:
        def f(sig):
            return _bs_call(s, k, t, r, sig) - s + k * math.exp(-r * t) - price
    try:
        return float(brentq(f, 1e-4, 5.0, xtol=1e-8, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


# ── single-contract IV ───────────────────────────────────────────────────────

def merton_iv(S0: float, K: float, T: float, r: float, opt_type: str,
              params: dict, n_cos: int = 512) -> float:
    """Black-Scholes implied vol of a single Merton-priced option."""
    sigma, lam = params["sigma"], params["lambda"]
    mu_j, sigma_j = params["mu_J"], params["sigma_J"]
    cum = merton_cumulants(S0, r, T, sigma, lam, mu_j, sigma_j)
    phi = lambda u: merton_cf(u, S0, r, T, sigma, lam, mu_j, sigma_j)
    p = cos_price_european(phi, S0, K, r, T, opt_type, n_cos, 10, cum)
    return _bs_iv(p, S0, K, T, r, opt_type)


def kou_iv(S0: float, K: float, T: float, r: float, opt_type: str,
           params: dict, n_cos: int = 512) -> float:
    """Black-Scholes implied vol of a single Kou-priced option."""
    sigma, lam = params["sigma"], params["lambda"]
    p_up, eta1, eta2 = params["p_up"], params["eta1"], params["eta2"]
    cum = kou_cumulants(S0, r, T, sigma, lam, p_up, eta1, eta2)
    phi = lambda u: kou_cf(u, S0, r, T, sigma, lam, p_up, eta1, eta2)
    pr = cos_price_european(phi, S0, K, r, T, opt_type, n_cos, 10, cum)
    return _bs_iv(pr, S0, K, T, r, opt_type)


# ── 2D surface ───────────────────────────────────────────────────────────────

def merton_iv_surface(S0: float, r: float,
                      K_grid: np.ndarray, T_grid: np.ndarray,
                      params: Optional[dict] = None,
                      opt_type_rule: str = "otm",
                      n_cos: int = 512) -> np.ndarray:
    """Build a Merton IV surface on a 2D grid.

    Parameters
    ----------
    S0, r        : market state
    K_grid       : 1D array of strikes
    T_grid       : 1D array of maturities (years)
    params       : Merton parameter dict (default: get_merton_params())
    opt_type_rule : 'otm' (puts when K<S0, calls when K>S0),
                    'call' all calls, 'put' all puts
    n_cos        : COS expansion terms

    Returns
    -------
    IV : np.ndarray, shape (len(T_grid), len(K_grid))
    """
    if params is None:
        params = get_merton_params()
    n_T, n_K = len(T_grid), len(K_grid)
    IV = np.full((n_T, n_K), np.nan)
    for i, T in enumerate(T_grid):
        for j, K in enumerate(K_grid):
            if opt_type_rule == "otm":
                opt = "put" if K < S0 else "call"
            else:
                opt = opt_type_rule
            IV[i, j] = merton_iv(S0, K, T, r, opt, params, n_cos)
    return IV


def kou_iv_surface(S0: float, r: float,
                   K_grid: np.ndarray, T_grid: np.ndarray,
                   params: Optional[dict] = None,
                   opt_type_rule: str = "otm",
                   n_cos: int = 512) -> np.ndarray:
    """Build a Kou IV surface on a 2D grid. See merton_iv_surface."""
    if params is None:
        params = get_kou_params()
    n_T, n_K = len(T_grid), len(K_grid)
    IV = np.full((n_T, n_K), np.nan)
    for i, T in enumerate(T_grid):
        for j, K in enumerate(K_grid):
            if opt_type_rule == "otm":
                opt = "put" if K < S0 else "call"
            else:
                opt = opt_type_rule
            IV[i, j] = kou_iv(S0, K, T, r, opt, params, n_cos)
    return IV


# ── demo ─────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """Quick smoke test. Run: python -m src.iv_surface"""
    print("=" * 60)
    print("Calibrated parameters available to W7")
    print("=" * 60)

    p_m = get_merton_params()
    print("\nMerton (unregularised, boundary optimum):")
    for k, v in p_m.items():
        print(f"  {k:>10} = {v}")

    try:
        p_reg = get_merton_params_regularised(alpha=0.01)
        print("\nMerton (regularised, alpha=0.01):")
        for k, v in p_reg.items():
            print(f"  {k:>10} = {v}")
    except FileNotFoundError:
        print("\nNo regularisation_path.csv -- skipping.")

    p_k = get_kou_params()
    print("\nKou:")
    for k, v in p_k.items():
        print(f"  {k:>10} = {v}")

    # Sample 4x7 surface
    print("\n" + "=" * 60)
    print("Sample Merton IV surface")
    print("=" * 60)
    S0, r = 735.60, 0.05
    K_grid = np.array([650, 690, 720, 735.60, 760, 790, 830])
    T_grid = np.array([0.10, 0.25, 0.50, 1.00])
    IV = merton_iv_surface(S0, r, K_grid, T_grid, params=p_m)

    header = "  T \\ K   " + " ".join(f"{k:>8.0f}" for k in K_grid)
    print(header)
    for i, T in enumerate(T_grid):
        row = f"  T={T:.2f}  " + " ".join(f"{iv:>8.4f}" for iv in IV[i])
        print(row)


if __name__ == "__main__":
    _demo()
