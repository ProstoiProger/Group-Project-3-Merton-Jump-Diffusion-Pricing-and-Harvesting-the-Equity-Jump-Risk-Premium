"""W6 — Merton & Kou calibration to QQQ implied volatility surface.

Calibrates four-parameter Merton and five-parameter Kou jump-diffusion models
to the cleaned QQQ option chain produced by W5.  Studies the identification
problem (lambda vs mu_J ridge) and computes the jump risk premium against
the physical jump intensities from W5.

Pipeline stages
---------------
  1. load_calibration_data()   — read cleaned options, apply W6 filters,
                                  recompute implied vols with Brent
  2. calibrate_merton()         — two-stage DE + L-BFGS-B optimisation
  3. calibrate_kou()            — same protocol for Kou
  4. identification_study()     — 2D loss surface + L1 regularisation path
  5. jump_risk_premium()        — risk-neutral vs physical jump variance

Entry point
-----------
    from src.calibration import write_outputs
    write_outputs(merton_params, kou_params, run, data, calib)
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution, minimize
from scipy.stats import norm

from .config import CalibrationConfig, DataConfig, KouParams, MertonParams, RunConfig
from .cos_engine import (
    kou_cf,
    kou_cumulants,
    merton_cf,
    merton_cumulants,
    merton_exact_call,
    cos_price_european,
)

LOGGER = logging.getLogger(__name__)


# ── Black-Scholes utilities (for IV inversion) ────────────────────────────────

def _bs_call(s: float, k: float, t: float, r: float, sigma: float) -> float:
    if t <= 0 or sigma <= 0:
        return max(s - k * math.exp(-r * t), 0.0)
    vt = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / vt
    d2 = d1 - vt
    return s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)


def _bs_implied_vol(price: float, s: float, k: float, t: float, r: float,
                    opt_type: str) -> float:
    """Invert call/put price to Black-Scholes IV via Brent."""
    if t <= 0 or price <= 0:
        return float("nan")
    if opt_type == "call":
        def f(sig):
            return _bs_call(s, k, t, r, sig) - price
        intrinsic = max(s - k * math.exp(-r * t), 0.0)
    else:
        def f(sig):
            return _bs_call(s, k, t, r, sig) - s + k * math.exp(-r * t) - price
        intrinsic = max(k * math.exp(-r * t) - s, 0.0)
    if price < intrinsic - 1e-4:
        return float("nan")
    try:
        return float(brentq(f, 1e-4, 5.0, xtol=1e-8, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


# ── data loader ───────────────────────────────────────────────────────────────

def load_calibration_data(merton: MertonParams, data: DataConfig,
                          calib: CalibrationConfig) -> pd.DataFrame:
    """Read the W5 cleaned options and prepare them for calibration.

    Operations
    ----------
    1. Read data/cleaned/qqq_options_clean.csv (from W5)
    2. Fetch the current spot from data/raw/qqq_daily.csv
    3. Apply W6 filters: T >= t_min, OTM only, near-ATM moneyness,
       sane IV range
    4. Recompute IV via Brent (vendor IV is unreliable)

    Returns a DataFrame with columns:
        S0, strike, T, type, option_price, iv_market, moneyness, expiry
    """
    options_path = data.data_cleaned / f"{data.ticker.lower()}_options_clean.csv"
    daily_path = data.data_raw / f"{data.ticker.lower()}_daily.csv"

    if not options_path.exists():
        raise FileNotFoundError(
            f"Options file not found: {options_path}\n"
            "Run `python main.py --w5` first."
        )
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily file not found: {daily_path}")

    df = pd.read_csv(options_path)
    daily = pd.read_csv(daily_path)
    s0 = float(daily["Close"].iloc[-1])
    r = merton.rate
    LOGGER.info("Loaded %d raw quotes; spot=%.2f, r=%.4f", len(df), s0, r)

    df = df.copy()
    df["S0"] = s0
    df["r"] = r
    df["moneyness"] = df["strike"] / s0

    n_initial = len(df)

    # Filter 1: minimum time to expiry
    df = df[df["T"] >= calib.t_min].copy()
    LOGGER.info("  after T >= %.3f:      %d", calib.t_min, len(df))

    # Filter 2: OTM only (puts with K<S0, calls with K>S0)
    is_otm = ((df["type"] == "put") & (df["strike"] < s0)) | \
             ((df["type"] == "call") & (df["strike"] > s0))
    df = df[is_otm].copy()
    LOGGER.info("  after OTM only:       %d", len(df))

    # Filter 3: near-ATM moneyness
    df = df[(df["moneyness"] >= calib.moneyness_min) &
            (df["moneyness"] <= calib.moneyness_max)].copy()
    LOGGER.info("  after moneyness band: %d", len(df))

    # Recompute IV with Brent (vendor IV is unreliable)
    df["iv_market"] = df.apply(
        lambda row: _bs_implied_vol(
            row["option_price"], s0, row["strike"], row["T"], r, row["type"]
        ),
        axis=1,
    )
    df = df.dropna(subset=["iv_market"]).copy()
    LOGGER.info("  after IV recomputable: %d", len(df))

    # Filter 4: realistic IV range
    df = df[(df["iv_market"] >= calib.iv_min) &
            (df["iv_market"] <= calib.iv_max)].copy()
    LOGGER.info("  after IV in [%.2f, %.2f]: %d  (dropped %d total)",
                calib.iv_min, calib.iv_max, len(df), n_initial - len(df))

    if len(df) < 30:
        raise RuntimeError(
            f"Only {len(df)} quotes survived filters — cannot calibrate. "
            "Try widening CALIB_* bounds in .env."
        )

    return df[["S0", "strike", "T", "type", "option_price",
               "iv_market", "moneyness", "expiry", "r"]].sort_values(
        ["T", "strike", "type"]).reset_index(drop=True)


# ── pricing helpers built on top of cos_engine ────────────────────────────────

def _merton_price(theta: np.ndarray, s0: float, k: float, t: float, r: float,
                  opt_type: str, n_cos: int) -> float:
    sigma, lam, mu_j, sigma_j = theta
    cum = merton_cumulants(s0, r, t, sigma, lam, mu_j, sigma_j)
    phi = lambda u: merton_cf(u, s0, r, t, sigma, lam, mu_j, sigma_j)
    return cos_price_european(phi, s0, k, r, t, opt_type, n_cos, 10, cum)


def _kou_price(theta: np.ndarray, s0: float, k: float, t: float, r: float,
               opt_type: str, n_cos: int) -> float:
    sigma, lam, p_up, eta1, eta2 = theta
    cum = kou_cumulants(s0, r, t, sigma, lam, p_up, eta1, eta2)
    phi = lambda u: kou_cf(u, s0, r, t, sigma, lam, p_up, eta1, eta2)
    return cos_price_european(phi, s0, k, r, t, opt_type, n_cos, 10, cum)


# ── loss functions ────────────────────────────────────────────────────────────

def _merton_loss(theta: np.ndarray, df: pd.DataFrame, n_cos: int = 512) -> float:
    sigma, lam, mu_j, sigma_j = theta
    if sigma <= 0 or lam < 0 or sigma_j <= 0:
        return 10.0
    s0 = df["S0"].iloc[0]
    sq = []
    for _, row in df.iterrows():
        try:
            p = _merton_price(theta, s0, row["strike"], row["T"], row["r"],
                              row["type"], n_cos)
            if not math.isfinite(p) or p <= 0:
                sq.append(0.25); continue
            iv = _bs_implied_vol(p, s0, row["strike"], row["T"], row["r"], row["type"])
            if not math.isfinite(iv):
                sq.append(0.25); continue
            sq.append((iv - row["iv_market"]) ** 2)
        except Exception:
            sq.append(0.25)
    return float(np.sqrt(np.mean(sq)))


def _kou_loss(theta: np.ndarray, df: pd.DataFrame, n_cos: int = 512) -> float:
    sigma, lam, p_up, eta1, eta2 = theta
    if sigma <= 0 or lam < 0 or not (0 < p_up < 1) or eta1 <= 1 or eta2 <= 0:
        return 10.0
    s0 = df["S0"].iloc[0]
    sq = []
    for _, row in df.iterrows():
        try:
            p = _kou_price(theta, s0, row["strike"], row["T"], row["r"],
                           row["type"], n_cos)
            if not math.isfinite(p) or p <= 0:
                sq.append(0.25); continue
            iv = _bs_implied_vol(p, s0, row["strike"], row["T"], row["r"], row["type"])
            if not math.isfinite(iv):
                sq.append(0.25); continue
            sq.append((iv - row["iv_market"]) ** 2)
        except Exception:
            sq.append(0.25)
    return float(np.sqrt(np.mean(sq)))


def _sub_sample(df: pd.DataFrame, per_expiry: int = 16) -> pd.DataFrame:
    """Stratified down-sampling by expiry for fast DE."""
    return (
        df.groupby("expiry", group_keys=False)
          .apply(lambda g: g.iloc[np.linspace(0, len(g) - 1,
                                              min(per_expiry, len(g)), dtype=int)])
          .reset_index(drop=True)
    )


# ── calibration drivers ───────────────────────────────────────────────────────

def calibrate_merton(df: pd.DataFrame, calib: CalibrationConfig,
                     run: RunConfig) -> dict:
    """Two-stage Merton calibration. Returns calibrated params + diagnostics."""
    bounds = [
        (calib.sigma_min, calib.sigma_max),
        (calib.lam_min, calib.lam_max),
        (calib.mu_j_min, calib.mu_j_max),
        (calib.sigma_j_min, calib.sigma_j_max),
    ]
    df_de = _sub_sample(df, per_expiry=16)
    LOGGER.info("Merton calibration on %d quotes (DE sub-sample %d)",
                len(df), len(df_de))

    t0 = time.time()
    LOGGER.info("Stage 1: differential_evolution (global)")
    de = differential_evolution(
        _merton_loss, bounds=bounds, args=(df_de, run.cos_n),
        seed=run.seed, maxiter=calib.de_maxiter, popsize=calib.de_popsize,
        tol=1e-5, polish=False, workers=1, disp=True,
    )
    LOGGER.info("  DE best RMSE = %.5f  theta = %s", de.fun, de.x)

    LOGGER.info("Stage 2: L-BFGS-B (local refinement on full %d quotes)", len(df))
    polish = minimize(
        _merton_loss, x0=de.x, args=(df, run.cos_n),
        method="L-BFGS-B", bounds=bounds,
        options={"disp": True, "maxiter": calib.lbfgs_maxiter, "ftol": 1e-7},
    )

    theta = polish.x
    elapsed = time.time() - t0
    LOGGER.info("Merton done in %.1fs.  Final RMSE = %.5f", elapsed, polish.fun)

    sigma_j = float(theta[3])
    mu_j = float(theta[2])
    return {
        "sigma": float(theta[0]),
        "lambda": float(theta[1]),
        "mu_J": mu_j,
        "sigma_J": sigma_j,
        "kappa": math.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0,
        "rmse": float(polish.fun),
        "elapsed_s": elapsed,
        "n_quotes": len(df),
    }


def calibrate_kou(df: pd.DataFrame, calib: CalibrationConfig,
                  run: RunConfig) -> dict:
    """Two-stage Kou calibration."""
    bounds = [
        (calib.sigma_min, calib.sigma_max),
        (calib.lam_min, calib.lam_max),
        (calib.p_up_min, calib.p_up_max),
        (calib.eta1_min, calib.eta1_max),
        (calib.eta2_min, calib.eta2_max),
    ]
    df_de = _sub_sample(df, per_expiry=16)
    LOGGER.info("Kou calibration on %d quotes (DE sub-sample %d)",
                len(df), len(df_de))

    t0 = time.time()
    LOGGER.info("Stage 1: differential_evolution (global)")
    de = differential_evolution(
        _kou_loss, bounds=bounds, args=(df_de, run.cos_n),
        seed=run.seed, maxiter=calib.de_maxiter, popsize=calib.de_popsize,
        tol=1e-5, polish=False, workers=1, disp=True,
    )
    LOGGER.info("  DE best RMSE = %.5f  theta = %s", de.fun, de.x)

    LOGGER.info("Stage 2: L-BFGS-B")
    polish = minimize(
        _kou_loss, x0=de.x, args=(df, run.cos_n),
        method="L-BFGS-B", bounds=bounds,
        options={"disp": True, "maxiter": calib.lbfgs_maxiter, "ftol": 1e-7},
    )

    theta = polish.x
    elapsed = time.time() - t0
    LOGGER.info("Kou done in %.1fs.  Final RMSE = %.5f", elapsed, polish.fun)
    return {
        "sigma": float(theta[0]),
        "lambda": float(theta[1]),
        "p_up": float(theta[2]),
        "eta1": float(theta[3]),
        "eta2": float(theta[4]),
        "rmse": float(polish.fun),
        "elapsed_s": elapsed,
        "n_quotes": len(df),
    }


def per_maturity_rmse(theta, df: pd.DataFrame, model: str, n_cos: int = 512) -> pd.DataFrame:
    s0 = df["S0"].iloc[0]
    pricer = _merton_price if model == "merton" else _kou_price
    rows = []
    for expiry, group in df.groupby("expiry"):
        errs = []
        for _, row in group.iterrows():
            try:
                p = pricer(np.array(theta), s0, row["strike"], row["T"],
                           row["r"], row["type"], n_cos)
                iv = _bs_implied_vol(p, s0, row["strike"], row["T"],
                                     row["r"], row["type"])
                if math.isfinite(iv):
                    errs.append(iv - row["iv_market"])
            except Exception:
                pass
        if errs:
            arr = np.array(errs)
            rows.append({
                "expiry": expiry, "n": len(arr),
                "T": float(group["T"].iloc[0]),
                "rmse": float(np.sqrt(np.mean(arr ** 2))),
                "mean_error": float(np.mean(arr)),
                "max_abs_error": float(np.max(np.abs(arr))),
            })
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


# ── identification study ──────────────────────────────────────────────────────

def identification_study(df: pd.DataFrame, merton_result: dict,
                         calib: CalibrationConfig, run: RunConfig,
                         out_dir: Path) -> dict:
    """2D loss surface in (lambda, mu_J) and L1-regularised calibration path."""
    LOGGER.info("Identification study: 2D loss surface in (lambda, mu_J)")
    sigma_fixed = merton_result["sigma"]
    sigma_j_fixed = merton_result["sigma_J"]

    df_sub = _sub_sample(df, per_expiry=16)
    n_lam, n_mu = 18, 18
    lambdas = np.logspace(np.log10(0.05), np.log10(5.0), n_lam)
    mu_js = np.linspace(-0.45, -0.02, n_mu)
    Z = np.full((n_mu, n_lam), np.nan)

    for i, mu_j in enumerate(mu_js):
        for j, lam in enumerate(lambdas):
            theta = np.array([sigma_fixed, lam, mu_j, sigma_j_fixed])
            Z[i, j] = _merton_loss(theta, df_sub, run.cos_n)
        LOGGER.info("  row %d/%d done", i + 1, n_mu)

    # Save grid
    pd.DataFrame({
        "lambda": np.repeat(lambdas, n_mu),
        "mu_J": np.tile(mu_js, n_lam),
        "rmse": Z.T.flatten(),
    }).to_csv(out_dir / "loss_grid.csv", index=False)

    # Iso-loss band
    z_min = float(np.nanmin(Z))
    band_mask = Z <= z_min + 0.002
    n_band = int(band_mask.sum())

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    LAM, MU = np.meshgrid(np.log10(lambdas), mu_js)
    cm = ax.contourf(LAM, MU, Z, levels=30, cmap="viridis")
    cs = ax.contour(LAM, MU, Z, levels=10, colors="white", alpha=0.4, linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.3f")
    ax.plot(np.log10(merton_result["lambda"]), merton_result["mu_J"],
            marker="*", color="red", markersize=18, markeredgecolor="white",
            markeredgewidth=1.2, label="calibrated optimum")
    ax.set_xlabel(r"$\log_{10}(\lambda)$")
    ax.set_ylabel(r"$\mu_J$")
    ax.set_title(r"Merton loss surface RMSE($\lambda$, $\mu_J$)"
                 + f" — band size = {n_band}/{n_lam*n_mu} cells")
    fig.colorbar(cm, ax=ax, label="vol-space RMSE")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_surface_lambda_muJ.png", dpi=150)
    plt.close(fig)
    LOGGER.info("  iso-loss band size: %d cells (RMSE within +0.002 of %.5f)",
                n_band, z_min)

    # Regularisation path
    LOGGER.info("Regularisation path: L1 penalty on |mu_J|")
    alphas = [0.0, 0.005, 0.01, 0.02, 0.05]
    bounds = [
        (calib.sigma_min, calib.sigma_max),
        (calib.lam_min, calib.lam_max),
        (calib.mu_j_min, calib.mu_j_max),
        (calib.sigma_j_min, calib.sigma_j_max),
    ]

    def reg_loss(theta, df_, n_cos, alpha):
        return _merton_loss(theta, df_, n_cos) + alpha * abs(theta[2])

    path_rows = []
    for alpha in alphas:
        de = differential_evolution(
            reg_loss, bounds=bounds, args=(df_sub, run.cos_n, alpha),
            seed=run.seed, maxiter=20, popsize=10, tol=1e-4,
            polish=False, workers=1, disp=False,
        )
        polish = minimize(
            reg_loss, x0=de.x, args=(df, run.cos_n, alpha),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-7},
        )
        theta = polish.x
        rmse_unreg = _merton_loss(theta, df, run.cos_n)
        path_rows.append({
            "alpha": alpha,
            "sigma": float(theta[0]),
            "lambda": float(theta[1]),
            "mu_J": float(theta[2]),
            "sigma_J": float(theta[3]),
            "rmse_unreg": rmse_unreg,
        })
        LOGGER.info("  alpha=%.4f -> sigma=%.4f, lam=%.4f, mu_J=%.4f, "
                    "sigma_J=%.4f, RMSE(unreg)=%.5f",
                    alpha, theta[0], theta[1], theta[2], theta[3], rmse_unreg)

    path_df = pd.DataFrame(path_rows)
    path_df.to_csv(out_dir / "regularisation_path.csv", index=False)

    # 4-panel regularisation figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col, title in [
        (axes[0, 0], "lambda", r"$\lambda$ vs $\alpha$"),
        (axes[0, 1], "mu_J", r"$\mu_J$ vs $\alpha$"),
        (axes[1, 0], "sigma", r"$\sigma$ vs $\alpha$"),
        (axes[1, 1], "rmse_unreg", r"RMSE(unreg) vs $\alpha$"),
    ]:
        ax.plot(path_df["alpha"], path_df[col], "o-")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("Regularisation path: L1 penalty pulls "
                 r"$\mu_J$ off the boundary")
    fig.tight_layout()
    fig.savefig(out_dir / "regularisation_path.png", dpi=150)
    plt.close(fig)

    return {
        "z_min": z_min,
        "band_size": n_band,
        "total_cells": int(n_lam * n_mu),
        "n_alphas": len(alphas),
    }


# ── jump risk premium ─────────────────────────────────────────────────────────

def jump_risk_premium(merton_result: dict, data: DataConfig,
                      out_dir: Path) -> dict:
    """Compute lambda^Q vs lambda^P and the jump variance risk premium.

    Uses W5 outputs:
        outputs/w5/jump_summary.csv         single-condition lambda^P
        outputs/w5/qqq_jump_daily_stats.csv jump returns for E[J^2]^P
    """
    LOGGER.info("Jump risk premium analysis")

    lam_Q = merton_result["lambda"]
    mu_j_Q = merton_result["mu_J"]
    sigma_j_Q = merton_result["sigma_J"]
    kappa_Q = merton_result["kappa"]
    EJ2_Q = mu_j_Q ** 2 + sigma_j_Q ** 2
    jump_var_Q = lam_Q * EJ2_Q

    summary_path = data.outputs_w5 / "jump_summary.csv"
    stats_path = data.outputs_w5 / "qqq_jump_daily_stats.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"W5 jump summary not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    sumdict = dict(zip(summary["metric"], summary["value"]))

    # The W5 BNS test produces TWO intensities -- use both for comparison
    lam_P_single = float(sumdict["annualized_rate_single"])
    lam_P_two = float(sumdict["annualized_rate_two"])

    EJ2_P_single = None
    EJ2_P_two = None
    if stats_path.exists():
        stats = pd.read_csv(stats_path)
        if "returns" in stats.columns and "jump_flag_single" in stats.columns:
            jr_single = pd.to_numeric(
                stats.loc[stats["jump_flag_single"].astype(bool), "returns"],
                errors="coerce"
            ).dropna().values
            if len(jr_single):
                EJ2_P_single = float(np.mean(jr_single ** 2))
        if "jump_flag_two" in stats.columns:
            jr_two = pd.to_numeric(
                stats.loc[stats["jump_flag_two"].astype(bool), "returns"],
                errors="coerce"
            ).dropna().values
            if len(jr_two):
                EJ2_P_two = float(np.mean(jr_two ** 2))

    # Fallback if jump returns weren't extracted
    if EJ2_P_single is None:
        EJ2_P_single = float(sumdict.get("avg_jump_size", 0.013)) ** 2
    if EJ2_P_two is None:
        EJ2_P_two = float(sumdict.get("avg_jump_size", 0.013)) ** 2

    jvrp_single = (lam_Q * EJ2_Q) / (lam_P_single * EJ2_P_single) \
        if lam_P_single * EJ2_P_single > 0 else float("nan")
    jvrp_two = (lam_Q * EJ2_Q) / (lam_P_two * EJ2_P_two) \
        if lam_P_two * EJ2_P_two > 0 else float("nan")

    LOGGER.info("  lambda^Q = %.4f,  lambda^P (single) = %.4f,  lambda^P (two) = %.4f",
                lam_Q, lam_P_single, lam_P_two)
    LOGGER.info("  E[J^2]^Q = %.5f,  E[J^2]^P (single) = %.5f,  E[J^2]^P (two) = %.5f",
                EJ2_Q, EJ2_P_single, EJ2_P_two)
    LOGGER.info("  jump_var^Q = %.5f", jump_var_Q)
    LOGGER.info("  JVRP (vs liberal P) = %.2f", jvrp_single)
    LOGGER.info("  JVRP (vs conservative P) = %.2f", jvrp_two)

    result = {
        "lambda_Q": lam_Q, "EJ2_Q": EJ2_Q, "jump_var_Q": jump_var_Q,
        "kappa_Q": kappa_Q,
        "lambda_P_single": lam_P_single, "EJ2_P_single": EJ2_P_single,
        "jump_var_P_single": lam_P_single * EJ2_P_single,
        "jvrp_single": jvrp_single,
        "lambda_P_two": lam_P_two, "EJ2_P_two": EJ2_P_two,
        "jump_var_P_two": lam_P_two * EJ2_P_two,
        "jvrp_two": jvrp_two,
    }
    pd.DataFrame([result]).to_csv(out_dir / "jump_risk_premium.csv", index=False)
    return result


# ── plots ─────────────────────────────────────────────────────────────────────

def _model_iv(theta, s0, k, T, r, opt_type, model, n_cos):
    pricer = _merton_price if model == "merton" else _kou_price
    p = pricer(np.array(theta), s0, k, T, r, opt_type, n_cos)
    return _bs_implied_vol(p, s0, k, T, r, opt_type)


def make_plots(df: pd.DataFrame, merton_res: dict, kou_res: dict,
               run: RunConfig, out_dir: Path) -> None:
    """Smile fit, RMSE bars, residual heatmap."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    theta_m = [merton_res["sigma"], merton_res["lambda"],
               merton_res["mu_J"], merton_res["sigma_J"]]
    theta_k = [kou_res["sigma"], kou_res["lambda"], kou_res["p_up"],
               kou_res["eta1"], kou_res["eta2"]]

    # Smile fit (1-month maturity, most data-rich)
    target_T = df["T"].min()
    nearest_expiry = df.loc[(df["T"] - target_T).abs().idxmin(), "expiry"]
    sub = df[df["expiry"] == nearest_expiry].sort_values("strike")
    s0 = sub["S0"].iloc[0]
    iv_m, iv_k = [], []
    for _, row in sub.iterrows():
        iv_m.append(_model_iv(theta_m, s0, row["strike"], row["T"],
                              row["r"], row["type"], "merton", run.cos_n))
        iv_k.append(_model_iv(theta_k, s0, row["strike"], row["T"],
                              row["r"], row["type"], "kou", run.cos_n))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(sub["moneyness"], sub["iv_market"], s=20, alpha=0.6,
               color="black", label="Market", zorder=3)
    ax.plot(sub["moneyness"], iv_m, color="C0", linewidth=2, label="Merton")
    ax.plot(sub["moneyness"], iv_k, color="C1", linewidth=2, linestyle="--",
            label="Kou")
    ax.axvline(1.0, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("Moneyness $K/S_0$")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"Smile fit — {nearest_expiry}  (T={sub['T'].iloc[0]:.3f}y)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "smile_fit_shortest.png", dpi=150)
    plt.close(fig)
    LOGGER.info("  saved smile_fit_shortest.png (expiry %s)", nearest_expiry)

    # All-maturity smile grid
    expiries = sorted(df["expiry"].unique())
    n = len(expiries)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for idx, exp in enumerate(expiries):
        ax = axes_flat[idx]
        sub = df[df["expiry"] == exp].sort_values("strike")
        s0 = sub["S0"].iloc[0]
        ivm, ivk = [], []
        for _, row in sub.iterrows():
            ivm.append(_model_iv(theta_m, s0, row["strike"], row["T"],
                                 row["r"], row["type"], "merton", run.cos_n))
            ivk.append(_model_iv(theta_k, s0, row["strike"], row["T"],
                                 row["r"], row["type"], "kou", run.cos_n))
        ax.scatter(sub["moneyness"], sub["iv_market"], s=10, alpha=0.5,
                   color="black", label="Market")
        ax.plot(sub["moneyness"], ivm, color="C0", linewidth=1.5, label="Merton")
        ax.plot(sub["moneyness"], ivk, color="C1", linewidth=1.5, linestyle="--",
                label="Kou")
        ax.axvline(1.0, color="grey", linestyle=":", alpha=0.4)
        ax.set_title(f"{exp} (T={sub['T'].iloc[0]:.3f}y)")
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("IV")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("QQQ implied vol smile — market vs Merton vs Kou")
    fig.tight_layout()
    fig.savefig(fig_dir / "smile_fit_all_maturities.png", dpi=150)
    plt.close(fig)

    # Per-maturity RMSE bars
    pm_m = per_maturity_rmse(theta_m, df, "merton", run.cos_n)
    pm_k = per_maturity_rmse(theta_k, df, "kou", run.cos_n)
    pm_m.to_csv(out_dir / "merton_rmse_by_expiry.csv", index=False)
    pm_k.to_csv(out_dir / "kou_rmse_by_expiry.csv", index=False)
    merged = pm_m.merge(pm_k[["expiry", "rmse"]], on="expiry",
                        suffixes=("_merton", "_kou"))
    merged = merged.sort_values("T")
    x = np.arange(len(merged))
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, merged["rmse_merton"], w, label="Merton", color="C0")
    ax.bar(x + w/2, merged["rmse_kou"], w, label="Kou", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\n(T={t:.2f})" for e, t in
                        zip(merged["expiry"], merged["T"])], fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("Per-maturity RMSE: Merton vs Kou")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "rmse_by_maturity.png", dpi=150)
    plt.close(fig)
    LOGGER.info("  saved rmse_by_maturity.png")


# ── pipeline entry point ──────────────────────────────────────────────────────

def write_outputs(merton_params: MertonParams, kou_params: KouParams,
                  run: RunConfig, data: DataConfig,
                  calib: CalibrationConfig,
                  output_dir: Path | None = None) -> None:
    """Run the full W6 calibration pipeline."""
    out = (output_dir or run.output_dir) / "w6"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load and filter data
    LOGGER.info("=" * 60)
    LOGGER.info("W6.1 — load and filter calibration data")
    LOGGER.info("=" * 60)
    df = load_calibration_data(merton_params, data, calib)
    df.to_csv(out / "calibration_dataset.csv", index=False)
    LOGGER.info("Final dataset: %d quotes, %d expiries",
                len(df), df["expiry"].nunique())

    # 2. Merton calibration
    LOGGER.info("=" * 60)
    LOGGER.info("W6.2 — Merton calibration")
    LOGGER.info("=" * 60)
    merton_result = calibrate_merton(df, calib, run)
    pd.DataFrame([merton_result]).to_csv(out / "merton_calibration.csv", index=False)
    LOGGER.info("Merton: sigma=%.4f, lam=%.4f, mu_J=%.4f, sigma_J=%.4f, RMSE=%.5f",
                merton_result["sigma"], merton_result["lambda"],
                merton_result["mu_J"], merton_result["sigma_J"],
                merton_result["rmse"])

    # 3. Identification study
    LOGGER.info("=" * 60)
    LOGGER.info("W6.3 — Identification study")
    LOGGER.info("=" * 60)
    id_result = identification_study(df, merton_result, calib, run, out)

    # 4. Kou calibration
    LOGGER.info("=" * 60)
    LOGGER.info("W6.4 — Kou calibration")
    LOGGER.info("=" * 60)
    kou_result = calibrate_kou(df, calib, run)
    pd.DataFrame([kou_result]).to_csv(out / "kou_calibration.csv", index=False)

    # 5. AIC comparison
    n = len(df)
    aic_m = 2 * 4 + n * math.log(merton_result["rmse"] ** 2)
    aic_k = 2 * 5 + n * math.log(kou_result["rmse"] ** 2)
    LOGGER.info("AIC: Merton = %.2f,  Kou = %.2f  ->  favored: %s",
                aic_m, aic_k, "Merton" if aic_m < aic_k else "Kou")
    pd.DataFrame([{
        "model": "Merton", "n_params": 4,
        "sigma": merton_result["sigma"], "lambda": merton_result["lambda"],
        "rmse": merton_result["rmse"], "aic": aic_m,
    }, {
        "model": "Kou", "n_params": 5,
        "sigma": kou_result["sigma"], "lambda": kou_result["lambda"],
        "rmse": kou_result["rmse"], "aic": aic_k,
    }]).to_csv(out / "merton_vs_kou_summary.csv", index=False)

    # 6. Jump risk premium
    LOGGER.info("=" * 60)
    LOGGER.info("W6.5 — Jump risk premium")
    LOGGER.info("=" * 60)
    premium = jump_risk_premium(merton_result, data, out)

    # 7. Plots
    LOGGER.info("=" * 60)
    LOGGER.info("W6.6 — Plots")
    LOGGER.info("=" * 60)
    make_plots(df, merton_result, kou_result, run, out)

    LOGGER.info("W6 complete — outputs in %s", out)
