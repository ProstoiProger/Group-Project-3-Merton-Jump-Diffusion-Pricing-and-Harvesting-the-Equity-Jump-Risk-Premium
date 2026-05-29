"""Data collection and jump-detection pipeline for QQQ (W5).

Consolidates the four W5/EDA modules into a single production module with
absolute (CWD-independent) paths and no interactive plt.show() calls.

Pipeline stages
---------------
  1. download_daily()      — QQQ daily OHLCV + log-returns via yfinance
  2. download_options()    — Option chain download and cleaning
  3. compute_rv_bpv()      — Rolling RV / BPV (BNS-style jump diagnostic)
  4. detect_jumps()        — Jump classification, lognormal fit, plots

Entry point
-----------
    from src.data_pipeline import run
    run(data_config)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as sc

from .config import DataConfig

LOGGER = logging.getLogger(__name__)

_MU1 = np.sqrt(2 / np.pi)   # E[|Z|] for Z ~ N(0,1), used in BPV scaling
_BNS_OMEGA = np.sqrt(np.pi ** 2 / 4 + np.pi - 5)   # asymptotic std of log-ratio


# ── stage 1: QQQ daily data ───────────────────────────────────────────────────

def download_daily(cfg: DataConfig) -> pd.DataFrame:
    """Download QQQ daily OHLCV and compute log-returns.

    Saves raw CSV to cfg.data_raw / qqq_daily.csv.
    Returns a DataFrame with columns Date, Close, returns (and others).
    """
    cfg.data_raw.mkdir(parents=True, exist_ok=True)
    cfg.outputs_w5.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading %s daily data (period=%s)", cfg.ticker, cfg.data_period)
    qqq = yf.download(
        cfg.ticker,
        period=cfg.data_period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if qqq.empty:
        raise RuntimeError(f"No data returned for {cfg.ticker}")

    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = [col[0] if isinstance(col, tuple) else col for col in qqq.columns]

    qqq = qqq.reset_index()
    qqq["returns"] = np.log(qqq["Close"] / qqq["Close"].shift(1))
    qqq = qqq.dropna().copy()

    out_path = cfg.data_raw / f"{cfg.ticker.lower()}_daily.csv"
    qqq.to_csv(out_path, index=False)
    LOGGER.info("Saved daily data to %s (%d rows)", out_path, len(qqq))
    return qqq


def plot_daily(df: pd.DataFrame, cfg: DataConfig) -> None:
    """Save price and returns time-series plots."""
    date_col = "Date" if "Date" in df.columns else df.columns[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[date_col], df["Close"])
    ax.set_title(f"{df.get('ticker', ['QQQ'])[0] if 'ticker' in df.columns else 'QQQ'} Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(cfg.outputs_w5 / "qqq_price.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[date_col], df["returns"])
    ax.set_title("QQQ Log Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(cfg.outputs_w5 / "qqq_returns.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["returns"], bins=50)
    ax.set_title("Distribution of QQQ Log Returns")
    ax.set_xlabel("Returns")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(cfg.outputs_w5 / "returns_histogram.png", dpi=150)
    plt.close(fig)


# ── stage 2: option chain data ────────────────────────────────────────────────

def _get_chain(ticker_symbol: str, expiry: str) -> pd.DataFrame:
    """Fetch one option chain. Robust to yfinance returning different
    column sets for calls vs puts (a known upstream bug)."""
    ticker = yf.Ticker(ticker_symbol)
    chain = ticker.option_chain(expiry)

    calls = chain.calls.copy()
    calls["type"] = "call"
    calls["expiry"] = expiry

    puts = chain.puts.copy()
    puts["type"] = "put"
    puts["expiry"] = expiry

    # Align schemas: keep only columns that exist in BOTH
    common_cols = [c for c in calls.columns if c in puts.columns]
    calls = calls[common_cols]
    puts = puts[common_cols]

    df = pd.concat([calls, puts], ignore_index=True)
    df["ticker"] = ticker_symbol
    return df


def download_options(cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and clean QQQ option chains across a broad term structure.

    Selects expiries closest to target maturities [1W..1Y] rather than the
    first N chronological (which are all weekly). Tolerates yfinance schema
    mismatches by skipping broken chains.
    """
    from datetime import datetime

    cfg.data_raw.mkdir(parents=True, exist_ok=True)
    cfg.data_cleaned.mkdir(parents=True, exist_ok=True)

    ticker = yf.Ticker(cfg.ticker)
    expiries = list(ticker.options)
    if not expiries:
        raise RuntimeError(f"No option expiries for {cfg.ticker}")

    target_days = [7, 14, 30, 60, 90, 180, 270, 365]
    today = datetime.now().date()
    selected = []
    for tgt in target_days:
        best = min(
            expiries,
            key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days - tgt),
        )
        if best not in selected:
            selected.append(best)
    LOGGER.info("Selected %d expiries spanning term structure: %s",
                len(selected), selected)

    chains = []
    for expiry in selected:
        try:
            chains.append(_get_chain(cfg.ticker, expiry))
            LOGGER.info("  ok: %s", expiry)
        except Exception as exc:
            LOGGER.warning("  SKIPPED %s: %s", expiry, exc)

    if not chains:
        raise RuntimeError("No option chains downloaded successfully")

    raw = pd.concat(chains, ignore_index=True)
    clean = _clean_options(raw, today)

    raw_path = cfg.data_raw / f"{cfg.ticker.lower()}_options_raw.csv"
    clean_path = cfg.data_cleaned / f"{cfg.ticker.lower()}_options_clean.csv"
    raw.to_csv(raw_path, index=False)
    clean.to_csv(clean_path, index=False)
    LOGGER.info("Options - raw: %d rows, clean: %d rows", len(raw), len(clean))
    return raw, clean


def _clean_options(df: pd.DataFrame, today=None) -> pd.DataFrame:
    """Apply data-quality filters to raw option data.

    Adds two columns useful for W6 calibration:
        T            time to expiry in years (calendar days / 365)
        price_source 'mid' if both bid&ask > 0, else 'lastPrice'
    """
    from datetime import datetime

    if today is None:
        today = datetime.now().date()

    df = df.copy()
    numeric_cols = ["strike", "lastPrice", "bid", "ask", "volume",
                    "openInterest", "impliedVolatility"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["strike", "expiry", "type"])

    if "bid" in df.columns:
        df = df[df["bid"].fillna(0) >= 0]
    if "ask" in df.columns:
        df = df[df["ask"].fillna(0) >= 0]
    if {"bid", "ask"}.issubset(df.columns):
        df = df[df["ask"].fillna(0) >= df["bid"].fillna(0)]

    df["mid_price"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
    valid_bidask = (df["bid"].fillna(0) > 0) & (df["ask"].fillna(0) > 0)
    valid_last = (df["lastPrice"].fillna(0) > 0) & (
        (df["openInterest"].fillna(0) > 0) | (df["volume"].fillna(0) > 0)
    )

    # Attach price_source and option_price BEFORE row filtering so masks align.
    df["price_source"] = np.where(valid_bidask, "mid", "lastPrice")
    df["option_price"] = np.where(valid_bidask, df["mid_price"], df["lastPrice"])

    df = df[valid_bidask | valid_last].copy()

    stale = (df["volume"].fillna(0) == 0) & (df["openInterest"].fillna(0) == 0) & (df["bid"].fillna(0) == 0)
    df = df[~stale].copy()

    valid_mid = df["mid_price"] > 0
    df["spread_pct"] = np.where(valid_mid, (df["ask"] - df["bid"]) / df["mid_price"], np.nan)
    df = df[df["spread_pct"].isna() | (df["spread_pct"] < 0.60)]
    df = df[df["option_price"] > 0.05]

    if "impliedVolatility" in df.columns:
        df = df[(df["impliedVolatility"] > 0) & (df["impliedVolatility"] < 3)]

    df["T"] = df["expiry"].apply(
        lambda e: max((datetime.strptime(e, "%Y-%m-%d").date() - today).days / 365.0, 1/365)
    )

    df = df.drop_duplicates()
    df = df.sort_values(["expiry", "type", "strike"]).reset_index(drop=True)
    return df


# ── stage 3: rolling RV / BPV ────────────────────────────────────────────────

def compute_rv_bpv(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Compute rolling realized variance, bipower variation, and BNS diagnostic.

    Columns added
    -------------
    RV, BPV, RV_to_BPV, bns_style_z, log_ratio, ratio_z,
    rolling_vol, jump_component, jump_flag_single, jump_flag_strict, jump_flag
    """
    window = cfg.bns_window
    out = df.copy()
    out["abs_return"] = out["returns"].abs()

    out["RV"] = out["returns"].pow(2).rolling(window, min_periods=window).sum()

    bpv_term = (out["abs_return"].shift(1) * out["abs_return"]).rolling(
        window - 1, min_periods=window - 1
    ).sum()
    out["BPV"] = (window / (window - 1)) * bpv_term / (_MU1 ** 2)

    safe_bpv = out["BPV"].clip(lower=1e-12)
    out["RV_to_BPV"] = out["RV"] / safe_bpv
    out["bns_style_z"] = np.sqrt(window) * (out["RV_to_BPV"] - 1) / _BNS_OMEGA
    out["jump_component"] = out["RV"] - out["BPV"]

    out["log_ratio"] = np.log(out["RV_to_BPV"].clip(lower=1e-12))
    out["log_ratio_mean"] = out["log_ratio"].rolling(window, min_periods=window).mean()
    out["log_ratio_std"] = out["log_ratio"].rolling(window, min_periods=window).std(ddof=0)
    out["ratio_z"] = (out["log_ratio"] - out["log_ratio_mean"]) / out["log_ratio_std"].replace(0, np.nan)

    out["rolling_vol"] = out["returns"].rolling(window, min_periods=window).std(ddof=0)

    z_thr = cfg.bns_z_threshold
    out["jump_flag_single"] = out["bns_style_z"].abs() > z_thr

    rolling_vol_safe = out["rolling_vol"].replace(0, np.nan)
    out["jump_flag_strict"] = out["jump_flag_single"] & (
        out["returns"].abs() > cfg.strict_vol_multiplier * rolling_vol_safe
    )
    out["jump_flag"] = out["jump_flag_single"]
    return out


def plot_rv_bpv(stats: pd.DataFrame, cfg: DataConfig) -> None:
    """Save RV/BPV time-series and log-ratio Z-score plots."""
    date_col = "date" if "date" in stats.columns else stats.columns[0]
    z_thr = cfg.bns_z_threshold

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stats[date_col], stats["RV"], label="Rolling RV")
    ax.plot(stats[date_col], stats["BPV"], label="Rolling BPV")
    ax.set_title(f"Rolling RV vs BPV (QQQ, {cfg.bns_window}-day window)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(cfg.outputs_w5 / "rv_bpv_timeseries.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stats[date_col], stats["ratio_z"], alpha=0.85, label="Log-ratio Z score")
    ax.axhline(z_thr, linestyle="--", color="red", label=f"±{z_thr} threshold")
    ax.axhline(-z_thr, linestyle="--", color="red")
    ax.set_title("RV/BPV Log-Ratio Variance-Ratio Jump Diagnostic")
    ax.set_xlabel("Date")
    ax.set_ylabel("Z score")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(cfg.outputs_w5 / "rv_bpv_ratio_z.png", dpi=150)
    plt.close(fig)


# ── stage 4: jump detection and characterisation ──────────────────────────────

def detect_jumps(stats: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Classify jump days and enrich the stats DataFrame.

    Adds abs_return, jump_score, jump_size_proxy, jump_flag_two columns
    (re-computes if not already present from compute_rv_bpv).
    """
    out = stats.copy()
    out["abs_return"] = out["returns"].abs()
    out["jump_score"] = out["bns_style_z"].abs()
    out["jump_size_proxy"] = out["abs_return"]

    if "jump_flag_single" not in out.columns:
        out["jump_flag_single"] = out["bns_style_z"].abs() > cfg.bns_z_threshold

    if "rolling_vol" not in out.columns or out["rolling_vol"].isna().all():
        out["rolling_vol"] = out["returns"].rolling(cfg.bns_window, min_periods=cfg.bns_window).std(ddof=0)

    out["jump_flag_two"] = out["jump_flag_single"] & (
        out["abs_return"] > cfg.strict_vol_multiplier * out["rolling_vol"]
    )
    out["jump_flag"] = out["jump_flag_single"]
    return out


def fit_lognormal(jump_abs_returns: pd.Series) -> dict:
    """Fit a lognormal to jump-size proxies (absolute returns on jump days)."""
    arr = jump_abs_returns.dropna().values
    arr = arr[arr > 0]

    if len(arr) < 5:
        return {}

    shape, loc, scale = sc.lognorm.fit(arr, floc=0)
    ks_stat, ks_p = sc.kstest(arr, "lognorm", args=(shape, loc, scale))

    return {
        "n_jump_obs": len(arr),
        "lognormal_sigma": shape,
        "lognormal_mu": np.log(scale),
        "lognormal_scale": scale,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
    }


def summarize_jumps(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute jump summary statistics table."""
    valid_days = int(daily["bns_style_z"].notna().sum())
    single = daily[daily["jump_flag_single"]]
    two = daily[daily.get("jump_flag_two", daily["jump_flag_single"])]

    return pd.DataFrame({
        "metric": [
            "total_days", "valid_days", "jump_days_single", "jump_days_two",
            "annualized_rate_single", "annualized_rate_two",
            "avg_jump_size", "median_jump_size", "max_jump_size",
        ],
        "value": [
            len(daily), valid_days, len(single), len(two),
            (len(single) / max(valid_days, 1)) * 252,
            (len(two) / max(valid_days, 1)) * 252,
            single["jump_size_proxy"].mean() if len(single) else 0.0,
            single["jump_size_proxy"].median() if len(single) else 0.0,
            single["jump_size_proxy"].max() if len(single) else 0.0,
        ],
    })


def plot_jump_results(daily: pd.DataFrame, lf: dict, cfg: DataConfig) -> None:
    """Save all W5 jump-detection plots."""
    date_col = "date" if "date" in daily.columns else daily.columns[0]
    z_thr = cfg.bns_z_threshold

    # Returns with jump highlights
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily[date_col], daily["returns"], lw=0.8, alpha=0.8, label="Daily Returns")
    ax.scatter(daily.loc[daily["jump_flag_single"], date_col],
               daily.loc[daily["jump_flag_single"], "returns"],
               color="orange", s=30, label="Single-condition jump", zorder=3)
    if "jump_flag_two" in daily.columns:
        ax.scatter(daily.loc[daily["jump_flag_two"], date_col],
                   daily.loc[daily["jump_flag_two"], "returns"],
                   color="red", marker="D", s=60, label="Two-condition jump", zorder=4)
    ax.set_title("QQQ Daily Returns with Jump Diagnostics")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(cfg.outputs_w5 / "daily_jump_days.png", dpi=150)
    plt.close(fig)

    # Z-score time series
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily[date_col], daily["bns_style_z"], lw=0.8, alpha=0.8, label="BNS Z score")
    ax.axhline(z_thr, linestyle="--", color="red", label=f"±{z_thr} threshold")
    ax.axhline(-z_thr, linestyle="--", color="red")
    ax.set_title("BNS-Style Jump Diagnostic Z Score")
    ax.set_xlabel("Date")
    ax.set_ylabel("Z score")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(cfg.outputs_w5 / "jump_statistic_timeseries.png", dpi=150)
    plt.close(fig)

    # RV/BPV ratio with highlighted jump days
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily[date_col], daily["RV_to_BPV"], lw=0.8, alpha=0.8, label="RV/BPV")
    ax.scatter(daily.loc[daily["jump_flag_single"], date_col],
               daily.loc[daily["jump_flag_single"], "RV_to_BPV"],
               color="orange", s=25, label="Single-condition jump", zorder=3)
    ax.set_title("Rolling RV/BPV Ratio with Jump Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("RV / BPV")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(cfg.outputs_w5 / "jump_days_highlighted.png", dpi=150)
    plt.close(fig)

    # Jump size histogram + lognormal fit
    jump_days = daily[daily["jump_flag_single"]]
    arr = jump_days["jump_size_proxy"].dropna()
    arr = arr[arr > 0]

    if len(arr) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(arr, bins=25, density=True, alpha=0.7, label="Empirical jump-size proxy")
        if lf:
            x = np.linspace(arr.min(), arr.max(), 300)
            pdf = sc.lognorm.pdf(x, lf["lognormal_sigma"], 0, lf["lognormal_scale"])
            ax.plot(x, pdf, "r-", lw=2, label=f"Lognormal fit (KS p={lf['ks_pvalue']:.3f})")
        ax.set_title("Jump-Size Proxy Distribution vs Lognormal Fit")
        ax.set_xlabel("|r_t| on jump days")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(cfg.outputs_w5 / "jump_size_distribution.png", dpi=150)
        plt.close(fig)

    # QQ plot
    if lf and len(arr) >= 5:
        emp = np.sort(arr.to_numpy())
        n = len(emp)
        p_vals = (np.arange(1, n + 1) - 0.5) / n
        theo = sc.lognorm.ppf(p_vals, lf["lognormal_sigma"], 0, lf["lognormal_scale"])

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(theo, emp, s=20)
        lims = [min(theo.min(), emp.min()), max(theo.max(), emp.max())]
        ax.plot(lims, lims, "--k", linewidth=1)
        ax.set_title("QQ Plot: Empirical Jump Sizes vs Lognormal")
        ax.set_xlabel("Theoretical lognormal quantiles")
        ax.set_ylabel("Empirical quantiles")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(cfg.outputs_w5 / "jump_lognormal_qq.png", dpi=150)
        plt.close(fig)


# ── pipeline entry point ──────────────────────────────────────────────────────

def run(cfg: DataConfig) -> None:
    """Execute all four W5 pipeline stages.

    Stage 1: Download QQQ daily data and plot price/returns.
    Stage 2: Download and clean QQQ option chains.
    Stage 3: Compute rolling RV/BPV and save diagnostics.
    Stage 4: Detect jumps, fit lognormal, save summary CSVs and plots.
    """
    cfg.data_raw.mkdir(parents=True, exist_ok=True)
    cfg.data_cleaned.mkdir(parents=True, exist_ok=True)
    cfg.outputs_w5.mkdir(parents=True, exist_ok=True)

    # Stage 1
    LOGGER.info("Stage 1: downloading %s daily data", cfg.ticker)
    qqq = download_daily(cfg)
    plot_daily(qqq, cfg)

    # Stage 2
    LOGGER.info("Stage 2: downloading %s option chains", cfg.ticker)
    try:
        download_options(cfg)
    except Exception as exc:
        LOGGER.warning("Option download skipped: %s", exc)

    # Stage 3
    LOGGER.info("Stage 3: computing rolling RV/BPV (window=%d)", cfg.bns_window)
    date_col = "Date" if "Date" in qqq.columns else qqq.columns[0]
    df_daily = qqq.rename(columns={date_col: "date"}).sort_values("date").reset_index(drop=True)
    df_daily = df_daily[np.isfinite(df_daily["returns"])].copy()

    stats = compute_rv_bpv(df_daily, cfg)
    stats.to_csv(cfg.outputs_w5 / "rv_bpv_timeseries.csv", index=False)
    stats[stats["jump_flag_single"]].to_csv(cfg.outputs_w5 / "rv_bpv_jump_days.csv", index=False)
    stats[stats["jump_flag_strict"]].to_csv(cfg.outputs_w5 / "rv_bpv_jump_days_strict.csv", index=False)
    plot_rv_bpv(stats, cfg)

    # Stage 4
    LOGGER.info("Stage 4: jump classification and lognormal fit")
    daily = detect_jumps(stats, cfg)

    summary = summarize_jumps(daily)
    lf = fit_lognormal(daily[daily["jump_flag_single"]]["jump_size_proxy"])

    daily.to_csv(cfg.outputs_w5 / "qqq_jump_daily_stats.csv", index=False)
    summary.to_csv(cfg.outputs_w5 / "jump_summary.csv", index=False)
    if lf:
        pd.DataFrame([lf]).to_csv(cfg.outputs_w5 / "lognormal_fit.csv", index=False)

    (daily[daily["jump_flag_single"]]
     .sort_values(["jump_score", "jump_size_proxy"], ascending=False)
     .to_csv(cfg.outputs_w5 / "major_jump_dates.csv", index=False))

    plot_jump_results(daily, lf, cfg)

    LOGGER.info("W5 complete — outputs in %s", cfg.outputs_w5)
    LOGGER.info("Jump summary:\n%s", summary.to_string(index=False))
