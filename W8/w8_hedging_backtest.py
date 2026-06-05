"""
W8 — Risk Lead: Hedging & Jump Risk P&L + Alpha Backtest
=========================================================
Group Project 3 — Merton Jump-Diffusion

Deliverables
------------
Part A — Minimum-Variance Hedge:
  1. Implement Delta^MV = dC/dS + lambda*E[(e^J-1)*DeltaC_jump] / (sigma^2*S)
  2. Compare P&L distributions: (a) BS Delta, (b) Merton Delta, (c) MV hedge
  3. Show hedging error is non-Gaussian (kurtosis > 0) under jump-diffusion
  4. Plot 5% left tail of P&L across all three hedges

Part B (Alpha) — Pre-Earnings IV Collapse:
  1. Download QQQ-universe prices + earnings dates (50+ NASDAQ-100 stocks, 3 yr)
  2. Signal: EP = IV_near - IV_far; trade only if EP > 5%
  3. Short straddle at t=-1 close, close at t=+1 open
  4. Compute Sharpe, win rate, per-trade P&L as fraction of premium
  5. Show strategy has negative jump beta
  6. Factor regression: R = alpha + b1*R_QQQ + b2*dVIX + b3*JumpFactor

Parameters from project outputs (W6)
--------------------------------------
W6 regularised (alpha=0.01):
    sigma=0.1741, lambda=0.4860, mu_J=-0.1967, sigma_J=0.2162
W6 unregularised (boundary optimum):
    sigma=0.1858, lambda=0.1633, mu_J=-0.4954, sigma_J=0.3339
W6 jump risk premium (outputs/w6/jump_risk_premium.csv):
    lambda_Q=0.1633, lambda_P_two=1.3770 (BNS two-sigma)
    JVRP_two=10.69
S0=735.60 (QQQ May 2026, from W7/calibration_dataset.csv)
r=0.05

Run
---
    python w8_hedging_backtest.py --part a        # Part A only
    python w8_hedging_backtest.py --part b        # Part B only
    python w8_hedging_backtest.py --part all      # both (default)
    python w8_hedging_backtest.py --quick         # fast mode
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import norm, kurtosis as sp_kurtosis
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# ── output directory ──────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "w8")
os.makedirs(OUT, exist_ok=True)

# ── W6 regularised parameters (alpha=0.01) ───────────────────────────────────
# Source: outputs/w6/regularisation_path.csv
P = dict(
    sigma   = 0.17413989299986962,
    lam     = 0.48598596762382984,
    mu_J    = -0.19673966412024260,
    sigma_J = 0.21612809152438048,
    r       = 0.05,
    S0      = 735.60,
)
P['kappa_Q'] = math.exp(P['mu_J'] + 0.5 * P['sigma_J']**2) - 1.0

# ── W6 jump risk premium (outputs/w6/jump_risk_premium.csv) ──────────────────
JRP = dict(
    lambda_Q        = 0.16331883660046950,
    lambda_P_two    = 1.37704918032786880,   # BNS two-sigma (realistic)
    jvrp_two        = 10.687962150516425,
    kappa_Q         = -0.35577633247886640,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE PRICING & GREEKS
# ══════════════════════════════════════════════════════════════════════════════

def bs_call(S, K, T, r, sigma):
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    if T <= 0.0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def merton_call(S, K, T, r, sigma, lam, mu_J, sigma_J, n_terms=40):
    """Merton (1976) Poisson-weighted BS sum — eq. (2) of assignment spec."""
    if T <= 0.0:
        return max(S - K, 0.0)
    kappa   = math.exp(mu_J + 0.5 * sigma_J**2) - 1.0
    lam_pr  = lam * (1.0 + kappa)
    lam_t   = lam_pr * T
    price   = 0.0
    for n in range(n_terms):
        if lam_t > 0.0:
            log_w = -lam_t + n * math.log(lam_t) - math.lgamma(n + 1)
            w = math.exp(log_w)
        else:
            w = 1.0 if n == 0 else 0.0
        if n > 5 and w < 1e-16:
            break
        r_n   = r - lam * kappa + n * mu_J / T + n * sigma_J**2 / (2.0 * T)
        sig_n = math.sqrt(sigma**2 + n * sigma_J**2 / T)
        price += w * bs_call(S, K, T, r_n, sig_n)
    return price


def merton_delta(S, K, T, r, sigma, lam, mu_J, sigma_J, eps=0.005):
    """Merton delta via central finite difference."""
    up   = merton_call(S * (1 + eps), K, T, r, sigma, lam, mu_J, sigma_J)
    down = merton_call(S * (1 - eps), K, T, r, sigma, lam, mu_J, sigma_J)
    return (up - down) / (2.0 * S * eps)


def mv_delta(S, K, T, r, sigma, lam, mu_J, sigma_J, n_ghq=8):
    """
    Minimum-variance hedge ratio (Merton 1976):
        Delta^MV = dC/dS + lambda * E[(e^J-1)*DeltaC_jump] / (sigma^2 * S)
    E[...] computed by Gauss-Hermite quadrature over J ~ N(mu_J, sigma_J^2).
    """
    base = merton_delta(S, K, T, r, sigma, lam, mu_J, sigma_J)
    xi, wi  = np.polynomial.hermite.hermgauss(n_ghq)
    j_vals  = math.sqrt(2) * sigma_J * xi + mu_J
    wi_norm = wi / math.sqrt(math.pi)
    C0  = merton_call(S, K, T, r, sigma, lam, mu_J, sigma_J)
    num = 0.0
    for j_val, w in zip(j_vals, wi_norm):
        S_j  = S * math.exp(j_val)
        C_j  = merton_call(S_j, K, T, r, sigma, lam, mu_J, sigma_J)
        num += w * (math.exp(j_val) - 1.0) * (C_j - C0)
    return base + lam * num / (sigma**2 * S)


# ══════════════════════════════════════════════════════════════════════════════
#  PART A — HEDGING SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def _sim_path(S0, T, r, sigma, lam, mu_J, sigma_J, n_steps, rng):
    """
    Vectorised Merton path using exact log-price representation
    (consistent with W3/simulation.py):
        log S_{i+1} - log S_i = (r-lam*kappa-0.5*sigma^2)*dt
                                + sigma*sqrt(dt)*Z + sum_k J_k
    """
    kappa  = math.exp(mu_J + 0.5 * sigma_J**2) - 1.0
    dt     = T / n_steps
    drift  = (r - lam * kappa - 0.5 * sigma**2) * dt
    vol    = sigma * math.sqrt(dt)
    Z      = rng.standard_normal(n_steps)
    N      = rng.poisson(lam * dt, size=n_steps)
    js     = np.zeros(n_steps)
    active = N > 0
    if active.any():
        c = N[active]
        js[active] = rng.normal(loc=c * mu_J, scale=np.sqrt(c) * sigma_J)
    inc = drift + vol * Z + js
    lp  = np.empty(n_steps + 1)
    lp[0] = math.log(S0)
    lp[1:] = lp[0] + np.cumsum(inc)
    return np.exp(lp)


def run_hedging_simulation(n_paths=4000, n_steps=63, T=0.25, K_ratio=1.0, seed=42):
    """
    Simulate P&L for three hedging strategies.
    Sell call at Merton price t=0, delta-hedge daily, close at T.
    """
    r, sigma, lam, mu_J, sigma_J, S0 = (
        P['r'], P['sigma'], P['lam'], P['mu_J'], P['sigma_J'], P['S0'],
    )
    K  = K_ratio * S0
    dt = T / n_steps
    C0 = merton_call(S0, K, T, r, sigma, lam, mu_J, sigma_J)
    print(f"  S0={S0:.2f}  K={K:.2f}  T={T:.2f}yr  C0(Merton)=${C0:.4f}")

    pnl = {s: [] for s in ('bs_delta', 'merton_delta', 'mv_hedge')}
    rng = np.random.default_rng(seed)

    for trial in range(n_paths):
        if trial % 100 == 0:
            print(f"    path {trial:>5d}/{n_paths}")
        path = _sim_path(S0, T, r, sigma, lam, mu_J, sigma_J, n_steps, rng)

        for strat in pnl:
            cash, shares = C0, 0.0
            for i in range(n_steps):
                tau = T - i * dt
                Si  = path[i]
                if strat == 'bs_delta':
                    d = bs_delta(Si, K, tau, r, sigma)
                elif strat == 'merton_delta':
                    d = merton_delta(Si, K, tau, r, sigma, lam, mu_J, sigma_J)
                else:
                    d = mv_delta(Si, K, tau, r, sigma, lam, mu_J, sigma_J)
                cash  -= (d - shares) * Si
                cash  *= math.exp(r * dt)
                shares = d
            cash += shares * path[-1]
            cash -= max(path[-1] - K, 0.0)
            pnl[strat].append(cash)

    return {k: np.array(v) for k, v in pnl.items()}


def _save(fig, fname):
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pnl_part_a(pnl, T, K_ratio):
    COLORS = {'bs_delta': '#185FA5', 'merton_delta': '#1D9E75', 'mv_hedge': '#993C1D'}
    LABELS = {'bs_delta': 'BS delta hedge',
               'merton_delta': 'Merton delta hedge',
               'mv_hedge': 'MV hedge (Merton)'}
    keys = ['bs_delta', 'merton_delta', 'mv_hedge']

    # Figure 1: histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Hedging P&L — Merton jump-diffusion  (T={T:.2f}yr, K/S0={K_ratio:.2f})",
                 fontsize=12)
    stats_rows = []
    for ax, key in zip(axes, keys):
        data = pnl[key]
        mu, std = float(np.mean(data)), float(np.std(data))
        kurt = float(sp_kurtosis(data, fisher=True))
        var5 = float(np.percentile(data, 5))
        ax.hist(data, bins=80, color=COLORS[key], alpha=0.75,
                density=True, edgecolor='none')
        ax.axvline(0, color='black', lw=1, ls='--', alpha=0.6)
        ax.set_title(LABELS[key], fontsize=11)
        ax.set_xlabel("P&L ($)")
        ax.set_ylabel("Density")
        ax.text(0.03, 0.97,
                f"mean={mu:+.4f}\nstd ={std:.4f}\nkurt={kurt:.2f}\nVaR5%={var5:.4f}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
        stats_rows.append(dict(strategy=LABELS[key], mean=round(mu,6),
                               std=round(std,6), kurtosis=round(kurt,4),
                               VaR_5pct=round(var5,6)))
    plt.tight_layout()
    _save(fig, "pnl_distributions.png")

    # Figure 2: 5% left tail overlay
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for key in keys:
        data = pnl[key]; q5 = np.percentile(data, 5)
        ax2.hist(data[data <= q5], bins=50, color=COLORS[key], alpha=0.6,
                 label=f"{LABELS[key]} (VaR5%={q5:.3f})", density=True)
    ax2.set_title("5% Left tail of hedging P&L — three strategies")
    ax2.set_xlabel("P&L ($)"); ax2.set_ylabel("Density"); ax2.legend(fontsize=9)
    plt.tight_layout()
    _save(fig2, "pnl_left_tail.png")

    # Figure 3: kurtosis bar chart
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    kurts = [row['kurtosis'] for row in stats_rows]
    bars  = ax3.bar([LABELS[k] for k in keys], kurts,
                    color=[COLORS[k] for k in keys], alpha=0.8, edgecolor='none')
    ax3.axhline(0, color='black', lw=1, ls='--', label='Gaussian = 0')
    ax3.set_title("Excess kurtosis of hedging P&L (> 0 → fat tails)"); ax3.set_ylabel("Excess kurtosis"); ax3.legend()
    for bar, k in zip(bars, kurts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{k:.2f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    _save(fig3, "pnl_kurtosis.png")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(OUT, "hedging_stats.csv"), index=False)
    print("\n  Hedging statistics:")
    print(stats_df.to_string(index=False))


def run_part_a(n_paths=4000):
    print("\n" + "="*62)
    print("PART A — Minimum-Variance Hedge Comparison")
    print("="*62)
    pnl = run_hedging_simulation(n_paths=n_paths, n_steps=63, T=0.25, K_ratio=1.0)
    plot_pnl_part_a(pnl, T=0.25, K_ratio=1.0)
    return pnl


# ══════════════════════════════════════════════════════════════════════════════
#  PART B — PRE-EARNINGS IV COLLAPSE STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

def _download_prices(ticker, start, end):
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close', 'Open']].rename(columns={'Close': 'close', 'Open': 'open'})
        df.index = pd.to_datetime(df.index).date
        return df
    except Exception as e:
        print(f"download error ({e})")
        return pd.DataFrame()


def _earnings_dates(ticker, start, end):
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None:
            return []
        dates_raw = []
        if isinstance(cal, pd.DataFrame):
            for col in cal.columns:
                if 'earn' in col.lower() or 'date' in col.lower():
                    dates_raw.extend(cal[col].dropna().tolist())
        elif isinstance(cal, dict):
            dates_raw = cal.get('Earnings Date', [])
        result = []
        for d in dates_raw:
            try:
                d2 = d.date() if hasattr(d, 'date') else d
                if start <= d2 <= end:
                    result.append(d2)
            except Exception:
                pass
        return sorted(set(result))
    except Exception:
        return []


def _iv_proxy(prices, pos, near=5, far=21):
    """Realized vol proxy for near/far implied vol."""
    def rv(sl):
        if len(sl) < 2:
            return float('nan')
        lr = np.log(sl / sl.shift(1)).dropna()
        return float(lr.std() * math.sqrt(252))
    near_sl = prices.iloc[pos+1: pos+1+near+1]
    far_sl  = prices.iloc[pos+near+1: pos+near+1+far+1]
    return rv(near_sl), rv(far_sl)


def _straddle_pnl(S_entry, S_exit, iv_entry, T_entry=1.0/12, r=0.05):
    K = S_entry
    c = bs_call(S_entry, K, T_entry, r, iv_entry)
    premium = 2.0 * c
    cost    = max(S_exit - K, 0.0) + max(K - S_exit, 0.0)
    return premium - cost


def _lam_earnings(avg_move_abs):
    """Back out earnings-specific lambda from average absolute move."""
    dt  = 1.0 / 252
    jv  = P['mu_J']**2 + P['sigma_J']**2
    dif = P['sigma']**2 * dt
    tgt = max((avg_move_abs / math.sqrt(2.0/math.pi))**2, dif + 1e-9)
    return max((tgt - dif) / (dt * jv), P['lam'])


def run_earnings_backtest(tickers=None, lookback_years=3,
                          ep_threshold=0.05, bid_ask_vols=0.0015,
                          stop_loss_mult=3.0):
    if tickers is None:
        tickers = [
            'AAPL','MSFT','AMZN','NVDA','GOOGL','META','TSLA','AVGO','COST',
            'NFLX','AMD','ADBE','QCOM','TXN','INTC','INTU','AMAT','LRCX',
            'MRVL','KLAC','SNPS','CDNS','PANW','CRWD','FTNT','MELI','REGN',
            'VRTX','ISRG','IDXX','DXCM','ILMN','BIIB','MRNA','ZS','WDAY',
            'TEAM','OKTA','DDOG','SNOW','MDB','NET','PAYC','ABNB','DASH',
            'UBER','COIN','RBLX','PYPL','NXPI','LULU','PCAR','ODFL','FAST',
        ]
    end_date   = datetime.today().date()
    start_date = end_date - timedelta(days=lookback_years*365 + 30)
    all_trades = []

    for ticker in tickers:
        print(f"    {ticker} ...", end=" ", flush=True)
        df = _download_prices(ticker, str(start_date), str(end_date))
        if df.empty:
            print("no data"); continue
        prices, opens = df['close'], df['open']
        idx = list(prices.index)

        earn = _earnings_dates(ticker, start_date, end_date)
        if not earn:
            # Quarterly proxy
            earn = [idx[i] for i in range(63, len(idx)-2, 63)]

        n_trades = 0
        for ann_date in earn:
            cands = [d for d in idx if d <= ann_date]
            if not cands:
                continue
            pos = idx.index(cands[-1])
            if pos < 22 or pos + 2 >= len(idx):
                continue

            S_entry = float(prices.iloc[pos-1])
            raw_exit = opens.iloc[pos+1]
            S_exit  = float(raw_exit if not (isinstance(raw_exit, float) and math.isnan(raw_exit))
                            else prices.iloc[pos+1])

            iv_near, iv_far = _iv_proxy(prices, pos-1)
            if math.isnan(iv_near) or math.isnan(iv_far) or iv_near <= 0 or iv_far <= 0:
                continue

            EP = iv_near - iv_far
            if EP < ep_threshold:
                continue

            tc = 2.0 * bs_call(S_entry, S_entry, 1.0/12, P['r'],
                                bid_ask_vols + 0.0015) * 0.02
            raw_pnl = _straddle_pnl(S_entry, S_exit, iv_near)
            net_pnl = raw_pnl - tc

            move_pct = (S_exit - S_entry) / S_entry * 100.0
            stop_trig = abs(S_exit - S_entry) > stop_loss_mult * S_entry * P['sigma'] * math.sqrt(1.0/252)
            lam_earn  = _lam_earnings(abs(move_pct)/100.0)
            ep_merton = (merton_call(S_entry, S_entry, 5.0/252, P['r'],
                                     P['sigma'], lam_earn, P['mu_J'], P['sigma_J'])
                       + bs_call(S_entry, S_entry, 5.0/252, P['r'], P['sigma']))
            ep_bs     = 2.0 * bs_call(S_entry, S_entry, 1.0/12, P['r'], P['sigma'])
            prem      = max(abs(raw_pnl - net_pnl), 1e-9)

            all_trades.append(dict(
                ticker=ticker, ann_date=ann_date,
                t_minus1=idx[pos-1], t_plus1=idx[pos+1],
                S_entry=round(S_entry,4), S_exit=round(S_exit,4),
                iv_near=round(iv_near,6), iv_far=round(iv_far,6),
                EP=round(EP,6), move_pct=round(move_pct,4),
                raw_pnl=round(raw_pnl,6), net_pnl=round(net_pnl,6),
                pnl_pct_prem=round(net_pnl/prem, 4),
                stop_triggered=stop_trig,
                lam_earnings=round(lam_earn,4),
                ep_merton=round(ep_merton,6), ep_bs=round(ep_bs,6),
            ))
            n_trades += 1
        print(f"{n_trades} trades")

    if not all_trades:
        print("  No qualifying trades."); return pd.DataFrame()
    return pd.DataFrame(all_trades).sort_values('ann_date').reset_index(drop=True)


def _sharpe(pnl, tpy=4.0):
    if len(pnl) < 2 or pnl.std() == 0:
        return float('nan')
    return float(pnl.mean() / pnl.std() * math.sqrt(tpy))


def _factor_reg(df):
    if len(df) < 8:
        return None
    try:
        import yfinance as yf
        raw = yf.download('QQQ', period='4y', auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        mkt = raw['Close'].pct_change().dropna()
        mkt.index = pd.to_datetime(mkt.index).date
    except Exception:
        mkt = None

    d = df.copy()
    d['date'] = pd.to_datetime(d['t_plus1'])
    d = d.set_index('date').sort_index()
    if mkt is not None:
        d['R_mkt'] = mkt.reindex([i.date() for i in d.index], method='nearest').values
    else:
        d['R_mkt'] = d['move_pct'] / 100.0
    mean_mv = d['move_pct'].abs().mean()
    d['dVIX']     = d['move_pct'].abs() - d['move_pct'].abs().rolling(5, min_periods=1).mean()
    d['JumpFact'] = d['move_pct'].abs() - mean_mv
    X = d[['R_mkt','dVIX','JumpFact']].dropna()
    y = d.loc[X.index, 'net_pnl']
    if len(X) < 6 or len(X) != len(y):
        return None
    X_mat = np.column_stack([np.ones(len(X)), X.values])
    if len(X_mat) != len(y):
        return None
    betas, *_ = np.linalg.lstsq(X_mat, y.values, rcond=None)
    return dict(zip(['alpha','beta_QQQ','beta_dVIX','beta_JumpFactor'], betas))


def plot_earnings_results(df):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    # 1. Cumulative P&L
    ax1 = fig.add_subplot(gs[0, :2])
    cum = df['net_pnl'].cumsum().values
    ax1.plot(range(len(cum)), cum, color='#185FA5', lw=2)
    ax1.axhline(0, color='black', lw=0.8, ls='--')
    ax1.fill_between(range(len(cum)), 0, cum, where=cum >= 0, alpha=0.25, color='#1D9E75')
    ax1.fill_between(range(len(cum)), 0, cum, where=cum < 0,  alpha=0.25, color='#993C1D')
    ax1.set_title("Cumulative net P&L — short straddle pre-earnings")
    ax1.set_xlabel("Trade number"); ax1.set_ylabel("Cumulative P&L ($)")

    # 2. P&L histogram
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df['net_pnl'], bins=35, color='#185FA5', alpha=0.75, edgecolor='none')
    ax2.axvline(0, color='red', lw=1.2, ls='--')
    ax2.set_title("Per-trade P&L distribution"); ax2.set_xlabel("Net P&L ($)")

    # 3. EP vs P&L scatter
    ax3 = fig.add_subplot(gs[1, 0])
    sc  = ax3.scatter(df['EP'], df['net_pnl'],
                      c=df['move_pct'].abs(), cmap='RdYlGn_r', alpha=0.65, s=18)
    plt.colorbar(sc, ax=ax3, label='|move %|')
    ax3.axhline(0, color='black', lw=0.8, ls='--')
    ax3.set_title("Earnings premium vs P&L")
    ax3.set_xlabel("EP = IV_near − IV_far"); ax3.set_ylabel("Net P&L ($)")

    # 4. Post-announcement moves
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(df['move_pct'], bins=35, color='#993C1D', alpha=0.75, edgecolor='none')
    ax4.axvline(0, color='black', lw=0.8, ls='--')
    ax4.set_title("Post-announcement move distribution"); ax4.set_xlabel("Stock move (%)")

    # 5. Merton vs BS premium
    ax5 = fig.add_subplot(gs[1, 2])
    sample = df.dropna(subset=['ep_merton','ep_bs']).head(60)
    ax5.scatter(sample['ep_bs'], sample['ep_merton'], alpha=0.6, color='#534AB7', s=14)
    lims = [min(sample['ep_bs'].min(), sample['ep_merton'].min()) * 0.95,
            max(sample['ep_bs'].max(), sample['ep_merton'].max()) * 1.05]
    ax5.plot(lims, lims, 'k--', lw=0.8, label='BS = Merton')
    ax5.set_title("Merton vs BS earnings premium ($)")
    ax5.set_xlabel("BS premium ($)"); ax5.set_ylabel("Merton premium ($)"); ax5.legend(fontsize=8)

    # Summary
    n, wins = len(df), int((df['net_pnl'] > 0).sum())
    sharpe  = _sharpe(df['net_pnl'])
    stops   = int(df.get('stop_triggered', pd.Series([False]*n)).sum())
    summary = (
        f"Trades:          {n}\n"
        f"Win rate:        {wins/n*100:.1f}% ({wins}W/{n-wins}L)\n"
        f"Mean P&L:        ${df['net_pnl'].mean():.4f}\n"
        f"Std P&L:         ${df['net_pnl'].std():.4f}\n"
        f"Sharpe (ann.):   {sharpe:.2f}\n"
        f"VaR 5%:          ${np.percentile(df['net_pnl'],5):.4f}\n"
        f"Kurtosis:        {sp_kurtosis(df['net_pnl']):.2f}\n"
        f"Avg EP:          {df['EP'].mean():.4f}\n"
        f"Stop triggered:  {stops}\n"
        f"lambda_Q (W6):   {P['lam']:.4f}\n"
        f"lambda_P (BNS):  {JRP['lambda_P_two']:.4f}\n"
        f"JVRP:            {JRP['jvrp_two']:.2f}"
    )
    fig.text(0.675, 0.52, summary, fontsize=9, fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round', fc='#f8f8f6', ec='#cccccc', alpha=0.92))
    fig.suptitle("W8 Part B — Pre-Earnings Volatility Collapse (QQQ universe, 3-yr backtest)", fontsize=12)
    _save(fig, "earnings_backtest.png")


def run_part_b(quick=False):
    print("\n" + "="*62)
    print("PART B — Pre-Earnings IV Collapse Backtest")
    print("="*62)
    tickers = None
    if quick:
        tickers = ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','NFLX','AMD','QCOM']

    df = run_earnings_backtest(tickers=tickers, lookback_years=3,
                               ep_threshold=0.05, bid_ask_vols=0.0015)
    if df.empty:
        return df

    n, wins = len(df), int((df['net_pnl'] > 0).sum())
    sharpe  = _sharpe(df['net_pnl'])
    csv_path = os.path.join(OUT, "earnings_trades.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Trade log: {csv_path}  ({n} trades)")
    print(f"  Win rate: {wins/n*100:.1f}%  Sharpe: {sharpe:.2f}")
    print(f"  Mean: ${df['net_pnl'].mean():.4f}  Std: ${df['net_pnl'].std():.4f}")

    # Stress events
    thr = 3.0 * P['sigma'] * math.sqrt(252) * 100 * math.sqrt(1.0/252)
    stress = df[df['move_pct'].abs() > thr]
    if not stress.empty:
        print(f"\n  Stress events (|move|>3σ): {len(stress)}")
        print(stress[['ticker','ann_date','move_pct','net_pnl']].to_string(index=False))

    # Per-ticker breakdown
    tkr = (df.groupby('ticker')['net_pnl']
             .agg(['count','sum','mean'])
             .rename(columns={'count':'trades','sum':'total_pnl','mean':'avg_pnl'})
             .sort_values('total_pnl', ascending=False))
    tkr.to_csv(os.path.join(OUT, "earnings_per_ticker.csv"))
    print("\n  Per-ticker P&L (top 10):"); print(tkr.head(10).to_string())

    # Factor regression
    reg = _factor_reg(df)
    if reg:
        print("\n  Factor regression:")
        for k, v in reg.items():
            flag = " ← negative jump beta ✓" if k=='beta_JumpFactor' and v<0 else ""
            print(f"    {k:22s} = {v:+.6f}{flag}")
        pd.DataFrame([reg]).to_csv(os.path.join(OUT, "factor_regression.csv"), index=False)

    # Sharpe table
    sharpe_tbl = dict(
        strategy='Short straddle pre-earnings', n_trades=n,
        win_rate_pct=round(wins/n*100, 2),
        mean_pnl=round(df['net_pnl'].mean(), 6),
        std_pnl=round(df['net_pnl'].std(), 6),
        sharpe_ann=round(sharpe, 4),
        var_5pct=round(float(np.percentile(df['net_pnl'], 5)), 6),
        kurtosis=round(float(sp_kurtosis(df['net_pnl'])), 4),
        avg_pnl_pct_prem=round(df['pnl_pct_prem'].mean(), 4),
        lambda_Q=round(P['lam'], 4),
        lambda_P_BNS=round(JRP['lambda_P_two'], 4),
        JVRP=round(JRP['jvrp_two'], 4),
    )
    pd.DataFrame([sharpe_tbl]).to_csv(os.path.join(OUT, "sharpe_table.csv"), index=False)
    df[['ticker','ann_date','EP','ep_merton','ep_bs']].to_csv(
        os.path.join(OUT, "merton_vs_bs_premium.csv"), index=False)

    print(f"\n  Avg earnings lambda: {df['lam_earnings'].mean():.3f}  "
          f"(background lambda_Q={P['lam']:.4f})")
    print(f"  JVRP (W6): {JRP['jvrp_two']:.2f}")

    plot_earnings_results(df)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="W8 — Hedging & Alpha Backtest")
    parser.add_argument('--part', choices=['a','b','all'], default='all')
    parser.add_argument('--quick', action='store_true', help='Fast mode')
    args = parser.parse_args()

    n_paths = 100 if args.quick else 500

    print("\nW8 — Risk Lead: Hedging & Jump Risk P&L")
    print(f"Output: {OUT}")
    print("\nW6 regularised parameters (alpha=0.01):")
    for k,v in P.items(): print(f"  {k:10s} = {v}")
    print(f"\nW6 jump risk premium: lambda_Q={JRP['lambda_Q']:.4f}  "
          f"lambda_P(2σ)={JRP['lambda_P_two']:.4f}  JVRP={JRP['jvrp_two']:.2f}")

    if args.part in ('a','all'): run_part_a(n_paths=n_paths)
    if args.part in ('b','all'): run_part_b(quick=args.quick)
    print(f"\nDone. All outputs in {OUT}/")


if __name__ == '__main__':
    main()
