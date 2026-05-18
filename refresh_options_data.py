"""Refresh QQQ option chain snapshot for W6 calibration (v3 — fallback to lastPrice)."""
from __future__ import annotations
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

RISK_FREE_RATE = 0.043
TICKER = "QQQ"
OUTPUT_PATH = Path("data/cleaned/qqq_options_snapshot.csv")


def bs_call(s, k, t, r, sigma):
    if t <= 0 or sigma <= 0:
        return max(s - k * math.exp(-r * t), 0.0)
    vt = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / vt
    d2 = d1 - vt
    return s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)


def bs_put(s, k, t, r, sigma):
    return bs_call(s, k, t, r, sigma) - s + k * math.exp(-r * t)


def implied_vol(price, s, k, t, r, opt_type):
    if t <= 0 or price <= 0:
        return float("nan")
    if opt_type == "call":
        intrinsic = max(s - k * math.exp(-r * t), 0.0)
        pricer = bs_call
    else:
        intrinsic = max(k * math.exp(-r * t) - s, 0.0)
        pricer = bs_put
    if price < intrinsic - 1e-4:
        return float("nan")
    try:
        return float(brentq(lambda sig: pricer(s, k, t, r, sig) - price,
                            1e-4, 5.0, xtol=1e-8, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


def choose_price(row):
    """Mid from bid/ask if both > 0, else lastPrice. Track the source."""
    bid = row.get("bid", 0) or 0
    ask = row.get("ask", 0) or 0
    last = row.get("lastPrice", 0) or 0
    if bid > 0 and ask > 0:
        return pd.Series({"price": 0.5 * (bid + ask), "price_source": "mid"})
    if last > 0:
        return pd.Series({"price": last, "price_source": "lastPrice"})
    return pd.Series({"price": np.nan, "price_source": "none"})


def main():
    import yfinance as yf
    print("Fetching QQQ spot...")
    tkr = yf.Ticker(TICKER)
    hist = tkr.history(period="5d", auto_adjust=False)
    s0 = float(hist["Close"].iloc[-1])
    snapshot_date = hist.index[-1].strftime("%Y-%m-%d")
    snap_dt = datetime.strptime(snapshot_date, "%Y-%m-%d").date()
    print(f"S0 = {s0:.2f}  snapshot = {snapshot_date}")

    expiries = tkr.options
    target_days = [7, 14, 30, 60, 90, 180, 270, 365]
    selected = []
    for tgt in target_days:
        best = min(expiries, key=lambda e:
                   abs((datetime.strptime(e, "%Y-%m-%d").date() - snap_dt).days - tgt))
        if best not in selected:
            selected.append(best)
    print(f"Selected expiries: {selected}")

    rows = []
    for expiry in selected:
        try:
            chain = tkr.option_chain(expiry)
        except Exception as e:
            print(f"skipped {expiry}: {e}")
            continue
        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            sub = df.copy()
            sub["type"] = opt_type
            sub["expiry"] = expiry
            rows.append(sub)

    raw = pd.concat(rows, ignore_index=True)
    print(f"\n--- DIAGNOSTIC FILTERING ---")
    print(f"Raw quotes:                       {len(raw)}")

    # Price selection with fallback
    prices = raw.apply(choose_price, axis=1)
    raw = pd.concat([raw, prices], axis=1)
    n_mid = (raw["price_source"] == "mid").sum()
    n_last = (raw["price_source"] == "lastPrice").sum()
    n_none = (raw["price_source"] == "none").sum()
    print(f"  with mid (bid&ask>0):           {n_mid}")
    print(f"  with lastPrice fallback:        {n_last}")
    print(f"  no usable price:                {n_none}")

    df = raw.dropna(subset=["price"]).copy()
    df = df[df["price"] >= 0.10].copy()
    print(f"After price >= $0.10:             {len(df)}")

    # Time and metadata
    df["snapshot_date"] = snapshot_date
    df["T"] = df["expiry"].apply(lambda e:
        max((datetime.strptime(e, "%Y-%m-%d").date() - snap_dt).days / 365.0, 1/365))
    df["S0"] = s0
    df["r"] = RISK_FREE_RATE

    # IV recompute from chosen price
    print("Recomputing IV with Brent...")
    df["iv_market"] = df.apply(lambda row:
        implied_vol(row["price"], s0, row["strike"], row["T"],
                    RISK_FREE_RATE, row["type"]), axis=1)
    df = df.dropna(subset=["iv_market"]).copy()
    print(f"After IV recomputable:            {len(df)}")

    df = df[(df["iv_market"] > 0.03) & (df["iv_market"] < 3.0)].copy()
    print(f"After IV in (0.03, 3.0):          {len(df)}")

    df["moneyness"] = df["strike"] / s0
    df = df[(df["moneyness"] > 0.5) & (df["moneyness"] < 1.6)].copy()
    print(f"After moneyness in (0.5, 1.6):    {len(df)}")

    out = df[["snapshot_date", "S0", "expiry", "T", "type", "strike",
              "moneyness", "bid", "ask", "lastPrice", "price", "price_source",
              "iv_market", "r"]]
    out = out.sort_values(["T", "strike", "type"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(out)} rows to {OUTPUT_PATH.resolve()}")
    print("\nRows per expiry x type:")
    print(out.groupby(["expiry", "type"]).size().unstack(fill_value=0))
    print("\nPrice source breakdown:")
    print(out["price_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
