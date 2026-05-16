import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf


DATA_RAW = Path("data/raw")
DATA_CLEANED = Path("data/cleaned")
OUTPUTS = Path("outputs")


def ensure_dirs() -> None:
    """Create project folders if they do not exist."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_CLEANED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)


def get_option_chain(ticker_symbol: str, expiry: str) -> pd.DataFrame:
    """Download calls and puts for one expiry and return a single dataframe."""
    ticker = yf.Ticker(ticker_symbol)
    chain = ticker.option_chain(expiry)

    calls = chain.calls.copy()
    calls["type"] = "call"
    calls["expiry"] = expiry

    puts = chain.puts.copy()
    puts["type"] = "put"
    puts["expiry"] = expiry

    options = pd.concat([calls, puts], ignore_index=True)
    options["ticker"] = ticker_symbol
    return options


def download_option_chains(ticker_symbol: str = "QQQ", n_expiries: int = 6) -> pd.DataFrame:
    """Download option chains for several expiries."""
    ticker = yf.Ticker(ticker_symbol)
    expiries = list(ticker.options)

    if not expiries:
        raise RuntimeError(f"No option expiries returned for {ticker_symbol}.")

    selected_expiries = expiries[:n_expiries]
    all_chains = []

    for expiry in selected_expiries:
        try:
            chain_df = get_option_chain(ticker_symbol, expiry)
            all_chains.append(chain_df)
        except Exception as exc:
            print(f"Skipping expiry {expiry}: {exc}")

    if not all_chains:
        raise RuntimeError("No option chains could be downloaded.")

    options_df = pd.concat(all_chains, ignore_index=True)
    return options_df


def clean_option_data(options_df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for option chain data."""
    df = options_df.copy()

    # Standardize numeric columns when possible
    numeric_cols = [
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "change",
        "percentChange",
        "volume",
        "openInterest",
        "impliedVolatility",
        "inTheMoney",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Core filters
    if {"bid", "ask"}.issubset(df.columns):
        df = df[(df["bid"].fillna(0) > 0) & (df["ask"].fillna(0) > 0)]
        df = df[df["ask"] >= df["bid"]]

    if {"volume", "openInterest"}.issubset(df.columns):
        df = df[(df["volume"].fillna(0) > 0) | (df["openInterest"].fillna(0) > 0)]

    if {"bid", "ask"}.issubset(df.columns):
        df["mid_price"] = (df["bid"] + df["ask"]) / 2

    df = df.dropna(subset=["strike", "expiry", "type"])
    return df.reset_index(drop=True)


def save_outputs(options_raw: pd.DataFrame, options_clean: pd.DataFrame, ticker_symbol: str = "QQQ") -> None:
    """Save raw and cleaned datasets."""
    raw_path = DATA_RAW / f"{ticker_symbol.lower()}_options_raw.csv"
    clean_path = DATA_CLEANED / f"{ticker_symbol.lower()}_options_clean.csv"

    options_raw.to_csv(raw_path, index=False)
    options_clean.to_csv(clean_path, index=False)

    print(f"Saved raw options to: {raw_path}")
    print(f"Saved cleaned options to: {clean_path}")


def main() -> None:
    ensure_dirs()

    ticker_symbol = "QQQ"
    print(f"Downloading option chains for {ticker_symbol}...")
    options_raw = download_option_chains(ticker_symbol=ticker_symbol, n_expiries=6)

    print("Cleaning option data...")
    options_clean = clean_option_data(options_raw)

    save_outputs(options_raw, options_clean, ticker_symbol=ticker_symbol)

    print("\nRaw shape:", options_raw.shape)
    print("Clean shape:", options_clean.shape)
    print("\nPreview:")
    print(options_clean.head())


if __name__ == "__main__":
    main()
