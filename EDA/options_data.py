from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf


DATA_RAW = Path("data/raw")
DATA_CLEANED = Path("data/cleaned")
OUTPUTS = Path("outputs/w5")


def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_CLEANED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)


def get_option_chain(
    ticker_symbol: str,
    expiry: str
) -> pd.DataFrame:

    ticker = yf.Ticker(ticker_symbol)
    chain = ticker.option_chain(expiry)

    calls = chain.calls.copy()
    calls["type"] = "call"
    calls["expiry"] = expiry

    puts = chain.puts.copy()
    puts["type"] = "put"
    puts["expiry"] = expiry

    options = pd.concat(
        [calls, puts],
        ignore_index=True
    )

    options["ticker"] = ticker_symbol

    return options


def download_option_chains(
    ticker_symbol: str = "QQQ",
    n_expiries: int = 6
) -> pd.DataFrame:

    ticker = yf.Ticker(ticker_symbol)

    expiries = list(ticker.options)

    if not expiries:
        raise RuntimeError(
            f"No option expiries returned for {ticker_symbol}"
        )

    selected_expiries = expiries[:n_expiries]

    all_chains = []

    for expiry in selected_expiries:

        print(f"Downloading expiry: {expiry}")

        try:
            chain_df = get_option_chain(
                ticker_symbol,
                expiry
            )

            all_chains.append(chain_df)

        except Exception as exc:
            print(
                f"Skipping expiry {expiry}: {exc}"
            )

    if not all_chains:
        raise RuntimeError(
            "No option chains downloaded."
        )

    options_df = pd.concat(
        all_chains,
        ignore_index=True
    )

    return options_df


def clean_option_data(
    options_df: pd.DataFrame
) -> pd.DataFrame:

    df = options_df.copy()

    original_rows = len(df)

    numeric_cols = [
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "change",
        "percentChange",
        "volume",
        "openInterest",
        "impliedVolatility"
    ]

    for col in numeric_cols:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # ------------------------------------------------
    # Basic cleaning
    # ------------------------------------------------

    df = df.dropna(
        subset=[
            "strike",
            "expiry",
            "type"
        ]
    )

    # remove negative quotes

    if "bid" in df.columns:
        df = df[df["bid"].fillna(0) >= 0]

    if "ask" in df.columns:
        df = df[df["ask"].fillna(0) >= 0]

    # ask >= bid

    if {"bid", "ask"}.issubset(df.columns):

        df = df[
            df["ask"].fillna(0)
            >=
            df["bid"].fillna(0)
        ]

    # ------------------------------------------------
    # Mid price
    # ------------------------------------------------

    df["mid_price"] = (
        df["bid"].fillna(0)
        +
        df["ask"].fillna(0)
    ) / 2

    valid_bidask = (
        (df["bid"].fillna(0) > 0)
        &
        (df["ask"].fillna(0) > 0)
    )

    valid_lastprice = (
        (df["lastPrice"].fillna(0) > 0)
        &
        (
            (df["openInterest"].fillna(0) > 0)
            |
            (df["volume"].fillna(0) > 0)
        )
    )

    # KEEP if:
    # 1) valid bid/ask
    # OR
    # 2) valid recent last trade

    keep_filter = (
        valid_bidask
        |
        valid_lastprice
    )

    df = df[keep_filter].copy()

    # ------------------------------------------------
    # Price source
    # ------------------------------------------------

    valid_bidask_filtered = (
        (df["bid"].fillna(0) > 0)
        &
        (df["ask"].fillna(0) > 0)
    )

    df["price_source"] = np.where(
        valid_bidask_filtered,
        "mid",
        "lastPrice"
    )

    df["option_price"] = np.where(
        valid_bidask_filtered,
        df["mid_price"],
        df["lastPrice"]
    )

    # ------------------------------------------------
    # Remove stale contracts
    # ------------------------------------------------

    stale_filter = (
        (df["volume"].fillna(0) == 0)
        &
        (df["openInterest"].fillna(0) == 0)
        &
        (df["bid"].fillna(0) == 0)
    )

    df = df[~stale_filter].copy()

    # ------------------------------------------------
    # Spread filter
    # ------------------------------------------------

    valid_mid = df["mid_price"] > 0

    spread_pct = np.where(
        valid_mid,
        (df["ask"] - df["bid"]) / df["mid_price"],
        np.nan
    )

    df["spread_pct"] = spread_pct

    df = df[
        (
            df["spread_pct"].isna()
        )
        |
        (
            df["spread_pct"] < 0.50
        )
    ]

    # ------------------------------------------------
    # Minimum option price
    # ------------------------------------------------

    df = df[
        df["option_price"] > 0.05
    ]

    # ------------------------------------------------
    # Implied volatility cleaning
    # ------------------------------------------------

    if "impliedVolatility" in df.columns:

        df = df[
            (df["impliedVolatility"] > 0)
            &
            (df["impliedVolatility"] < 3)
        ]

    # ------------------------------------------------
    # Remove duplicates
    # ------------------------------------------------

    df = df.drop_duplicates()

    # ------------------------------------------------
    # Sorting
    # ------------------------------------------------

    df = df.sort_values(
        ["expiry", "type", "strike"]
    )

    df = df.reset_index(drop=True)

    removed_rows = original_rows - len(df)

    print("\n===== CLEANING SUMMARY =====")

    print(
        "Original rows:",
        original_rows
    )

    print(
        "Rows after cleaning:",
        len(df)
    )

    print(
        "Removed rows:",
        removed_rows
    )

    return df


def print_data_quality_summary(
    clean_df: pd.DataFrame
) -> None:

    print("\n===== OPTION DATA QUALITY =====")

    print(
        "Total cleaned contracts:",
        len(clean_df)
    )

    maturities = clean_df["expiry"].nunique()

    print(
        "Unique maturities:",
        maturities
    )

    strikes_per_expiry = (
        clean_df.groupby("expiry")["strike"]
        .nunique()
        .sort_index()
    )

    print("\n===== STRIKES PER MATURITY =====")

    print(strikes_per_expiry)

    print(
        "\nMinimum strikes across maturities:",
        strikes_per_expiry.min()
    )

    print(
        "\nPrice source counts:"
    )

    print(
        clean_df["price_source"]
        .value_counts()
    )


def save_outputs(
    options_raw: pd.DataFrame,
    options_clean: pd.DataFrame,
    ticker_symbol: str = "QQQ"
) -> None:

    raw_path = (
        DATA_RAW
        / f"{ticker_symbol.lower()}_options_raw.csv"
    )

    clean_path = (
        DATA_CLEANED
        / f"{ticker_symbol.lower()}_options_clean.csv"
    )

    options_raw.to_csv(
        raw_path,
        index=False
    )

    options_clean.to_csv(
        clean_path,
        index=False
    )

    print(f"\nSaved raw options to:\n{raw_path}")

    print(f"\nSaved cleaned options to:\n{clean_path}")


def main() -> None:

    ensure_dirs()

    ticker_symbol = "QQQ"

    print(
        f"\nDownloading option chains for {ticker_symbol}..."
    )

    options_raw = download_option_chains(
        ticker_symbol=ticker_symbol,
        n_expiries=6
    )

    print("\nCleaning option data...")

    options_clean = clean_option_data(
        options_raw
    )

    save_outputs(
        options_raw,
        options_clean,
        ticker_symbol=ticker_symbol
    )

    print(
        "\n===== DATASET SHAPES ====="
    )

    print(
        "Raw shape:",
        options_raw.shape
    )

    print(
        "Clean shape:",
        options_clean.shape
    )

    print_data_quality_summary(
        options_clean
    )

    print("\n===== PREVIEW =====")

    print(
        options_clean.head()
    )

    print(
        "\nNOTE:"
    )

    print(
        "For best option quote quality, "
        "re-run the download during "
        "U.S. market trading hours."
    )


if __name__ == "__main__":
    main()