import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_RAW = Path("data/raw")
OUTPUTS = Path("outputs")
DATA_RAW.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)


def download_qqq_daily(period="3y"):
    qqq = yf.download(
        "QQQ",
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in qqq.columns
        ]

    qqq = qqq.reset_index()
    qqq["returns"] = np.log(qqq["Close"] / qqq["Close"].shift(1))
    qqq = qqq.dropna().copy()

    qqq.to_csv(DATA_RAW / "qqq_daily.csv", index=False)
    return qqq


def plot_prices(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"])
    plt.title("QQQ Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "qqq_price.png", dpi=150)
    plt.show()


def plot_returns(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["returns"])
    plt.title("QQQ Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "qqq_returns.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    qqq = download_qqq_daily()
    print(qqq.head())
    plot_prices(qqq)
    plot_returns(qqq)

    print("\nDataset Info:")
    print(qqq.info())
    print("\nSummary Statistics:")
    print(qqq.describe())
    print("\nMissing Values:")
    print(qqq.isnull().sum())

    plt.figure(figsize=(10, 5))
    plt.hist(qqq["returns"], bins=50)
    plt.title("Distribution of QQQ Returns")
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "returns_histogram.png", dpi=150)
    plt.show()

    volatility = qqq["returns"].std()
    print(f"Volatility: {volatility}")

    largest_jumps = qqq["returns"].abs().sort_values(ascending=False).head(10)
    print("\nLargest Jumps:")
    print(largest_jumps)

    threshold = 2 * volatility
    jumps = qqq[qqq["returns"].abs() > threshold]

    print("\nDetected Jumps:")
    print(jumps[["Date", "Close", "returns"]].head())
    print(f"\nNumber of jumps detected: {len(jumps)}")

    plt.figure(figsize=(12, 6))
    plt.plot(qqq["Date"], qqq["returns"], label="Returns")
    plt.scatter(
        jumps["Date"],
        jumps["returns"],
        color="red",
        label="Jumps"
    )

    plt.title("Detected Jumps in QQQ Returns")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "detected_jumps.png", dpi=150)
    plt.show()