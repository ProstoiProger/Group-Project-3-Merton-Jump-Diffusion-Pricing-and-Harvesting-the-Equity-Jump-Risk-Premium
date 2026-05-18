from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    FILE_DIR = Path(__file__).resolve().parent
except NameError:
    FILE_DIR = Path.cwd()

ROOT = FILE_DIR.parent if FILE_DIR.name == "EDA" else FILE_DIR
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs" / "w5"
DATA_RAW.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
WINDOW = 20
MU1 = np.sqrt(2 / np.pi)

def load_daily_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    date_col = next((c for c in ["Date", "date", "Datetime", "datetime"] if c in df.columns), None)
    if date_col is None:
        raise ValueError("No date column found.")
    if "returns" not in df.columns:
        raise ValueError("No returns column found in daily CSV.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["returns"] = pd.to_numeric(df["returns"], errors="coerce")
    df = df.dropna(subset=[date_col, "returns"]).copy()
    df = df.rename(columns={date_col: "date"})
    df = df.sort_values("date").reset_index(drop=True)
    df = df[np.isfinite(df["returns"])].copy()

    return df


def compute_rolling_rv_bpv(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    out = df.copy()
    out["abs_return"] = out["returns"].abs()
    out["RV"] = out["returns"].pow(2).rolling(window, min_periods=window).sum()
    out["BPV"] = (
        (out["abs_return"].shift(1) * out["abs_return"])
        .rolling(window - 1, min_periods=window - 1)
        .sum()
        / (MU1 ** 2)
    )
    out["RV_to_BPV"] = out["RV"] / out["BPV"]
    out["jump_component"] = out["RV"] - out["BPV"]
    ratio_mean = out["RV_to_BPV"].rolling(window, min_periods=window).mean()
    ratio_std = out["RV_to_BPV"].rolling(window, min_periods=window).std(ddof=0).replace(0, np.nan)
    out["ratio_z"] = (out["RV_to_BPV"] - ratio_mean) / ratio_std

    return out


def plot_rv_bpv(stats: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(stats["date"], stats["RV"], label="Rolling RV")
    plt.plot(stats["date"], stats["BPV"], label="Rolling BPV")
    plt.title("Rolling RV vs BPV")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "rv_bpv_timeseries.png", dpi=150)
    plt.show()


def plot_ratio_z(stats: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(stats["date"], stats["ratio_z"], label="RV/BPV standardized score")
    plt.axhline(2.0, linestyle="--", label="+2 threshold")
    plt.axhline(-2.0, linestyle="--", label="-2 threshold")
    plt.title("Rolling RV/BPV Standardized Score")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "rv_bpv_ratio_z.png", dpi=150)
    plt.show()


def main():
    input_file = DATA_RAW / "qqq_daily.csv"
    df = load_daily_data(input_file)
    stats = compute_rolling_rv_bpv(df, window=WINDOW)
    stats.to_csv(OUTPUTS / "rv_bpv_timeseries.csv", index=False)
    jump_days = stats[
        stats["ratio_z"].abs() > 2
    ].copy()
    jump_days.to_csv(
        OUTPUTS / "rv_bpv_jump_days.csv",
        index=False
    )
    print("\n===== RV/BPV SUMMARY =====")
    print(
    "\nAverage RV:",
    stats["RV"].mean()
    )
    print(
    "Average BPV:",
    stats["BPV"].mean()
    )
    print(
    "Average jump component:",
    stats["jump_component"].mean()
    )
    print(
    "Detected jump days:",
    len(jump_days)
    )
    print(stats[["date", "RV", "BPV", "RV_to_BPV", "ratio_z"]].tail(10).to_string(index=False))
    plot_rv_bpv(stats)
    plot_ratio_z(stats)
    print("\nSaved to:", OUTPUTS)


if __name__ == "__main__":
    main()