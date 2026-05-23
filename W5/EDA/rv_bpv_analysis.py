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

WINDOW = 20                      # rolling window in trading days
MU1 = np.sqrt(2 / np.pi)         # E[|Z|] for Z ~ N(0,1)
VAR_RATIO_Z_THRESHOLD = 2.0      # descriptive flagging threshold
STRICT_VOL_MULTIPLIER = 3.0      # second confirmation rule for jump days


def load_daily_data(path: Path) -> pd.DataFrame:
    """
    Load daily QQQ data and standardize the columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    date_col = next(
        (c for c in ["Date", "date", "Datetime", "datetime"] if c in df.columns),
        None
    )
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
    """
    Compute rolling RV, BPV, log-ratio diagnostic, and jump flags.

    RV_t  = sum_{i in window} r_i^2

    BPV_t = (n/(n-1)) * (1/mu1^2) * sum_{i in window-1} |r_i| |r_{i-1}|

    log_ratio_t = ln(RV_t / BPV_t)

    Z_t = (log_ratio_t - mean_w(log_ratio_t)) / std_w(log_ratio_t)
    """
    out = df.copy()

    out["abs_return"] = out["returns"].abs()

    # Realized Variance
    out["RV"] = out["returns"].pow(2).rolling(
        window,
        min_periods=window
    ).sum()

    # Bipower Variation with finite-sample correction n/(n-1)
    bpv_term = (
        out["abs_return"].shift(1) * out["abs_return"]
    ).rolling(
        window - 1,
        min_periods=window - 1
    ).sum()

    out["BPV"] = (window / (window - 1)) * bpv_term / (MU1 ** 2)

    safe_bpv = out["BPV"].clip(lower=1e-12)
    out["RV_to_BPV"] = out["RV"] / safe_bpv
    OMEGA = np.sqrt((np.pi ** 2) / 4 + np.pi - 5)

    out["bns_style_z"] = (
        np.sqrt(window)
        * (out["RV_to_BPV"] - 1)
        / OMEGA
    )
    out["jump_component"] = out["RV"] - out["BPV"]

    # Rolling log-ratio diagnostic
    out["log_ratio"] = np.log(out["RV_to_BPV"].clip(lower=1e-12))

    out["log_ratio_mean"] = out["log_ratio"].rolling(
        window,
        min_periods=window
    ).mean()

    out["log_ratio_std"] = out["log_ratio"].rolling(
        window,
        min_periods=window
    ).std(ddof=0)

    out["ratio_z"] = (
        out["log_ratio"] - out["log_ratio_mean"]
    ) / out["log_ratio_std"].replace(0, np.nan)

    # Rolling daily volatility for the second confirmation filter
    out["rolling_vol"] = out["returns"].rolling(
        window,
        min_periods=window
    ).std(ddof=0)

    # Single-condition descriptive jump flag
    out["jump_flag_single"] = out["bns_style_z"].abs() > VAR_RATIO_Z_THRESHOLD

    # Two-condition confirmation filter
    rolling_vol_safe = out["rolling_vol"].replace(0, np.nan)
    out["jump_flag_strict"] = (
        out["jump_flag_single"]
        & (out["returns"].abs() > STRICT_VOL_MULTIPLIER * rolling_vol_safe)
    )

    # Backward-compatible alias
    out["jump_flag"] = out["jump_flag_single"]

    return out


def plot_rv_bpv(stats: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(stats["date"], stats["RV"], label="Rolling RV")
    plt.plot(stats["date"], stats["BPV"], label="Rolling BPV")
    plt.title("Rolling Realized Variance vs Bipower Variation (QQQ, 20-day window)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "rv_bpv_timeseries.png", dpi=150)
    plt.close()


def plot_log_ratio_z(stats: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(
        stats["date"],
        stats["ratio_z"],
        label="Rolling log-ratio Z score",
        alpha=0.85
    )
    plt.axhline(
        VAR_RATIO_Z_THRESHOLD,
        linestyle="--",
        color="red",
        label=f"+{VAR_RATIO_Z_THRESHOLD} flagging threshold"
    )
    plt.axhline(
        -VAR_RATIO_Z_THRESHOLD,
        linestyle="--",
        color="red",
        label=f"-{VAR_RATIO_Z_THRESHOLD} flagging threshold"
    )
    plt.title(
        "Rolling RV/BPV Log-Ratio Variance-Ratio Jump Diagnostic\n"
        "(empirical daily z-score; descriptive flagging only)"
    )
    plt.xlabel("Date")
    plt.ylabel("Z score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "rv_bpv_ratio_z.png", dpi=150)
    plt.close()


def print_summary(stats: pd.DataFrame) -> None:
    jump_days_single = stats[stats["jump_flag_single"]].copy()
    jump_days_strict = stats[stats["jump_flag_strict"]].copy()

    print("\n===== RV/BPV LOG-RATIO VARIANCE-RATIO SUMMARY =====")
    print(f"Average RV:                 {stats['RV'].mean():.6f}")
    print(f"Average BPV:                {stats['BPV'].mean():.6f}")
    print(f"Average jump component:     {stats['jump_component'].mean():.6f}")
    print(f"Single-condition jumps:     {len(jump_days_single)}")
    print(f"Two-condition jumps:        {len(jump_days_strict)}")
    print()

    tail_cols = ["date", "RV", "BPV", "RV_to_BPV", "log_ratio", "ratio_z"]
    print(stats[tail_cols].tail(10).to_string(index=False))


def main() -> None:
    input_file = DATA_RAW / "qqq_daily.csv"

    df = load_daily_data(input_file)
    stats = compute_rolling_rv_bpv(df, window=WINDOW)

    stats.to_csv(OUTPUTS / "rv_bpv_timeseries.csv", index=False)

    jump_days_single = stats[stats["jump_flag_single"]].copy()
    jump_days_single.to_csv(OUTPUTS / "rv_bpv_jump_days.csv", index=False)

    jump_days_strict = stats[stats["jump_flag_strict"]].copy()
    jump_days_strict.to_csv(OUTPUTS / "rv_bpv_jump_days_strict.csv", index=False)

    print_summary(stats)

    plot_rv_bpv(stats)
    plot_log_ratio_z(stats)

    print("\nSaved to:", OUTPUTS)


if __name__ == "__main__":
    main()