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


def compute_daily_jump_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abs_return"] = out["returns"].abs()

    threshold = out["returns"].std() * 2.0
    out["jump_flag"] = out["abs_return"] > threshold
    out["jump_size_proxy"] = out["abs_return"]

    return out


def summarize_jumps(daily: pd.DataFrame) -> pd.DataFrame:
    jump_days = daily[daily["jump_flag"]].copy()

    summary = pd.DataFrame({
        "metric": [
            "total_days",
            "jump_days",
            "jump_frequency",
            "avg_jump_size_proxy",
            "median_jump_size_proxy",
            "max_jump_size_proxy",
        ],
        "value": [
            len(daily),
            len(jump_days),
            len(jump_days) / max(len(daily), 1),
            jump_days["jump_size_proxy"].mean() if len(jump_days) else 0.0,
            jump_days["jump_size_proxy"].median() if len(jump_days) else 0.0,
            jump_days["jump_size_proxy"].max() if len(jump_days) else 0.0,
        ]
    })

    return summary


def plot_returns_and_jumps(daily: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(daily["date"], daily["returns"], label="Daily Returns")
    plt.scatter(
        daily.loc[daily["jump_flag"], "date"],
        daily.loc[daily["jump_flag"], "returns"],
        color="red",
        label="Jump Days",
        zorder=3
    )
    plt.title("QQQ Daily Returns with Jump Days")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "daily_jump_days.png", dpi=150)
    plt.show()


def plot_jump_histogram(daily: pd.DataFrame) -> None:
    jump_days = daily[daily["jump_flag"]]
    plt.figure(figsize=(10, 5))
    plt.hist(jump_days["jump_size_proxy"], bins=30)
    plt.title("Jump Size Distribution")
    plt.xlabel("Jump Size Proxy")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "jump_size_distribution.png", dpi=150)
    plt.show()


def main():
    input_file = DATA_RAW / "qqq_daily.csv"

    df = load_daily_data(input_file)
    daily = compute_daily_jump_proxy(df)
    summary = summarize_jumps(daily)

    daily.to_csv(OUTPUTS / "qqq_jump_daily_stats.csv", index=False)
    summary.to_csv(OUTPUTS / "jump_summary.csv", index=False)

    daily[daily["jump_flag"]].sort_values("jump_size_proxy", ascending=False).to_csv(
        OUTPUTS / "major_jump_dates.csv",
        index=False
    )

    print("\n===== JUMP DETECTION SUMMARY =====")
    print(summary.to_string(index=False))

    print("\n===== TOP JUMP DAYS =====")
    cols_to_show = ["date", "returns", "abs_return", "jump_size_proxy"]
    print(
        daily[daily["jump_flag"]]
        .sort_values("jump_size_proxy", ascending=False)[cols_to_show]
        .head(10)
        .to_string(index=False)
    )

    plot_returns_and_jumps(daily)
    plot_jump_histogram(daily)

    print("\nAll W5 outputs saved to:")
    print(OUTPUTS)


if __name__ == "__main__":
    main()