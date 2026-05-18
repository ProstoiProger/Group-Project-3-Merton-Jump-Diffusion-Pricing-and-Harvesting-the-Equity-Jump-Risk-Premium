from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    FILE_DIR = Path(__file__).resolve().parent
except NameError:
    FILE_DIR = Path.cwd()

ROOT = FILE_DIR.parent if FILE_DIR.name == "EDA" else FILE_DIR
OUTPUTS = ROOT / "outputs" / "w5"
OUTPUTS.mkdir(parents=True, exist_ok=True)

BNS_Z_THRESHOLD = 2.0


def load_rv_bpv_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "date",
        "returns",
        "RV",
        "BPV",
        "RV_to_BPV",
        "ratio_z",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in rv_bpv_timeseries.csv: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = ["returns", "RV", "BPV", "RV_to_BPV", "ratio_z", "jump_component"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "returns", "RV", "BPV", "RV_to_BPV", "ratio_z"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    return df


def mark_jumps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["abs_return"] = out["returns"].abs()
    out["jump_flag"] = out["ratio_z"].abs() > BNS_Z_THRESHOLD
    out["jump_score"] = out["ratio_z"].abs()
    out["jump_size_proxy"] = out["abs_return"]

    return out


def summarize_jumps(daily: pd.DataFrame) -> pd.DataFrame:
    jump_days = daily[daily["jump_flag"]].copy()

    summary = pd.DataFrame(
        {
            "metric": [
                "total_days",
                "jump_days",
                "jump_frequency",
                "annualized_jump_rate",
                "avg_jump_size_proxy",
                "median_jump_size_proxy",
                "max_jump_size_proxy",
            ],
            "value": [
                len(daily),
                len(jump_days),
                len(jump_days) / max(len(daily), 1),
                (len(jump_days) / max(len(daily), 1)) * 252,
                jump_days["jump_size_proxy"].mean() if len(jump_days) else 0.0,
                jump_days["jump_size_proxy"].median() if len(jump_days) else 0.0,
                jump_days["jump_size_proxy"].max() if len(jump_days) else 0.0,
            ],
        }
    )

    return summary


def plot_returns_and_jumps(daily: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(daily["date"], daily["returns"], label="Daily Returns")
    plt.scatter(
        daily.loc[daily["jump_flag"], "date"],
        daily.loc[daily["jump_flag"], "returns"],
        color="red",
        label="Jump Days",
        zorder=3,
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


def plot_diagnostics(daily: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(daily["date"], daily["RV_to_BPV"], label="RV/BPV")
    plt.scatter(
        daily.loc[daily["jump_flag"], "date"],
        daily.loc[daily["jump_flag"], "RV_to_BPV"],
        color="red",
        label="Jump Days",
        zorder=3,
    )
    plt.title("Rolling RV/BPV with Jump Days")
    plt.xlabel("Date")
    plt.ylabel("RV/BPV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "jump_days_highlighted.png", dpi=150)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(daily["date"], daily["ratio_z"], label="RV/BPV standardized score")
    plt.axhline(BNS_Z_THRESHOLD, linestyle="--", label=f"+{BNS_Z_THRESHOLD} threshold")
    plt.axhline(-BNS_Z_THRESHOLD, linestyle="--", label=f"-{BNS_Z_THRESHOLD} threshold")
    plt.title("Rolling RV/BPV Standardized Score")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "jump_statistic_timeseries.png", dpi=150)
    plt.show()


def main():
    input_file = OUTPUTS / "rv_bpv_timeseries.csv"

    df = load_rv_bpv_data(input_file)
    daily = mark_jumps(df)
    summary = summarize_jumps(daily)

    daily.to_csv(OUTPUTS / "qqq_jump_daily_stats.csv", index=False)
    summary.to_csv(OUTPUTS / "jump_summary.csv", index=False)

    major_jump_dates = (
        daily[daily["jump_flag"]]
        .sort_values(["jump_score", "jump_size_proxy"], ascending=False)
        .copy()
    )
    major_jump_dates.to_csv(OUTPUTS / "major_jump_dates.csv", index=False)

    print("\n===== JUMP DETECTION SUMMARY =====")
    print(summary.to_string(index=False))

    print("\n===== TOP JUMP DAYS =====")
    cols_to_show = ["date", "returns", "abs_return", "RV_to_BPV", "ratio_z", "jump_score"]
    print(
        major_jump_dates[cols_to_show]
        .head(10)
        .to_string(index=False)
    )

    plot_returns_and_jumps(daily)
    plot_jump_histogram(daily)
    plot_diagnostics(daily)

    print("\nAll W5 outputs saved to:")
    print(OUTPUTS)


if __name__ == "__main__":
    main()