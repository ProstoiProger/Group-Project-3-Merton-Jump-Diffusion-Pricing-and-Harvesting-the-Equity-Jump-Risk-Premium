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

def load_5m_data(path: Path) -> pd.DataFrame:
    """
    Load QQQ 5-minute data and standardize columns.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(path)

    datetime_candidates = [
        "Datetime",
        "datetime",
        "Date",
        "date",
        "timestamp",
        "Timestamp"
    ]

    dt_col = next(
        (c for c in datetime_candidates if c in df.columns),
        None
    )

    if dt_col is None:
        raise ValueError(
            "No datetime column found."
        )

    df[dt_col] = pd.to_datetime(
        df[dt_col],
        errors="coerce"
    )

    df = df.dropna(subset=[dt_col]).copy()

    df = df.rename(
        columns={dt_col: "datetime"}
    )

    price_candidates = [
        "Close",
        "close",
        "Adj Close",
        "adj close",
        "Price",
        "price"
    ]

    price_col = next(
        (c for c in price_candidates if c in df.columns),
        None
    )

    if price_col is None and "returns" not in df.columns:
        raise ValueError(
            "No price column found."
        )

 
    if "returns" not in df.columns:

        df[price_col] = pd.to_numeric(
            df[price_col],
            errors="coerce"
        )

        df = df.dropna(
            subset=[price_col]
        ).copy()

        df = df.sort_values(
            "datetime"
        ).copy()

        df["returns"] = np.log(
            df[price_col] /
            df[price_col].shift(1)
        )

    else:

        df["returns"] = pd.to_numeric(
            df["returns"],
            errors="coerce"
        )

    df = df.dropna(subset=["returns"]).copy()

    # remove inf values
    df = df[np.isfinite(df["returns"])]

    # sort
    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    return df


def compute_daily_rv_bpv(
    df: pd.DataFrame
) -> pd.DataFrame:

    mu1 = np.sqrt(2 / np.pi)

    df = df.copy()

    df["date"] = df["datetime"].dt.date

    rows = []

    for date, g in df.groupby("date"):

        r = g["returns"].dropna().to_numpy()

        n = len(r)

        # skip very small samples
        if n < 3:
            continue


        rv = np.sum(r ** 2)


        bpv = (
            (1 / mu1**2)
            * np.sum(
                np.abs(r[:-1]) *
                np.abs(r[1:])
            )
        )

        bpv = max(bpv, 1e-12)

        r2 = r ** 2

        var_r2 = (
            np.var(r2, ddof=1)
            if n > 1 else 0.0
        )

        omega_hat = (
            np.sqrt(max(var_r2, 1e-12))
            / bpv
        )

        omega_hat = max(
            omega_hat,
            1e-12
        )


        z = (
            np.sqrt(n)
            * ((rv / bpv) - 1)
            / omega_hat
        )

        jump_flag = abs(z) > 1.96

        rows.append({
            "date": pd.to_datetime(date),
            "n_intraday_obs": n,
            "RV": rv,
            "BPV": bpv,
            "RV_to_BPV": rv / bpv,
            "omega_hat": omega_hat,
            "Z": z,
            "jump_flag": jump_flag,
            "jump_size_proxy": max(rv - bpv, 0.0)
        })

    daily = pd.DataFrame(rows)

    daily = daily.sort_values(
        "date"
    ).reset_index(drop=True)

    return daily


def summarize_jumps(
    daily: pd.DataFrame
) -> pd.DataFrame:

    jump_days = daily[
        daily["jump_flag"]
    ].copy()

    summary = pd.DataFrame({

        "metric": [
            "total_days",
            "jump_days",
            "jump_frequency",
            "avg_jump_size_proxy",
            "median_Z",
            "max_Z"
        ],

        "value": [

            len(daily),

            len(jump_days),

            len(jump_days) / max(len(daily), 1),

            jump_days[
                "jump_size_proxy"
            ].mean()
            if len(jump_days)
            else 0.0,

            daily["Z"].median()
            if len(daily)
            else np.nan,

            daily["Z"].max()
            if len(daily)
            else np.nan
        ]
    })

    return summary



def plot_rv_bpv(
    daily: pd.DataFrame
) -> None:

    plt.figure(figsize=(12, 6))

    plt.plot(
        daily["date"],
        daily["RV"],
        label="RV"
    )

    plt.plot(
        daily["date"],
        daily["BPV"],
        label="BPV"
    )

    plt.title("Daily RV vs BPV")
    plt.xlabel("Date")
    plt.ylabel("Value")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUTS / "rv_bpv_timeseries.png",
        dpi=150
    )

    plt.show()


def plot_jump_stat(
    daily: pd.DataFrame
) -> None:

    plt.figure(figsize=(12, 6))

    plt.plot(
        daily["date"],
        daily["Z"],
        label="Z Statistic"
    )

    plt.axhline(
        1.96,
        linestyle="--",
        label="+1.96 Threshold"
    )

    plt.axhline(
        -1.96,
        linestyle="--",
        label="-1.96 Threshold"
    )

    plt.title("BNS Jump Test Statistic")

    plt.xlabel("Date")
    plt.ylabel("Z")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUTS / "jump_statistic_timeseries.png",
        dpi=150
    )

    plt.show()


def plot_jump_days(
    daily: pd.DataFrame
) -> None:

    plt.figure(figsize=(12, 6))

    plt.plot(
        daily["date"],
        daily["RV_to_BPV"],
        label="RV/BPV"
    )

    jump_days = daily[
        daily["jump_flag"]
    ]

    plt.scatter(
        jump_days["date"],
        jump_days["RV_to_BPV"],
        color="red",
        label="Jump Days",
        zorder=3
    )

    # annotate biggest jumps
    top_jumps = jump_days.nlargest(
        5,
        "jump_size_proxy"
    )

    for _, row in top_jumps.iterrows():

        plt.annotate(
            row["date"].strftime("%Y-%m-%d"),
            (
                row["date"],
                row["RV_to_BPV"]
            ),
            fontsize=8
        )

    plt.title("Detected Jump Days")

    plt.xlabel("Date")
    plt.ylabel("RV/BPV")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUTS / "jump_days_highlighted.png",
        dpi=150
    )

    plt.show()


def plot_jump_histogram(
    daily: pd.DataFrame
) -> None:

    jump_days = daily[
        daily["jump_flag"]
    ]

    plt.figure(figsize=(10, 5))

    plt.hist(
        jump_days["jump_size_proxy"],
        bins=30
    )

    plt.title("Jump Size Distribution")

    plt.xlabel("Jump Size Proxy")
    plt.ylabel("Frequency")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUTS / "jump_size_distribution.png",
        dpi=150
    )

    plt.show()



def main():

    input_file = DATA_RAW / "qqq_5m.csv"
    df = load_5m_data(input_file)
    daily = compute_daily_rv_bpv(df)
    summary = summarize_jumps(daily)
    daily.to_csv(
        OUTPUTS / "qqq_jump_daily_stats.csv",
        index=False
    )

    summary.to_csv(
        OUTPUTS / "jump_summary.csv",
        index=False
    )

    daily[
        daily["jump_flag"]
    ].sort_values(
        "Z",
        ascending=False
    ).to_csv(
        OUTPUTS / "major_jump_dates.csv",
        index=False
    )

    print("\n===== JUMP DETECTION SUMMARY =====")

    print(
        summary.to_string(index=False)
    )

    print("\n===== TOP JUMP DAYS =====")

    cols_to_show = [
        "date",
        "RV",
        "BPV",
        "RV_to_BPV",
        "Z",
        "jump_size_proxy"
    ]

    print(
        daily[
            daily["jump_flag"]
        ]
        .sort_values(
            "Z",
            ascending=False
        )[cols_to_show]
        .head(10)
        .to_string(index=False)
    )

    plot_rv_bpv(daily)

    plot_jump_stat(daily)

    plot_jump_days(daily)

    plot_jump_histogram(daily)

    print("\nAll W5 outputs saved to:")
    print(OUTPUTS)

if __name__ == "__main__":
    main()