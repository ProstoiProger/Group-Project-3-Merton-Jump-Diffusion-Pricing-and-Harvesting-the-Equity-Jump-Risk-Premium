from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sc


try:
    FILE_DIR = Path(__file__).resolve().parent
except NameError:
    FILE_DIR = Path.cwd()

ROOT = FILE_DIR.parent if FILE_DIR.name == "EDA" else FILE_DIR
OUTPUTS = ROOT / "outputs" / "w5"
OUTPUTS.mkdir(parents=True, exist_ok=True)

VAR_RATIO_Z_THRESHOLD = 2.0
ROLLING_VOL_MULTIPLIER = 3.0
MIN_LOGNORMAL_OBS = 5
ROLLING_WINDOW = 20


def _save_and_show(filename: str, show: bool = True) -> None:
    """
    Save the current figure to OUTPUTS and optionally show it.
    """
    plt.tight_layout()
    plt.savefig(OUTPUTS / filename, dpi=150)
    if show:
        plt.show()
    plt.close()


def load_rv_bpv_data(path: Path) -> pd.DataFrame:
    """
    Load rolling RV/BPV diagnostics produced by rv_bpv_analysis.py.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "date",
        "returns",
        "RV",
        "BPV",
        "RV_to_BPV",
        "bns_style_z",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in rv_bpv_timeseries.csv: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "returns",
        "RV",
        "BPV",
        "RV_to_BPV",
        "bns_style_z",
        "ratio_z",
        "jump_component",
        "rolling_vol",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "date",
            "returns",
            "RV",
            "BPV",
            "RV_to_BPV",
            "bns_style_z",
        ]
    ).copy()

    df = df.sort_values("date").reset_index(drop=True)
    return df


def mark_jumps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create jump flags and jump scores.
    """
    out = df.copy()

    out["abs_return"] = out["returns"].abs()

    out["jump_flag_single"] = (
        out["bns_style_z"].abs() > VAR_RATIO_Z_THRESHOLD
    )

    if (
        "rolling_vol" not in out.columns
        or out["rolling_vol"].isna().all()
    ):
        out["rolling_vol"] = (
            out["returns"]
            .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
            .std(ddof=0)
        )

    out["jump_flag_two"] = (
        out["jump_flag_single"]
        & (
            out["abs_return"]
            > ROLLING_VOL_MULTIPLIER * out["rolling_vol"]
        )
    )

    out["jump_score"] = out["bns_style_z"].abs()
    out["jump_size_proxy"] = out["abs_return"]

    # backward compatibility
    out["jump_flag"] = out["jump_flag_single"]

    return out


def summarize_jumps(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Create jump summary statistics.
    """
    valid_days = daily["bns_style_z"].notna().sum()

    single = daily[daily["jump_flag_single"]]
    two = daily[daily["jump_flag_two"]]

    summary = pd.DataFrame(
        {
            "metric": [
                "total_days",
                "valid_days_after_rolling_window",
                "jump_days_single_condition",
                "jump_days_two_condition",
                "jump_frequency_single",
                "jump_frequency_two",
                "annualized_jump_rate_single",
                "annualized_jump_rate_two",
                "avg_jump_size_proxy",
                "median_jump_size_proxy",
                "max_jump_size_proxy",
            ],
            "value": [
                len(daily),
                valid_days,
                len(single),
                len(two),
                len(single) / max(valid_days, 1),
                len(two) / max(valid_days, 1),
                (len(single) / max(valid_days, 1)) * 252,
                (len(two) / max(valid_days, 1)) * 252,
                single["jump_size_proxy"].mean() if len(single) else 0.0,
                single["jump_size_proxy"].median() if len(single) else 0.0,
                single["jump_size_proxy"].max() if len(single) else 0.0,
            ],
        }
    )
    return summary


def return_distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute return distribution diagnostics.
    """
    r = df["returns"].dropna()
    jb = sc.jarque_bera(r)

    jb_stat = float(jb.statistic)
    jb_p = float(jb.pvalue)

    stats_df = pd.DataFrame(
        {
            "metric": [
                "count",
                "mean_daily",
                "std_daily",
                "skewness",
                "excess_kurtosis",
                "jarque_bera_stat",
                "jarque_bera_pvalue",
                "min_return",
                "max_return",
            ],
            "value": [
                len(r),
                r.mean(),
                r.std(),
                r.skew(),
                r.kurtosis(),
                jb_stat,
                jb_p,
                r.min(),
                r.max(),
            ],
        }
    )
    return stats_df


def fit_lognormal_jump_sizes(jump_abs_returns: pd.Series) -> dict:
    """
    Fit a lognormal distribution to jump-size proxies.

    NOTE:
    Absolute daily return is only a proxy for jump size.
    It still contains diffusion volatility.
    """
    arr = jump_abs_returns.dropna().values
    arr = arr[arr > 0]

    if len(arr) < MIN_LOGNORMAL_OBS:
        return {}

    shape, loc, scale = sc.lognorm.fit(arr, floc=0)
    ks_stat, ks_p = sc.kstest(arr, "lognorm", args=(shape, loc, scale))

    return {
        "n_jump_obs": len(arr),
        "lognormal_sigma": shape,
        "lognormal_mu": np.log(scale),
        "lognormal_loc": loc,
        "lognormal_scale": scale,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_p,
    }


def plot_returns_and_jumps(daily: pd.DataFrame, show: bool = True) -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(
        daily["date"],
        daily["returns"],
        label="Daily Returns",
        lw=0.8,
        alpha=0.8,
    )

    plt.scatter(
        daily.loc[daily["jump_flag_single"], "date"],
        daily.loc[daily["jump_flag_single"], "returns"],
        color="orange",
        s=30,
        label="Single-condition jump",
        zorder=3,
    )

    plt.scatter(
        daily.loc[daily["jump_flag_two"], "date"],
        daily.loc[daily["jump_flag_two"], "returns"],
        color="red",
        marker="D",
        s=60,
        label="Two-condition jump",
        zorder=4,
    )

    plt.title("QQQ Daily Returns with Jump Diagnostics")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True)
    _save_and_show("daily_jump_days.png", show=show)


def plot_jump_histogram(
    daily: pd.DataFrame,
    lf: dict | None = None,
    show: bool = True
) -> None:
    """
    Plot jump-size histogram with fitted lognormal density.
    Saves:
    - jump_size_distribution.png
    - jump_lognormal_fit.png
    """
    jump_days = daily[daily["jump_flag_single"]]
    arr = jump_days["jump_size_proxy"].dropna()
    arr = arr[arr > 0]

    if len(arr) == 0:
        return

    if lf is None:
        lf = fit_lognormal_jump_sizes(arr)

    plt.figure(figsize=(10, 6))
    plt.hist(
        arr,
        bins=25,
        density=True,
        alpha=0.7,
        label="Empirical jump-size proxy",
    )

    if lf:
        x = np.linspace(arr.min(), arr.max(), 300)
        pdf = sc.lognorm.pdf(
            x,
            lf["lognormal_sigma"],
            lf["lognormal_loc"],
            lf["lognormal_scale"],
        )
        plt.plot(
            x,
            pdf,
            "r-",
            lw=2,
            label=(
                f"Lognormal fit "
                f"(KS p-value = {lf['ks_pvalue']:.3f})"
            ),
        )

    plt.title("Jump-Size Proxy Distribution vs Lognormal Fit")
    plt.xlabel("|r_t| on flagged days")
    plt.ylabel("Density")
    plt.legend(fontsize=8)
    plt.grid(True)
    _save_and_show("jump_size_distribution.png", show=show)

    # save a second copy with an explicit lognormal name
    # so the report can reference it directly
    plt.figure(figsize=(10, 6))
    plt.hist(
        arr,
        bins=25,
        density=True,
        alpha=0.7,
        label="Empirical jump-size proxy",
    )
    if lf:
        x = np.linspace(arr.min(), arr.max(), 300)
        pdf = sc.lognorm.pdf(
            x,
            lf["lognormal_sigma"],
            lf["lognormal_loc"],
            lf["lognormal_scale"],
        )
        plt.plot(
            x,
            pdf,
            "r-",
            lw=2,
            label=(
                f"Lognormal fit "
                f"(KS p-value = {lf['ks_pvalue']:.3f})"
            ),
        )
    plt.title("Jump-Size Proxy Distribution vs Lognormal Fit")
    plt.xlabel("|r_t| on flagged days")
    plt.ylabel("Density")
    plt.legend(fontsize=8)
    plt.grid(True)
    _save_and_show("jump_lognormal_fit.png", show=show)


def plot_lognormal_qq(
    daily: pd.DataFrame,
    lf: dict | None = None,
    show: bool = True
) -> None:
    """
    QQ plot for the fitted lognormal distribution.
    Saves:
    - jump_lognormal_qq.png
    """
    jump_days = daily[daily["jump_flag_single"]]
    arr = jump_days["jump_size_proxy"].dropna()
    arr = arr[arr > 0]

    if len(arr) < MIN_LOGNORMAL_OBS:
        return

    if lf is None:
        lf = fit_lognormal_jump_sizes(arr)

    if not lf:
        return

    emp = np.sort(arr.to_numpy())
    n = len(emp)
    p = (np.arange(1, n + 1) - 0.5) / n

    theo = sc.lognorm.ppf(
        p,
        lf["lognormal_sigma"],
        lf["lognormal_loc"],
        lf["lognormal_scale"],
    )

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(theo, emp, s=20)

    lims = [
        min(np.min(theo), np.min(emp)),
        max(np.max(theo), np.max(emp)),
    ]
    plt.plot(lims, lims, linestyle="--", color="black", linewidth=1)

    plt.title("QQ Plot: Empirical Jump Sizes vs Lognormal")
    plt.xlabel("Theoretical lognormal quantiles")
    plt.ylabel("Empirical jump-size quantiles")
    plt.grid(True)
    _save_and_show("jump_lognormal_qq.png", show=show)


def plot_diagnostics(daily: pd.DataFrame, show: bool = True) -> None:
    """
    Plot RV/BPV diagnostics and Z-score.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(
        daily["date"],
        daily["RV_to_BPV"],
        label="RV/BPV",
        lw=0.8,
        alpha=0.8,
    )

    plt.scatter(
        daily.loc[daily["jump_flag_single"], "date"],
        daily.loc[daily["jump_flag_single"], "RV_to_BPV"],
        color="orange",
        s=25,
        label="Single-condition jump",
        zorder=3,
    )

    plt.scatter(
        daily.loc[daily["jump_flag_two"], "date"],
        daily.loc[daily["jump_flag_two"], "RV_to_BPV"],
        color="red",
        marker="D",
        s=50,
        label="Two-condition jump",
        zorder=4,
    )

    plt.title("Rolling RV/BPV Ratio with Jump Days")
    plt.xlabel("Date")
    plt.ylabel("RV / BPV")
    plt.legend()
    plt.grid(True)
    _save_and_show("jump_days_highlighted.png", show=show)

    plt.figure(figsize=(12, 6))
    plt.plot(
        daily["date"],
        daily["bns_style_z"],
        label="Log-Ratio Z score",
        lw=0.8,
        alpha=0.8,
    )

    plt.axhline(
        VAR_RATIO_Z_THRESHOLD,
        linestyle="--",
        color="red",
        label=f"+{VAR_RATIO_Z_THRESHOLD} threshold",
    )

    plt.axhline(
        -VAR_RATIO_Z_THRESHOLD,
        linestyle="--",
        color="red",
        label=f"-{VAR_RATIO_Z_THRESHOLD} threshold",
    )

    plt.title("Rolling RV/BPV BNS-Style Jump Diagnostic")
    plt.xlabel("Date")
    plt.ylabel("Z score")
    plt.legend()
    plt.grid(True)
    _save_and_show("jump_statistic_timeseries.png", show=show)

    plt.figure(figsize=(12, 6))
    plt.plot(
        daily["date"],
        daily["RV"],
        label="Rolling RV",
    )
    plt.plot(
        daily["date"],
        daily["BPV"],
        label="Rolling BPV",
    )
    plt.title("Rolling RV vs BPV")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    _save_and_show("rv_bpv_timeseries.png", show=show)


def main():
    input_file = OUTPUTS / "rv_bpv_timeseries.csv"

    df = load_rv_bpv_data(input_file)
    daily = mark_jumps(df)

    summary = summarize_jumps(daily)
    dist_stats = return_distribution_stats(daily)

    daily.to_csv(OUTPUTS / "qqq_jump_daily_stats.csv", index=False)
    summary.to_csv(OUTPUTS / "jump_summary.csv", index=False)
    dist_stats.to_csv(OUTPUTS / "return_dist_stats.csv", index=False)

    major_jump_dates = (
        daily[daily["jump_flag_single"]]
        .sort_values(["jump_score", "jump_size_proxy"], ascending=False)
        .copy()
    )
    major_jump_dates.to_csv(OUTPUTS / "major_jump_dates.csv", index=False)

    lf = fit_lognormal_jump_sizes(
        daily[daily["jump_flag_single"]]["jump_size_proxy"]
    )
    lf_df = pd.DataFrame([lf]) if lf else pd.DataFrame()
    lf_df.to_csv(OUTPUTS / "lognormal_fit.csv", index=False)

    # Plots now show AND save
    plot_returns_and_jumps(daily, show=True)
    plot_jump_histogram(daily, lf, show=True)
    plot_lognormal_qq(daily, lf, show=True)
    plot_diagnostics(daily, show=True)

    print("\n===== JUMP DETECTION SUMMARY =====")
    print(summary.to_string(index=False))

    print("\n===== RETURN DISTRIBUTION STATISTICS =====")
    print(dist_stats.to_string(index=False))

    print("\n===== LOGNORMAL FIT =====")
    if lf:
        for k, v in lf.items():
            print(f"{k}: {v:.6g}")
    else:
        print("Not enough jump observations for a stable lognormal fit.")

    print("\n===== TOP 10 JUMP DAYS =====")
    cols_to_show = [
        "date",
        "returns",
        "abs_return",
        "RV_to_BPV",
        "bns_style_z",
        "jump_flag_two",
    ]
    print(
        major_jump_dates[cols_to_show]
        .head(10)
        .to_string(index=False)
    )

    print("\nAll W5 outputs saved to:")
    print(OUTPUTS)


if __name__ == "__main__":
    main()