"""Pre-earnings implied-volatility strategy backtest (W8-B).

Harvests the equity jump risk premium by selling short-dated options before
earnings announcements and capturing the subsequent implied-vol collapse.

Strategy logic
--------------
  1. Identify QQQ earnings calendar dates.
  2. Enter short straddle (sell ATM call + put) N days before earnings.
  3. Exit position on the earnings date or the day after.
  4. Record per-trade P&L, net of bid-ask spread.

Status: stub — implementation pending for the W8 workstream.

Planned outputs
---------------
- Per-trade P&L DataFrame
- Cumulative equity curve
- Sharpe ratio, max drawdown, win rate
- Factor regression (market beta, VIX beta)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import DataConfig, MertonParams, RunConfig

LOGGER = logging.getLogger(__name__)


def load_earnings_calendar(ticker: str) -> pd.DataFrame:
    """Download or load earnings announcement dates for a ticker.

    Returns a DataFrame with at least a 'date' column.
    Status: stub — raises NotImplementedError.
    """
    raise NotImplementedError("W8 earnings calendar not yet implemented")


def run_strategy(
    options: pd.DataFrame,
    earnings_dates: pd.DataFrame,
    entry_days_before: int = 5,
) -> pd.DataFrame:
    """Execute the pre-earnings implied-vol collapse strategy.

    Parameters
    ----------
    options:
        Cleaned option chain with historical prices.
    earnings_dates:
        DataFrame of earnings announcement dates.
    entry_days_before:
        How many trading days before earnings to enter the short straddle.

    Returns
    -------
    DataFrame with per-trade P&L.
    Status: stub — raises NotImplementedError.
    """
    raise NotImplementedError("W8 strategy not yet implemented")


def compute_performance(trades: pd.DataFrame) -> dict[str, float]:
    """Compute Sharpe ratio, max drawdown, win rate, and annualized return.

    Status: stub — raises NotImplementedError.
    """
    raise NotImplementedError("W8 performance metrics not yet implemented")


def write_outputs(
    params: MertonParams,
    config: RunConfig,
    data_cfg: DataConfig,
    output_dir: Path | None = None,
) -> None:
    """Run the full W8-B backtest pipeline and write outputs.

    Status: stub — raises NotImplementedError until W8 is implemented.
    """
    raise NotImplementedError("W8 backtest pipeline not yet implemented")
