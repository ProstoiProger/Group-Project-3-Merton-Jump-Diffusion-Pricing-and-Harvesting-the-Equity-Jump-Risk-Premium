"""Minimum-variance hedging under the Merton jump-diffusion model (W8-A).

Derives and implements the minimum-variance hedge ratio for options written on
a Merton jump-diffusion underlying.  Unlike the complete-market Black-Scholes
setting, the jump component cannot be fully hedged; the residual jump variance
drives the P&L distribution fat tails documented in the project write-up.

Status: stub — implementation pending for the W8 workstream.

Planned outputs
---------------
- Minimum-variance hedge ratio delta_mv(S, K, T, params)
- P&L distribution under delta-hedging (comparison: GBM vs Merton)
- Tail-risk metrics: kurtosis, VaR, CVaR
- Hedge effectiveness table
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import MertonParams, RunConfig

LOGGER = logging.getLogger(__name__)


def min_variance_delta(
    params: MertonParams,
    strike: float | None = None,
) -> float:
    """Return the minimum-variance hedge ratio for a European call.

    Under the Merton model the hedge ratio differs from the Black-Scholes delta
    because of the unhedgeable jump component.  The derivation (W8 theory)
    yields a modified delta that minimises the instantaneous P&L variance.

    Status: stub — raises NotImplementedError.
    """
    raise NotImplementedError("W8 min-variance delta not yet implemented")


def simulate_hedged_pnl(
    params: MertonParams,
    config: RunConfig,
    rebalance_freq: int = 1,
) -> pd.DataFrame:
    """Simulate P&L from a delta-hedging strategy over a discrete rebalancing grid.

    Parameters
    ----------
    rebalance_freq:
        Number of trading days between rebalances.

    Returns
    -------
    DataFrame with columns: path, terminal_pnl, max_drawdown.

    Status: stub — raises NotImplementedError.
    """
    raise NotImplementedError("W8 hedged P&L simulation not yet implemented")


def write_outputs(
    params: MertonParams,
    config: RunConfig,
    output_dir: Path | None = None,
) -> None:
    """Run the full W8-A hedging analysis and write outputs.

    Status: stub — raises NotImplementedError until W8 is implemented.
    """
    raise NotImplementedError("W8 hedging pipeline not yet implemented")
