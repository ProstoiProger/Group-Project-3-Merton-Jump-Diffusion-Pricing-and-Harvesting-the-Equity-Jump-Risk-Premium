"""Model calibration to the QQQ implied-volatility surface (W6).

Calibrates the Merton (1976) and Kou (2002) jump-diffusion parameters to
market option prices using non-linear least squares on the implied-vol surface.

Status: stub — implementation pending for the W6 workstream.

Planned outputs
---------------
- Calibrated MertonParams and KouParams
- Calibration residuals and RMSE
- Fitted vs market implied-vol surface comparison
- Jump risk premium: lambda^Q (risk-neutral) vs lambda^P (physical, from W5)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import DataConfig, KouParams, MertonParams, RunConfig

LOGGER = logging.getLogger(__name__)


def calibrate_merton(
    options: pd.DataFrame,
    s0: float,
    r: float,
    initial_guess: MertonParams | None = None,
) -> MertonParams:
    """Calibrate Merton (sigma, lambda, mu_j, sigma_j) to market implied vols.

    Parameters
    ----------
    options:
        Cleaned option chain with columns: strike, expiry, type, option_price.
    s0:
        Current spot price.
    r:
        Risk-free rate (annualized).
    initial_guess:
        Starting parameter vector; defaults to .env values.

    Returns
    -------
    MertonParams with calibrated values.
    """
    raise NotImplementedError("W6 calibration not yet implemented")


def calibrate_kou(
    options: pd.DataFrame,
    s0: float,
    r: float,
    merton_calibrated: MertonParams,
    initial_kou: KouParams | None = None,
) -> KouParams:
    """Calibrate Kou (p_up, eta1, eta2) given a calibrated Merton baseline.

    Returns
    -------
    KouParams with calibrated values.
    """
    raise NotImplementedError("W6 calibration not yet implemented")


def measure_jump_risk_premium(
    lambda_q: float,
    lambda_p: float,
) -> dict[str, float]:
    """Compute the equity jump risk premium: lambda^Q / lambda^P.

    Parameters
    ----------
    lambda_q:
        Risk-neutral jump intensity from calibration.
    lambda_p:
        Physical jump intensity from W5 BNS detection.

    Returns
    -------
    Dictionary with keys: lambda_q, lambda_p, jrp_ratio, jrp_log.
    """
    raise NotImplementedError("W6 jump risk premium not yet implemented")


def write_outputs(
    params: MertonParams,
    kou: KouParams,
    config: RunConfig,
    data_cfg: DataConfig,
    output_dir: Path | None = None,
) -> None:
    """Run the full W6 calibration pipeline and write outputs.

    Status: stub — raises NotImplementedError until W6 is implemented.
    """
    raise NotImplementedError("W6 calibration pipeline not yet implemented")
