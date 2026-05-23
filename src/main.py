"""Unified pipeline entry point for the Merton Jump-Diffusion project.

Group 3: Merton Jump-Diffusion Pricing and Harvesting the Equity Jump Risk Premium

Stages
------
  --w3   Monte Carlo engine       (src/mc_engine.py)
  --w4   COS / Fourier engine     (src/cos_engine.py)
  --w5   Data collection + jump detection  (src/data_pipeline.py)

All parameters are loaded from the project-root .env file.
Run with no flags to execute all three stages.

Usage
-----
    python main.py              # all stages
    python main.py --w3         # Monte Carlo only
    python main.py --w4         # COS engine only
    python main.py --w5         # data pipeline only
    python main.py --w3 --w4    # any combination
    python main.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging

import matplotlib
matplotlib.use("Agg")

from .config import load_config
from .mc_engine import write_outputs as mc_write_outputs
from .cos_engine import write_outputs as cos_write_outputs
from .data_pipeline import run as data_run

LOGGER = logging.getLogger(__name__)


def run_w3() -> None:
    """Monte Carlo pricing engine (W3)."""
    merton, _kou, run, _data = load_config()
    LOGGER.info("=" * 60)
    LOGGER.info("W3 — Monte Carlo Engine")
    LOGGER.info("params: %s", merton)
    LOGGER.info("=" * 60)
    mc_write_outputs(merton, run)
    LOGGER.info("W3 complete.")


def run_w4() -> None:
    """COS / Fourier pricing engine (W4)."""
    merton, kou, run, _data = load_config()
    LOGGER.info("=" * 60)
    LOGGER.info("W4 — COS Pricing Engine")
    LOGGER.info("params: %s", merton)
    LOGGER.info("=" * 60)
    cos_write_outputs(merton, kou, run)
    LOGGER.info("W4 complete.")


def run_w5() -> None:
    """Data collection and jump detection (W5)."""
    _merton, _kou, _run, data = load_config()
    LOGGER.info("=" * 60)
    LOGGER.info("W5 — Data Collection & Jump Detection")
    LOGGER.info("ticker: %s  period: %s", data.ticker, data.data_period)
    LOGGER.info("=" * 60)
    data_run(data)
    LOGGER.info("W5 complete.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Group 3 — Merton Jump-Diffusion pipeline (run all stages with no flags)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--w3", action="store_true", help="Run W3: Monte Carlo engine.")
    parser.add_argument("--w4", action="store_true", help="Run W4: COS pricing engine.")
    parser.add_argument("--w5", action="store_true", help="Run W5: data collection + jump detection.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        metavar="LEVEL",
        help="Logging verbosity: DEBUG, INFO, WARNING (default: INFO).",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    run_all = not (args.w3 or args.w4 or args.w5)

    if run_all or args.w3:
        run_w3()
    if run_all or args.w4:
        run_w4()
    if run_all or args.w5:
        run_w5()

    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline complete.")


if __name__ == "__main__":
    main()
