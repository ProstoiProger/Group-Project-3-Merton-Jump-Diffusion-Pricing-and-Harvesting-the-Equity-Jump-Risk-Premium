"""Configuration primitives for the W3 Monte Carlo pipeline.

This module groups the parameter containers, environment-file loading, command
line argument parsing, and logging configuration in one place.  Keeping these
concerns separate from the numerical kernels makes the pipeline easier to
re-use from notebooks and unit tests.

Notation is consistent with the W2 derivations:

    dX_t = (r - lambda * kappa - 0.5 * sigma**2) dt + sigma dW_t + J dN_t,
    kappa = E[exp(J)-1] = exp(mu_J + 0.5 * sigma_J**2) - 1.

The adjusted intensity ``lambda_prime = lambda * (1 + kappa)`` appears only in
the analytic Merton pricing formula after the algebraic transformation derived
in W2.  The Monte Carlo simulator always uses the original Poisson intensity.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


@dataclass(frozen=True)
class MarketParams:
    """Risk-neutral Merton jump-diffusion parameters."""

    s0: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    rate: float = 0.05
    sigma: float = 0.20
    jump_intensity: float = 1.00
    jump_mean: float = -0.10
    jump_vol: float = 0.25

    @property
    def kappa(self) -> float:
        """Expected percentage jump size, E[exp(J)-1]."""
        return math.exp(self.jump_mean + 0.5 * self.jump_vol**2) - 1.0

    @property
    def lambda_prime(self) -> float:
        """Adjusted intensity used only in the Merton analytic formula."""
        return self.jump_intensity * (1.0 + self.kappa)

    def with_overrides(self, **overrides: float) -> "MarketParams":
        """Return a copy of the parameters with selected fields replaced."""
        values = {
            "s0": self.s0,
            "strike": self.strike,
            "maturity": self.maturity,
            "rate": self.rate,
            "sigma": self.sigma,
            "jump_intensity": self.jump_intensity,
            "jump_mean": self.jump_mean,
            "jump_vol": self.jump_vol,
        }
        values.update(overrides)
        return MarketParams(**values)


@dataclass(frozen=True)
class RunConfig:
    """Configuration for the reproducible experiment pipeline."""

    paths: int = 50_000
    steps: int = 252
    seed: int = 42
    replications: int = 5
    merton_terms: int = 150
    output_dir: Path = Path("results")
    log_level: str = "INFO"


@dataclass(frozen=True)
class SmileScenario:
    """Named parameter overrides used in the implied-volatility smile comparison."""

    label: str
    overrides: dict[str, float] = field(default_factory=dict)


def configure_logging(level: str) -> None:
    """Configure consistent console logging for command-line runs."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_dotenv_from_cwd() -> None:
    """Load .env from the current working directory if it exists.

    The project uses the conventional python-dotenv pattern: put a `.env` file
    next to the script or run the script from a folder containing `.env`.  CLI
    arguments still override values loaded from the environment.
    """
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a default."""
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a default."""
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_path(name: str, default: str) -> Path:
    """Read a filesystem path from the environment."""
    return Path(os.getenv(name, default)).expanduser().resolve()


def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface.

    Configuration is loaded automatically from `.env` using python-dotenv.  CLI
    arguments override environment variables.
    """
    parser = argparse.ArgumentParser(description="Run Merton jump-diffusion Monte Carlo experiments.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for CSV and PDF outputs.")
    parser.add_argument("--paths", type=int, default=None, help="Number of Monte Carlo paths.")
    parser.add_argument("--steps", type=int, default=None, help="Number of time steps for stability checks.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed.")
    parser.add_argument("--replications", type=int, default=None, help="Number of replications for RMSE studies.")
    parser.add_argument("--path-sizes", type=str, default=None, help="Comma-separated path sizes.")
    parser.add_argument("--step-sizes", type=str, default=None, help="Comma-separated step sizes.")
    parser.add_argument("--strikes", type=str, default=None, help="Comma-separated strikes for smile plot.")
    parser.add_argument("--log-level", type=str, default=None, help="Logging level.")
    return parser


def load_configuration(
    args: argparse.Namespace,
) -> tuple[MarketParams, RunConfig, list[int], list[int], list[float]]:
    """Load configuration from `.env` and CLI arguments.

    Precedence order is: CLI arguments > environment variables > defaults.
    """
    load_dotenv_from_cwd()

    params = MarketParams(
        s0=_env_float("W3_S0", 100.0),
        strike=_env_float("W3_STRIKE", 100.0),
        maturity=_env_float("W3_MATURITY", 1.0),
        rate=_env_float("W3_RATE", 0.05),
        sigma=_env_float("W3_SIGMA", 0.20),
        jump_intensity=_env_float("W3_JUMP_INTENSITY", 1.0),
        jump_mean=_env_float("W3_JUMP_MEAN", -0.10),
        jump_vol=_env_float("W3_JUMP_VOL", 0.25),
    )

    output_dir = args.output_dir if args.output_dir is not None else _env_path("W3_OUTPUT_DIR", "results")
    log_level = args.log_level or os.getenv("W3_LOG_LEVEL", "INFO")
    config = RunConfig(
        paths=args.paths or _env_int("W3_PATHS", 50_000),
        steps=args.steps or _env_int("W3_STEPS", 252),
        seed=args.seed or _env_int("W3_SEED", 42),
        replications=args.replications or _env_int("W3_REPLICATIONS", 5),
        merton_terms=_env_int("W3_MERTON_TERMS", 150),
        output_dir=Path(output_dir).expanduser().resolve(),
        log_level=log_level,
    )

    path_sizes = _parse_int_list(args.path_sizes or os.getenv("W3_PATH_SIZES", "1000,5000,10000,25000,50000"))
    step_sizes = _parse_int_list(args.step_sizes or os.getenv("W3_STEP_SIZES", "1,12,52,252"))
    strikes = _parse_float_list(args.strikes or os.getenv("W3_STRIKES", "70,80,90,100,110,120,130"))

    return params, config, path_sizes, step_sizes, strikes
