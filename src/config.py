"""Unified configuration for the Merton Jump-Diffusion pipeline.

All parameters are read from a .env file placed in the project root.
Precedence: environment variables > .env values > dataclass defaults.

Usage
-----
    from src.config import load_config
    merton, kou, run, data = load_config()
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parent.parent


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_env() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)
    else:
        env_file = ROOT / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=str(env_file), override=False)


def _f(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if not v else float(v)


def _i(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if not v else int(v)


def _s(name: str, default: str) -> str:
    return os.getenv(name, default)


# ── parameter dataclasses ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class MertonParams:
    """Risk-neutral Merton jump-diffusion parameters.

    Notation follows the W2 derivation:
        dX_t = (r - lambda*kappa - 0.5*sigma^2) dt + sigma dW_t + J dN_t
        kappa = E[exp(J) - 1] = exp(mu_j + 0.5*sigma_j^2) - 1
    """

    s0: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    rate: float = 0.05
    sigma: float = 0.20
    lam: float = 1.0
    mu_j: float = -0.10
    sigma_j: float = 0.25

    @property
    def kappa(self) -> float:
        """Expected percentage jump: E[exp(J) - 1]."""
        return math.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1.0

    @property
    def lam_prime(self) -> float:
        """Risk-neutral adjusted intensity for the Merton analytic formula."""
        return self.lam * (1.0 + self.kappa)

    def replace(self, **kwargs: float) -> "MertonParams":
        """Return a copy with selected fields replaced."""
        fields = {
            "s0": self.s0, "strike": self.strike, "maturity": self.maturity,
            "rate": self.rate, "sigma": self.sigma, "lam": self.lam,
            "mu_j": self.mu_j, "sigma_j": self.sigma_j,
        }
        fields.update(kwargs)
        return MertonParams(**fields)


@dataclass(frozen=True)
class KouParams:
    """Kou (2002) double-exponential model extra parameters.

    Jump size Y = log(J):
      Up-jump   (prob p_up):      Y ~ Exp(eta1),  mean = 1/eta1
      Down-jump (prob 1 - p_up):  Y ~ Exp(eta2),  mean = 1/eta2 (negative)
    """

    p_up: float = 0.40
    eta1: float = 10.0
    eta2: float = 5.0


@dataclass(frozen=True)
class RunConfig:
    """Monte Carlo and COS engine runtime settings."""

    paths: int = 50_000
    steps: int = 252
    seed: int = 42
    replications: int = 5
    merton_terms: int = 150
    cos_n: int = 512
    cos_l: int = 10
    output_dir: Path = ROOT / "outputs"


@dataclass(frozen=True)
class DataConfig:
    """Data collection and jump-detection settings for W5."""

    ticker: str = "QQQ"
    data_period: str = "3y"
    n_expiries: int = 6
    bns_window: int = 20
    bns_z_threshold: float = 2.0
    strict_vol_multiplier: float = 3.0
    data_raw: Path = ROOT / "data" / "raw"
    data_cleaned: Path = ROOT / "data" / "cleaned"
    outputs_w5: Path = ROOT / "outputs" / "w5"


# ── public API ────────────────────────────────────────────────────────────────

def load_config() -> tuple[MertonParams, KouParams, RunConfig, DataConfig]:
    """Load all configuration from .env + environment variables.

    Call once at application startup.  Downstream code should consume
    the returned dataclasses rather than calling os.getenv directly.
    """
    _load_env()

    merton = MertonParams(
        s0=_f("S0", 100.0),
        strike=_f("STRIKE", 100.0),
        maturity=_f("MATURITY", 1.0),
        rate=_f("RATE", 0.05),
        sigma=_f("SIGMA", 0.20),
        lam=_f("LAMBDA", 1.0),
        mu_j=_f("MU_J", -0.10),
        sigma_j=_f("SIGMA_J", 0.25),
    )

    kou = KouParams(
        p_up=_f("P_UP", 0.40),
        eta1=_f("ETA1", 10.0),
        eta2=_f("ETA2", 5.0),
    )

    run = RunConfig(
        paths=_i("PATHS", 50_000),
        steps=_i("STEPS", 252),
        seed=_i("SEED", 42),
        replications=_i("REPLICATIONS", 5),
        merton_terms=_i("MERTON_TERMS", 150),
        cos_n=_i("COS_N", 512),
        cos_l=_i("COS_L", 10),
        output_dir=Path(_s("OUTPUT_DIR", str(ROOT / "outputs"))),
    )

    data = DataConfig(
        ticker=_s("TICKER", "QQQ"),
        data_period=_s("DATA_PERIOD", "3y"),
        n_expiries=_i("N_EXPIRIES", 6),
        bns_window=_i("BNS_WINDOW", 20),
        bns_z_threshold=_f("BNS_Z_THRESHOLD", 2.0),
        strict_vol_multiplier=_f("STRICT_VOL_MULTIPLIER", 3.0),
        data_raw=Path(_s("DATA_RAW", str(ROOT / "data" / "raw"))),
        data_cleaned=Path(_s("DATA_CLEANED", str(ROOT / "data" / "cleaned"))),
        outputs_w5=Path(_s("OUTPUTS_W5", str(ROOT / "outputs" / "w5"))),
    )

    return merton, kou, run, data
