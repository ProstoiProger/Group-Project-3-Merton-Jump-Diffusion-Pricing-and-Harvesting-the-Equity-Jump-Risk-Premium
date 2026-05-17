"""Plot renderers for the W3 Monte Carlo pipeline.

The plotting layer consumes the data frames produced in :mod:`experiments`
and persists the corresponding figures as PDF files.  Separating the rendering
from the experiment logic keeps both halves easy to test in isolation.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_convergence_plot(convergence: pd.DataFrame, stability: pd.DataFrame, output_path: Path) -> None:
    """Save convergence and stability diagnostics as a PDF figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].loglog(convergence["paths"], convergence["rmse"], marker="o", label="MC RMSE")
    reference = convergence["rmse"].iloc[0] * math.sqrt(convergence["paths"].iloc[0]) / np.sqrt(
        convergence["paths"].to_numpy()
    )
    axes[0].loglog(convergence["paths"], reference, linestyle="--", label=r"$1/\sqrt{N}$ reference")
    axes[0].set_title("Convergence in Number of Paths")
    axes[0].set_xlabel("Number of paths")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()

    axes[1].loglog(stability["steps"], stability["rmse"], marker="s")
    axes[1].set_title("Time-Grid Stability")
    axes[1].set_xlabel("Number of time steps")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_smile_plot(smile: pd.DataFrame, output_path: Path) -> None:
    """Save the implied-volatility smile comparison as a PDF figure.

    Each scenario produced by :func:`experiments.build_smile_table` is rendered
    as a separate curve, which makes the asymmetric Merton smile directly
    comparable to the flat Black--Scholes line and to the more pronounced
    skew induced by larger negative jumps.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    scenario_styles = {
        "gbm": {"linestyle": "--", "marker": "x", "label": r"GBM ($\lambda = 0$)"},
        "merton_baseline": {"marker": "o", "label": "Merton baseline"},
        "merton_large_negative_jumps": {"marker": "s", "label": r"Merton, $\mu_J = -0.20$"},
    }

    for scenario_name, group in smile.groupby("scenario"):
        style = scenario_styles.get(str(scenario_name), {"marker": "o", "label": str(scenario_name)})
        ax.plot(group["moneyness"], group["implied_vol"], **style)

    ax.set_title("Implied Volatility Smile")
    ax.set_xlabel("Moneyness K/S0")
    ax.set_ylabel("Implied volatility")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_sensitivity_plot(sensitivity: pd.DataFrame, output_path: Path) -> None:
    """Save the control-variate sensitivity plot as a PDF figure."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(sensitivity["jump_vol"], sensitivity["variance_reduction_factor"], marker="o")
    ax.axhline(1.0, linestyle=":", linewidth=0.8)
    ax.set_title("Control-Variate Efficiency vs Jump-Size Dispersion")
    ax.set_xlabel(r"Jump volatility $\sigma_J$")
    ax.set_ylabel("Variance reduction factor")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
