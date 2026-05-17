"""Experiment builders for the W3 Monte Carlo pipeline.

Each builder returns a :class:`pandas.DataFrame` that can be persisted as CSV
and consumed by the plotting layer.  The experiments cover the deliverables
required by the project brief: a baseline validation against the Merton
analytic formula, the variance-reduction table, a control-variate sensitivity
study, RMSE convergence in the number of paths, time-grid stability, and the
implied-volatility smile comparison.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from config import MarketParams, RunConfig, SmileScenario
from pricing import implied_volatility, merton_exact_call
from simulation import (
    merton_mc_call,
    price_from_terminal,
    simulate_terminal_prices_grid,
)


def build_validation_table(params: MarketParams, config: RunConfig) -> pd.DataFrame:
    """Compare exact Merton price to the baseline MC estimate."""
    exact = merton_exact_call(params, n_terms=config.merton_terms)
    mc = merton_mc_call(params, config)
    return pd.DataFrame(
        [
            {"method": "Merton exact", "price": exact, "standard_error": 0.0, "absolute_error": 0.0},
            {
                "method": "Monte Carlo",
                "price": mc.price,
                "standard_error": mc.standard_error,
                "absolute_error": abs(mc.price - exact),
            },
        ]
    )


def build_variance_reduction_table(params: MarketParams, config: RunConfig) -> pd.DataFrame:
    """Evaluate variance-reduction methods against the exact Merton price.

    The ``variance_reduction_factor`` column reports (SE_plain / SE_method)**2,
    so values above one indicate effective variance reduction.  Moment matching
    is a bias-correction device rather than a variance-reduction technique, so
    its factor is expected to be close to unity.
    """
    exact = merton_exact_call(params, n_terms=config.merton_terms)
    methods: Sequence[tuple[str, dict[str, bool]]] = [
        ("plain", {}),
        ("antithetic", {"antithetic": True}),
        ("moment_matching", {"moment_match": True}),
        ("control_variate", {"control_variate": True}),
        ("antithetic_moment_matching", {"antithetic": True, "moment_match": True}),
        ("antithetic_control_variate", {"antithetic": True, "control_variate": True}),
    ]

    rows: list[dict[str, float | str]] = []
    plain_se: float | None = None
    for name, kwargs in methods:
        result = merton_mc_call(params, config, **kwargs)
        if name == "plain":
            plain_se = result.standard_error
        se_ratio = (result.standard_error / plain_se) if plain_se else 1.0
        vr_factor = (1.0 / se_ratio) ** 2 if se_ratio > 0.0 else float("nan")
        rows.append(
            {
                "method": name,
                "price": result.price,
                "standard_error": result.standard_error,
                "absolute_error": abs(result.price - exact),
                "se_ratio_vs_plain": se_ratio,
                "variance_reduction_factor": vr_factor,
            }
        )

    return pd.DataFrame(rows)


def build_sensitivity_table(params: MarketParams, config: RunConfig) -> pd.DataFrame:
    """Report control-variate efficiency across jump-vol regimes.

    The control variate captures the diffusive component of the payoff
    variance; as the jump-size dispersion grows the residual jump variance
    dominates and the achievable reduction necessarily decreases.  The table
    makes that dependence explicit for the project write-up.
    """
    jump_vol_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rows: list[dict[str, float | str]] = []
    for jump_vol in jump_vol_grid:
        scenario_params = params.with_overrides(jump_vol=jump_vol)
        plain = merton_mc_call(scenario_params, config)
        cv = merton_mc_call(scenario_params, config, control_variate=True)
        se_ratio = cv.standard_error / plain.standard_error if plain.standard_error > 0.0 else float("nan")
        vr_factor = (1.0 / se_ratio) ** 2 if se_ratio and se_ratio > 0.0 else float("nan")
        rows.append(
            {
                "jump_vol": jump_vol,
                "plain_standard_error": plain.standard_error,
                "control_standard_error": cv.standard_error,
                "variance_reduction_factor": vr_factor,
            }
        )

    return pd.DataFrame(rows)


def run_convergence_study(
    params: MarketParams,
    config: RunConfig,
    path_sizes: Iterable[int],
) -> pd.DataFrame:
    """Run RMSE convergence in the number of Monte Carlo paths."""
    exact = merton_exact_call(params, n_terms=config.merton_terms)
    rows: list[dict[str, float | int]] = []

    for paths in path_sizes:
        estimates = []
        for replication in range(config.replications):
            run_config = RunConfig(
                paths=paths,
                steps=config.steps,
                seed=config.seed + replication,
                replications=config.replications,
                merton_terms=config.merton_terms,
                output_dir=config.output_dir,
                log_level=config.log_level,
            )
            estimates.append(merton_mc_call(params, run_config).price)

        estimates_array = np.asarray(estimates)
        rows.append(
            {
                "paths": paths,
                "mean_price": float(estimates_array.mean()),
                "rmse": float(np.sqrt(np.mean((estimates_array - exact) ** 2))),
            }
        )

    return pd.DataFrame(rows)


def run_time_grid_stability(
    params: MarketParams,
    config: RunConfig,
    step_sizes: Iterable[int],
) -> pd.DataFrame:
    """Check numerical stability as the time grid changes."""
    exact = merton_exact_call(params, n_terms=config.merton_terms)
    rows: list[dict[str, float | int]] = []

    for steps in step_sizes:
        estimates = []
        for replication in range(config.replications):
            terminal_prices = simulate_terminal_prices_grid(
                params,
                paths=config.paths,
                steps=steps,
                seed=config.seed + replication,
            )
            estimates.append(price_from_terminal(terminal_prices, params).price)

        estimates_array = np.asarray(estimates)
        rows.append(
            {
                "steps": steps,
                "mean_price": float(estimates_array.mean()),
                "rmse": float(np.sqrt(np.mean((estimates_array - exact) ** 2))),
            }
        )

    return pd.DataFrame(rows)


def build_smile_table(
    params: MarketParams,
    strikes: Iterable[float],
    n_terms: int,
    scenarios: Sequence[SmileScenario] | None = None,
) -> pd.DataFrame:
    """Compute implied vols across strikes for several model scenarios.

    The reference scenarios distinguish a pure Black--Scholes diffusion
    (``lambda = 0``), the Merton baseline, and a Merton variant with large
    negative jumps to illustrate the resulting smile asymmetry.
    """
    if scenarios is None:
        scenarios = (
            SmileScenario(label="gbm", overrides={"jump_intensity": 0.0}),
            SmileScenario(label="merton_baseline", overrides={}),
            SmileScenario(label="merton_large_negative_jumps", overrides={"jump_mean": -0.20}),
        )

    strike_list = list(strikes)
    rows: list[dict[str, float | str]] = []
    for scenario in scenarios:
        scenario_params = params.with_overrides(**scenario.overrides)
        for strike in strike_list:
            price = merton_exact_call(scenario_params, strike=strike, n_terms=n_terms)
            iv = implied_volatility(
                price,
                scenario_params.s0,
                strike,
                scenario_params.maturity,
                scenario_params.rate,
            )
            rows.append(
                {
                    "scenario": scenario.label,
                    "strike": strike,
                    "moneyness": strike / scenario_params.s0,
                    "model_price": price,
                    "implied_vol": iv,
                }
            )

    return pd.DataFrame(rows)
