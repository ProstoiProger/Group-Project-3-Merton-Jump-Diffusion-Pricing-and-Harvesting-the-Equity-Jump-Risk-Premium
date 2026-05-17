"""Command-line entry point for the W3 Merton jump-diffusion pipeline.

The module wires together the configuration loader, the experiment builders,
and the plotting layer into a single reproducible pipeline.  Running the
script from the project root reproduces every CSV and PDF deliverable used
in the write-up.

Notation is consistent with the W2 derivations:

    dX_t = (r - lambda * kappa - 0.5 * sigma**2) dt + sigma dW_t + J dN_t,
    kappa = E[exp(J)-1] = exp(mu_J + 0.5 * sigma_J**2) - 1.
"""

from __future__ import annotations

import logging

from config import (
    MarketParams,
    RunConfig,
    build_arg_parser,
    configure_logging,
    load_configuration,
)
from experiments import (
    build_sensitivity_table,
    build_smile_table,
    build_validation_table,
    build_variance_reduction_table,
    run_convergence_study,
    run_time_grid_stability,
)
from plots import save_convergence_plot, save_sensitivity_plot, save_smile_plot

LOGGER = logging.getLogger(__name__)


def write_outputs(
    params: MarketParams,
    config: RunConfig,
    path_sizes: list[int],
    step_sizes: list[int],
    strikes: list[float],
) -> None:
    """Run the full W3 pipeline and write CSV/PDF outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing W3 outputs to %s", config.output_dir)

    validation = build_validation_table(params, config)
    variance = build_variance_reduction_table(params, config)
    sensitivity = build_sensitivity_table(params, config)
    convergence = run_convergence_study(params, config, path_sizes)
    stability = run_time_grid_stability(params, config, step_sizes)
    smile = build_smile_table(params, strikes, config.merton_terms)

    validation.to_csv(config.output_dir / "validation_summary.csv", index=False)
    variance.to_csv(config.output_dir / "variance_reduction_table.csv", index=False)
    sensitivity.to_csv(config.output_dir / "control_variate_sensitivity.csv", index=False)
    convergence.to_csv(config.output_dir / "path_convergence.csv", index=False)
    stability.to_csv(config.output_dir / "time_grid_stability.csv", index=False)
    smile.to_csv(config.output_dir / "implied_vol_smile.csv", index=False)

    save_convergence_plot(convergence, stability, config.output_dir / "convergence_study.pdf")
    save_smile_plot(smile, config.output_dir / "implied_vol_smile.pdf")
    save_sensitivity_plot(sensitivity, config.output_dir / "control_variate_sensitivity.pdf")

    LOGGER.info("Validation summary:\n%s", validation.to_string(index=False))
    LOGGER.info("Variance reduction summary:\n%s", variance.to_string(index=False))
    LOGGER.info("Control-variate sensitivity:\n%s", sensitivity.to_string(index=False))


def main() -> None:
    """CLI entry point for the W3 Monte Carlo pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()
    params, config, path_sizes, step_sizes, strikes = load_configuration(args)
    configure_logging(config.log_level)

    LOGGER.info("Starting Merton MC pipeline")
    LOGGER.info("Market parameters: %s", params)
    LOGGER.info("Run configuration: %s", config)
    write_outputs(params, config, path_sizes, step_sizes, strikes)
    LOGGER.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
