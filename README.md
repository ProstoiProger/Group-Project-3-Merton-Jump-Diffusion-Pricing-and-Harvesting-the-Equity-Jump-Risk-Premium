# Group 3 — Merton Jump-Diffusion: Pricing and Harvesting the Equity Jump Risk Premium

**Stochastic Processes: Computational Finance — Spring 2026**

---

## Overview

This project implements the full quantitative finance research pipeline for the Merton (1976) jump-diffusion model applied to NASDAQ-100 (QQQ) equity options. We derive the model from first principles, build two independent pricing engines (Monte Carlo and COS/Fourier), collect and clean real market data, detect jumps in historical returns, calibrate the model to the implied volatility surface, and backtest a pre-earnings volatility strategy.

**Asset:** QQQ / IWM European options  
**Core model:** Merton (1976) jump-diffusion  
**Extension:** Kou (2002) double-exponential jump model

---

## Team

| # | Name | Role |
|---|------|------|
| 1 | Azamat Orazgali (BDA-2402) | Team Lead, W2 Theory Lead, and Integration Coordinator: Led the team workflow, assigned and monitored progress across workstreams, developed the W2 mathematical derivations, supported and finalized parts of the W3 Monte Carlo pipeline, organized the GitHub project structure, validated results before merge, and integrated the W3, W4, and W5 components into the final LaTeX paper and the final code |
| 2 | Bekzat Duisenkhan (BDA-2402) | W1 Research Lead: Responsible for the research direction of the work, including searching, selecting, and analyzing scientific articles related to stochastic volatility, jump-diffusion models, and option pricing. Based on the reviewed literature, developed the mathematical and financial foundations of the project, prepared the theoretical derivations for W1, and helped define the overall direction of the implementation |
| 3 | Aldiyar Zheksenaly (MCS-2401) | W1 Research Lead: Responsible for collecting and organizing academic sources related to stochastic volatility, jump-diffusion models, and quantitative finance. Conducted comparative analysis of selected research papers, summarized key methodologies and model assumptions, and prepared the theoretical background used in the W1 section |
| 4 | Shynggys Yessenbay (BDA-2401) | W2 Theory Lead: Helped develop and review the W2 mathematical derivations for the Merton jump-diffusion model, including notation consistency and theoretical validation |
| 5 | Almas Nurkaliev (BDA-2401) | W3 Monte Carlo Engineer: Developed the Monte Carlo pricing engine for the Merton jump-diffusion model, implementing the simulation pipeline across three core modules (config.py, pricing.py, simulation.py). Delivered all three required variance reduction methods (antithetic variates, moment matching, Black-Scholes control variate), convergence diagnostics, and the implied volatility smile experiments underlying the W3 section of the final paper |
| 6 | Yelnazar Nurym (MCS-2401) | W4 Fourier Engineer: Built and validated a fast, accurate pricing engine for European options under the Merton jump-diffusion model using the COS (Characteristic Option Pricing) method. Took the characteristic function, converted it into fast option prices using Fourier series and output prices in <1 millisecond per option |
| 7 | Dilnaz Yesseniyazova (BDA-2408) | W5 Data Scientist: Built the empirical data pipeline and prepared the raw QQQ data for Week 5. Downloaded 3 years of daily trading data from Yahoo Finance, calculated log returns, and formatted datasets for jump-detection analysis. Managed the option-chain component by collecting and cleaning QQQ option data across six periods, applying retention and no-arbitrage filters, and preparing the final cleaned dataset for Week 6. Also created exploratory analysis outputs, including price plots, return distributions, and descriptive statistics |
| 8 | Baimukhammed Turarbek (BDA-2408) | W5 Data Scientist: Developed the methodology and statistical framework for jump detection, including the RV/BPV structure, variance-ratio diagnostics, and confirmation filters to reduce false positives. Conducted the main empirical jump analysis, interpreted the results, and compared conservative and liberal jump estimates. Also analyzed jump-size distributions using a lognormal fit, KS testing, and Q-Q plots in relation to the Merton model |
| 9 | Nazar Nurzhankyzy (BDA-2409) | W6 Calibration Engineer: Calibrated Merton (1976) and Kou (2002) jump-diffusion models to a 424-quote QQQ option surface (six maturities, May 2026). Designed the two-stage `differential_evolution` + `L-BFGS-B` calibration pipeline, performed an identification study (2D loss surface and L1-regularisation path) showing that $(\lambda, \mu_J)$ are not uniquely identified from option prices, computed the jump variance risk premium against the W5 BNS-style physical intensities, and built the `src/iv_surface.py` handoff API used by W7. Also extended the W5 data-pipeline so that the option-chain loader produces a broad and clean term structure suitable for calibration |
| 10 | Madiyar Gabit (BDA-2406) | W7 Vol Analyst: Gathered all participants into a group, worked on presentations W1–W5, has the role of W7 included in the Final Part of the project |
| 11 | Temirkhan Smagulov (MCS-2401) | W8 Risk Lead: Validating Literature Review, has the role of W8 included in the Final Part of the project |

### Workstream Roles

| Workstream | Responsible |
|------------|-------------|
| W1 — Research Lead (Literature Review) | Aldiyar Zheksenaly, Bekzat Duisenkhan |
| W2 — Theory Lead (Mathematical Derivations) | Azamat Orazgali, Shynggys Yessenbay |
| W3 — MC Engineer (Monte Carlo Engine) | Almas Nurkaliev, Azamat Orazgali |
| W4 — Fourier Engineer (COS Engine) | Yelnazar Nurym |
| W5 — Data Scientist (QQQ Data & Jump Detection) | Dilnaz Yesseniyazova, Baimukhammed Turarbek |
| W6 — Calibration Engineer | Nazar Nurzhankyzy |
| W7 — Vol Analyst (Implied Vol Surface) | Madiyar Gabit |
| W8 — Risk Lead (Hedging & Alpha Backtest) | Temirkhan Smagulov |

---

## Project Structure

```
.
├── main.py                   # root entry point — forwards to src/main.py
├── requirements.txt          # all Python dependencies
├── .env.example              # parameter template (copy to .env and edit)
├── .env                      # local parameters — git-ignored
│
├── src/                      # production pipeline package
│   ├── config.py             # all parameters read from .env
│   ├── mc_engine.py          # W3: Monte Carlo engine (simulation, pricing, experiments)
│   ├── cos_engine.py         # W4: COS / Fourier engine (Merton CF, Kou CF, digitals)
│   ├── data_pipeline.py      # W5: data download, RV/BPV, jump detection
│   ├── calibration.py        # W6: Merton + Kou calibration, identification study, JVRP
│   ├── iv_surface.py         # W6 -> W7 handoff: model IV at any (K, T)
│   └── main.py               # CLI entry point (--w3, --w4, --w5, --w6)
│
├── data/
│   ├── raw/                  # downloaded CSVs (git-ignored)
│   └── cleaned/              # cleaned option data (git-ignored)
│
├── outputs/                  # all generated plots and CSVs (git-ignored)
│   ├── w3/                   # W3 results
│   ├── w4/                   # W4 results
│   ├── w5/                   # W5 results
│   └── w6/                   # W6 results (calibration CSVs, loss surface, smile fits)
│
├── notebooks/                # exploratory Jupyter notebooks
├── paper/                    # final compiled paper
├── slides/                   # presentation slides
│
├── W1/                       # Literature review
│   └── Merton Jump-Diffusion_Review.pdf
│
├── W2/                       # Mathematical derivations
│   ├── Jump models.tex
│   └── W2 Theory Lead.pdf
│
├── W3/                       # Original Monte Carlo source files
│   ├── config.py
│   ├── simulation.py
│   ├── pricing.py
│   ├── experiments.py
│   ├── plots.py
│   └── merton_mc.py
│
├── W4/                       # Original COS source files
│   └── CallPut_COS_Method.py
│
├── W5/                       # Original data pipeline source files
│   └── EDA/
│       ├── initial_eda.py
│       ├── options_data.py
│       ├── rv_bpv_analysis.py
│       └── jump_detection.py
│
├── W6/                       # Calibration deliverables
│   ├── W6_report.tex         # standalone LaTeX report
│   ├── W6_report.pdf         # compiled PDF
│   └── W6_section.tex        # drop-in section for main paper
│
└── Latex/                    # Compiled paper
    ├── Group3_full_W1-W5.tex # endterm version (W1-W5)
    └── Group3_full_W1-W6.tex # final version (W1-W6)
```

---

## Installation

```bash
# clone the repository
git clone <repo-url>
cd <repo-folder>

# create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# install all dependencies
pip install -r requirements.txt

# configure parameters (optional — defaults are already set)
cp .env.example .env
# edit .env to override any model or pipeline parameters
```

**Python version:** 3.10 or later

---

## Running the Pipeline

```bash
# run all four stages (W3 + W4 + W5 + W6)
python main.py

# run individual stages
python main.py --w3          # Monte Carlo engine
python main.py --w4          # COS / Fourier engine
python main.py --w5          # data collection + jump detection
python main.py --w6          # Merton + Kou calibration (requires W5 outputs)

# combine any two
python main.py --w3 --w4

# verbose output
python main.py --log-level DEBUG
```

> **Pipeline order:** W6 reads the cleaned options chain produced by W5
> (`data/cleaned/qqq_options_clean.csv`) and the jump-detection summary
> (`outputs/w5/jump_summary.csv`). Run `--w5` at least once before `--w6`.
> The full W6 stage takes ~2 hours on a laptop (DE on 96-quote sub-sample,
> followed by L-BFGS-B on 424 quotes, then identification study with
> 18x18 grid and 5-point regularisation path).

### What each stage produces

| Stage | Output location | Contents |
|-------|----------------|----------|
| W3 | `outputs/w3/` | `validation_summary.csv`, `variance_reduction_table.csv`, `convergence_study.pdf`, `implied_vol_smile.pdf`, `control_variate_sensitivity.pdf` |
| W4 | `outputs/w4/` | `W4_results.png` (4-panel: spectral convergence, COS vs MC, Merton vs Kou smile, digital options) |
| W5 | `outputs/w5/` | RV/BPV timeseries, jump detection plots, jump summary CSVs, lognormal fit; `data/raw/`: downloaded QQQ daily and options CSVs |
| W6 | `outputs/w6/` | `merton_calibration.csv`, `kou_calibration.csv`, `merton_vs_kou_summary.csv` (AIC); per-maturity RMSE; `loss_surface_lambda_muJ.png`, `regularisation_path.png`, `jump_risk_premium.csv`; smile fits in `outputs/w6/figures/` |

> **Note:** The W5 stage downloads live data from Yahoo Finance.  
> Re-run during U.S. market hours for the freshest option quotes.

---

## Configuration

All model and pipeline parameters are stored in `.env` (git-ignored).  
Copy `.env.example` to `.env` to get started — the defaults reproduce the baseline results.

| Variable | Default | Description |
|----------|---------|-------------|
| `S0` | 100.0 | Initial spot price |
| `STRIKE` | 100.0 | ATM strike for single-option experiments |
| `MATURITY` | 1.0 | Time to expiry in years |
| `RATE` | 0.05 | Risk-free rate |
| `SIGMA` | 0.20 | Diffusion volatility |
| `LAMBDA` | 1.0 | Jump intensity (jumps per year) |
| `MU_J` | -0.10 | Mean log-jump size |
| `SIGMA_J` | 0.25 | Std dev of log-jump size |
| `P_UP` | 0.40 | Kou: probability of up-jump |
| `ETA1` | 10.0 | Kou: up-jump decay rate |
| `ETA2` | 5.0 | Kou: down-jump decay rate |
| `PATHS` | 50000 | Monte Carlo paths |
| `SEED` | 42 | Random seed |
| `COS_N` | 512 | COS terms |
| `TICKER` | QQQ | Underlying for W5 |
| `DATA_PERIOD` | 3y | History window for W5 |
| `CALIB_SIGMA_MIN` / `MAX` | 0.05 / 0.40 | W6: diffusion vol bounds |
| `CALIB_LAMBDA_MIN` / `MAX` | 0.01 / 8.00 | W6: jump intensity bounds |
| `CALIB_MUJ_MIN` / `MAX` | -0.50 / 0.05 | W6: mean log-jump bounds |
| `CALIB_SIGMAJ_MIN` / `MAX` | 0.01 / 0.40 | W6: log-jump std bounds |
| `CALIB_DE_MAXITER` | 40 | W6: DE generations |
| `CALIB_DE_POPSIZE` | 12 | W6: DE population multiplier |
| `CALIB_REG_ALPHA` | 0.01 | W6: L1 penalty default |

---

## Model

Under the risk-neutral measure **Q**, the Merton jump-diffusion is:

```
dS_t / S_t = (r − λμ̄) dt + σ dW_t + (e^{J_k} − 1) dN_t
```

where:
- `σ` — diffusion volatility
- `λ` — jump intensity (jumps per year)
- `J_k ~ N(μ_J, σ_J²)` — log-jump sizes
- `μ̄ = exp(μ_J + σ_J²/2) − 1` — martingale correction

**Analytic pricing formula** (Poisson-weighted Black–Scholes sum):

```
C^Merton = Σ_{n=0}^∞  e^{-λ'T} (λ'T)^n / n!  ·  C^BS(S, K, T, r_n, σ_n)
```

---

## Key References

- Merton, R.C. (1976). Option Pricing When Underlying Stock Returns are Discontinuous. *Journal of Financial Economics*, 3, 125–144.
- Kou, S.G. (2002). A Jump-Diffusion Model for Option Pricing. *Management Science*, 48(8), 1086–1101.
- Carr, P. & Wu, L. (2003). What Type of Process Underlies Options? *Journal of Finance*, 58(6), 2581–2610.
- Eraker, B., Johannes, M., & Polson, N. (2003). The Impact of Jumps in Returns and Volatility. *Journal of Finance*, 58(3), 1269–1300.
- Barndorff-Nielsen, O.E. & Shephard, N. (2006). Econometrics of Testing for Jumps Using Bipower Variation. *Journal of Financial Econometrics*, 4(1), 1–30.
