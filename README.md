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
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |
| 6 | | |
| 7 | | |
| 8 | | |
| 9 | | |
| 10 | | |
| 11 | | |

### Workstream Roles

| Workstream | Responsible |
|------------|-------------|
| W1 — Research Lead (Literature Review) | |
| W2 — Theory Lead (Mathematical Derivations) | |
| W3 — MC Engineer (Monte Carlo Engine) | |
| W4 — Fourier Engineer (COS Engine) | |
| W5 — Data Scientist (QQQ Data & Jump Detection) | |
| W6 — Calibration Engineer | |
| W7 — Vol Analyst (Implied Vol Surface) | |
| W8 — Risk Lead (Hedging & Alpha Backtest) | |

---

## Project Structure

```
.
├── main.py                   # unified pipeline entry point (W3 + W4 + W5)
├── requirements.txt          # all Python dependencies
│
├── W1/                       # Literature review
│   ├── Merton_1976_*.pdf/tex
│   ├── Kou_2002_*.pdf/tex
│   ├── Carr_Wu_2003_*.pdf/tex
│   ├── Eraker_2003_*.pdf/tex
│   ├── Hawkes_*.pdf/tex
│   ├── deep_learning_*.pdf/tex
│   └── Merton Jump-Diffusion_Review.pdf   # compiled review
│
├── W2/                       # Mathematical derivations
│   ├── Jump models.tex
│   └── W2 Theory Lead.pdf
│
├── W3/                       # Monte Carlo pricing engine
│   ├── config.py             # parameter containers and CLI
│   ├── simulation.py         # Merton terminal-price simulator
│   ├── pricing.py            # Black-Scholes and Merton exact formula
│   ├── experiments.py        # variance reduction, convergence, smile
│   ├── plots.py              # figure renderers
│   ├── merton_mc.py          # pipeline entry point
│   └── results/              # CSV and PDF outputs
│
├── W4/                       # COS / Fourier pricing engine
│   ├── CallPut_COS_Method.py # COS engine: Merton CF, Kou CF, digitals
│   └── results/              # validation plots
│
├── W5/                       # Real data and jump detection
│   ├── EDA/
│   │   ├── initial_eda.py    # QQQ price history download and EDA
│   │   ├── options_data.py   # option chain download and cleaning
│   │   ├── rv_bpv_analysis.py# rolling RV / BPV computation (BNS test)
│   │   └── jump_detection.py # jump classification and lognormal fit
│   ├── data/
│   │   ├── raw/              # qqq_daily.csv, qqq_options_raw.csv
│   │   └── cleaned/          # qqq_options_clean.csv
│   └── outputs/w5/           # plots and CSV results
│
└── Latex/                    # Compiled paper (W1–W5)
    └── Group3_full_W1-W5.tex
```

---

## Installation

```bash
# clone the repository
git clone <repo-url>
cd <repo-folder>

# create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# install all dependencies
pip install -r requirements.txt
```

**Python version:** 3.10 or later

---

## Running the Pipeline

```bash
# run all three stages (W3 + W4 + W5)
python main.py

# run individual stages
python main.py --w3          # Monte Carlo engine
python main.py --w4          # COS / Fourier engine
python main.py --w5          # data collection + jump detection

# combine any two
python main.py --w3 --w4

# verbose output
python main.py --log-level DEBUG
```

### What each stage produces

| Stage | Output location | Contents |
|-------|----------------|----------|
| W3 | `W3/results/` | `validation_summary.csv`, `variance_reduction_table.csv`, `convergence_study.pdf`, `implied_vol_smile.pdf`, `control_variate_sensitivity.pdf` |
| W4 | `W4/results/` | `W4_results.png` (4-panel: spectral convergence, COS vs MC, Merton vs Kou smile, digital options) |
| W5 | `W5/outputs/` | QQQ price and returns plots; `W5/outputs/w5/`: RV/BPV timeseries, jump detection plots, jump summary CSVs, lognormal fit |

> **Note:** The W5 stage downloads live data from Yahoo Finance.  
> Re-run during U.S. market hours for the freshest option quotes.

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
