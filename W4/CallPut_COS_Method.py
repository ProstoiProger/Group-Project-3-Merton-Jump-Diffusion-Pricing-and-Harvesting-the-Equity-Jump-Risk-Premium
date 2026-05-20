import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. CHARACTERISTIC FUNCTIONS
# ==========================================

def merton_char_func(u, S0, r, T, sigma, lam, mu_J, sigma_J):
    """
    Characteristic function of log(S_T) under Merton Jump-Diffusion.
    Jumps: log-jump sizes ~ Normal(mu_J, sigma_J^2)
    """
    x0 = np.log(S0)
    mu_bar = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1          # martingale correction

    drift     = 1j * u * (r - 0.5 * sigma ** 2 - lam * mu_bar)
    diffusion = -0.5 * (sigma ** 2) * (u ** 2)
    jumps     = lam * (np.exp(1j * u * mu_J - 0.5 * (sigma_J ** 2) * (u ** 2)) - 1)

    psi = drift + diffusion + jumps
    return np.exp(1j * u * x0 + psi * T)


def kou_char_func(u, S0, r, T, sigma, lam, p_up, eta1, eta2):
    """
    Characteristic function of log(S_T) under Kou Double-Exponential Jump-Diffusion.

    Jump size Y = log(J):
      - Up-jump   with prob p_up:   Y ~ Exponential(eta1), mean = 1/eta1
      - Down-jump with prob 1-p_up: Y ~ Exponential(eta2), mean = 1/eta2 (negative side)

    CF of a single jump:
      m(u) = p_up * eta1/(eta1 - iu) + (1-p_up) * eta2/(eta2 + iu)

    Martingale correction:
      kappa = lam * (p_up*eta1/(eta1-1) + (1-p_up)*eta2/(eta2+1) - 1)
    """
    x0 = np.log(S0)

    # Martingale correction (mean of e^Y under jump measure)
    kappa = lam * (p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1)

    drift     = 1j * u * (r - 0.5 * sigma ** 2 - kappa)
    diffusion = -0.5 * (sigma ** 2) * (u ** 2)

    # CF of a single Kou jump
    m_u = (p_up * eta1 / (eta1 - 1j * u) +
           (1 - p_up) * eta2 / (eta2 + 1j * u))
    jumps = lam * (m_u - 1)

    psi = drift + diffusion + jumps
    return np.exp(1j * u * x0 + psi * T)


# ==========================================
# 2. COS PAYOFF HELPER INTEGRALS
# ==========================================

def chi_k(k, a, b, c, d):
    """Integral of e^x * cos(k*pi*(x-a)/(b-a)) over [c,d]."""
    k_pi = k * np.pi / (b - a)
    expr1 = np.cos(k_pi * (d - a)) * np.exp(d) - np.cos(k_pi * (c - a)) * np.exp(c)
    expr2 = (k_pi * np.sin(k_pi * (d - a)) * np.exp(d)
             - k_pi * np.sin(k_pi * (c - a)) * np.exp(c))
    return (expr1 + expr2) / (1.0 + k_pi ** 2)


def psi_k(k, a, b, c, d):
    """Integral of cos(k*pi*(x-a)/(b-a)) over [c,d]."""
    if k == 0:
        return d - c
    k_pi = k * np.pi / (b - a)
    return (np.sin(k_pi * (d - a)) - np.sin(k_pi * (c - a))) / k_pi


# ==========================================
# 3. GENERIC COS PRICING ENGINE
# ==========================================

def _cos_bounds(char_func_cumulants, L=10):
    """Compute truncation bounds [a,b] from cumulants (c1, c2, c4)."""
    c1, c2, c4 = char_func_cumulants
    a = c1 - L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    b = c1 + L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    return a, b


def cos_price_european(phi_func, S0, K, r, T, option_type='call', N=4096, L=10,
                       cumulants=None):
    """
    Generic COS pricer for European Call / Put.

    Parameters
    ----------
    phi_func : callable
        Characteristic function phi(u) of log(S_T).
    option_type : 'call' or 'put'
    cumulants  : (c1, c2, c4) tuple for domain truncation.
                 If None, a wide default is used.
    """
    if cumulants is None:
        # Fallback wide bounds
        x0 = np.log(S0)
        a, b = x0 - L * 0.5, x0 + L * 0.5
    else:
        a, b = _cos_bounds(cumulants, L)

    c = np.log(K)          # lower limit of in-the-money region
    d = b if option_type == 'call' else c

    k   = np.arange(0, N)
    u   = k * np.pi / (b - a)
    phi = phi_func(u)

    chi_vals = np.array([chi_k(ki, a, b, c if option_type == 'call' else a,
                               b if option_type == 'call' else c) for ki in k])
    psi_vals = np.array([psi_k(ki, a, b, c if option_type == 'call' else a,
                               b if option_type == 'call' else c) for ki in k])

    if option_type == 'call':
        U_k = (2.0 / (b - a)) * (chi_vals - K * psi_vals)
    else:  # put
        U_k = (2.0 / (b - a)) * (-chi_vals + K * psi_vals)

    inner         = np.real(phi * np.exp(-1j * u * a)) * U_k
    inner[0]     *= 0.5
    price         = np.exp(-r * T) * np.sum(inner)
    return max(0.0, price)


def cos_price_digital(phi_func, S0, K, r, T, digital_type='cash_call',
                      N=4096, L=10, cumulants=None):
    """
    COS pricer for Digital (Binary) options.

    digital_type options
    --------------------
    'cash_call'  : pays $1 if S_T > K   (cash-or-nothing call)
    'cash_put'   : pays $1 if S_T < K   (cash-or-nothing put)
    'asset_call' : pays S_T if S_T > K  (asset-or-nothing call)
    'asset_put'  : pays S_T if S_T < K  (asset-or-nothing put)

    Payoff coefficients
    -------------------
    Cash-or-nothing call  : U_k = (2/(b-a)) * psi_k(ln K, b)
    Cash-or-nothing put   : U_k = (2/(b-a)) * psi_k(a, ln K)
    Asset-or-nothing call : U_k = (2/(b-a)) * chi_k(ln K, b)
    Asset-or-nothing put  : U_k = (2/(b-a)) * chi_k(a, ln K)
    """
    if cumulants is None:
        x0 = np.log(S0)
        a, b = x0 - L * 0.5, x0 + L * 0.5
    else:
        a, b = _cos_bounds(cumulants, L)

    c = np.log(K)
    k = np.arange(0, N)
    u = k * np.pi / (b - a)
    phi = phi_func(u)

    if digital_type == 'cash_call':
        coeff = np.array([psi_k(ki, a, b, c, b) for ki in k])
    elif digital_type == 'cash_put':
        coeff = np.array([psi_k(ki, a, b, a, c) for ki in k])
    elif digital_type == 'asset_call':
        coeff = np.array([chi_k(ki, a, b, c, b) for ki in k])
    elif digital_type == 'asset_put':
        coeff = np.array([chi_k(ki, a, b, a, c) for ki in k])
    else:
        raise ValueError(f"Unknown digital_type: {digital_type}")

    U_k          = (2.0 / (b - a)) * coeff
    inner        = np.real(phi * np.exp(-1j * u * a)) * U_k
    inner[0]    *= 0.5
    price        = np.exp(-r * T) * np.sum(inner)
    return max(0.0, price)


# ==========================================
# 4. MERTON WRAPPERS (cumulants pre-computed)
# ==========================================

def merton_cumulants(S0, r, T, sigma, lam, mu_J, sigma_J):
    mu_bar = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1
    c1 = np.log(S0) + (r - 0.5 * sigma ** 2 - lam * mu_bar) * T + lam * T * mu_J
    c2 = (sigma ** 2) * T + lam * T * (mu_J ** 2 + sigma_J ** 2)
    c4 = lam * T * (mu_J ** 4 + 6 * mu_J ** 2 * sigma_J ** 2 + 3 * sigma_J ** 4)
    return c1, c2, c4


def cos_method_merton_call(S0, K, r, T, sigma, lam, mu_J, sigma_J, N=4096, L=10):
    cum = merton_cumulants(S0, r, T, sigma, lam, mu_J, sigma_J)
    phi = lambda u: merton_char_func(u, S0, r, T, sigma, lam, mu_J, sigma_J)
    return cos_price_european(phi, S0, K, r, T, 'call', N, L, cum)


def cos_method_merton_put(S0, K, r, T, sigma, lam, mu_J, sigma_J, N=4096, L=10):
    cum = merton_cumulants(S0, r, T, sigma, lam, mu_J, sigma_J)
    phi = lambda u: merton_char_func(u, S0, r, T, sigma, lam, mu_J, sigma_J)
    return cos_price_european(phi, S0, K, r, T, 'put', N, L, cum)


def cos_method_merton_digital(S0, K, r, T, sigma, lam, mu_J, sigma_J,
                              digital_type='cash_call', N=4096, L=10):
    cum = merton_cumulants(S0, r, T, sigma, lam, mu_J, sigma_J)
    phi = lambda u: merton_char_func(u, S0, r, T, sigma, lam, mu_J, sigma_J)
    return cos_price_digital(phi, S0, K, r, T, digital_type, N, L, cum)


# ==========================================
# 5. KOU WRAPPERS (cumulants pre-computed)
# ==========================================

def kou_cumulants(S0, r, T, sigma, lam, p_up, eta1, eta2):
    kappa = lam * (p_up * eta1 / (eta1 - 1) + (1 - p_up) * eta2 / (eta2 + 1) - 1)
    # First cumulant (mean of log-price)
    c1 = np.log(S0) + (r - 0.5 * sigma ** 2 - kappa) * T + lam * T * (p_up / eta1 - (1 - p_up) / eta2)
    # Second cumulant (variance of log-price)
    c2 = sigma ** 2 * T + lam * T * (2 * p_up / eta1 ** 2 + 2 * (1 - p_up) / eta2 ** 2)
    # Fourth cumulant (excess kurtosis contribution)
    c4 = lam * T * (24 * p_up / eta1 ** 4 + 24 * (1 - p_up) / eta2 ** 4)
    return c1, c2, c4


def cos_method_kou_call(S0, K, r, T, sigma, lam, p_up, eta1, eta2, N=4096, L=10):
    cum = kou_cumulants(S0, r, T, sigma, lam, p_up, eta1, eta2)
    phi = lambda u: kou_char_func(u, S0, r, T, sigma, lam, p_up, eta1, eta2)
    return cos_price_european(phi, S0, K, r, T, 'call', N, L, cum)


# ==========================================
# 6. EXACT BENCHMARK FORMULAS
# ==========================================

def black_scholes_call(S, K, r, T, sigma):
    if T <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def merton_exact_call(S0, K, r, T, sigma, lam, mu_J, sigma_J, n_max=50):
    """Exact Merton call via Poisson-weighted Black-Scholes series."""
    mu_bar  = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1
    lam_prime = lam * (1 + mu_bar)
    price = 0.0
    for n in range(n_max):
        weight  = (np.exp(-lam_prime * T) * (lam_prime * T) ** n) / math.factorial(n)
        r_n     = r - lam * mu_bar + (n * mu_J) / T + (n * sigma_J ** 2) / (2 * T)
        sigma_n = np.sqrt(sigma ** 2 + (n * sigma_J ** 2) / T)
        price  += weight * black_scholes_call(S0, K, r_n, T, sigma_n)
    return price


def bs_digital_cash_call(S, K, r, T, sigma):
    """Black-Scholes cash-or-nothing call: pays $1 if S_T > K."""
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * stats.norm.cdf(d2)


# ==========================================
# 7. MONTE CARLO VALIDATION  (Criterion 2b)
# ==========================================

def monte_carlo_merton_call(S0, K, r, T, sigma, lam, mu_J, sigma_J,
                            n_paths=200_000, seed=42):
    """
    Monte Carlo price for European Call under Merton model.
    Returns (price, std_error) for error-band validation.
    """
    rng = np.random.default_rng(seed)
    mu_bar = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1

    # Simulate terminal log-price
    Z       = rng.standard_normal(n_paths)
    N_jumps = rng.poisson(lam * T, n_paths)

    # Sum of log-jumps for each path
    log_jumps = np.array([
        np.sum(rng.normal(mu_J, sigma_J, n)) if n > 0 else 0.0
        for n in N_jumps
    ])

    log_ST = (np.log(S0)
              + (r - 0.5 * sigma ** 2 - lam * mu_bar) * T
              + sigma * np.sqrt(T) * Z
              + log_jumps)

    ST      = np.exp(log_ST)
    payoffs = np.maximum(ST - K, 0.0)
    disc    = np.exp(-r * T)

    price     = disc * np.mean(payoffs)
    std_error = disc * np.std(payoffs) / np.sqrt(n_paths)
    return price, std_error


# ==========================================
# 8. IMPLIED VOLATILITY (for smile plots)
# ==========================================

def implied_vol(price, S, K, r, T, option_type='call', tol=1e-8, max_iter=200):
    """Newton-Raphson implied volatility solver."""
    sigma = 0.3
    for _ in range(max_iter):
        if option_type == 'call':
            f = black_scholes_call(S, K, r, T, sigma) - price
        else:
            f = (black_scholes_call(S, K, r, T, sigma)
                 - S + K * np.exp(-r * T)) - price

        d1    = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega  = S * stats.norm.pdf(d1) * np.sqrt(T)
        if abs(vega) < 1e-12:
            break
        sigma -= f / vega
        sigma  = max(sigma, 1e-6)
        if abs(f) < tol:
            break
    return sigma


# ==========================================
# 9. MAIN VALIDATION & PLOTS
# ==========================================

if __name__ == "__main__":

    # ── Shared parameters ────────────────────────────────────────────
    S0      = 100.0
    K       = 105.0
    r       = 0.05
    T       = 0.5
    sigma   = 0.20
    lam     = 2.0
    mu_J    = -0.05
    sigma_J = 0.10

    # Kou-specific parameters
    p_up = 0.4       # probability of up-jump
    eta1 = 10.0      # up-jump decay  (mean up-jump  = 1/eta1 = 10%)
    eta2 = 5.0       # down-jump decay (mean down-jump = 1/eta2 = 20%)

    strikes = np.linspace(80, 120, 25)

    # ════════════════════════════════════════════════════════════════
    # CRITERION 1 — COS with Merton CF  (already present, confirmed)
    # ════════════════════════════════════════════════════════════════
    exact_price = merton_exact_call(S0, K, r, T, sigma, lam, mu_J, sigma_J)
    cos_price   = cos_method_merton_call(S0, K, r, T, sigma, lam, mu_J, sigma_J, N=512)
    print("=" * 60)
    print("CRITERION 1 — Merton CF + COS")
    print(f"  Exact Merton price : {exact_price:.8f}")
    print(f"  COS price (N=512)  : {cos_price:.8f}")

    # ════════════════════════════════════════════════════════════════
    # CRITERION 2a — COS vs Exact Merton: error < 10^-8
    # ════════════════════════════════════════════════════════════════
    N_vals  = [16, 32, 64, 128, 256, 512, 1024]
    errors  = []
    print("\n" + "=" * 60)
    print("CRITERION 2a — Convergence: COS vs Exact Merton")
    print(f"{'N':<8} {'COS Price':<16} {'|Error|':<16} {'Pass <1e-8'}")
    for N in N_vals:
        cp  = cos_method_merton_call(S0, K, r, T, sigma, lam, mu_J, sigma_J, N=N)
        err = abs(cp - exact_price)
        errors.append(err)
        flag = "✓" if err < 1e-8 else "✗"
        print(f"{N:<8} {cp:<16.8f} {err:<16.2e} {flag}")

    # ════════════════════════════════════════════════════════════════
    # CRITERION 2b — COS vs Monte Carlo (within MC error bands)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CRITERION 2b — COS vs Monte Carlo (200k paths)")
    mc_price, mc_se = monte_carlo_merton_call(S0, K, r, T, sigma, lam, mu_J, sigma_J)
    cos_512         = cos_method_merton_call(S0, K, r, T, sigma, lam, mu_J, sigma_J, N=512)
    lo, hi          = mc_price - 2 * mc_se, mc_price + 2 * mc_se
    inside          = lo <= cos_512 <= hi
    print(f"  MC price   : {mc_price:.6f}  ±2σ = [{lo:.6f}, {hi:.6f}]")
    print(f"  COS price  : {cos_512:.6f}")
    print(f"  Inside MC band: {'✓ YES' if inside else '✗ NO'}")

    # ════════════════════════════════════════════════════════════════
    # CRITERION 3 — Kou COS engine + Merton vs Kou smile comparison
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CRITERION 3 — Kou COS Engine")
    kou_price = cos_method_kou_call(S0, K, r, T, sigma, lam, p_up, eta1, eta2, N=512)
    print(f"  Kou COS call price (K={K}): {kou_price:.6f}")

    # Implied-vol smile for Merton and Kou
    merton_ivols = []
    kou_ivols    = []
    for k_strike in strikes:
        mp = cos_method_merton_call(S0, k_strike, r, T, sigma, lam, mu_J, sigma_J, N=512)
        kp = cos_method_kou_call   (S0, k_strike, r, T, sigma, lam, p_up, eta1, eta2, N=512)
        try:
            merton_ivols.append(implied_vol(mp, S0, k_strike, r, T) * 100)
        except Exception:
            merton_ivols.append(np.nan)
        try:
            kou_ivols.append(implied_vol(kp, S0, k_strike, r, T) * 100)
        except Exception:
            kou_ivols.append(np.nan)

    # ════════════════════════════════════════════════════════════════
    # CRITERION 4 — Digital option pricing using COS
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CRITERION 4 — Digital Option Pricing via COS")

    dig_types = ['cash_call', 'cash_put', 'asset_call', 'asset_put']
    for dt in dig_types:
        cos_dig = cos_method_merton_digital(S0, K, r, T, sigma, lam, mu_J, sigma_J,
                                            digital_type=dt, N=512)
        print(f"  {dt:<14}: {cos_dig:.6f}")

    # BS benchmark for cash-or-nothing call (pure diffusion reference)
    bs_dig = bs_digital_cash_call(S0, K, r, T, sigma)
    cos_dig_cash = cos_method_merton_digital(S0, K, r, T, sigma, 0.0, mu_J, sigma_J,
                                             digital_type='cash_call', N=512)
    print(f"\n  BS cash-call (lam=0) benchmark : {bs_dig:.6f}")
    print(f"  COS cash-call (lam=0)          : {cos_dig_cash:.6f}")
    print(f"  Difference                     : {abs(bs_dig - cos_dig_cash):.2e}")

    # ════════════════════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('W4 — COS Pricing Engine Validation', fontsize=14, fontweight='bold')

    # ── Plot 1: Convergence (Criterion 2a) ──────────────────────────
    ax = axes[0, 0]
    ax.semilogy(N_vals, errors, 'o-', color='darkblue', linewidth=2)
    ax.axhline(1e-8, color='red', linestyle='--', label='Target: $10^{-8}$')
    ax.set_title('Criterion 2a: Spectral Convergence')
    ax.set_xlabel('N (COS terms)')
    ax.set_ylabel('|COS − Exact| (log scale)')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)

    # ── Plot 2: MC validation (Criterion 2b) ────────────────────────
    ax = axes[0, 1]
    mc_prices, mc_ses = [], []
    for k_strike in strikes:
        mc_p, mc_e = monte_carlo_merton_call(S0, k_strike, r, T, sigma,
                                             lam, mu_J, sigma_J, n_paths=100_000)
        mc_prices.append(mc_p)
        mc_ses.append(mc_e)
    cos_smile = [cos_method_merton_call(S0, ks, r, T, sigma, lam, mu_J, sigma_J, N=512)
                 for ks in strikes]
    mc_arr = np.array(mc_prices)
    se_arr = np.array(mc_ses)
    ax.fill_between(strikes, mc_arr - 2*se_arr, mc_arr + 2*se_arr,
                    alpha=0.3, color='orange', label='MC ±2σ band')
    ax.plot(strikes, mc_arr,   'o', color='orange', markersize=3, label='MC price')
    ax.plot(strikes, cos_smile, '-', color='darkblue', linewidth=2, label='COS (N=512)')
    ax.set_title('Criterion 2b: COS vs Monte Carlo')
    ax.set_xlabel('Strike K')
    ax.set_ylabel('Call Price')
    ax.legend(fontsize=8)
    ax.grid(True, ls='--', alpha=0.5)

    # ── Plot 3: Merton vs Kou Implied-Vol Smile (Criterion 3) ───────
    ax = axes[1, 0]
    ax.plot(strikes, merton_ivols, 'b-o', markersize=4, linewidth=2, label='Merton')
    ax.plot(strikes, kou_ivols,    'r-s', markersize=4, linewidth=2, label='Kou')
    ax.set_title('Criterion 3: Merton vs Kou Implied-Vol Smile')
    ax.set_xlabel('Strike K')
    ax.set_ylabel('Implied Volatility (%)')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)

    # ── Plot 4: Digital option prices across strikes (Criterion 4) ──
    ax = axes[1, 1]
    for dt, col in zip(['cash_call', 'cash_put', 'asset_call', 'asset_put'],
                       ['blue', 'red', 'green', 'purple']):
        digs = [cos_method_merton_digital(S0, ks, r, T, sigma, lam, mu_J, sigma_J,
                                          digital_type=dt, N=512) for ks in strikes]
        ax.plot(strikes, digs, color=col, linewidth=2, label=dt)
    ax.set_title('Criterion 4: Digital Options via COS')
    ax.set_xlabel('Strike K')
    ax.set_ylabel('Price')
    ax.legend(fontsize=8)
    ax.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('W4_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to W4_results.png")
