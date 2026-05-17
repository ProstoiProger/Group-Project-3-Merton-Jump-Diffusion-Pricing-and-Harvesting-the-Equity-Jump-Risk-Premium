import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math


# ==========================================
# 1. CHARACTERISTIC FUNCTION (MERTON)
# ==========================================
def merton_char_func(u, S0, r, T, sigma, lam, mu_J, sigma_J):
    """
    Computes the characteristic function phi(u) for the log-asset price
    under the Merton Jump-Diffusion model.
    """
    x0 = np.log(S0)

    # Drift corrector (ensures the martingale property)
    mu_bar = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1

    # Characteristic exponent components
    drift = 1j * u * (r - 0.5 * sigma ** 2 - lam * mu_bar)
    diffusion = -0.5 * (sigma ** 2) * (u ** 2)
    jumps = lam * (np.exp(1j * u * mu_J - 0.5 * (sigma_J ** 2) * (u ** 2)) - 1)

    psi = drift + diffusion + jumps
    return np.exp(1j * u * x0 + psi * T)


# ==========================================
# 2. COS METHOD PAYOFF COEFFICIENTS (CALL)
# ==========================================
def chi_k(k, a, b, c, d):
    """Helper function chi_k for the COS method payoff coefficients."""
    k_pi = k * np.pi / (b - a)

    # Using the analytical integration formula
    expr1 = np.cos(k_pi * (d - a)) * np.exp(d) - np.cos(k_pi * (c - a)) * np.exp(c)
    expr2 = k_pi * np.sin(k_pi * (d - a)) * np.exp(d) - k_pi * np.sin(k_pi * (c - a)) * np.exp(c)

    return (expr1 + expr2) / (1.0 + k_pi ** 2)


def psi_k(k, a, b, c, d):
    """Helper function psi_k for the COS method payoff coefficients."""
    # Handle k = 0 separately to avoid division by zero
    if k == 0:
        return d - c

    k_pi = k * np.pi / (b - a)
    return (np.sin(k_pi * (d - a)) - np.sin(k_pi * (c - a))) / k_pi


# ==========================================
# 3. CORE COS METHOD PRICING ENGINE
# ==========================================
def cos_method_merton_call(S0, K, r, T, sigma, lam, mu_J, sigma_J, N=4096, L=10):
    """
    Prices a European Call option using the Fang-Osterlee COS Method
    under Merton Jump-Diffusion.
    """
    # Cumulants for domain truncation [a, b]
    mu_bar = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1
    c1 = np.log(S0) + (r - 0.5 * sigma ** 2 - lam * mu_bar) * T + lam * T * mu_J
    c2 = (sigma ** 2) * T + lam * T * (mu_J ** 2 + sigma_J ** 2)
    c4 = lam * T * (mu_J ** 4 + 6 * (mu_J ** 2) * (sigma_J ** 2) + 3 * sigma_J ** 4)

    a = c1 - L * np.sqrt(c2 + np.sqrt(c4))
    b = c1 + L * np.sqrt(c2 + np.sqrt(c4))

    # Integration limits for a Call Option (from ln(K) up to b)
    # Note: In standard COS notation, we scale limits by S0 to match payoff definitions
    c = np.log(K / S0)
    d = b - np.log(S0)

    # Vectorized loop preparation
    k = np.arange(0, N)
    u = k * np.pi / (b - a)

    # Evaluate characteristic function (shift the asset variable to match payoff scaling)
    # phi(u) defined with x0 = ln(S0), but inside formula we evaluate relative to x0 = 0 to align with standard payoff
    phi = merton_char_func(u, 1.0, r, T, sigma, lam, mu_J, sigma_J)

    # Compute payoff coefficients (U_k) scaled by S0
    # Vectorized computation of chi and psi arrays
    chi_vals = np.array([chi_k(ki, a - np.log(S0), b - np.log(S0), c, d) for ki in k])
    psi_vals = np.array([psi_k(ki, a - np.log(S0), b - np.log(S0), c, d) for ki in k])

    U_k = (2.0 / (b - a)) * S0 * (chi_vals - psi_vals)

    # Apply COS method summation
    inner_term = np.real(phi * np.exp(-1j * u * (a - np.log(S0)))) * U_k
    inner_term[0] = inner_term[0] * 0.5  # First term weighting factor (1/2)

    call_price = np.exp(-r * T) * np.sum(inner_term)
    return max(0.0, call_price)


# ==========================================
# 4. EXACT MERTON ANALYTICAL FORMULA (W2/W4 Verification)
# ==========================================
def black_scholes_call(S, K, r, T, sigma):
    """Standard Black-Scholes Formula."""
    if T <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def merton_exact_call(S0, K, r, T, sigma, lam, mu_J, sigma_J, n_max=50):
    """
    Computes the exact analytical price of a Merton Call Option
    using a truncated infinite sum of Poisson-weighted Black-Scholes prices.
    """
    mu_bar = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1
    lam_prime = lam * (1 + mu_bar)

    price = 0.0
    for n in range(n_max):
        # Poisson probability weight
        weight = (np.exp(-lam_prime * T) * (lam_prime * T) ** n) / math.factorial(n)

        # Adjusted risk-free rate and volatility for the n-th jump state
        r_n = r - lam * mu_bar + (n * mu_J) / T + (n * sigma_J ** 2) / (2 * T)
        sigma_n = np.sqrt(sigma ** 2 + (n * sigma_J ** 2) / T)

        # Evaluate Black-Scholes for this state
        bs_price = black_scholes_call(S0, K, r_n, T, sigma_n)
        price += weight * bs_price

    return price


# ==========================================
# 5. VALIDATION ENVIRONMENT & CONVERGENCE TESTS
# ==========================================
if __name__ == "__main__":
    # Parameters from project description typical range
    S0 = 100.0  # Current asset price
    K = 105.0  # Out-of-the-money Call strike
    r = 0.05  # Risk-free rate (5%)
    T = 0.5  # Maturity (6 months)

    sigma = 0.20  # Diffusion volatility
    lam = 2.0  # Jump intensity (2 jumps per year)
    mu_J = -0.05  # Mean log-jump size (negative jumps reflect crashes)
    sigma_J = 0.10  # Jump size standard deviation

    # Calculate Exact Price Benchmark
    exact_price = merton_exact_call(S0, K, r, T, sigma, lam, mu_J, sigma_J)
    print(f"--- Benchmark Pricing Output ---")
    print(f"Merton Exact Pricing Formula: {exact_price:.6f}\n")

    # Test COS Method convergence by varying N
    N_test_values = [16, 32, 64, 128, 256, 512, 1024]
    cos_prices = []
    errors = []

    print(f"--- COS Method Convergence Profile ---")
    print(f"{'N terms':<10}{'COS Price':<15}{'Absolute Error':<15}")
    for N in N_test_values:
        cos_p = cos_method_merton_call(S0, K, r, T, sigma, lam, mu_J, sigma_J, N=N)
        err = abs(cos_p - exact_price)
        cos_prices.append(cos_p)
        errors.append(err)
        print(f"{N:<10}{cos_p:<15.6f}{err:<15.4e}")

    # Generate Deliverable Plots for Project Submission
    plt.figure(figsize=(10, 5))

    # Plot 1: Exponential Convergence
    plt.subplot(1, 2, 1)
    plt.semilogy(N_test_values, errors, 'o-', color='darkblue', linewidth=2)
    plt.title('Spectral Convergence of COS Engine')
    plt.xlabel('Number of Expansion Terms (N)')
    plt.ylabel('Absolute Error (Log Scale)')
    plt.grid(True, which="both", ls="--")

    # Plot 2: Volatility Smile Check (COS Pricing vs Strikes)
    strikes = np.linspace(80, 120, 20)
    cos_smile_prices = [cos_method_merton_call(S0, k, r, T, sigma, lam, mu_J, sigma_J, N=512) for k in strikes]
    exact_smile_prices = [merton_exact_call(S0, k, r, T, sigma, lam, mu_J, sigma_J) for k in strikes]

    plt.subplot(1, 2, 2)
    plt.plot(strikes, exact_smile_prices, 'r-', label='Exact Formula', linewidth=2)
    plt.plot(strikes, cos_smile_prices, 'bo', label='COS Engine (N=512)', markersize=4)
    plt.title('Pricing Engine Across Strikes Profile')
    plt.xlabel('Strike Price (K)')
    plt.ylabel('Option Premium')
    plt.legend()
    plt.grid(True, ls="--")

    plt.tight_layout()
    plt.show()