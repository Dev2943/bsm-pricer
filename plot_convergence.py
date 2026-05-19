"""
Convergence comparison: Newton-Raphson vs Bisection for implied volatility.

This script demonstrates the central theoretical difference between the two
solvers by plotting their absolute error against iteration count on a
logarithmic scale.

Expected behavior:
    - Newton-Raphson: ERROR PLUMMETS — quadratic convergence. The number of
      correct digits roughly doubles each step until floating-point precision
      is hit (around 1e-16).
    - Bisection: ERROR DROPS LINEARLY (on log scale, this looks like a
      straight line going down). Each iteration halves the interval, gaining
      one bit (~0.3 decimal digits) of precision.

Run with: python3 plot_convergence.py
Produces: convergence_plot.png in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt

from black_scholes import BlackScholesInputs, OptionType, price
from greeks import vega


# ----- Instrumented Newton-Raphson: tracks sigma at every iteration -----
def newton_with_history(market_price, S, K, T, r, option_type,
                        initial_guess=0.2, tolerance=1e-15, max_iterations=50):
    """Same Newton-Raphson as implied_vol.py, but returns the full history."""
    history = [initial_guess]
    sigma = initial_guess

    for _ in range(max_iterations):
        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)
        error = price(inputs, option_type) - market_price

        if abs(error) < tolerance:
            break

        current_vega = vega(inputs)
        sigma_new = sigma - error / current_vega

        if sigma_new <= 0:
            sigma_new = sigma / 2

        sigma = sigma_new
        history.append(sigma)

    return history


# ----- Instrumented Bisection: tracks midpoint at every iteration -----
def bisection_with_history(market_price, S, K, T, r, option_type,
                           sigma_low=1e-4, sigma_high=5.0,
                           tolerance=1e-15, max_iterations=80):
    """Same Bisection as implied_vol.py, but returns the full history."""
    history = []

    inputs_low = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma_low)
    inputs_high = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma_high)
    f_low = price(inputs_low, option_type) - market_price
    f_high = price(inputs_high, option_type) - market_price

    for _ in range(max_iterations):
        mid = (sigma_low + sigma_high) / 2
        history.append(mid)

        inputs_mid = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=mid)
        f_mid = price(inputs_mid, option_type) - market_price

        if abs(f_mid) < tolerance:
            break

        if f_mid * f_low < 0:
            sigma_high = mid
            f_high = f_mid
        else:
            sigma_low = mid
            f_low = f_mid

    return history


# ----- Run both solvers on the same problem -----
if __name__ == "__main__":
    # Standard test case: ATM 1-year call at 25% vol
    S, K, T, r = 100, 100, 1.0, 0.05
    sigma_true = 0.25
    opt_type = OptionType.CALL

    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma_true)
    market_price = price(inputs, opt_type)

    print(f"Test case: S={S}, K={K}, T={T}, r={r}, true sigma={sigma_true}")
    print(f"Market price = {market_price:.6f}\n")

    # Get convergence histories
    newton_hist = newton_with_history(market_price, S, K, T, r, opt_type)
    bisect_hist = bisection_with_history(market_price, S, K, T, r, opt_type)

    # Compute absolute errors at each step
    newton_errors = [abs(s - sigma_true) for s in newton_hist]
    bisect_errors = [abs(s - sigma_true) for s in bisect_hist]

    # Floor errors at 1e-17 so log scale doesn't break on zero
    newton_errors = [max(e, 1e-17) for e in newton_errors]
    bisect_errors = [max(e, 1e-17) for e in bisect_errors]

    print(f"Newton converged in {len(newton_hist) - 1} iterations")
    print(f"  Final error: {newton_errors[-1]:.2e}")
    print(f"Bisection converged in {len(bisect_hist)} iterations")
    print(f"  Final error: {bisect_errors[-1]:.2e}\n")

    # ----- Build the plot -----
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(
        range(len(newton_errors)), newton_errors,
        marker="o", markersize=8, linewidth=2,
        label=f"Newton-Raphson ({len(newton_hist) - 1} iterations)",
        color="#C44E52",
    )
    ax.semilogy(
        range(len(bisect_errors)), bisect_errors,
        marker="s", markersize=5, linewidth=2,
        label=f"Bisection ({len(bisect_hist)} iterations)",
        color="#4C72B0",
    )

    # Reference line at machine epsilon
    ax.axhline(y=1e-16, color="gray", linestyle="--", alpha=0.5,
               label="Machine epsilon (~1e-16)")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$|\sigma_n - \sigma_{true}|$", fontsize=12)
    ax.set_title(
        f"Implied Volatility Solver Convergence\n"
        f"S={S}, K={K}, T={T}y, r={r:.0%}, true $\\sigma$={sigma_true}",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=1e-18)

    # Annotation calling out the convergence behavior
    ax.text(
        0.02, 0.05,
        "Newton: quadratic convergence — digits double per step\n"
        "Bisection: linear convergence — interval halves per step",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig("convergence_plot.png", dpi=150, bbox_inches="tight")
    print("Saved convergence_plot.png")

    # Also display interactively if running in an environment that supports it
    try:
        plt.show()
    except Exception:
        pass
