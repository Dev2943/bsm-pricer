"""
Implied volatility solvers.

Given a market price for an option, find the volatility sigma such that
Black-Scholes outputs exactly that price.

Implements two methods:
    - Newton-Raphson:  fast (quadratic convergence) but can diverge
    - Bisection:       slow (linear convergence) but bulletproof


"""

import numpy as np

from black_scholes import BlackScholesInputs, OptionType, price
from greeks import vega


# ----------------------------- ARBITRAGE BOUNDS (provided) -----------------------------
def _check_arbitrage_bounds(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
) -> None:
    """Validate that the market price falls within no-arbitrage bounds.

    For a call:  max(S - K*exp(-rT), 0) <= C <= S
    For a put:   max(K*exp(-rT) - S, 0) <= P <= K*exp(-rT)

    If violated, no positive sigma can reproduce the price — raises ValueError.
    """
    discount = K * np.exp(-r * T)

    if option_type == OptionType.CALL:
        lower = max(S - discount, 0.0)
        upper = S
    else:  # PUT
        lower = max(discount - S, 0.0)
        upper = discount

    if market_price < lower - 1e-10 or market_price > upper + 1e-10:
        raise ValueError(
            f"Market price {market_price:.4f} violates arbitrage bounds "
            f"[{lower:.4f}, {upper:.4f}] for a {option_type.value}. "
            f"No valid implied volatility exists."
        )


# ----------------------------- NEWTON-RAPHSON -----------------------------
def implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    initial_guess: float = 0.2,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:

    _check_arbitrage_bounds(market_price, S, K, T, r, option_type)

    sigma = initial_guess

    for iteration in range(max_iterations):

        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)

        current_price = price(inputs, option_type)

        error = current_price - market_price

        if abs(error) < tolerance:
            return sigma

        current_vega = vega(inputs)

        if abs(current_vega) < 1e-10:
            return implied_vol_bisection(
                market_price, S, K, T, r, option_type,
                tolerance=tolerance, max_iterations=max_iterations * 2,
            )

        sigma_new = sigma - error / current_vega

        if sigma_new <= 0:
            sigma_new = sigma / 2

        sigma = sigma_new

    raise RuntimeError(
        f"Newton-Raphson did not converge in {max_iterations} iterations. "
        f"Last sigma = {sigma:.6f}, last error = {error:.6e}"
    )


# ----------------------------- BISECTION -----------------------------
def implied_vol_bisection(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    sigma_low: float = 1e-4,
    sigma_high: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> float:

    _check_arbitrage_bounds(market_price, S, K, T, r, option_type)

    inputs_low = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma_low)
    inputs_high = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma_high)

    price_low = price(inputs_low, option_type)

    price_high = price(inputs_high, option_type)

    f_low = price_low - market_price
    f_high = price_high - market_price

    if f_low * f_high > 0:
        raise ValueError(
            f"Root not bracketed in [{sigma_low}, {sigma_high}]. "
            f"f(low) = {f_low:.4f}, f(high) = {f_high:.4f}. "
            f"Widen the bounds or check input validity."
        )

    for iteration in range(max_iterations):

        mid = (sigma_low + sigma_high) / 2

        inputs_mid = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=mid)

        price_mid = price(inputs_mid, option_type)

        if abs(price_mid - market_price) < tolerance:
            return mid

        f_mid = price_mid - market_price

        if f_mid * f_low < 0:
            sigma_high = mid
            f_high = f_mid
        else:
            sigma_low = mid
            f_low = f_mid

    raise RuntimeError(
        f"Bisection did not converge in {max_iterations} iterations."
    )


# ----------------------------- ROUND-TRIP SANITY CHECK -----------------------------
if __name__ == "__main__":
    # The fundamental test: if we price at a known sigma, then ask the solver
    # what sigma reproduces that price, we should get back the original.
    print("Round-trip test: price at known sigma, then recover sigma from price.\n")

    test_cases = [
        # (S, K, T, r, sigma_true, option_type)
        (100, 100, 1.0, 0.05, 0.25, OptionType.CALL),    # ATM call
        (100, 80,  0.5, 0.03, 0.40, OptionType.CALL),    # ITM call
        (100, 120, 2.0, 0.05, 0.15, OptionType.PUT),     # ITM put
        (100, 100, 0.1, 0.05, 0.80, OptionType.CALL),    # short-dated, high vol
        (100, 100, 5.0, 0.02, 0.10, OptionType.PUT),     # long-dated, low vol
    ]

    for S, K, T, r, sigma_true, opt_type in test_cases:
        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma_true)
        market_price = price(inputs, opt_type)

        iv_newton = implied_vol_newton(market_price, S, K, T, r, opt_type)
        iv_bisect = implied_vol_bisection(market_price, S, K, T, r, opt_type)

        print(f"S={S}, K={K}, T={T}, r={r}, sigma_true={sigma_true}, {opt_type.value}")
        print(f"  market_price = {market_price:.6f}")
        print(f"  Newton IV    = {iv_newton:.10f}  (error: {abs(iv_newton - sigma_true):.2e})")
        print(f"  Bisection IV = {iv_bisect:.10f}  (error: {abs(iv_bisect - sigma_true):.2e})")
        print()
