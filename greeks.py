"""
Black-Scholes Greeks: sensitivities of option price to inputs.

This file computes the five primary Greeks for European options:
    - Delta (dV/dS)
    - Gamma (d2V/dS2)
    - Vega  (dV/dsigma)
    - Theta (dV/dT, often negated to be 'time decay')
    - Rho   (dV/dr)

"""

import numpy as np
from scipy.stats import norm

from black_scholes import BlackScholesInputs, OptionType, _d1, _d2


# ----------------------------- DELTA -----------------------------
def delta(inputs: BlackScholesInputs, option_type: OptionType) -> float:
    """First derivative of option price with respect to underlying price.

    Formulas:
        delta_call = N(d1)
        delta_put  = N(d1) - 1
    """
    d1 = _d1(inputs)

    
    if option_type == OptionType.CALL:
        return norm.cdf(d1)

    elif option_type == OptionType.PUT:
        return norm.cdf(d1) - 1
    
    else:
        raise ValueError(f"Unknown option type: {option_type}")


# ----------------------------- GAMMA -----------------------------
def gamma(inputs: BlackScholesInputs) -> float:
    """Second derivative of option price with respect to underlying price."""

    d1 = _d1(inputs)

    return norm.pdf(d1) / (
        inputs.S * inputs.sigma * np.sqrt(inputs.T)
    )


# ----------------------------- VEGA -----------------------------
def vega(inputs: BlackScholesInputs) -> float:
    """First derivative of option price with respect to volatility."""

    d1 = _d1(inputs)

    return inputs.S * norm.pdf(d1) * np.sqrt(inputs.T)


# ----------------------------- THETA -----------------------------
def theta(inputs: BlackScholesInputs, option_type: OptionType) -> float:
    """First derivative of option price with respect to time-to-maturity."""

    d1 = _d1(inputs)
    d2 = _d2(inputs)

    common_term = (
        -inputs.S
        * norm.pdf(d1)
        * inputs.sigma
        / (2 * np.sqrt(inputs.T))
    )

    if option_type == OptionType.CALL:
        return (
            common_term
            - inputs.r
            * inputs.K
            * np.exp(-inputs.r * inputs.T)
            * norm.cdf(d2)
        )

    elif option_type == OptionType.PUT:
        return (
            common_term
            + inputs.r
            * inputs.K
            * np.exp(-inputs.r * inputs.T)
            * norm.cdf(-d2)
        )

    else:
        raise ValueError(f"Unknown option type: {option_type}")


# ----------------------------- RHO -----------------------------
def rho(inputs: BlackScholesInputs, option_type: OptionType) -> float:
    """First derivative of option price with respect to risk-free rate."""

    d2 = _d2(inputs)

    if option_type == OptionType.CALL:
        return (
            inputs.K
            * inputs.T
            * np.exp(-inputs.r * inputs.T)
            * norm.cdf(d2)
        )

    elif option_type == OptionType.PUT:
        return (
            -inputs.K
            * inputs.T
            * np.exp(-inputs.r * inputs.T)
            * norm.cdf(-d2)
        )

    else:
        raise ValueError(f"Unknown option type: {option_type}")

# ----------------------------- ALL GREEKS -----------------------------
def all_greeks(inputs: BlackScholesInputs, option_type: OptionType) -> dict[str, float]:
    """Return all five Greeks in a single dict."""
    return {
        "delta": delta(inputs, option_type),
        "gamma": gamma(inputs),
        "vega": vega(inputs),
        "theta": theta(inputs, option_type),
        "rho": rho(inputs, option_type),
    }


# ----------------------------- SANITY CHECK -----------------------------
if __name__ == "__main__":
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

    print("Greeks for S=100, K=100, T=1y, r=5%, sigma=20%\n")

    print("CALL:")
    for name, value in all_greeks(inputs, OptionType.CALL).items():
        print(f"  {name:6s}: {value:10.4f}")

    print("\nPUT:")
    for name, value in all_greeks(inputs, OptionType.PUT).items():
        print(f"  {name:6s}: {value:10.4f}")

    print("\nReference values from the lesson:")
    print("  CALL: delta=0.6368, gamma=0.0188, vega=37.524, theta=-6.414, rho=53.232")
    print("  PUT:  delta=-0.3632, gamma=0.0188, vega=37.524, theta=-1.658, rho=-41.890")
