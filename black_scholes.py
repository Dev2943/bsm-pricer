"""
Black-Scholes European option pricer.

Implements the closed-form Black-Scholes-Merton formula for European calls
and puts on a non-dividend-paying stock.

Assumptions baked into this model (KNOW THESE — interviewers ask):
    1. Underlying follows geometric Brownian motion: dS = rS dt + sigma S dW
    2. Constant risk-free rate r
    3. Constant volatility sigma
    4. No dividends
    5. No transaction costs or taxes
    6. European exercise only (no early exercise)
    7. Continuous trading possible
    8. No arbitrage opportunities
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm


class OptionType(Enum):
    """European option types."""
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class BlackScholesInputs:
    """Container for Black-Scholes pricing inputs.

    Attributes
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price.
    T : float
        Time to maturity in years (e.g., 0.5 for six months).
    r : float
        Continuously compounded risk-free rate (e.g., 0.05 for 5%).
    sigma : float
        Annualized volatility of the underlying (e.g., 0.20 for 20%).
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float

    def __post_init__(self) -> None:
        # Validate inputs — silent garbage in => silent garbage out is a quant sin.
        if self.S <= 0:
            raise ValueError(f"Underlying price S must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike K must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to maturity T must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {self.sigma}")


def _d1(inputs: BlackScholesInputs) -> float:
    """Compute d1 from the Black-Scholes formula.

    d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
    """
    numerator = np.log(inputs.S / inputs.K) + (inputs.r + 0.5 * inputs.sigma ** 2) * inputs.T
    denominator = inputs.sigma * np.sqrt(inputs.T)
    return numerator / denominator


def _d2(inputs: BlackScholesInputs) -> float:
    """Compute d2 from the Black-Scholes formula.

    d2 = d1 - sigma * sqrt(T)
    """
    return _d1(inputs) - inputs.sigma * np.sqrt(inputs.T)


def price(inputs: BlackScholesInputs, option_type: OptionType) -> float:
    """Price a European option using the Black-Scholes formula.

    Parameters
    ----------
    inputs : BlackScholesInputs
        Model inputs (S, K, T, r, sigma).
    option_type : OptionType
        CALL or PUT.

    Returns
    -------
    float
        Theoretical option price under Black-Scholes assumptions.

    Formulas
    --------
    Call:  C = S * N(d1) - K * exp(-rT) * N(d2)
    Put:   P = K * exp(-rT) * N(-d2) - S * N(-d1)
    """
    d1 = _d1(inputs)
    d2 = _d2(inputs)
    discount = np.exp(-inputs.r * inputs.T)

    if option_type == OptionType.CALL:
        return inputs.S * norm.cdf(d1) - inputs.K * discount * norm.cdf(d2)
    elif option_type == OptionType.PUT:
        return inputs.K * discount * norm.cdf(-d2) - inputs.S * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option type: {option_type}")


if __name__ == "__main__":
    # Quick sanity check when run directly.
    inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
    call_price = price(inputs, OptionType.CALL)
    put_price = price(inputs, OptionType.PUT)

    print(f"Inputs: S=100, K=100, T=1y, r=5%, sigma=20%")
    print(f"  Call: ${call_price:.4f}")
    print(f"  Put:  ${put_price:.4f}")
    print(f"  C - P = {call_price - put_price:.4f}")
    print(f"  S - K*exp(-rT) = {inputs.S - inputs.K * np.exp(-inputs.r * inputs.T):.4f}")
    print(f"  (These last two should match — put-call parity)")
