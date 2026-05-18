"""
Test suite for the Black-Scholes pricer.

Three categories of tests:
    1. Put-call parity — must hold by no-arbitrage, for ANY valid inputs.
    2. Known benchmark values — published textbook results (Hull).
    3. Boundary conditions — deep ITM, deep OTM, near-zero T.

Run with: pytest test_pricer.py -v
"""

import numpy as np
import pytest

from black_scholes import BlackScholesInputs, OptionType, price


# ---------- 1. Put-call parity ----------
# Parity: C - P = S - K * exp(-rT). Must hold for every valid input set.

@pytest.mark.parametrize(
    "S, K, T, r, sigma",
    [
        (100, 100, 1.0, 0.05, 0.20),    # ATM, 1 year
        (100, 80, 0.5, 0.03, 0.30),     # ITM call
        (100, 120, 2.0, 0.05, 0.25),    # OTM call
        (50, 50, 0.25, 0.01, 0.40),     # short-dated
        (1000, 1000, 5.0, 0.02, 0.15),  # large notional
    ],
)
def test_put_call_parity(S, K, T, r, sigma):
    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)
    call = price(inputs, OptionType.CALL)
    put = price(inputs, OptionType.PUT)
    expected = S - K * np.exp(-r * T)
    assert abs((call - put) - expected) < 1e-10, (
        f"Parity violated: C - P = {call - put:.10f}, expected {expected:.10f}"
    )


# ---------- 2. Known benchmark values ----------
# From Hull, "Options, Futures, and Other Derivatives" (9th ed.) — classic
# textbook example: S=42, K=40, r=10%, sigma=20%, T=0.5.
# Hull reports Call = 4.76, Put = 0.81.

def test_hull_textbook_call():
    inputs = BlackScholesInputs(S=42, K=40, T=0.5, r=0.10, sigma=0.20)
    call = price(inputs, OptionType.CALL)
    assert abs(call - 4.7594) < 0.001, f"Got {call}, expected ~4.7594"


def test_hull_textbook_put():
    inputs = BlackScholesInputs(S=42, K=40, T=0.5, r=0.10, sigma=0.20)
    put = price(inputs, OptionType.PUT)
    assert abs(put - 0.8086) < 0.001, f"Got {put}, expected ~0.8086"


# ---------- 3. Boundary conditions ----------

def test_deep_itm_call_approaches_intrinsic():
    """A deep ITM call should price close to S - K * exp(-rT)."""
    inputs = BlackScholesInputs(S=200, K=100, T=1.0, r=0.05, sigma=0.20)
    call = price(inputs, OptionType.CALL)
    intrinsic_pv = inputs.S - inputs.K * np.exp(-inputs.r * inputs.T)
    assert abs(call - intrinsic_pv) < 0.5


def test_deep_otm_call_approaches_zero():
    """A deep OTM call should price close to zero."""
    inputs = BlackScholesInputs(S=50, K=200, T=1.0, r=0.05, sigma=0.20)
    call = price(inputs, OptionType.CALL)
    assert 0 <= call < 0.01


def test_deep_otm_put_approaches_zero():
    """A deep OTM put (high S, low K) should price close to zero."""
    inputs = BlackScholesInputs(S=200, K=50, T=1.0, r=0.05, sigma=0.20)
    put = price(inputs, OptionType.PUT)
    assert 0 <= put < 0.01


def test_prices_are_non_negative():
    """Option prices can never be negative under no-arbitrage."""
    np.random.seed(42)
    for _ in range(50):
        inputs = BlackScholesInputs(
            S=np.random.uniform(10, 500),
            K=np.random.uniform(10, 500),
            T=np.random.uniform(0.01, 5),
            r=np.random.uniform(0, 0.10),
            sigma=np.random.uniform(0.05, 1.0),
        )
        assert price(inputs, OptionType.CALL) >= 0
        assert price(inputs, OptionType.PUT) >= 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        BlackScholesInputs(S=-1, K=100, T=1.0, r=0.05, sigma=0.20)
    with pytest.raises(ValueError):
        BlackScholesInputs(S=100, K=0, T=1.0, r=0.05, sigma=0.20)
    with pytest.raises(ValueError):
        BlackScholesInputs(S=100, K=100, T=0, r=0.05, sigma=0.20)
    with pytest.raises(ValueError):
        BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0)
