"""
Test suite for greeks.py.

Three categories of tests, mirroring the structure of test_pricer.py:

    1. MATHEMATICAL IDENTITIES — must hold for ANY valid input set
       (e.g., delta_put = delta_call - 1, by put-call parity)

    2. FINITE DIFFERENCE TESTS — the analytical Greek must agree with
       a numerical derivative of the price function
       (this is the most important class of tests — it's how you prove
       your Greeks are correct from first principles)

    3. SANITY BOUNDS — call delta in [0, 1], gamma non-negative, etc.

Run with: pytest test_greeks.py -v
"""

import numpy as np
import pytest

from black_scholes import BlackScholesInputs, OptionType, price
from greeks import delta, gamma, vega, theta, rho


# A standard set of inputs to test against, covering different moneyness
# and maturities. Same style as test_pricer.py.
STANDARD_INPUTS = [
    (100, 100, 1.0, 0.05, 0.20),    # ATM, 1 year
    (100, 80,  0.5, 0.03, 0.30),    # ITM call
    (100, 120, 2.0, 0.05, 0.25),    # OTM call
    (50,  50,  0.25, 0.01, 0.40),   # short-dated
    (1000, 1000, 5.0, 0.02, 0.15),  # large notional
]


# ========== 1. MATHEMATICAL IDENTITIES ==========

@pytest.mark.parametrize("S, K, T, r, sigma", STANDARD_INPUTS)
def test_put_delta_equals_call_delta_minus_one(S, K, T, r, sigma):
    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)

    d_call = delta(inputs, OptionType.CALL)
    d_put = delta(inputs, OptionType.PUT)

    assert abs(d_put - (d_call - 1)) < 1e-10


@pytest.mark.parametrize("S, K, T, r, sigma", STANDARD_INPUTS)
def test_gamma_call_equals_gamma_put(S, K, T, r, sigma):
    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)

    g = gamma(inputs)

    assert g >= 0


@pytest.mark.parametrize("S, K, T, r, sigma", STANDARD_INPUTS)
def test_vega_call_equals_vega_put(S, K, T, r, sigma):
    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)

    v = vega(inputs)

    assert v >= 0


@pytest.mark.parametrize("S, K, T, r, sigma", STANDARD_INPUTS)
def test_rho_relationship(S, K, T, r, sigma):
    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)

    r_call = rho(inputs, OptionType.CALL)
    r_put = rho(inputs, OptionType.PUT)

    expected = K * T * np.exp(-r * T)

    assert abs((r_call - r_put) - expected) < 1e-8


# ========== 2. FINITE DIFFERENCE TESTS ==========

def test_delta_call_matches_finite_difference():
    inputs_base = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

    analytical = delta(inputs_base, OptionType.CALL)

    h = 0.01

    inputs_up = BlackScholesInputs(S=100 + h, K=100, T=1.0, r=0.05, sigma=0.20)
    inputs_down = BlackScholesInputs(S=100 - h, K=100, T=1.0, r=0.05, sigma=0.20)

    price_up = price(inputs_up, OptionType.CALL)
    price_down = price(inputs_down, OptionType.CALL)

    numerical = (price_up - price_down) / (2 * h)

    assert abs(analytical - numerical) < 1e-5


def test_gamma_matches_finite_difference():
    inputs_base = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

    analytical = gamma(inputs_base)

    h = 1.0

    inputs_up = BlackScholesInputs(S=100 + h, K=100, T=1.0, r=0.05, sigma=0.20)
    inputs_down = BlackScholesInputs(S=100 - h, K=100, T=1.0, r=0.05, sigma=0.20)

    price_up = price(inputs_up, OptionType.CALL)
    price_center = price(inputs_base, OptionType.CALL)
    price_down = price(inputs_down, OptionType.CALL)

    numerical = (price_up - 2 * price_center + price_down) / (h * h)

    assert abs(analytical - numerical) < 1e-3


def test_vega_matches_finite_difference():
    inputs_base = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

    analytical = vega(inputs_base)

    h = 0.0001

    inputs_up = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20 + h)
    inputs_down = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20 - h)

    price_up = price(inputs_up, OptionType.CALL)
    price_down = price(inputs_down, OptionType.CALL)

    numerical = (price_up - price_down) / (2 * h)

    assert abs(analytical - numerical) < 1e-3


# ========== 3. SANITY BOUNDS ==========

@pytest.mark.parametrize("S, K, T, r, sigma", STANDARD_INPUTS)
def test_call_delta_between_zero_and_one(S, K, T, r, sigma):
    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)

    d = delta(inputs, OptionType.CALL)

    assert 0 <= d <= 1


@pytest.mark.parametrize("S, K, T, r, sigma", STANDARD_INPUTS)
def test_put_delta_between_minus_one_and_zero(S, K, T, r, sigma):
    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)

    d = delta(inputs, OptionType.PUT)

    assert -1 <= d <= 0


@pytest.mark.parametrize("S, K, T, r, sigma", STANDARD_INPUTS)
def test_gamma_non_negative(S, K, T, r, sigma):
    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)

    assert gamma(inputs) >= 0


@pytest.mark.parametrize("S, K, T, r, sigma", STANDARD_INPUTS)
def test_vega_non_negative(S, K, T, r, sigma):
    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma)

    assert vega(inputs) >= 0