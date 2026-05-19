"""
Test suite for implied_vol.py.

Categories:
    1. ROUND-TRIP TESTS — price at a known sigma, then verify the solver
       recovers it. This is the fundamental correctness test for any IV solver.
    2. ARBITRAGE BOUND TESTS — out-of-bounds prices must raise ValueError.
    3. AGREEMENT TESTS — Newton and bisection must agree on the answer.

Run with: pytest test_implied_vol.py -v
"""

import numpy as np
import pytest

from black_scholes import BlackScholesInputs, OptionType, price
from implied_vol import implied_vol_newton, implied_vol_bisection


# (S, K, T, r, sigma_true, option_type) — wide coverage of the parameter space
ROUND_TRIP_CASES = [
    (100, 100, 1.0,  0.05, 0.25, OptionType.CALL),    # ATM call, 1y
    (100,  80, 0.5,  0.03, 0.40, OptionType.CALL),    # ITM call, short-dated
    (100, 120, 2.0,  0.05, 0.15, OptionType.PUT),     # ITM put, long-dated
    (100, 100, 0.1,  0.05, 0.80, OptionType.CALL),    # very short, high vol
    (100, 100, 5.0,  0.02, 0.10, OptionType.PUT),     # very long, low vol
    ( 50,  60, 1.0,  0.04, 0.30, OptionType.CALL),    # OTM call
    (200, 180, 0.75, 0.05, 0.20, OptionType.PUT),     # OTM put
]


# ========== 1. ROUND-TRIP TESTS ==========

@pytest.mark.parametrize("S, K, T, r, sigma_true, opt_type", ROUND_TRIP_CASES)
def test_newton_roundtrip(S, K, T, r, sigma_true, opt_type):

    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma_true)

    market_price = price(inputs, opt_type)

    recovered = implied_vol_newton(
        market_price, S, K, T, r, opt_type
    )

    assert abs(recovered - sigma_true) < 1e-6


@pytest.mark.parametrize("S, K, T, r, sigma_true, opt_type", ROUND_TRIP_CASES)
def test_bisection_roundtrip(S, K, T, r, sigma_true, opt_type):

    inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma_true)

    market_price = price(inputs, opt_type)

    recovered = implied_vol_bisection(
        market_price, S, K, T, r, opt_type
    )

    assert abs(recovered - sigma_true) < 1e-6


# ========== 2. ARBITRAGE BOUND TESTS ==========

def test_call_price_above_underlying_raises():

    with pytest.raises(ValueError):

        implied_vol_newton(
            150.0,   # impossible call price
            100,     # S
            100,     # K
            1.0,     # T
            0.05,    # r
            OptionType.CALL,
        )


def test_call_price_below_intrinsic_raises():

    with pytest.raises(ValueError):

        implied_vol_newton(
            10.0,
            100,
            80,
            1.0,
            0.05,
            OptionType.CALL,
        )


def test_put_price_above_strike_pv_raises():

    with pytest.raises(ValueError):

        implied_vol_newton(
            200.0,
            100,
            100,
            1.0,
            0.05,
            OptionType.PUT,
        )


# ========== 3. AGREEMENT TESTS ==========

@pytest.mark.parametrize("S, K, T, r, sigma_true, opt_type", ROUND_TRIP_CASES)
def test_newton_and_bisection_agree(
    S, K, T, r, sigma_true, opt_type
):

    inputs = BlackScholesInputs(
        S=S, K=K, T=T, r=r, sigma=sigma_true
    )

    market_price = price(inputs, opt_type)

    iv_newton = implied_vol_newton(
        market_price, S, K, T, r, opt_type
    )

    iv_bisect = implied_vol_bisection(
        market_price, S, K, T, r, opt_type
    )

    assert abs(iv_newton - iv_bisect) < 1e-6