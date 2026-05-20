"""
Volatility Smile from Real Market Data.

Pulls the SPY option chain from Yahoo Finance, runs our implied volatility
solver on each call across strikes for a single expiry, and plots the
resulting smile/skew.

This demonstrates two things at once:
    1. The IV solver works on real market quotes (not just synthetic BS prices)
    2. Real markets show a skew that BS cannot reproduce — proof BS is wrong

Caveats (covered in day4_vol_smile_lesson.md, Part 5):
    - SPY pays dividends; this BS model does not (small IV bias)
    - SPY options are American; this BS model is European (small ITM bias)
    - We use the mid-price; wide spreads can distort it (we filter)
    - We use calendar days / 365 for T; trading-days conventions also exist

Run with: python3 vol_smile.py
Produces: volatility_smile.png
"""

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from black_scholes import OptionType
from implied_vol import implied_vol_newton


# ----------------------------- CONFIG -----------------------------
TICKER = "SPY"
RISK_FREE_RATE = 0.045  # ~current 1-3 month T-bill yield; hardcoded for simplicity
TARGET_DAYS_TO_EXPIRY = 30  # we'll pick the available expiry closest to this
MIN_BID = 0.05  # filter: bids below this are too illiquid to be meaningful
MAX_SPREAD_RATIO = 0.50  # filter: (ask - bid) / mid > 50% → toss it


# ----------------------------- DATA FETCH (provided) -----------------------------
def fetch_chain(ticker: str, target_days: int):
    """Fetch the option chain for the expiry closest to target_days from now.

    Returns: (calls_df, spot_price, time_to_expiry_in_years)
    """
    tkr = yf.Ticker(ticker)

    # Current spot price (most recent close)
    spot = tkr.history(period="1d")["Close"].iloc[-1]

    # All available expiries, as datetime objects
    expiries = [datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                for d in tkr.options]
    today = datetime.now(timezone.utc)

    # Find the expiry closest to our target days
    target_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
    best_expiry = min(
        expiries,
        key=lambda d: abs((d - target_date).days - target_days),
    )
    days_to_expiry = (best_expiry - target_date).days
    expiry_str = best_expiry.strftime("%Y-%m-%d")

    print(f"Spot price ({ticker}): ${spot:.2f}")
    print(f"Selected expiry: {expiry_str} ({days_to_expiry} days out)")

    chain = tkr.option_chain(expiry_str)
    calls = chain.calls

   
    # We have `days_to_expiry` as an integer number of days.
    # Convert to years using the calendar convention: T = days / 365
    T = days_to_expiry / 365

    return calls, float(spot), T


# ----------------------------- IV COMPUTATION -----------------------------
def compute_smile(calls_df, spot: float, T: float, r: float):
    """Run the IV solver on every viable call in the chain.

    Returns: (strikes, ivs) — two parallel lists of floats
    """
    strikes = []
    ivs = []

    for _, row in calls_df.iterrows():
        strike = float(row["strike"])
        bid = float(row["bid"])
        ask = float(row["ask"])

        # ----- Filter 1: skip if bid is too low (illiquid) -----
        if bid < MIN_BID:
            continue

        # ----- Filter 2: skip if spread is too wide -----
        mid = (bid + ask) / 2

        spread_ratio = (ask - bid) / mid if mid > 0 else float("inf")
        if spread_ratio > MAX_SPREAD_RATIO:
            continue

        # ----- Filter 3: skip prices outside arbitrage bounds -----
        # The IV solver itself will raise ValueError if so; we catch and skip.
        try:
            iv = implied_vol_newton(
                market_price=mid,
                S=spot,
                K=strike,
                T=T,
                r=r,
                option_type=OptionType.CALL,
            )

        except (ValueError, RuntimeError):
            # ValueError = arbitrage bound violation
            # RuntimeError = solver failed to converge
            continue

        # ----- Filter 4: drop absurd IV values (data errors) -----
        if iv < 0.01 or iv > 3.0:
            continue

        strikes.append(strike)
        ivs.append(iv)

    return strikes, ivs


# ----------------------------- PLOTTING (provided) -----------------------------
def plot_smile(strikes, ivs, spot, T, expiry_label):
    """Render the smile/skew plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot IV as percentage points (multiply by 100)
    ax.plot(strikes, [iv * 100 for iv in ivs],
            marker="o", markersize=5, linewidth=1.5,
            color="#C44E52", label="Market IV")

    # Vertical line at spot price
    ax.axvline(spot, color="gray", linestyle="--", alpha=0.6,
               label=f"Spot = ${spot:.2f}")

    ax.set_xlabel("Strike ($)", fontsize=12)
    ax.set_ylabel("Implied Volatility (%)", fontsize=12)
    ax.set_title(
        f"SPY Volatility Skew — {expiry_label} expiry (T = {T*365:.0f} days)\n"
        f"Real market quotes inverted through Black-Scholes",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Annotation explaining what we're seeing
    ax.text(
        0.02, 0.05,
        "If BS were correct, this curve would be flat.\n"
        "The downward slope is the equity 'skew' — \n"
        "OTM puts are priced as crash insurance.",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig("volatility_smile.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved volatility_smile.png ({len(strikes)} valid strikes)")

    try:
        plt.show()
    except Exception:
        pass


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print(f"Fetching {TICKER} option chain...\n")

    calls, spot, T = fetch_chain(TICKER, TARGET_DAYS_TO_EXPIRY)

    print(f"Number of calls in chain: {len(calls)}")

    strikes, ivs = compute_smile(
        calls,
        spot,
        T,
        RISK_FREE_RATE,
    )

    print(f"After filtering: {len(strikes)} viable strikes\n")

    if len(strikes) < 5:
        print("Too few strikes survived filtering. Something is wrong.")
        print("Maybe market is closed? yfinance rate-limited?")
        raise SystemExit(1)

    # Pretty print a few values
    print("Sample of the smile:")
    for k, iv in list(zip(strikes, ivs))[::max(1, len(strikes) // 10)]:
        marker = " <-- ATM" if abs(k - spot) < 1 else ""
        print(f"  Strike ${k:.0f}  →  IV {iv*100:.2f}%{marker}")

    # Use one of the expiry dates we'll fish back from the data
    plot_smile(strikes, ivs, spot, T, expiry_label=f"~{T*365:.0f}d")
