# Black-Scholes European Option Pricer

A clean implementation of the Black-Scholes-Merton model for pricing European calls and puts on a non-dividend-paying stock. Built as the foundation for a broader quantitative finance project portfolio.

## What this does

- Prices European call and put options using the closed-form BSM formula
- Validates against put-call parity (no-arbitrage sanity check)
- Validates against Hull textbook benchmark values
- Validates boundary conditions (deep ITM, deep OTM)

Greeks, implied volatility solver, and an interactive UI are coming in subsequent commits.

## The math

A European call on a non-dividend stock under BSM:

```
C = S * N(d1) - K * exp(-rT) * N(d2)
P = K * exp(-rT) * N(-d2) - S * N(-d1)

d1 = [ln(S/K) + (r + sigma^2 / 2) * T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
```

Where:
- `S` = current underlying price
- `K` = strike price
- `T` = time to maturity (years)
- `r` = continuously compounded risk-free rate
- `sigma` = annualized volatility
- `N(.)` = standard normal CDF

### Derivation sketch

The result follows from three ingredients:

1. **Geometric Brownian Motion** for the underlying: `dS = rS dt + sigma S dW` under the risk-neutral measure Q.
2. **Risk-neutral pricing**: a derivative's value today equals `exp(-rT) * E^Q[payoff at T]`.
3. **Log-normal distribution of S_T**: `ln(S_T) ~ N(ln(S_0) + (r - sigma^2/2)T, sigma^2 T)`.

Plug the log-normal density into `exp(-rT) * E^Q[max(S_T - K, 0)]` and the result simplifies to the formula above.

### Put-call parity

By a static no-arbitrage argument (long call + short put + cash `K*exp(-rT)` replicates long stock):

```
C - P = S - K * exp(-rT)
```

This must hold for any valid input set. We test it across five parameter combinations.

## Assumptions of the model

These are the standard BSM assumptions. Each one is violated in real markets to some degree — knowing where and how matters more than memorizing the list:

1. Underlying follows geometric Brownian motion (real markets show jumps, vol clustering, fat tails)
2. Constant risk-free rate (rates evolve stochastically; matters more for long-dated options)
3. Constant volatility (real implied vol shows a smile/skew across strikes and a term structure across maturities)
4. No dividends (extendable; this implementation omits for simplicity)
5. No transaction costs or taxes
6. European exercise only (American exercise requires a separate model — binomial tree or LSM)
7. Continuous trading
8. No arbitrage

## Project structure

```
project_1_black_scholes/
├── black_scholes.py    # Core pricer
├── test_pricer.py      # Test suite
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
python black_scholes.py        # Quick sanity check
pytest test_pricer.py -v       # Full test suite
```

Expected output of the sanity check:

```
Inputs: S=100, K=100, T=1y, r=5%, sigma=20%
  Call: $10.4506
  Put:  $5.5735
  C - P = 4.8771
  S - K*exp(-rT) = 4.8771
```

## Design choices

- **Frozen dataclass for inputs** — immutable, hashable, gives a clean API with validation in `__post_init__`.
- **Enum for option type** — prevents typos like `"Call"` vs `"call"` causing silent bugs.
- **Validation on construction** — fail fast on bad inputs rather than producing nonsense prices.
- **scipy.stats.norm.cdf** rather than a hand-rolled approximation — the underlying erf is well-optimized; reimplementing it would be a wheel-reinvention with no upside for a project at this stage.

## Limitations and what's next

What this pricer **cannot** do today, in order of how I'll address them:

1. **No Greeks** — coming day 2 (delta, gamma, vega, theta, rho).
2. **No implied volatility solver** — coming day 3 (Newton-Raphson and bisection, with convergence comparison).
3. **No dividends** — extendable by replacing `S` with `S * exp(-qT)` where `q` is the dividend yield.
4. **European only** — American option pricing requires a binomial tree or Longstaff-Schwartz Monte Carlo (see Project 2).
5. **Constant volatility** — real markets show implied vol smile/skew. Heston (stochastic vol) and Dupire (local vol) extensions are out of scope here but conceptually understood.

## References

- Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities," *Journal of Political Economy*, 81(3).
- Merton, R. (1973). "Theory of Rational Option Pricing," *Bell Journal of Economics*, 4(1).
- Hull, J. *Options, Futures, and Other Derivatives*, 9th ed. — Chapter 15 is the standard reference; benchmark values in `test_pricer.py` come from his Example 15.6.

---

Part of a broader quantitative finance project portfolio. Project 2 (Monte Carlo with variance reduction) builds on this foundation.
