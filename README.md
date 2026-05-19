# Black-Scholes Options Pricer with Greeks and Implied Volatility

A clean, tested implementation of the Black-Scholes-Merton model for European options, including all five primary Greeks and two numerical solvers for implied volatility. Built as the foundation for a broader quantitative finance project portfolio.

## Highlights

- Closed-form Black-Scholes pricer for European calls and puts
- Five Greeks: **delta, gamma, vega, theta, rho** with full mathematical derivations in the code
- Implied volatility solvers using both **Newton-Raphson** and **bisection**, with automatic fallback when Newton diverges
- 55+ tests including put-call parity, finite-difference Greek verification, and round-trip IV consistency
- Convergence comparison demonstrating quadratic (Newton) vs. linear (bisection) convergence

## Convergence — Newton vs. Bisection

![Convergence plot](convergence_plot.png)

Newton-Raphson converges to machine precision (~1e-16) in 3 iterations because each step roughly doubles the number of correct digits. Bisection takes 53 iterations to reach the same precision because it gains only one bit per step. Both methods are implemented; Newton is used by default with bisection as a safety fallback.

---

## The math

### Pricing formula

For a European call on a non-dividend-paying stock under Black-Scholes-Merton assumptions:

```
C = S * N(d1) - K * exp(-rT) * N(d2)
P = K * exp(-rT) * N(-d2) - S * N(-d1)

d1 = [ln(S/K) + (r + sigma^2 / 2) * T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
```

Where `N(.)` is the standard normal CDF.

### Put-call parity

By a static no-arbitrage argument, the call and put prices must satisfy:

```
C - P = S - K * exp(-rT)
```

This identity is enforced as a test across multiple parameter sets.

### Greeks

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| Delta_call | `N(d1)` | dC/dS — sensitivity to underlying |
| Delta_put | `N(d1) - 1` | dP/dS |
| Gamma | `phi(d1) / (S * sigma * sqrt(T))` | d²V/dS² — same for call and put |
| Vega | `S * phi(d1) * sqrt(T)` | dV/dsigma — same for call and put |
| Theta_call | `-S*phi(d1)*sigma/(2*sqrt(T)) - r*K*exp(-rT)*N(d2)` | -dV/dt (time decay) |
| Theta_put | `-S*phi(d1)*sigma/(2*sqrt(T)) + r*K*exp(-rT)*N(-d2)` | |
| Rho_call | `K * T * exp(-rT) * N(d2)` | dV/dr |
| Rho_put | `-K * T * exp(-rT) * N(-d2)` | |

Where `phi(.)` is the standard normal PDF.

### Implied volatility

The Black-Scholes formula is monotonic but transcendental in sigma, so there is no closed-form inversion. Given a market price, we solve `BS(sigma) = C_market` numerically.

**Newton-Raphson iteration:**

```
sigma_{n+1} = sigma_n - (BS(sigma_n) - C_market) / vega(sigma_n)
```

The derivative is vega — which is already implemented for the Greeks. Quadratic convergence in well-behaved regions; ~5-10 iterations to machine precision in practice.

**Bisection** is the bulletproof fallback: bracket the root in `[sigma_low, sigma_high]`, repeatedly halve the interval, gain one bit of precision per step.

### No-arbitrage bounds

Before solving for IV, we validate that the quoted price falls within the no-arbitrage range:

```
For a call: max(S - K*exp(-rT), 0) <= C <= S
For a put:  max(K*exp(-rT) - S, 0) <= P <= K*exp(-rT)
```

Prices outside this range have no valid implied volatility — the solver raises `ValueError`.

---

## Assumptions of the model

These are the standard BSM assumptions. Each is violated in real markets — knowing where matters more than memorizing the list.

1. Underlying follows geometric Brownian motion (real markets exhibit jumps, vol clustering, fat tails)
2. Constant risk-free rate (rates evolve stochastically; matters more for long-dated options)
3. Constant volatility (real implied vol shows a smile/skew across strikes and a term structure across maturities)
4. No dividends (this implementation; extendable via the q-adjustment)
5. No transaction costs or taxes
6. European exercise only
7. Continuous trading
8. No arbitrage

The persistence of the implied-volatility smile in real markets is direct evidence that assumption #3 is violated, and is the entire motivation for stochastic-vol models like Heston.

---

## Project structure

```
bsm-pricer/
├── black_scholes.py     # Core pricer (Day 1)
├── greeks.py            # Five Greeks (Day 2)
├── implied_vol.py       # Newton-Raphson + Bisection solvers (Day 3)
├── plot_convergence.py  # Generates convergence_plot.png
├── test_pricer.py       # 12 tests
├── test_greeks.py       # 43 tests (with parametrization)
├── test_implied_vol.py  # Round-trip and arbitrage bound tests
├── convergence_plot.png # Centerpiece visualization
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt

# Pricer + sanity check
python3 black_scholes.py

# Greeks + reference values
python3 greeks.py

# IV solver round-trip
python3 implied_vol.py

# Convergence plot
python3 plot_convergence.py

# Full test suite (55+ tests)
pytest -v
```

## API examples

### Pricing

```python
from black_scholes import BlackScholesInputs, OptionType, price

inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
call_price = price(inputs, OptionType.CALL)   # 10.4506
put_price = price(inputs, OptionType.PUT)     # 5.5735
```

### Greeks

```python
from greeks import delta, gamma, vega, theta, rho, all_greeks

inputs = BlackScholesInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

delta(inputs, OptionType.CALL)   # 0.6368
gamma(inputs)                    # 0.0188 (same for call and put)
vega(inputs)                     # 37.524
theta(inputs, OptionType.CALL)   # -6.414 per year
rho(inputs, OptionType.CALL)     # 53.232 per 100% rate change

# All in one dict:
all_greeks(inputs, OptionType.CALL)
```

### Implied volatility

```python
from implied_vol import implied_vol_newton, implied_vol_bisection

# Given a market quote of $12.34 for the same ATM 1-year call:
iv = implied_vol_newton(12.34, S=100, K=100, T=1.0, r=0.05,
                        option_type=OptionType.CALL)
# Returns sigma ≈ 0.2502
```

---

## Tests

55+ tests across three categories:

| File | Tests | What they verify |
|------|-------|------------------|
| `test_pricer.py` | 12 | Put-call parity across parameter sets, Hull textbook benchmarks, boundary conditions (deep ITM/OTM), input validation |
| `test_greeks.py` | 43 | Algebraic identities (`delta_put = delta_call - 1`, `gamma_call = gamma_put`), **finite-difference verification of each Greek against the analytical formula**, sanity bounds |
| `test_implied_vol.py` | varies | Round-trip consistency (price → IV → recovered sigma), arbitrage bound violations raise `ValueError`, Newton and bisection agree |

The finite-difference Greek tests are particularly meaningful: they verify the analytical Greek formulas against numerical derivatives of the price function, directly from the definition `f'(x) ≈ [f(x+h) - f(x-h)] / (2h)`. If the two agree, the analytical formula is correct from first principles.

---

## Limitations and what's next

What this implementation **does not** do, and how it would be extended:

1. **No dividends.** Easily added by replacing `S` with `S * exp(-qT)` for continuous dividend yield `q`.
2. **European exercise only.** American options would require a binomial tree (CRR) or Longstaff-Schwartz Monte Carlo.
3. **Constant volatility.** Real markets show implied vol smile/skew. Heston (stochastic vol) and Dupire (local vol) are the standard extensions.
4. **No jumps in the underlying.** Merton's jump-diffusion model addresses this.
5. **Single asset.** Multi-asset and basket options would require Cholesky-decomposed correlated Brownian motions.

These are addressed by subsequent projects in the broader portfolio (Monte Carlo, exotic option pricing, volatility modeling).

---

## References

- Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities," *Journal of Political Economy*, 81(3).
- Merton, R. (1973). "Theory of Rational Option Pricing," *Bell Journal of Economics*, 4(1).
- Hull, J. *Options, Futures, and Other Derivatives*, 9th ed. — Chapter 15 (pricing), Chapter 19 (Greeks), Chapter 20 (implied volatility).
- Press, Teukolsky, Vetterling, Flannery. *Numerical Recipes*, Chapter 9 — root finding methods.

---

Part of a broader quantitative finance project portfolio.
