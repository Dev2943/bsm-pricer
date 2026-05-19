# Day 3 — Implied Volatility

The capstone of Project 1. This is where the pricer becomes useful for real-world work.

---

## Part 1 — The problem

Black-Scholes takes σ (volatility) as an **input** and gives you a price. But in the real world, traders observe **prices** on exchanges and want to know what volatility the market is "implying."

Given:
- A market price for an option, call it `C_market`
- All other inputs (S, K, T, r) — these are observable

Find:
- The σ that makes Black-Scholes output exactly `C_market`

Formally: solve `BlackScholes(S, K, T, r, σ) = C_market` for σ.

**Why this matters:**

1. **Implied vol is the trader's actual variable.** Nobody quotes options by dollar price; they quote by IV. "I'm paying 22 vol for the January 100 calls" is how you'd hear it on a desk.
2. **The vol surface (IV across strikes and maturities) is the central object in derivatives trading.** It's how you spot mispricings, build vol strategies, calibrate exotic models.
3. **IV is the bridge between the model and the market.** It's the model's "confession" of what σ has to be for it to match reality.

There's no closed-form solution. We need a numerical root finder.

---

## Part 2 — Newton-Raphson method

The standard approach. You probably saw this in calculus.

**The idea:** suppose you want to find `x` such that `f(x) = 0`. Start with a guess `x_0`. Approximate `f` near `x_0` by its tangent line, and find where the tangent hits zero. That's your next guess `x_1`. Repeat.

The update rule:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

For IV, our function is:

$$f(\sigma) = BS(\sigma) - C_{market}$$

We want `f(σ) = 0`, i.e., the BS price equals the market price.

The derivative of `f` with respect to σ is... **vega**. We already implemented vega yesterday.

$$f'(\sigma) = \frac{\partial BS}{\partial \sigma} = \nu(\sigma)$$

So the IV iteration is:

$$\sigma_{n+1} = \sigma_n - \frac{BS(\sigma_n) - C_{market}}{\nu(\sigma_n)}$$

**This is the entire algorithm.** Start with a guess (σ₀ = 0.2 is a fine default), iterate the formula until the difference between BS(σ) and C_market is tiny, return σ.

### Why Newton-Raphson is so fast

For well-behaved functions, Newton-Raphson has **quadratic convergence**: the number of correct decimal places roughly doubles each step. In practice, IV converges in 5-10 iterations to machine precision. Compare to bisection (linear convergence, ~50 iterations).

### When Newton-Raphson fails

1. **Vega is near zero.** For deep ITM or deep OTM options, vega is tiny, and dividing by a tiny number explodes your step size. The algorithm overshoots and may diverge.
2. **Bad starting guess.** If σ₀ is wildly off, the tangent line approximation is poor and the iteration may go to negative σ or wander.
3. **Function has flat regions.** Same problem as #1.

Production code handles these with bounds (clamp σ > 0), maximum iteration limits, and falling back to bisection if Newton diverges.

---

## Part 3 — Bisection method (the safe backup)

Slower but bulletproof. Works as long as you can bracket the root — i.e., find σ_low and σ_high such that:

$$f(\sigma_{low}) < 0 < f(\sigma_{high})$$

(or vice versa).

**The algorithm:**

1. Compute `mid = (low + high) / 2`
2. Evaluate `f(mid)`
3. If `f(mid)` has the same sign as `f(low)`, the root is in `[mid, high]` — set `low = mid`
4. Otherwise the root is in `[low, mid]` — set `high = mid`
5. Repeat until `high - low` is tiny

**Convergence is linear** — each iteration halves the interval. So you gain one bit of precision per step, or ~3.3 decimal digits per 10 steps. To get 6 decimals of precision, you need ~20 iterations.

For IV, reasonable initial bounds are σ_low = 0.001 (essentially no vol) and σ_high = 5.0 (500% vol — extreme but covers all realistic cases). If the BS price at σ_low is above the market and the price at σ_high is below, your bounds are bad and IV can't be found (likely the input price violates arbitrage bounds).

---

## Part 4 — Arbitrage bounds (a sanity check before you solve)

Not every "price" admits a valid IV. Static no-arbitrage gives us bounds on what an option price can be:

For a European call:

$$\max(S - K e^{-rT}, 0) \leq C \leq S$$

For a European put:

$$\max(K e^{-rT} - S, 0) \leq P \leq K e^{-rT}$$

If a quoted price violates these bounds, no positive σ can produce it. Your solver should detect this and raise an informative error rather than spinning forever.

---

## Part 5 — Your task

Create a new file `implied_vol.py` in your project folder. Implement two functions:

```python
def implied_vol_newton(
    market_price: float,
    inputs_partial,        # S, K, T, r — but NOT sigma
    option_type: OptionType,
    initial_guess: float = 0.2,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Solve for IV using Newton-Raphson."""


def implied_vol_bisection(
    market_price: float,
    inputs_partial,
    option_type: OptionType,
    sigma_low: float = 1e-4,
    sigma_high: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> float:
    """Solve for IV using bisection. Slower but robust."""
```

I'll give you the scaffolded file in the next message. Key implementation notes:

- Each function should return σ, the implied volatility
- If the input price violates arbitrage bounds, raise `ValueError` with a clear message
- If the solver fails to converge, raise `RuntimeError` with the iteration count
- The Newton method should fall back to bisection if it diverges or hits non-positive σ (this is what production code does)

---

## Part 6 — How to verify your solver

The classic test: **round-trip consistency.**

1. Pick any σ_true (say, 0.25).
2. Use your pricer to compute the BS price at that σ.
3. Feed that price back into your IV solver.
4. The solver should return σ ≈ σ_true to ~6 decimal places.

If this doesn't work, your solver is broken.

Do this across:
- A range of moneyness (deep ITM, ATM, deep OTM)
- A range of maturities (1 week, 1 year, 5 years)
- A range of true volatilities (5%, 20%, 80%)
- Both calls and puts

If all roundtrips pass, your solver is correct.

---

## Part 7 — Newton vs bisection convergence comparison

This is the killer chart for your README.

After both solvers work, write a small script that:

1. Picks a single test case (e.g., S=100, K=100, T=1, r=0.05, true σ = 0.25)
2. Computes the BS price
3. Runs Newton-Raphson and records `(iteration, current_sigma)` at each step
4. Runs bisection and does the same
5. Plots both on a log-y axis (y = absolute error from true σ vs iteration count)

Newton's curve will plummet — error roughly squared each step until machine precision. Bisection's curve will be a straight line — error halved each step. Visually striking. Recruiters love this.

---

## Part 8 — Interview questions

The IV solver is a beloved interview topic. Be ready for:

1. **Why Newton-Raphson and not bisection by default?** (quadratic convergence vs linear)
2. **When does Newton-Raphson fail?** (vega near zero — deep ITM/OTM)
3. **What's vega's role in your iteration?** (it's the derivative, the "slope" used for the tangent line)
4. **Why is there no closed form for IV?** (BS is monotonic in σ but transcendental; you can't algebraically isolate σ)
5. **What's the no-arbitrage bound on a call price? Why?** (max(S - K·e^(-rT), 0) ≤ C ≤ S — sketch the static replication argument)
6. **How do you detect that a quoted price has no valid IV?** (it's outside the arbitrage bounds OR the solver hits the σ_high cap)
7. **What is the implied volatility smile?** (IV plotted across strikes for fixed expiry isn't flat — it's typically U-shaped; this contradicts a core BS assumption and is the entire reason stochastic vol models exist)
8. **Why don't traders quote prices in dollars?** (because vol is the more stable / interpretable variable; a $5 option might be cheap or expensive depending on vol, but "22 vol" has clear meaning)

---

## What to do now

1. **Read Parts 1–4 carefully.** Make sure you understand the algorithm before you see the code.
2. Tell me you've read it and I'll send the scaffolded file with TODOs.

We're going to do this the same way as the Greeks: I scaffold, you fill in the math. The work this time is bigger (loop logic, edge cases) but the pattern is identical — translate equations into code.

Ready when you are.
