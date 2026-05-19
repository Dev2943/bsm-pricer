# Day 2 — The Greeks

A complete reference for the Black-Scholes Greeks: what they are, where they come from, how to implement them, and the questions interviewers will use to test whether you actually understand them.

---

## Part 1 — What Greeks are and why they matter

The Black-Scholes formula gives you one number: the price of an option. But a price by itself is useless for managing risk. If you're a trader holding 10,000 call options, you don't just want to know what they're worth today — you want to know:

- If the stock moves $1, what happens to my position?
- If volatility spikes, what happens?
- How much money am I losing every day just from time passing?

Greeks answer these questions. Mathematically, every Greek is a **partial derivative** of the option price with respect to one of its inputs.

| Greek | Symbol | Definition | Plain English |
|---|---|---|---|
| Delta | Δ | ∂V/∂S | Price sensitivity to underlying move |
| Gamma | Γ | ∂²V/∂S² | How fast delta changes |
| Vega | ν | ∂V/∂σ | Price sensitivity to volatility |
| Theta | Θ | ∂V/∂T | Price sensitivity to time |
| Rho | ρ | ∂V/∂r | Price sensitivity to interest rate |

Three real-world uses to keep in your head:

1. **Hedging.** A market maker who sold you a call wants to neutralize their delta — they'll buy Δ shares of stock for every option to make the position delta-neutral. As the stock moves, delta changes (that's gamma), so they have to re-hedge constantly.

2. **Position expression.** "Long gamma" is a bet on big moves in either direction. "Long vega" is a bet on rising implied volatility. Traders express specific views by structuring positions with specific Greek profiles.

3. **Risk reporting.** Every options desk produces a "Greeks report" daily showing the aggregate Δ, Γ, ν, Θ, ρ exposure of the book. This is how risk managers know what they're sitting on.

---

## Part 2 — Delta, in full

Delta is the easiest Greek conceptually but it teaches you the trick that makes all the others tractable. Pay attention.

### The formula

For a European call: **Δ_call = N(d₁)**

For a European put: **Δ_put = N(d₁) - 1**

Where d₁ is the same one from the pricer:

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$

### Intuition

- Call delta is between 0 and 1. Deep ITM call → Δ → 1 (moves dollar-for-dollar with stock). Deep OTM call → Δ → 0 (worthless, doesn't react). At-the-money → Δ ≈ 0.5.
- Put delta is between -1 and 0. Deep ITM put → Δ → -1. Deep OTM put → Δ → 0.
- Put-call parity gives you the relationship Δ_put = Δ_call - 1 by direct differentiation.

### The derivation — and the trick

Start with C = S·N(d₁) - K·e^(-rT)·N(d₂). Take the partial derivative with respect to S:

$$\frac{\partial C}{\partial S} = N(d_1) + S \cdot \phi(d_1) \cdot \frac{\partial d_1}{\partial S} - K e^{-rT} \cdot \phi(d_2) \cdot \frac{\partial d_2}{\partial S}$$

where φ(·) = N'(·) is the standard normal PDF.

Since d₂ = d₁ - σ√T, we have ∂d₂/∂S = ∂d₁/∂S. So the last two terms factor:

$$= N(d_1) + \frac{\partial d_1}{\partial S}\left[S \phi(d_1) - K e^{-rT} \phi(d_2)\right]$$

**Now the magic identity:**

$$S \phi(d_1) = K e^{-rT} \phi(d_2)$$

This is true for all S, K, r, σ, T. It's a consequence of the algebraic structure of d₁ and d₂. (Proof: expand e^(-d₂²/2) using d₂ = d₁ - σ√T and the definition of d₁; the cross terms collapse exactly.)

The bracket vanishes, leaving:

$$\boxed{\frac{\partial C}{\partial S} = N(d_1)}$$

**Why this matters for interviews:** if you can state and prove this identity, you immediately signal that you understand BS at a structural level, not just memorization. The identity reappears in every other Greek derivation.

---

## Part 3 — The other four

### Gamma

**Γ = φ(d₁) / (S · σ · √T)** — same for both calls and puts.

Derivation: differentiate Δ_call = N(d₁) once more with respect to S. The chain rule gives Γ = φ(d₁) · ∂d₁/∂S, and ∂d₁/∂S = 1/(S·σ·√T).

Key facts:
- Gamma is always positive for long calls and long puts (convexity of the payoff)
- Gamma is highest near ATM and decays as you move ITM or OTM
- Gamma is highest near expiration for ATM options (they're "knife-edge" sensitive)
- Long gamma costs you theta — there's no free lunch

### Vega

**ν = S · φ(d₁) · √T** — same for both calls and puts.

Note: vega is reported per 1.00 change in volatility (i.e., per 100 percentage points). To express "per 1 percentage point change in vol," divide by 100. This is a common reporting choice — be explicit about it.

Key facts:
- Vega is always positive for long calls and long puts
- Vega is largest for ATM, long-dated options
- Vega → 0 as T → 0 (no time for vol to matter) or as options go deep ITM/OTM

### Theta

**Θ_call = - [S · φ(d₁) · σ / (2√T)] - r · K · e^(-rT) · N(d₂)**

**Θ_put = - [S · φ(d₁) · σ / (2√T)] + r · K · e^(-rT) · N(-d₂)**

Key facts:
- Theta is the messiest Greek; pay attention to signs and conventions
- Theta is typically negative for long options (you lose money as time passes — "time decay")
- Theta is reported per year by default. Divide by 365 to get per-calendar-day decay, or by 252 to get per-trading-day decay. State your convention.
- Theta acceleration: for ATM options, theta gets worse (more negative) as expiration approaches

### Rho

**ρ_call = K · T · e^(-rT) · N(d₂)**

**ρ_put = -K · T · e^(-rT) · N(-d₂)**

Often reported per 1 percentage point change in r, so divide by 100. Rho is the least-watched Greek for equity options (rates move slowly) but matters for long-dated options and is critical for fixed-income derivatives.

---

## Part 4 — Your implementation task

Create a new file `greeks.py` in the same project folder, alongside `black_scholes.py`.

### Structure

The file should expose five functions, each taking the `BlackScholesInputs` dataclass and the `OptionType` enum (both imported from `black_scholes`). Match the style of the existing code.

```python
def delta(inputs: BlackScholesInputs, option_type: OptionType) -> float:
    ...

def gamma(inputs: BlackScholesInputs) -> float:
    ...  # same for call and put — no option_type needed

def vega(inputs: BlackScholesInputs) -> float:
    ...

def theta(inputs: BlackScholesInputs, option_type: OptionType) -> float:
    ...

def rho(inputs: BlackScholesInputs, option_type: OptionType) -> float:
    ...
```

Plus a convenience function:

```python
def all_greeks(inputs: BlackScholesInputs, option_type: OptionType) -> dict[str, float]:
    """Return all five Greeks for an option in a single dict."""
    ...
```

Include a `__main__` block at the bottom that prints the Greeks for the same example as `black_scholes.py` (S=100, K=100, T=1, r=5%, σ=20%) for both a call and a put. This is your sanity check.

### Hints (read only if stuck)

- `scipy.stats.norm.pdf(x)` gives you φ(x). You're already using `norm.cdf(x)` for N(x).
- Don't recompute d₁ and d₂ inside every function. Either reuse `_d1` and `_d2` from `black_scholes.py` (you'll need to import them — Python allows underscored imports), or compute them once at the top of each Greek function.
- Use the dataclass attributes directly: `inputs.S`, `inputs.K`, `inputs.T`, `inputs.r`, `inputs.sigma`.
- Pay attention to **signs** in theta and rho — easy place to lose points.

### Self-check before writing tests

Run your `__main__` and check that you get something close to these reference values for S=100, K=100, T=1, r=5%, σ=20%:

| Greek | Call | Put |
|---|---|---|
| Delta | 0.6368 | -0.3632 |
| Gamma | 0.0188 | 0.0188 |
| Vega | 37.524 | 37.524 |
| Theta | -6.414 | -1.658 |
| Rho | 53.232 | -41.890 |

If your numbers don't match within ~0.01, something is off. Debug before moving on.

---

## Part 5 — Tests you should write

Create `test_greeks.py` with these tests. Each one catches a different class of bug.

### Mathematical identities — these must hold for ANY valid inputs

1. **Put delta vs call delta:** `delta_put == delta_call - 1`. From put-call parity.
2. **Gamma identity:** `gamma_call == gamma_put`. Same for calls and puts; the second derivative of C - P with respect to S is zero because C - P is linear in S.
3. **Vega identity:** `vega_call == vega_put`. Same reasoning.
4. **Rho relationship:** From put-call parity, `rho_call - rho_put == K · T · e^(-rT)`.

### Numerical verification — finite differences must approximate analytical

This is the most important class of test. The Greeks are *defined* as derivatives, so they must match numerical derivatives of the price function:

5. **Delta vs finite difference:** Compute `(price(S + h) - price(S - h)) / (2h)` for h = 0.01. Should match `delta(S)` to 4-5 decimals.
6. **Gamma vs finite difference:** Compute `(price(S + h) - 2·price(S) + price(S - h)) / h²` for h = 1.0. Should match `gamma(S)`.
7. **Vega vs finite difference:** Same idea, with σ.
8. Same for theta and rho.

### Sanity bounds

9. **Call delta is in [0, 1].** Test for a range of moneyness.
10. **Put delta is in [-1, 0].**
11. **Gamma is non-negative.**
12. **Vega is non-negative.**

---

## Part 6 — Interview questions you should be able to answer

Don't move on to Project 2 until you can answer each of these in 30–60 seconds out loud.

1. What's the delta of an at-the-money call? Why?
2. A trader is "long gamma." What does that mean and what kind of market do they want?
3. Why does theta accelerate (get more negative) as an ATM option approaches expiration?
4. You sell a 1-month call and want to delta-hedge. The stock then makes a big move. Are you happy or sad?
5. What's the delta of a put with strike 90 when the stock is at 200? Why is your answer obvious?
6. A client says "I want to bet that volatility goes up but I have no view on direction." What do you build them?
7. Two options have the same delta: ATM 1-week call and OTM 1-year call. Which has higher gamma? Higher vega?
8. Derive the relationship between call delta and put delta from put-call parity.
9. State the magic identity S·φ(d₁) = K·e^(-rT)·φ(d₂) and sketch why it's true.
10. Your portfolio is delta-neutral and gamma-positive. The market is flat for a month. Are you making or losing money? Why?

If you can do all 10 cleanly, you're ahead of most candidates for risk and junior QR interviews on this topic.

---

## Part 7 — Bonus stretch goal (optional but high signal)

After your Greeks are working, add a simple visualization script `greeks_plot.py` that uses matplotlib to plot:

1. **Delta and Gamma vs Spot** — for an ATM option, with three lines: 1 week to expiry, 1 month, 1 year. You'll see the gamma "knife edge" effect clearly.
2. **Vega vs Spot** — same setup. Notice how vega is highest ATM.
3. **A Greeks surface** — delta as a function of S (x-axis) and T (y-axis), with a 3D surface or heatmap.

These plots are gold for recruiters when they screenshot well in your README. They also force you to confront whether your Greeks behave as theory predicts — which is its own debugging exercise.

---

## What to do now

1. Open `bsm-pricer/` in VS Code with the venv activated.
2. Read Part 1 and Part 2 carefully. Make sure you can explain the magic identity.
3. Take a piece of paper. **Derive delta_call = N(d₁) yourself** without looking at Part 2. Then check.
4. Write `greeks.py` from the structure in Part 4. Start with delta and gamma; they're easiest. Then vega, then theta, then rho.
5. Run your `__main__` and check against the reference table.
6. Write `test_greeks.py` with at least the mathematical identity tests (1–4) and the delta finite difference test (5).
7. Run `pytest` — make sure everything passes (old tests + new tests).
8. Commit and push:

```bash
git add greeks.py test_greeks.py
git commit -m "Day 2: Greeks (delta, gamma, vega, theta, rho) with identity and FD tests"
git push
```

When you're done — or stuck — show me your `greeks.py` and your test output. I'll grill you on the interview questions and we'll plug holes before moving to Day 3 (implied volatility solver).

Do not skip the derivation in step 3. The math is the point.
