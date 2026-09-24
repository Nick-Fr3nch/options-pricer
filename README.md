## Web App

Run the interactive Streamlit UI:

    streamlit run app.py

Opens a browser at http://localhost:8501 with three tabs:

- **Pricer** — Black-Scholes and CRR prices for any input
- **Greeks** — Delta, Gamma, Vega, Theta, Rho
- **Live IV Smile** — pull any ticker's option chain and plot the smile

# Options Pricer

A from-scratch options pricing library in Python:
Black-Scholes, Greeks, implied volatility (Newton-Raphson + bisection),
and a Cox-Ross-Rubinstein binomial tree with American early exercise.
Includes a command-line interface, a full test suite, and a live-market IV smile notebook.

![Implied Volatility Smile](notebooks/iv_smile.png)

## Features

- Black-Scholes European call/put pricing with continuous dividend yield
- Greeks: Delta, Gamma, Vega, Theta, Rho - validated against finite differences
- Implied volatility: Newton-Raphson with bisection fallback
- Binomial tree (CRR): European and American options
- CLI for pricing, Greeks, IV, and tree pricing
- 29 unit tests (known values, put-call parity, finite differences, edge cases)
- IV smile notebook using live option chains via yfinance

## Quick Start

    git clone https://github.com/Nick-Fr3nch/options-pricer.git
    cd options-pricer
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Usage

Price a European call:

    python cli.py price --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --type call

Show the Greeks:

    python cli.py greeks --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --type call

Implied volatility round-trip:

    python cli.py iv --price 10.450584 --S 100 --K 100 --T 1 --r 0.05 --type call

American put via binomial tree:

    python cli.py tree --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --type put --american --N 1000

## Model Overview

Black-Scholes:

    C = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
    d1 = (ln(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

Greeks: Delta, Gamma, Vega, Theta, Rho. Analytic values validated against central finite differences.

Implied volatility: Newton-Raphson using Vega as the derivative, bisection fallback on [1e-6, 5.0].

Binomial tree (CRR): u = exp(sigma*sqrt(dt)), d = 1/u, p = (exp((r-q)*dt) - d) / (u - d). Backward induction with max(continuation, intrinsic) for American exercise.

## Project Layout

    options-pricer/
      pricer/
        black_scholes.py
        greeks.py
        implied_vol.py
        binomial.py
      tests/
      notebooks/
        iv_smile.ipynb
      cli.py
      requirements.txt
      README.md

## Tests

    pytest -v

29 tests, all passing. Covers known analytic values, put-call parity, finite-difference Greeks, IV round-trips, no-arbitrage bounds, CRR convergence, and American early exercise.

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
- Cox, J., Ross, S., & Rubinstein, M. (1979). Option Pricing: A Simplified Approach.
- Hull, J. Options, Futures, and Other Derivatives.

## License

MIT
