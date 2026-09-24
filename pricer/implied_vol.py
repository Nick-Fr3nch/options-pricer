"""
Implied volatility solver.

Given a market price, find sigma such that BS(sigma) = market_price.

Primary method: Newton-Raphson, using Vega as the derivative.
Fallback: bisection on [1e-6, 5.0] if Newton fails.
"""

from math import exp, sqrt

from pricer.black_scholes import bs_price
from pricer.greeks import bs_greeks


def _arb_bounds(S, K, T, r, q, option):
    """Lower and upper no-arbitrage bounds for a European option."""
    if option == "call":
        lower = max(S * exp(-q * T) - K * exp(-r * T), 0.0)
        upper = S * exp(-q * T)
    else:
        lower = max(K * exp(-r * T) - S * exp(-q * T), 0.0)
        upper = K * exp(-r * T)
    return lower, upper


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option: str = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Return implied volatility for a European option.

    Raises ValueError if the price violates no-arbitrage bounds.
    """
    option = option.lower()

    if T <= 0:
        raise ValueError("Cannot compute implied vol at expiry.")

    lower, upper = _arb_bounds(S, K, T, r, q, option)
    if market_price < lower - 1e-8 or market_price > upper + 1e-8:
        raise ValueError(
            f"Price {market_price:.6f} outside bounds [{lower:.6f}, {upper:.6f}]"
        )

    # Initial guess: Brenner-Subrahmanyam approximation, C ~ 0.4 * S * sigma * sqrt(T)
    sigma = 0.2
    if market_price > 0:
        sigma = market_price / (0.4 * S * sqrt(T))
    sigma = max(1e-4, min(sigma, 5.0))

    # --- Newton-Raphson ---
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, q, option)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        vega = bs_greeks(S, K, T, r, sigma, q, option)["vega"]

        if vega < 1e-12:
            break  # fall through to bisection

        sigma_new = sigma - diff / vega

        if sigma_new <= 0.0 or sigma_new > 5.0:
            break  # fall through to bisection

        sigma = sigma_new

    # --- Bisection fallback ---
    low, high = 1e-6, 5.0
    f_low = bs_price(S, K, T, r, low, q, option) - market_price
    f_high = bs_price(S, K, T, r, high, q, option) - market_price

    if f_low * f_high > 0:
        raise ValueError("No implied vol in bisection bracket [1e-6, 5.0].")

    for _ in range(200):
        mid = 0.5 * (low + high)
        f_mid = bs_price(S, K, T, r, mid, q, option) - market_price

        if abs(f_mid) < tol or (high - low) < tol:
            return mid

        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)