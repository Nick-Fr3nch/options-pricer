"""
Black-Scholes closed-form pricer for European options.

Formula:
    C = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
    P = K * exp(-rT) * N(-d2) - S * exp(-qT) * N(-d1)

where:
    d1 = (ln(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
"""

from math import log, sqrt, exp, erf, pi


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function N(x)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function n(x)."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float):
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
) -> float:
    """
    Price a European call or put using Black-Scholes.

    Parameters
    ----------
    S : spot price
    K : strike price
    T : time to expiry in years
    r : risk-free rate (continuous)
    sigma : volatility (annualized)
    q : dividend yield (continuous), default 0
    option : "call" or "put"
    """
    option = option.lower()

    # At or past expiry, the option is worth its intrinsic value.
    if T <= 0:
        if option == "call":
            return max(S - K, 0.0)
        elif option == "put":
            return max(K - S, 0.0)
        else:
            raise ValueError("option must be 'call' or 'put'")

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    if option == "call":
        return S * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    elif option == "put":
        return K * exp(-r * T) * norm_cdf(-d2) - S * exp(-q * T) * norm_cdf(-d1)
    else:
        raise ValueError("option must be 'call' or 'put'")