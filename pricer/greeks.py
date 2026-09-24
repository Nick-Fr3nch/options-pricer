"""
The Greeks: sensitivities of the Black-Scholes price to each input.

Delta  = dV/dS
Gamma  = d2V/dS2
Vega   = dV/dsigma
Theta  = dV/dt      (usually reported as -dV/dT)
Rho    = dV/dr
"""

from math import exp, sqrt

from pricer.black_scholes import norm_cdf, norm_pdf, _d1_d2


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
) -> dict:
    """Return a dict of Delta, Gamma, Vega, Theta, Rho for a European option."""
    option = option.lower()
    if T <= 0:
        raise ValueError("Greeks are undefined at expiry (T <= 0).")

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm_pdf(d1)
    disc_q = exp(-q * T)
    disc_r = exp(-r * T)

    if option == "call":
        delta = disc_q * norm_cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt(T))
            - r * K * disc_r * norm_cdf(d2)
            + q * S * disc_q * norm_cdf(d1)
        )
        rho = K * T * disc_r * norm_cdf(d2)
    elif option == "put":
        delta = disc_q * (norm_cdf(d1) - 1.0)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2.0 * sqrt(T))
            + r * K * disc_r * norm_cdf(-d2)
            - q * S * disc_q * norm_cdf(-d1)
        )
        rho = -K * T * disc_r * norm_cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt(T))
    vega = S * disc_q * pdf_d1 * sqrt(T)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }