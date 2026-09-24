"""
Cox-Ross-Rubinstein binomial tree.

Handles European and American options. For American options, at each node
we take max(continuation value, intrinsic value).
"""

from math import exp, sqrt

import numpy as np


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    N: int = 500,
    option: str = "call",
    american: bool = False,
) -> float:
    """
    Price an option using a CRR binomial tree.

    Parameters
    ----------
    N : number of time steps (higher = more accurate)
    american : if True, allow early exercise at every node
    """
    option = option.lower()
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")

    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    p = (exp((r - q) * dt) - d) / (u - d)
    disc = exp(-r * dt)

    # Terminal stock prices at step N: S * u^j * d^(N-j)
    j = np.arange(N + 1)
    stock = S * (u ** j) * (d ** (N - j))

    if option == "call":
        values = np.maximum(stock - K, 0.0)
    else:
        values = np.maximum(K - stock, 0.0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        # Stock prices at step i for nodes j = 0..i
        j = np.arange(i + 1)
        stock_i = S * (u ** j) * (d ** (i - j))

        # Continuation value: discounted expected value of children
        values = disc * (p * values[1 : i + 2] + (1.0 - p) * values[0 : i + 1])

        if american:
            if option == "call":
                intrinsic = np.maximum(stock_i - K, 0.0)
            else:
                intrinsic = np.maximum(K - stock_i, 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])