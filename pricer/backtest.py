"""
Delta-hedging backtest.

Simulates selling an option and hedging it with shares of the underlying
at a fixed rebalancing frequency. Reports final P&L.

Economic interpretation:
  - We SELL the option (collect premium), and hedge by holding delta shares.
  - This makes us short gamma and short vega, long theta.
  - If realized vol > implied vol: we lose (short gamma).
  - If realized vol < implied vol: we win.
  - If realized == implied: P&L should be near zero, minus a small
    discrete-hedging cost that shrinks as rebalance frequency increases.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from pricer.black_scholes import bs_price
from pricer.greeks import bs_greeks


@dataclass
class HedgeResult:
    pnl: float
    pnl_path: List[float] = field(default_factory=list)
    stock_path: List[float] = field(default_factory=list)
    n_rebalances: int = 0


def simulate_gbm_path(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma_realized: float,
    N: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate a GBM path of stock prices with N+1 points (including S0)."""
    rng = np.random.default_rng(seed)
    dt = T / N
    z = rng.standard_normal(N)
    log_returns = (r - q - 0.5 * sigma_realized ** 2) * dt + sigma_realized * np.sqrt(dt) * z
    log_path = np.concatenate([[0.0], np.cumsum(log_returns)])
    return S0 * np.exp(log_path)


def delta_hedge_backtest(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_implied: float,
    sigma_realized: float,
    q: float = 0.0,
    option: str = "call",
    N_steps: int = 252,
    N_rebalances: int = 21,
    seed: Optional[int] = None,
) -> HedgeResult:
    """
    Sell one option at t=0 and delta-hedge it until expiry.

    P&L = (final portfolio value) - (initial portfolio value).
    Initial portfolio value is zero: we collect premium and use it to
    buy the initial hedge; the rest sits in cash.
    """
    dt = T / N_steps
    S_path = simulate_gbm_path(S0, T, r, q, sigma_realized, N_steps, seed)

    # Step 0: sell option, buy initial hedge
    premium = bs_price(S0, K, T, r, sigma_implied, q, option)
    g0 = bs_greeks(S0, K, T, r, sigma_implied, q, option)
    shares = g0["delta"]
    cash = premium - shares * S0

    pnl_path = []
    # We track mark-to-market value of (cash + shares*S - short_option)
    # at each subsequent step.
    rebalance_indices = set(
        np.linspace(0, N_steps - 1, N_rebalances + 1, dtype=int).tolist()
    )

    for i in range(1, N_steps + 1):
        S = S_path[i]
        cash *= np.exp(r * dt)  # accrue interest on cash
        t_remaining = T - i * dt

        # Option value now (using IMPLIED vol for marking)
        if t_remaining > 0:
            option_value = bs_price(S, K, t_remaining, r, sigma_implied, q, option)
        else:
            option_value = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)

        # Mark-to-market value of our position (we are SHORT the option)
        m2m = cash + shares * S - option_value
        pnl_path.append(m2m)

        # Rebalance after marking (skip final step)
        if i in rebalance_indices and i < N_steps and t_remaining > 0:
            g = bs_greeks(S, K, t_remaining, r, sigma_implied, q, option)
            target = g["delta"]
            trade = target - shares
            cash -= trade * S
            shares = target

    final_pnl = pnl_path[-1]

    return HedgeResult(
        pnl=final_pnl,
        pnl_path=pnl_path,
        stock_path=S_path.tolist(),
        n_rebalances=len(rebalance_indices),
    )


def run_many_paths(
    n_paths: int = 500,
    **kwargs,
) -> dict:
    """Run the backtest over many simulated paths, return stats."""
    pnls = []
    for seed in range(n_paths):
        res = delta_hedge_backtest(seed=seed, **kwargs)
        pnls.append(res.pnl)
    pnls = np.array(pnls)
    return {
        "mean": float(pnls.mean()),
        "std": float(pnls.std()),
        "sharpe": float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0,
        "min": float(pnls.min()),
        "max": float(pnls.max()),
        "n_paths": n_paths,
    }