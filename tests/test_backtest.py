import pytest

from pricer.backtest import delta_hedge_backtest, run_many_paths


def test_hedged_pnl_near_zero_when_vols_match():
    result = run_many_paths(
        300, S0=100, K=100, T=1, r=0.05,
        sigma_implied=0.20, sigma_realized=0.20,
        N_steps=252, N_rebalances=21, option="call",
    )
    assert abs(result["mean"]) < 1.0


def test_short_gamma_loses_when_realized_above_implied():
    result = run_many_paths(
        300, S0=100, K=100, T=1, r=0.05,
        sigma_implied=0.20, sigma_realized=0.35,
        N_steps=252, N_rebalances=21, option="call",
    )
    assert result["mean"] < 0.0


def test_short_gamma_wins_when_realized_below_implied():
    result = run_many_paths(
        300, S0=100, K=100, T=1, r=0.05,
        sigma_implied=0.20, sigma_realized=0.10,
        N_steps=252, N_rebalances=21, option="call",
    )
    assert result["mean"] > 0.0


def test_backtest_returns_path():
    res = delta_hedge_backtest(
        S0=100, K=100, T=1, r=0.05,
        sigma_implied=0.20, sigma_realized=0.20,
        N_steps=252, N_rebalances=21, seed=42,
    )
    assert len(res.pnl_path) == 252
    assert len(res.stock_path) == 253
    assert res.n_rebalances > 0