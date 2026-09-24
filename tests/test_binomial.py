import pytest

from pricer.binomial import crr_price
from pricer.black_scholes import bs_price


S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20


def test_european_call_converges_to_bs():
    tree = crr_price(S, K, T, r, sigma, N=1000, option="call")
    bs = bs_price(S, K, T, r, sigma, option="call")
    assert tree == pytest.approx(bs, abs=5e-3)


def test_european_put_converges_to_bs():
    tree = crr_price(S, K, T, r, sigma, N=1000, option="put")
    bs = bs_price(S, K, T, r, sigma, option="put")
    assert tree == pytest.approx(bs, abs=5e-3)


def test_convergence_improves_with_n():
    # Error at N=1000 should be smaller than at N=100
    bs = bs_price(S, K, T, r, sigma, option="call")
    err_100 = abs(crr_price(S, K, T, r, sigma, N=100, option="call") - bs)
    err_1000 = abs(crr_price(S, K, T, r, sigma, N=1000, option="call") - bs)
    assert err_1000 < err_100


def test_american_put_worth_more_than_european():
    euro = crr_price(S, K, T, r, sigma, N=500, option="put", american=False)
    amer = crr_price(S, K, T, r, sigma, N=500, option="put", american=True)
    assert amer > euro


def test_american_call_equals_european_no_dividend():
    euro = crr_price(S, K, T, r, sigma, N=500, option="call", american=False)
    amer = crr_price(S, K, T, r, sigma, N=500, option="call", american=True)
    assert amer == pytest.approx(euro, abs=1e-3)


def test_invalid_option_type():
    with pytest.raises(ValueError):
        crr_price(S, K, T, r, sigma, N=100, option="banana")