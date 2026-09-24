import math
import pytest

from pricer.black_scholes import bs_price


def test_call_known_value():
    price = bs_price(100, 100, 1, 0.05, 0.20, option="call")
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_put_known_value():
    price = bs_price(100, 100, 1, 0.05, 0.20, option="put")
    assert price == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    C = bs_price(S, K, T, r, sigma, option="call")
    P = bs_price(S, K, T, r, sigma, option="put")
    assert C - P == pytest.approx(S - K * math.exp(-r * T), abs=1e-10)


def test_expired_call():
    assert bs_price(110, 100, 0, 0.05, 0.20, option="call") == 10.0
    assert bs_price(90, 100, 0, 0.05, 0.20, option="call") == 0.0


def test_expired_put():
    assert bs_price(90, 100, 0, 0.05, 0.20, option="put") == 10.0
    assert bs_price(110, 100, 0, 0.05, 0.20, option="put") == 0.0


def test_call_bounds():
    # Call must be between intrinsic and S*exp(-qT)
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    price = bs_price(S, K, T, r, sigma, option="call")
    assert price >= max(S - K * math.exp(-r * T), 0.0)
    assert price <= S


def test_invalid_option_type():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.05, 0.20, option="banana")