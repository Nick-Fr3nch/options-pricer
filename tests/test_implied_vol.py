import pytest

from pricer.black_scholes import bs_price
from pricer.implied_vol import implied_vol


def test_round_trip_call():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    price = bs_price(S, K, T, r, sigma, option="call")
    iv = implied_vol(price, S, K, T, r, option="call")
    assert iv == pytest.approx(sigma, abs=1e-6)


def test_round_trip_put():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    price = bs_price(S, K, T, r, sigma, option="put")
    iv = implied_vol(price, S, K, T, r, option="put")
    assert iv == pytest.approx(sigma, abs=1e-6)


def test_round_trip_deep_otm():
    S, K, T, r, sigma = 100, 150, 1, 0.05, 0.35
    price = bs_price(S, K, T, r, sigma, option="call")
    iv = implied_vol(price, S, K, T, r, option="call")
    assert iv == pytest.approx(sigma, abs=1e-5)


def test_round_trip_deep_itm():
    S, K, T, r, sigma = 150, 100, 1, 0.05, 0.30
    price = bs_price(S, K, T, r, sigma, option="call")
    iv = implied_vol(price, S, K, T, r, option="call")
    assert iv == pytest.approx(sigma, abs=1e-5)


def test_price_below_intrinsic_raises():
    with pytest.raises(ValueError):
        implied_vol(0.0, 100, 90, 1, 0.05, option="call")


def test_price_above_stock_raises():
    with pytest.raises(ValueError):
        implied_vol(200.0, 100, 100, 1, 0.05, option="call")


def test_at_expiry_raises():
    with pytest.raises(ValueError):
        implied_vol(5.0, 100, 100, 0, 0.05, option="call")