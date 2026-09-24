import pytest

from pricer.black_scholes import bs_price
from pricer.greeks import bs_greeks


S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20


def test_call_delta_known_value():
    g = bs_greeks(S, K, T, r, sigma, option="call")
    assert g["delta"] == pytest.approx(0.6368, abs=1e-3)


def test_put_delta_between_minus_one_and_zero():
    g = bs_greeks(S, K, T, r, sigma, option="put")
    assert -1.0 <= g["delta"] <= 0.0


def test_call_delta_minus_put_delta_equals_discount_factor():
    # With q=0, call_delta - put_delta = 1
    gc = bs_greeks(S, K, T, r, sigma, option="call")
    gp = bs_greeks(S, K, T, r, sigma, option="put")
    assert gc["delta"] - gp["delta"] == pytest.approx(1.0, abs=1e-10)


def test_gamma_matches_finite_difference():
    h = 1e-4
    fd = (
        bs_price(S + h, K, T, r, sigma, option="call")
        - 2 * bs_price(S, K, T, r, sigma, option="call")
        + bs_price(S - h, K, T, r, sigma, option="call")
    ) / (h * h)
    g = bs_greeks(S, K, T, r, sigma, option="call")
    assert g["gamma"] == pytest.approx(fd, rel=1e-4)


def test_vega_matches_finite_difference():
    h = 1e-4
    fd = (
        bs_price(S, K, T, r, sigma + h, option="call")
        - bs_price(S, K, T, r, sigma - h, option="call")
    ) / (2 * h)
    g = bs_greeks(S, K, T, r, sigma, option="call")
    assert g["vega"] == pytest.approx(fd, rel=1e-4)


def test_delta_matches_finite_difference():
    h = 1e-4
    fd = (
        bs_price(S + h, K, T, r, sigma, option="call")
        - bs_price(S - h, K, T, r, sigma, option="call")
    ) / (2 * h)
    g = bs_greeks(S, K, T, r, sigma, option="call")
    assert g["delta"] == pytest.approx(fd, rel=1e-4)


def test_gamma_same_for_call_and_put():
    gc = bs_greeks(S, K, T, r, sigma, option="call")["gamma"]
    gp = bs_greeks(S, K, T, r, sigma, option="put")["gamma"]
    assert gc == pytest.approx(gp, rel=1e-12)


def test_vega_same_for_call_and_put():
    vc = bs_greeks(S, K, T, r, sigma, option="call")["vega"]
    vp = bs_greeks(S, K, T, r, sigma, option="put")["vega"]
    assert vc == pytest.approx(vp, rel=1e-12)


def test_greeks_at_expiry_raise():
    with pytest.raises(ValueError):
        bs_greeks(S, K, 0, r, sigma, option="call")