"""
Streamlit web app for the options pricer.

Run with:
    streamlit run app.py
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yfinance as yf
import time

from pricer.black_scholes import bs_price
from pricer.binomial import crr_price
from pricer.greeks import bs_greeks
from pricer.implied_vol import implied_vol


st.set_page_config(page_title="Options Pricer", layout="wide")
st.title("Options Pricer")
st.caption("Black-Scholes · Greeks · Implied Vol · Binomial Tree · Live IV Smile")

tab1, tab2, tab3 = st.tabs(["Pricer", "Greeks", "Live IV Smile"])

# ---------- Tab 1: Pricer ----------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Inputs")
        S = st.number_input("Spot (S)", value=100.0, step=1.0)
        K = st.number_input("Strike (K)", value=100.0, step=1.0)
        T = st.number_input("Time to expiry (years)", value=1.0, step=0.01, min_value=0.01)
        r = st.number_input("Risk-free rate", value=0.05, step=0.005, format="%.4f")
        q = st.number_input("Dividend yield", value=0.0, step=0.005, format="%.4f")
        sigma = st.number_input("Volatility", value=0.20, step=0.01, min_value=0.001)
        option = st.selectbox("Option type", ["call", "put"])

    with col2:
        st.subheader("Results")
        bs = bs_price(S, K, T, r, sigma, q, option)
        st.metric("Black-Scholes price", f"{bs:.4f}")

        with st.expander("Binomial tree (CRR)"):
            N = st.slider("Number of steps", 50, 2000, 500, step=50)
            american = st.checkbox("American exercise", value=False)
            tree = crr_price(S, K, T, r, sigma, q, N=N, option=option, american=american)
            st.metric(f"CRR price (N={N})", f"{tree:.4f}")
            st.caption(f"Difference vs BS: {tree - bs:+.4f}")

# ---------- Tab 2: Greeks ----------
with tab2:
    st.subheader("Greeks")
    g = bs_greeks(S, K, T, r, sigma, q, option)
    cols = st.columns(5)
    cols[0].metric("Delta", f"{g['delta']:+.4f}")
    cols[1].metric("Gamma", f"{g['gamma']:.4f}")
    cols[2].metric("Vega (per 1.00)", f"{g['vega']:.4f}")
    cols[3].metric("Theta (per year)", f"{g['theta']:.4f}")
    cols[4].metric("Rho (per 1.00)", f"{g['rho']:+.4f}")

    st.caption("Vega / Rho per 1% change: divide by 100. Theta per day: divide by 365.")

# ---------- Tab 3: Live IV Smile ----------
with tab3:
    st.subheader("Live Implied Volatility Smile")

    ticker_symbol = st.text_input("Ticker", value="AAPL")
    r_market = st.number_input("Risk-free rate for IV calc", value=0.05, step=0.005, format="%.4f")

    if ticker_symbol:
        ticker = yf.Ticker(ticker_symbol)

        S_live = None
        all_expirations = ()
        last_error = None

        for attempt in range(3):
            try:
                hist = ticker.history(period="1d")
                if hist.empty:
                    last_error = "No price data"
                    time.sleep(1.0)
                    continue
                S_live = float(hist["Close"].iloc[-1])

                opts = ticker.options
                if not opts:
                    last_error = "No options data"
                    time.sleep(1.5)
                    continue

                all_expirations = opts
                break
            except Exception as e:
                last_error = str(e)
                time.sleep(1.5)

        if S_live is None or not all_expirations:
            st.error(
                f"Could not fetch market data for {ticker_symbol}. "
                f"Yahoo Finance may be rate-limiting the cloud server. "
                f"Try again in a moment, or try a different ticker like MSFT or SPY. "
                f"(Last error: {last_error})"
            )
        else:
            today = datetime.today()
            future_expirations = [
                e for e in all_expirations
                if (datetime.strptime(e, "%Y-%m-%d") - today).days >= 1
            ]

            if not future_expirations:
                st.error("No future expirations available for this ticker.")
            else:
                expiry = st.selectbox("Expiry", future_expirations)

                if st.button("Fetch and compute"):
                    # ... rest of your existing code inside the button block ...
                    pass