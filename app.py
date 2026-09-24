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

    if st.button("Fetch and compute"):
        with st.spinner("Pulling option chain..."):
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                st.error("Could not fetch spot price. Try a different ticker.")
                st.stop()

            S_live = float(hist["Close"].iloc[-1])
            expirations = ticker.options
            if not expirations:
                st.error("No options found for this ticker.")
                st.stop()

            expiry = st.selectbox("Expiry", expirations)
            chain = ticker.option_chain(expiry)
            calls = chain.calls.copy()
            puts = chain.puts.copy()

            today = datetime.today()
            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
            T_live = (expiry_dt - today).days / 365.0

            def iv_row(row, opt_type):
                K_live = row["strike"]
                bid, ask = row["bid"], row["ask"]
                if bid <= 0 or ask <= 0:
                    return np.nan
                mid = 0.5 * (bid + ask)
                try:
                    return implied_vol(mid, S_live, K_live, T_live, r_market, 0.0, opt_type)
                except Exception:
                    return np.nan

            calls["iv"] = calls.apply(lambda r_: iv_row(r_, "call"), axis=1)
            puts["iv"] = puts.apply(lambda r_: iv_row(r_, "put"), axis=1)

            otm_calls = calls[(calls["strike"] > S_live) & calls["iv"].notna()]
            otm_puts = puts[(puts["strike"] < S_live) & puts["iv"].notna()]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(otm_calls["strike"], otm_calls["iv"] * 100, "o-", label="OTM Calls", markersize=4)
            ax.plot(otm_puts["strike"], otm_puts["iv"] * 100, "o-", label="OTM Puts", markersize=4)
            ax.axvline(S_live, color="black", linestyle="--", label=f"Spot = {S_live:.2f}")
            ax.set_xlabel("Strike")
            ax.set_ylabel("Implied Volatility (%)")
            ax.set_title(f"IV Smile — {ticker_symbol} {expiry} (T = {T_live:.3f}y)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            st.write(f"Spot: **{S_live:.2f}** | Expiry: **{expiry}** | Days: **{(expiry_dt - today).days}**")