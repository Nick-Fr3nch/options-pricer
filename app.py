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

    if ticker_symbol:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1d")

        if hist.empty:
            st.error(f"Could not fetch spot price for {ticker_symbol}.")
        else:
            S_live = float(hist["Close"].iloc[-1])
            today = datetime.today()

            all_expirations = ticker.options
            future_expirations = [
                e for e in all_expirations
                if (datetime.strptime(e, "%Y-%m-%d") - today).days >= 1
            ]

            if not future_expirations:
                st.error("No future expirations available for this ticker.")
            else:
                expiry = st.selectbox("Expiry", future_expirations)

                if st.button("Fetch and compute"):
                    with st.spinner("Pulling option chain..."):
                        chain = ticker.option_chain(expiry)
                        calls = chain.calls.copy()
                        puts = chain.puts.copy()

                        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
                        T_live = (expiry_dt - today).days / 365.0

                        def get_mid(row):
                            """Prefer mid of bid/ask; fall back to lastPrice."""
                            bid, ask = row["bid"], row["ask"]
                            last = row.get("lastPrice", 0.0)
                            if bid > 0 and ask > 0:
                                return 0.5 * (bid + ask), "bid/ask"
                            if last > 0:
                                return float(last), "last"
                            return np.nan, "none"

                        def iv_row(row, opt_type):
                            K_live = row["strike"]
                            price, source = get_mid(row)
                            if np.isnan(price) or price <= 0:
                                return np.nan
                            try:
                                return implied_vol(price, S_live, K_live, T_live, r_market, 0.0, opt_type)
                            except Exception:
                                return np.nan

                        calls["iv"] = calls.apply(lambda r_: iv_row(r_, "call"), axis=1)
                        puts["iv"] = puts.apply(lambda r_: iv_row(r_, "put"), axis=1)

                        # Diagnostics
                        with st.expander("Data diagnostics"):
                            st.write(f"Spot: {S_live:.2f}")
                            st.write(f"Expiry: {expiry}")
                            st.write(f"T (years): {T_live:.4f} ({(expiry_dt - today).days} days)")
                            st.write(f"Total calls: {len(calls)}, total puts: {len(puts)}")
                            st.write(f"Calls with bid>0 & ask>0: {((calls['bid'] > 0) & (calls['ask'] > 0)).sum()}")
                            st.write(f"Calls with lastPrice>0: {(calls['lastPrice'] > 0).sum()}")
                            st.write(f"Calls with IV computed: {calls['iv'].notna().sum()}")
                            st.write(f"Puts with IV computed: {puts['iv'].notna().sum()}")
                            st.write("Sample rows:")
                            st.dataframe(calls[["strike", "bid", "ask", "lastPrice", "iv"]].head(10))

                        otm_calls = calls[(calls["strike"] > S_live) & calls["iv"].notna()]
                        otm_puts = puts[(puts["strike"] < S_live) & puts["iv"].notna()]

                        if otm_calls.empty and otm_puts.empty:
                            st.warning("No valid IVs computed. Check the diagnostics panel above.")
                        else:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.plot(otm_calls["strike"], otm_calls["iv"] * 100, "o-",
                                    label="OTM Calls", markersize=4)
                            ax.plot(otm_puts["strike"], otm_puts["iv"] * 100, "o-",
                                    label="OTM Puts", markersize=4)
                            ax.axvline(S_live, color="black", linestyle="--",
                                       label=f"Spot = {S_live:.2f}")
                            ax.set_xlabel("Strike")
                            ax.set_ylabel("Implied Volatility (%)")
                            ax.set_title(f"IV Smile — {ticker_symbol} {expiry} (T = {T_live:.3f}y)")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)

                            st.write(f"Spot: **{S_live:.2f}** | Expiry: **{expiry}** | Days: **{(expiry_dt - today).days}**")