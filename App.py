"""
NIFTY50 Equity Portfolio Analyser
----------------------------------
Compares three strategies over a user-selected date range:
  1. Equal-weight benchmark across all NIFTY50 constituents
  2. Momentum strategy — top-N stocks by n-year historical return
  3. NIFTY50 index (^NSEI) as a passive reference

Run with:  streamlit run App.py
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import pandas as pd
import altair as alt
import streamlit as st

from topstock import top10stocks, NIFTY50_SYMBOLS
from utils import cagr, annualised_volatility, sharpe_ratio, max_drawdown

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NIFTY50 Portfolio Analyser", layout="wide")
st.title("📈 NIFTY50 Equity Portfolio Analyser")
st.markdown(
    "Compare a **momentum-based** stock selection strategy against the "
    "equal-weight NIFTY50 benchmark and the index itself."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Parameters")

start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

performance_days = st.sidebar.slider(
    "Momentum Lookback (trading days)", min_value=20, max_value=252, value=100
)
top_stock_count = st.sidebar.slider(
    "Number of Top Stocks", min_value=3, max_value=20, value=10
)
initial_equity = st.sidebar.number_input(
    "Initial Capital (₹)", value=1_000_000, step=100_000, min_value=10_000
)
n_years = st.sidebar.slider(
    "Years to rank top stocks over", min_value=1, max_value=7, value=1
)

start_str = start_date.strftime('%Y-%m-%d')
end_str   = end_date.strftime('%Y-%m-%d')

# ---------------------------------------------------------------------------
# Data loading  (cached so re-runs don't re-download)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_stock_prices(start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        NIFTY50_SYMBOLS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    return data['Close']


@st.cache_data(show_spinner=False)
def load_nifty(start: str, end: str) -> pd.Series:
    data = yf.download('^NSEI', start=start, end=end, auto_adjust=True, progress=False)
    return data['Close'].squeeze()


@st.cache_data(show_spinner=False)
def get_top_stocks(n_years: int, top_n: int) -> list:
    return top10stocks(n_years, top_n)


with st.spinner("Fetching market data from Yahoo Finance…"):
    try:
        stock_prices = load_stock_prices(start_str, end_str)
        nifty_prices = load_nifty(start_str, end_str)
        sample_stocks = get_top_stocks(n_years, top_stock_count)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

if stock_prices.empty or nifty_prices.empty:
    st.error("No data returned for the selected date range. Try widening the range.")
    st.stop()

# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

# 1. Equal-weight benchmark — all 50 NIFTY stocks
benchmark_prices = stock_prices.dropna(axis=1, how='all')
benchmark_weights = initial_equity / len(benchmark_prices.columns)
normalised_benchmark = benchmark_prices / benchmark_prices.iloc[0]
benchmark_portfolio = benchmark_weights * normalised_benchmark.sum(axis=1)

# 2. Momentum strategy — top-N stocks selected above
available_samples = [s for s in sample_stocks if s in stock_prices.columns]
if not available_samples:
    st.error("None of the selected momentum stocks have data in this date range.")
    st.stop()

sample_prices = stock_prices[available_samples].dropna(how='all')
sample_weights = initial_equity / len(available_samples)
normalised_sample = sample_prices / sample_prices.iloc[0]
sample_portfolio = sample_weights * normalised_sample.sum(axis=1)

# 3. NIFTY index
nifty_portfolio = (initial_equity / nifty_prices.iloc[0]) * nifty_prices

# ---------------------------------------------------------------------------
# Equity curve chart
# ---------------------------------------------------------------------------
st.subheader("Portfolio Equity Curve")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(benchmark_portfolio.index, benchmark_portfolio,
        label='NIFTY50 Equal-Weight Benchmark', linewidth=1.8)
ax.plot(sample_portfolio.index, sample_portfolio,
        label=f'Top-{len(available_samples)} Momentum Strategy', linewidth=1.8)
ax.plot(nifty_portfolio.index, nifty_portfolio,
        label='NIFTY Index (^NSEI)', linewidth=1.5, linestyle='--', alpha=0.8)
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value (₹)')
ax.set_title('Equity Curve Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# Performance metrics table
# ---------------------------------------------------------------------------
st.subheader("Performance Metrics")

metrics_df = pd.DataFrame({
    'Strategy': [
        'NIFTY50 Equal-Weight',
        f'Top-{len(available_samples)} Momentum',
        'NIFTY Index (^NSEI)',
    ],
    'CAGR': [
        f"{cagr(benchmark_portfolio)*100:.2f}%",
        f"{cagr(sample_portfolio)*100:.2f}%",
        f"{cagr(nifty_portfolio)*100:.2f}%",
    ],
    'Ann. Volatility': [
        f"{annualised_volatility(benchmark_portfolio)*100:.2f}%",
        f"{annualised_volatility(sample_portfolio)*100:.2f}%",
        f"{annualised_volatility(nifty_portfolio)*100:.2f}%",
    ],
    'Sharpe Ratio': [
        f"{sharpe_ratio(benchmark_portfolio):.2f}",
        f"{sharpe_ratio(sample_portfolio):.2f}",
        f"{sharpe_ratio(nifty_portfolio):.2f}",
    ],
    'Max Drawdown': [
        f"{max_drawdown(benchmark_portfolio)*100:.2f}%",
        f"{max_drawdown(sample_portfolio)*100:.2f}%",
        f"{max_drawdown(nifty_portfolio)*100:.2f}%",
    ],
})

st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Selected stocks + individual return bar chart
# ---------------------------------------------------------------------------
st.subheader(f"Top-{len(available_samples)} Momentum Holdings")
st.write(", ".join(available_samples))

stock_returns = {}
for sym in available_samples:
    series = stock_prices[sym].dropna()
    if len(series) >= 2:
        ret = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100
        stock_returns[sym] = round(float(ret), 2)

returns_df = (
    pd.DataFrame(list(stock_returns.items()), columns=['Stock', 'Total Return (%)'])
    .sort_values('Total Return (%)', ascending=False)
    .reset_index(drop=True)
)

bar_chart = (
    alt.Chart(returns_df)
    .mark_bar()
    .encode(
        x=alt.X('Stock:N', sort='-y', title='Stock'),
        y=alt.Y('Total Return (%):Q', title='Total Return (%)'),
        color=alt.condition(
            alt.datum['Total Return (%)'] > 0,
            alt.value('#2ecc71'),
            alt.value('#e74c3c'),
        ),
        tooltip=['Stock', 'Total Return (%)'],
    )
    .properties(
        title=f'Individual Stock Returns — Top {len(available_samples)} Holdings',
        height=350,
    )
)

st.altair_chart(bar_chart, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Data sourced from Yahoo Finance via yfinance. "
    "Past performance is not indicative of future results. "
    "This tool is for educational purposes only."
)
