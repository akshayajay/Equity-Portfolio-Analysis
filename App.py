import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials
import datetime
import pandas as pd
from topstock import top10stocks
import altair as alt

# Streamlit app title and description
st.title("Equity Portfolio Analysis")
st.write("This app calculates and visualizes the equity curves of different portfolio strategies.")

# User-adjustable parameters
start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1), min_value=datetime.date(2020, 1, 1), max_value=datetime.date.today())
end_date = st.date_input("End Date", value=datetime.date.today(), min_value=datetime.date(2021, 1, 1), max_value=datetime.date.today())
performance_days = st.slider("Performance Days", min_value=1, max_value=500, value=100)
top_stock_count = st.slider("Top Stock Count", min_value=1, max_value=10, value=10)
initial_equity = st.number_input("Initial Equity", min_value=1000, max_value=10000000, value=1000000, step=1000)
n_years = st.number_input("Number of years from 2015 for which historical data will be analysed to pick the top ten stocks", min_value=1, max_value=8, value=1, step=1)
stock_symbols = top10stocks(n_years)

# Convert np.datetime64 to datetime.date
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')

# Download Nifty index data
yahoo_financials_nifty = YahooFinancials('^NSEI')
nifty_data = yahoo_financials_nifty.get_historical_price_data(start_date, end_date, 'daily')
nifty_prices = pd.DataFrame({symbol: data['adjclose'] for symbol, data in nifty_data.items()})

# Download stock data for the selected symbols
yahoo_financials_stocks = YahooFinancials(stock_symbols)
stock_data = yahoo_financials_stocks.get_historical_price_data(start_date, end_date, 'daily')
stock_prices = pd.DataFrame({symbol: data['adjclose'] for symbol, data in stock_data.items()})

# Calculate the equity curve for the benchmark strategy
benchmark_symbols = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS',             
                                 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',             
                                 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',             
                                 'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS',             
                                 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS',             
                                 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS',             
                                 'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']

yahoo_financials_benchmark = YahooFinancials(benchmark_symbols)
benchmark_data = yahoo_financials_benchmark.get_historical_price_data(start_date, end_date, 'daily')
benchmark_prices = pd.DataFrame({symbol: data['adjclose'] for symbol, data in benchmark_data.items()})

# Align the stock prices and benchmark prices DataFrame
stock_prices = stock_prices.loc[:, benchmark_prices.columns]

# Calculate the equity curve for the sample strategy
returns = stock_prices.pct_change(periods=performance_days) + 1
sample_returns = returns.iloc[-1].sort_values(ascending=False)
sample_stocks = sample_returns[:top_stock_count].index.tolist()
sample_weights = initial_equity / top_stock_count
sample_portfolio = sample_weights * (stock_prices[sample_stocks] / stock_prices.iloc[0]).sum(axis=1)

# Calculate the equity curve for the Nifty index
nifty_weights = initial_equity / nifty_prices.iloc[0]
nifty_portfolio = nifty_weights * nifty_prices

# Plot the equity curves
plt.figure(figsize=(10, 6))
plt.plot(benchmark_portfolio.index, benchmark_portfolio, label='Benchmark')
plt.plot(sample_portfolio.index, sample_portfolio, label='Sample Strategy')
plt.plot(nifty_portfolio.index, nifty_portfolio, label='Nifty Index')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.title('Portfolio Equity Curve')
plt.legend()
plt.grid(True)

# Display the plots
st.pyplot(plt)

# Display the stocks selected for the sample strategy
st.write("Stocks selected for the sample strategy:")
st.write(top10stocks(n_years))

# Calculate performance metrics
benchmark_cagr = (benchmark_portfolio[-1] / benchmark_portfolio[0]) ** (252 / len(benchmark_portfolio)) - 1
sample_cagr = (sample_portfolio[-1] / sample_portfolio[0]) ** (252 / len(sample_portfolio)) - 1
nifty_cagr = (nifty_portfolio[-1] / nifty_portfolio[0]) ** (252 / len(nifty_portfolio)) - 1


benchmark_volatility = np.std(benchmark_portfolio.pct_change()) * np.sqrt(252)
sample_volatility = np.std(sample_portfolio.pct_change()) * np.sqrt(252)
nifty_volatility = np.std(nifty_portfolio.pct_change()) * np.sqrt(252)

benchmark_sharpe_ratio = benchmark_cagr / benchmark_volatility
sample_sharpe_ratio = sample_cagr / sample_volatility
nifty_sharpe_ratio = nifty_cagr / nifty_volatility

# Create a DataFrame for the performance metrics
data = {
    'Metric': ['Benchmark CAGR', 'Sample Strategy CAGR', 'Nifty Index CAGR',
               'Benchmark Volatility', 'Sample Strategy Volatility', 'Nifty Index Volatility',
               'Benchmark Sharpe Ratio', 'Sample Strategy Sharpe Ratio', 'Nifty Index Sharpe Ratio'],
    'Value': [benchmark_cagr, sample_cagr, nifty_cagr,
              benchmark_volatility, sample_volatility, nifty_volatility,
              benchmark_sharpe_ratio, sample_sharpe_ratio, nifty_sharpe_ratio]
}
df = pd.DataFrame(data)

# Display the table
st.write("Performance Metrics:")
st.table(df)

# Create DataFrame for sample portfolio's CAGR
sample_cagr_df = pd.DataFrame({'Company': sample_portfolio.index, 'CAGR': sample_portfolio.values})

# Bar graph for sample portfolio's CAGR
bar_chart_sample_cagr = alt.Chart(sample_cagr_df).mark_bar().encode(
    y='CAGR',
    x=alt.X('Company', sort='-x')
)

st.write("Sample Portfolio - CAGR")
st.altair_chart(bar_chart_sample_cagr, use_container_width=True)
