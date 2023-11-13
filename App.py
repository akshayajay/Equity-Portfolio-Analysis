import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import pandas as pd
from topstock import top10stocks
import altair as alt

# User-adjustable parameters
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date.today()
performance_days = 100
top_stock_count = 10
initial_equity = 1000000
n_years = 1
stock_symbols = top10stocks(n_years)

# Convert np.datetime64 to datetime.date
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Download Nifty index data
nifty_data = yf.download('^NSEI', start=start_date_str, end=end_date_str, progress=False)

# Download stock data for the selected symbols
stock_data = yf.download(stock_symbols, start=start_date_str, end=end_date_str, progress=False)

# Calculate the equity curve for the benchmark strategy
stock_prices = stock_data['Adj Close']
benchmark_symbols = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS',             
                                 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',             
                                 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',             
                                 'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS',             
                                 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS',             
                                 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS',             
                                 'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']

benchmark_data = yf.download(benchmark_symbols, start=start_date_str, end=end_date_str, progress=False)
benchmark_prices = benchmark_data['Adj Close']
benchmark_weights = initial_equity / len(benchmark_symbols)
benchmark_portfolio = benchmark_weights * (benchmark_prices / benchmark_prices.iloc[0]).sum(axis=1)

# Calculate the equity curve for the sample strategy
returns = stock_prices.pct_change(periods=performance_days) + 1
sample_returns = returns.iloc[-1].sort_values(ascending=False)
sample_stocks = sample_returns[:top_stock_count].index.tolist()
sample_weights = initial_equity / top_stock_count
sample_portfolio = sample_weights * (stock_prices[sample_stocks] / stock_prices.iloc[0]).sum(axis=1)

# Calculate the equity curve for the Nifty index
nifty_weights = initial_equity / nifty_data['Adj Close'].iloc[0]
nifty_portfolio = nifty_weights * nifty_data['Adj Close']

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

# Display the plot
plt.show()

# Display the stocks selected for the sample strategy
print("Stocks selected for the sample strategy:")
print(top10stocks(n_years))

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
print("Performance Metrics:")
print(df)

# Create DataFrame for sample portfolio's CAGR
sample_cagr_df = pd.DataFrame({'Company': sample_portfolio.index, 'CAGR': sample_portfolio.values})

# Bar graph for sample portfolio's CAGR
bar_chart_sample_cagr = alt.Chart(sample_cagr_df).mark_bar().encode(
    y='CAGR',
    x=alt.X('Company', sort='-x')
)

# Display the bar chart
print("Sample Portfolio - CAGR")
bar_chart_sample_cagr.show()
