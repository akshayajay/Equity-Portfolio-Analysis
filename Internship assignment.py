import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Set the parameters
start_date = '2020-10-01'
end_date = '2023-06-13'
performance_days = 100
top_stock_count = 10
initial_equity = 1000000

# Download Nifty index data
nifty_data = yf.download('^NSEI', start=start_date, end=end_date, progress=False)

# Download stock data for the selected symbols
stock_symbols = ['RELIANCE.NS', 'HCLTECH.NS', 'TATAMOTORS.NS', 'M&M.NS', 'EICHERMOT.NS', 'JSWSTEEL.NS', 'BAJFINANCE.NS',
                 'APOLLOHOSP.NS', 'WIPRO.NS', 'ADANIENT.NS']
stock_data = yf.download(stock_symbols, start=start_date, end=end_date, progress=False)

benchmark_symbols = ['HDFCLIFE.NS', 'TATAMOTORS.NS', 'HCLTECH.NS', 'POWERGRID.NS', 'EICHERMOT.NS', 'SBILIFE.NS',
                   'BAJAJ-AUTO.NS', 'TECHM.NS', 'KOTAKBANK.NS', 'HINDALCO.NS', 'DRREDDY.NS', 'WIPRO.NS', 'INFY.NS',
                   'AXISBANK.NS', 'NTPC.NS', 'BRITANNIA.NS', 'TCS.NS', 'LT.NS', 'ADANIENT.NS', 'NESTLEIND.NS',
                   'TATASTEEL.NS', 'GRASIM.NS', 'HEROMOTOCO.NS', 'BHARTIARTL.NS', 'TATACONSUM.NS', 'HDFC.NS',
                   'APOLLOHOSP.NS', 'JSWSTEEL.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'COALINDIA.NS', 'ADANIPORTS.NS',
                   'CIPLA.NS', 'TITAN.NS', 'ITC.NS', 'INDUSINDBK.NS', 'RELIANCE.NS', 'ONGC.NS', 'HINDUNILVR.NS',
                   'UPL.NS', 'SBIN.NS', 'ASIANPAINT.NS', 'ULTRACEMCO.NS', 'MARUTI.NS', 'BPCL.NS', 'DIVISLAB.NS',
                   'SUNPHARMA.NS', 'M&M.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS']
benchmark_data = yf.download(benchmark_symbols, start=start_date, end=end_date, progress=False)

# Calculate the equity curve for the benchmark strategy
stock_prices = stock_data['Adj Close']
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
plt.show()

# Display the stocks selected for the sample strategy
print("Stocks selected for the sample strategy:")
print(sample_stocks)

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

# Display performance metrics
print("\nPerformance Metrics:")
print("Benchmark CAGR: {:.2%}".format(benchmark_cagr))
print("Sample Strategy CAGR: {:.2%}".format(sample_cagr))
print("Nifty Index CAGR: {:.2%}".format(nifty_cagr))
print()
print("Benchmark Volatility: {:.2%}".format(benchmark_volatility))
print("Sample Strategy Volatility: {:.2%}".format(sample_volatility))
print("Nifty Index Volatility: {:.2%}".format(nifty_volatility))
print()
print("Benchmark Sharpe Ratio: {:.2f}".format(benchmark_sharpe_ratio))
print("Sample Strategy Sharpe Ratio: {:.2f}".format(sample_sharpe_ratio))
print("Nifty Index Sharpe Ratio: {:.2f}".format(nifty_sharpe_ratio))
