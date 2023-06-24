from nsepy import get_history
import datetime
import pandas as pd

# Define the list of company symbols
stock_symbols = ['ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
                 'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
                 'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO',
                 'HINDUNILVR', 'HDFC', 'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK',
                 'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE',
                 'SBIN', 'SUNPHARMA', 'TCS', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN',
                 'UPL', 'ULTRACEMCO', 'WIPRO']

# User input for the number of years
def top10stocks(n_years):
start_date = datetime.date.today() - datetime.timedelta(days=n_years*365)
end_date = datetime.date.today()

# Download historical data for the specified companies
stock_data = {}
for symbol in stock_symbols:
    try:
        data = get_history(symbol=symbol, index=False, start=start_date, end=end_date)
        if not data.empty:  # Check if data is available for the symbol
            stock_data[symbol] = data
        else:
            print(f"No data available for symbol: {symbol}")
    except:
        print(f"Failed to download data for symbol: {symbol}")

# Calculate the percentage returns for each company
  returns_dict = {}
  for symbol, data in stock_data.items():
      company_data = data['Close']
      start_price = company_data.iloc[0]
      end_price = company_data.iloc[-1]
      returns = (end_price - start_price) / start_price * 100
      returns_dict[symbol] = returns
  
  # Sort the companies based on returns
  sorted_returns = sorted(returns_dict.items(), key=lambda x: x[1], reverse=True)
  
  # Get the top 10 performing companies
  top_10_companies = [company[0] for company in sorted_returns[:10]]
  return(top_10_companies)
  
