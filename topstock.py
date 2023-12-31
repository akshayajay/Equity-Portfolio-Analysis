import yfinance as yf
import datetime

# Define the list of company symbols
def top10stocks(n_years):
    stock_symbols = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS',
                     'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
                     'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
                     'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'ITC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS',
                     'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS',
                     'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS',
                     'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']
    
    # User input for the number of years
    start_year = 2015
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(start_year + n_years - 1, 12, 31)
    
    # Restrict the end date to June 22, 2023
    end_date = min(end_date, datetime.date.today())
    
    # Download historical data for the specified companies
    stock_data = {}
    for symbol in stock_symbols:
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if len(data) > 0:  # Check if data is available for the symbol
                stock_data[symbol] = data
            else:
                print(f"No data available for symbol: {symbol}")
        except:
            print(f"Failed to download data for symbol: {symbol}")
    
    # Calculate the percentage returns for each company
    returns_dict = {}
    for symbol, data in stock_data.items():
        company_data = data['Adj Close']
        start_price = company_data.iloc[0]
        end_price = company_data.iloc[-1]
        returns = (end_price - start_price) / start_price * 100
        returns_dict[symbol] = returns
    
     # Sort the companies based on returns
    sorted_returns = sorted(returns_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top 10 performing companies
    top_10_companies = [company[0] for company in sorted_returns[:10]]
    return top_10_companies 
