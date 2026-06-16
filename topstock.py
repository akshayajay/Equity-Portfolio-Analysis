import yfinance as yf
import datetime

NIFTY50_SYMBOLS = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS',
    'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
    'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS',
    'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS',
    'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS',
    'SHRIRAMFIN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS',
]


def top10stocks(n_years: int, top_n: int = 10) -> list:
    """
    Return the top-performing NIFTY50 stocks over the past n_years,
    ranked by total price return.

    Args:
        n_years: Number of historical years to evaluate.
        top_n:   How many top stocks to return (default 10).

    Returns:
        List of ticker symbols (e.g. ['RELIANCE.NS', 'TCS.NS', ...]).
    """
    end_date = datetime.date.today()
    start_date = datetime.date(end_date.year - n_years, end_date.month, end_date.day)

    # Single batch download is far faster than looping per symbol
    data = yf.download(
        NIFTY50_SYMBOLS,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        auto_adjust=True,   # 'Close' column becomes the adjusted close
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "No market data returned. Check your internet connection or date range."
        )

    prices = data['Close']

    returns = {}
    for symbol in prices.columns:
        series = prices[symbol].dropna()
        if len(series) < 2:
            continue
        total_return = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100
        returns[symbol] = float(total_return)

    if not returns:
        raise ValueError("Could not compute returns for any symbol.")

    sorted_stocks = sorted(returns, key=lambda s: returns[s], reverse=True)
    return sorted_stocks[:top_n]
