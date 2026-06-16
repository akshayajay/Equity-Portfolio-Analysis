"""
Portfolio performance metrics used by the Streamlit app.

Extracted into a separate module so they can be unit-tested independently
of the Streamlit runtime.
"""

import numpy as np
import pandas as pd


def cagr(series: pd.Series) -> float:
    """
    Compound Annual Growth Rate, annualised to 252 trading days.

    Returns 0.0 if the series is too short or starts at zero.
    """
    series = series.dropna()
    if len(series) < 2 or series.iloc[0] == 0:
        return 0.0
    return float((series.iloc[-1] / series.iloc[0]) ** (252 / len(series)) - 1)


def annualised_volatility(series: pd.Series) -> float:
    """
    Annualised standard deviation of daily returns (252 trading days).
    """
    daily_returns = series.pct_change().dropna()
    return float(daily_returns.std() * np.sqrt(252))


def sharpe_ratio(series: pd.Series, risk_free: float = 0.065) -> float:
    """
    Sharpe ratio using a 6.5% risk-free rate (approx. Indian 10-yr gilt).

    Returns 0.0 if annualised volatility is zero.
    """
    c = cagr(series)
    v = annualised_volatility(series)
    return float((c - risk_free) / v) if v != 0 else 0.0


def max_drawdown(series: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (negative number, e.g. -0.35 = -35%).
    """
    series = series.dropna()
    if series.empty:
        return 0.0
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return float(drawdown.min())
