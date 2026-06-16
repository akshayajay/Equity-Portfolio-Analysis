"""
tests/test_utils.py
-------------------
Unit tests for portfolio performance metric functions in utils.py.

All tests use synthetic pandas Series — no network calls, no yfinance dependency.
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import cagr, annualised_volatility, sharpe_ratio, max_drawdown


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def flat_series(value: float = 100.0, n: int = 252) -> pd.Series:
    """A constant price series — zero return, zero volatility."""
    return pd.Series([value] * n)


def growing_series(start: float = 100.0, end: float = 200.0, n: int = 252) -> pd.Series:
    """A linearly growing series from start to end over n days."""
    return pd.Series(np.linspace(start, end, n))


def declining_series(start: float = 100.0, end: float = 50.0, n: int = 252) -> pd.Series:
    """A linearly declining series."""
    return pd.Series(np.linspace(start, end, n))


def drawdown_series() -> pd.Series:
    """
    Series that goes 100 → 150 → 75, giving a known max drawdown of -50%.
    """
    up   = np.linspace(100, 150, 100)
    down = np.linspace(150, 75, 100)
    return pd.Series(np.concatenate([up, down]))


# ---------------------------------------------------------------------------
# cagr()
# ---------------------------------------------------------------------------

class TestCagr:
    def test_doubling_portfolio_positive(self):
        """A portfolio that doubles over 252 days should have a positive CAGR."""
        series = growing_series(100, 200, 252)
        result = cagr(series)
        assert result > 0, "CAGR should be positive for a doubling portfolio."

    def test_cagr_doubling_approx_100_percent(self):
        """Doubling in exactly 252 trading days → CAGR ≈ 100%."""
        series = growing_series(100, 200, 252)
        result = cagr(series)
        assert abs(result - 1.0) < 0.01, f"Expected ~1.0 (100%), got {result:.4f}"

    def test_declining_portfolio_negative(self):
        """A portfolio that halves should have a negative CAGR."""
        series = declining_series(100, 50, 252)
        result = cagr(series)
        assert result < 0, "CAGR should be negative for a declining portfolio."

    def test_flat_series_zero_cagr(self):
        """A constant series should give CAGR ≈ 0."""
        result = cagr(flat_series())
        assert abs(result) < 1e-6, f"Expected ~0.0, got {result}"

    def test_too_short_series_returns_zero(self):
        """Series with fewer than 2 data points should return 0.0."""
        result = cagr(pd.Series([100.0]))
        assert result == 0.0

    def test_zero_start_returns_zero(self):
        """Series starting at zero should return 0.0 (avoid division by zero)."""
        result = cagr(pd.Series([0.0, 50.0, 100.0]))
        assert result == 0.0

    def test_returns_float(self):
        result = cagr(growing_series())
        assert isinstance(result, float)

    def test_nan_values_are_ignored(self):
        """NaN values at the start should be dropped without error."""
        s = pd.Series([np.nan, np.nan, 100.0, 110.0, 120.0])
        result = cagr(s)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# annualised_volatility()
# ---------------------------------------------------------------------------

class TestAnnualisedVolatility:
    def test_flat_series_zero_volatility(self):
        """A constant price series should have (near) zero volatility."""
        result = annualised_volatility(flat_series())
        assert result < 1e-6, f"Expected ~0.0, got {result}"

    def test_volatile_series_positive(self):
        """A series with actual price movement should have positive volatility."""
        rng = np.random.default_rng(42)
        prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, 252))
        result = annualised_volatility(pd.Series(prices))
        assert result > 0, "Volatility should be positive for a moving series."

    def test_returns_float(self):
        result = annualised_volatility(growing_series())
        assert isinstance(result, float)

    def test_higher_noise_higher_volatility(self):
        """Higher daily noise → higher annualised volatility."""
        rng = np.random.default_rng(0)
        low_noise  = 100 * np.cumprod(1 + rng.normal(0, 0.005, 252))
        high_noise = 100 * np.cumprod(1 + rng.normal(0, 0.02, 252))
        assert annualised_volatility(pd.Series(high_noise)) > \
               annualised_volatility(pd.Series(low_noise))


# ---------------------------------------------------------------------------
# sharpe_ratio()
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_sharpe_for_strong_growth(self):
        """A strongly growing, low-volatility series should yield a positive Sharpe."""
        # Build a series that doubles smoothly in 252 days (CAGR ≈ 100%, low vol)
        series = growing_series(100, 200, 252)
        result = sharpe_ratio(series)
        assert result > 0, f"Expected positive Sharpe, got {result}"

    def test_negative_sharpe_for_declining(self):
        """A declining series should produce a negative Sharpe ratio."""
        series = declining_series(100, 50, 252)
        result = sharpe_ratio(series)
        assert result < 0, f"Expected negative Sharpe, got {result}"

    def test_zero_volatility_returns_zero(self):
        """A flat (zero-vol) series should return 0.0 without raising."""
        result = sharpe_ratio(flat_series())
        assert result == 0.0

    def test_custom_risk_free_rate(self):
        """A higher risk-free rate should reduce (or flip) the Sharpe ratio."""
        series = growing_series(100, 115, 252)  # modest growth
        sharpe_low_rf  = sharpe_ratio(series, risk_free=0.0)
        sharpe_high_rf = sharpe_ratio(series, risk_free=0.20)
        assert sharpe_low_rf > sharpe_high_rf

    def test_returns_float(self):
        result = sharpe_ratio(growing_series())
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# max_drawdown()
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_known_drawdown(self):
        """
        Series: 100 → 150 → 75.
        Peak = 150, trough = 75  →  drawdown = (75-150)/150 = -0.50.
        """
        series = drawdown_series()
        result = max_drawdown(series)
        assert abs(result - (-0.5)) < 0.01, f"Expected -0.50, got {result:.4f}"

    def test_monotonically_rising_no_drawdown(self):
        """A series that only goes up should have a drawdown of 0."""
        series = growing_series(100, 200, 252)
        result = max_drawdown(series)
        assert result >= -1e-6, f"Expected ~0, got {result}"

    def test_returns_negative_or_zero(self):
        """max_drawdown should always be ≤ 0."""
        rng = np.random.default_rng(7)
        prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, 252))
        result = max_drawdown(pd.Series(prices))
        assert result <= 0.0

    def test_empty_series_returns_zero(self):
        result = max_drawdown(pd.Series([], dtype=float))
        assert result == 0.0

    def test_returns_float(self):
        result = max_drawdown(growing_series())
        assert isinstance(result, float)
