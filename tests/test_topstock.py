"""
tests/test_topstock.py
-----------------------
Unit tests for the top10stocks() function in topstock.py.

All yfinance.download calls are mocked — no internet connection required.
"""

import datetime
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from topstock import top10stocks, NIFTY50_SYMBOLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_download(symbols: list, returns_map: dict, n_days: int = 252):
    """
    Build a fake yf.download() return value.

    returns_map: { 'TICKER.NS': 0.5 }  → that ticker grows 50% over n_days.
    Tickers not in returns_map get a flat line (0% return).
    """
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")

    price_data = {}
    for sym in symbols:
        growth = returns_map.get(sym, 0.0)
        prices = np.linspace(100.0, 100.0 * (1 + growth), n_days)
        price_data[sym] = prices

    # yfinance returns a MultiIndex DataFrame: (metric, ticker)
    arrays = [
        ["Close"] * len(symbols),
        symbols,
    ]
    multi_index = pd.MultiIndex.from_arrays(arrays, names=["Price", "Ticker"])
    df = pd.DataFrame(
        np.column_stack([price_data[s] for s in symbols]),
        index=dates,
        columns=multi_index,
    )
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTop10Stocks:

    def _make_return_map(self, n: int = 50) -> dict:
        """Give each NIFTY50 symbol a unique return so ranking is deterministic."""
        return {sym: (i + 1) * 0.01 for i, sym in enumerate(NIFTY50_SYMBOLS[:n])}

    @patch("topstock.yf.download")
    def test_returns_list(self, mock_dl):
        """top10stocks() should return a list."""
        returns_map = self._make_return_map()
        mock_dl.return_value = make_mock_download(NIFTY50_SYMBOLS, returns_map)
        result = top10stocks(1)
        assert isinstance(result, list)

    @patch("topstock.yf.download")
    def test_returns_10_by_default(self, mock_dl):
        """Default call should return exactly 10 symbols."""
        returns_map = self._make_return_map()
        mock_dl.return_value = make_mock_download(NIFTY50_SYMBOLS, returns_map)
        result = top10stocks(1)
        assert len(result) == 10

    @patch("topstock.yf.download")
    def test_respects_top_n_parameter(self, mock_dl):
        """top_n parameter should control how many symbols come back."""
        returns_map = self._make_return_map()
        mock_dl.return_value = make_mock_download(NIFTY50_SYMBOLS, returns_map)
        for n in [3, 5, 15]:
            result = top10stocks(1, top_n=n)
            assert len(result) == n, f"Expected {n} stocks, got {len(result)}"

    @patch("topstock.yf.download")
    def test_all_symbols_are_valid_nifty50(self, mock_dl):
        """Every returned symbol should be a recognised NIFTY50 constituent."""
        returns_map = self._make_return_map()
        mock_dl.return_value = make_mock_download(NIFTY50_SYMBOLS, returns_map)
        result = top10stocks(1)
        for sym in result:
            assert sym in NIFTY50_SYMBOLS, f"{sym} is not a NIFTY50 constituent."

    @patch("topstock.yf.download")
    def test_highest_return_stock_is_first(self, mock_dl):
        """
        The stock with the largest return should always appear first.
        We give WIPRO.NS a 999% return so it must rank #1.
        """
        returns_map = self._make_return_map()
        returns_map["WIPRO.NS"] = 9.99          # 999% — clear winner
        mock_dl.return_value = make_mock_download(NIFTY50_SYMBOLS, returns_map)
        result = top10stocks(1)
        assert result[0] == "WIPRO.NS", (
            f"Expected WIPRO.NS first, got {result[0]}"
        )

    @patch("topstock.yf.download")
    def test_no_duplicate_symbols(self, mock_dl):
        """Returned list should contain no duplicate tickers."""
        returns_map = self._make_return_map()
        mock_dl.return_value = make_mock_download(NIFTY50_SYMBOLS, returns_map)
        result = top10stocks(1)
        assert len(result) == len(set(result)), "Duplicate symbols found in result."

    @patch("topstock.yf.download")
    def test_empty_data_raises_value_error(self, mock_dl):
        """An empty DataFrame from yfinance should raise ValueError, not crash silently."""
        mock_dl.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No market data"):
            top10stocks(1)

    @patch("topstock.yf.download")
    def test_calls_yfinance_once(self, mock_dl):
        """Should use a single batch download, not one call per symbol."""
        returns_map = self._make_return_map()
        mock_dl.return_value = make_mock_download(NIFTY50_SYMBOLS, returns_map)
        top10stocks(1)
        assert mock_dl.call_count == 1, (
            f"Expected 1 yfinance call, got {mock_dl.call_count}. "
            "Use batch download instead of per-symbol loop."
        )

    @patch("topstock.yf.download")
    def test_date_range_matches_n_years(self, mock_dl):
        """
        The date range passed to yfinance should span approximately n_years back.
        """
        returns_map = self._make_return_map()
        mock_dl.return_value = make_mock_download(NIFTY50_SYMBOLS, returns_map)

        today = datetime.date.today()
        top10stocks(2)

        call_kwargs = mock_dl.call_args
        start_str = call_kwargs.kwargs.get("start") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else None
        if start_str:
            start = datetime.date.fromisoformat(start_str)
            days_back = (today - start).days
            # Should be approximately 2 years (allow ±10 days)
            assert 720 <= days_back <= 740, (
                f"Expected ~730 days back for n_years=2, got {days_back}."
            )


# ---------------------------------------------------------------------------
# NIFTY50_SYMBOLS sanity checks (no mocking needed)
# ---------------------------------------------------------------------------

class TestNifty50Symbols:
    def test_symbol_count(self):
        """Should have exactly 50 symbols."""
        assert len(NIFTY50_SYMBOLS) == 50, (
            f"Expected 50 symbols, found {len(NIFTY50_SYMBOLS)}."
        )

    def test_no_duplicates(self):
        assert len(NIFTY50_SYMBOLS) == len(set(NIFTY50_SYMBOLS)), (
            "Duplicate symbols found in NIFTY50_SYMBOLS."
        )

    def test_all_end_with_ns(self):
        bad = [s for s in NIFTY50_SYMBOLS if not s.endswith(".NS")]
        assert not bad, f"Symbols missing .NS suffix: {bad}"
