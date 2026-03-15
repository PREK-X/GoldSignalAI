"""
GoldSignalAI — tests/test_data_pipeline.py
==========================================
Unit tests for the data infrastructure pipeline:
  - Polygon fetcher
  - Fallback hierarchy (Polygon → MT5 → yfinance)
  - Data validator
  - Data processor
"""

import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.validator import validate_ohlcv, DataValidationError
from data.polygon_fetcher import fetch_polygon_data, _results_to_dataframe


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 200, start_price: float = 2000.0) -> pd.DataFrame:
    """Create a valid synthetic OHLCV DataFrame for testing."""
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [base_time + timedelta(minutes=15 * i) for i in range(n)]

    np.random.seed(42)
    prices = start_price + np.cumsum(np.random.randn(n) * 2)

    df = pd.DataFrame({
        "open":   prices,
        "high":   prices + np.abs(np.random.randn(n) * 3),
        "low":    prices - np.abs(np.random.randn(n) * 3),
        "close":  prices + np.random.randn(n) * 1,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=pd.DatetimeIndex(times, tz="UTC"))

    # Ensure OHLCV consistency: high >= max(open,close), low <= min(open,close)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# TEST: VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class TestValidator:
    """Tests for data/validator.py"""

    def test_valid_data_passes(self):
        """Clean data should pass all checks."""
        df = _make_ohlcv(200)
        result = validate_ohlcv(df, min_bars=100)
        assert len(result) == 200

    def test_nan_values_rejected(self):
        """NaN in OHLC columns should raise DataValidationError."""
        df = _make_ohlcv(200)
        df.iloc[50, df.columns.get_loc("close")] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            validate_ohlcv(df)

    def test_high_less_than_low_rejected(self):
        """Rows where high < low should raise DataValidationError."""
        df = _make_ohlcv(200)
        # Force high < low on one row
        idx = 100
        df.iloc[idx, df.columns.get_loc("high")] = df.iloc[idx]["low"] - 10
        with pytest.raises(DataValidationError, match="high < low"):
            validate_ohlcv(df)

    def test_non_increasing_timestamps_rejected(self):
        """Duplicate or out-of-order timestamps should raise."""
        df = _make_ohlcv(200)
        # Duplicate a timestamp
        new_index = df.index.tolist()
        new_index[50] = new_index[49]  # make timestamp go backwards/equal
        df.index = pd.DatetimeIndex(new_index, tz="UTC")
        with pytest.raises(DataValidationError, match="not strictly increasing"):
            validate_ohlcv(df)

    def test_insufficient_bars_rejected(self):
        """Too few bars should raise DataValidationError."""
        df = _make_ohlcv(10)
        with pytest.raises(DataValidationError, match="Insufficient data"):
            validate_ohlcv(df, min_bars=100)

    def test_negative_prices_rejected(self):
        """Non-positive prices should raise DataValidationError."""
        df = _make_ohlcv(200)
        df.iloc[0, df.columns.get_loc("close")] = -1.0
        with pytest.raises(DataValidationError, match="non-positive"):
            validate_ohlcv(df)

    def test_missing_columns_rejected(self):
        """Missing required columns should raise DataValidationError."""
        df = _make_ohlcv(200)
        df = df.drop(columns=["volume"])
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_ohlcv(df)

    def test_empty_dataframe_rejected(self):
        """Empty DataFrame should raise DataValidationError."""
        df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="empty"):
            validate_ohlcv(df)


# ─────────────────────────────────────────────────────────────────────────────
# TEST: POLYGON FETCHER
# ─────────────────────────────────────────────────────────────────────────────

class TestPolygonFetcher:
    """Tests for data/polygon_fetcher.py"""

    def test_results_to_dataframe(self):
        """Polygon JSON results should convert to a proper DataFrame."""
        # Simulate Polygon API response results
        base_ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        results = [
            {
                "t": base_ts + i * 900_000,  # 15-min intervals in ms
                "o": 2000.0 + i,
                "h": 2005.0 + i,
                "l": 1995.0 + i,
                "c": 2002.0 + i,
                "v": 1000 + i,
                "n": 50,
            }
            for i in range(100)
        ]

        df = _results_to_dataframe(results)

        assert len(df) == 100
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None  # UTC-aware
        assert df["open"].iloc[0] == 2000.0
        assert df["close"].iloc[-1] == 2101.0

    def test_empty_results_returns_empty_df(self):
        """Empty results list should return empty DataFrame."""
        df = _results_to_dataframe([])
        assert df.empty

    @patch("data.polygon_fetcher.Config")
    def test_missing_api_key_raises(self, mock_config):
        """Missing API key should raise ValueError."""
        mock_config.POLYGON_API_KEY = ""
        mock_config.HISTORICAL_YEARS = 2
        with pytest.raises(ValueError, match="POLYGON_API_KEY"):
            fetch_polygon_data()

    @patch("data.polygon_fetcher.requests.get")
    @patch("data.polygon_fetcher.Config")
    def test_successful_fetch(self, mock_config, mock_get):
        """Successful API response should return a DataFrame."""
        mock_config.POLYGON_API_KEY = "test_key"
        mock_config.HISTORICAL_YEARS = 2

        base_ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "t": base_ts + i * 900_000,
                    "o": 2000.0, "h": 2005.0, "l": 1995.0, "c": 2002.0, "v": 500,
                }
                for i in range(50)
            ],
            "resultsCount": 50,
        }
        mock_get.return_value = mock_response

        df = fetch_polygon_data(
            symbol="C:XAUUSD",
            timeframe="M15",
            start_date="2025-01-01",
            end_date="2025-01-15",
        )

        assert df is not None
        assert len(df) == 50
        assert "close" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# TEST: FALLBACK HIERARCHY
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackHierarchy:
    """Tests for the fallback system in data/fetcher.py"""

    @patch("data.fetcher._fetch_polygon")
    @patch("data.fetcher._fetch_yfinance")
    def test_fallback_to_yfinance(self, mock_yf, mock_polygon):
        """When Polygon fails, yfinance should be tried."""
        from data.fetcher import get_market_data

        mock_polygon.return_value = None  # Polygon fails

        df_yf = _make_ohlcv(100)
        mock_yf.return_value = df_yf

        result = get_market_data("XAUUSD", "M15", 100)

        assert result is not None
        assert len(result) == 100
        mock_polygon.assert_called_once()
        mock_yf.assert_called_once()

    @patch("data.fetcher._fetch_polygon")
    def test_polygon_primary_success(self, mock_polygon):
        """When Polygon succeeds, no fallback should be attempted."""
        from data.fetcher import get_market_data

        df_polygon = _make_ohlcv(200)
        mock_polygon.return_value = df_polygon

        result = get_market_data("XAUUSD", "M15", 200)

        assert result is not None
        assert len(result) == 200
        mock_polygon.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# TEST: PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessor:
    """Tests for data/processor.py"""

    def test_process_returns_correct_structure(self):
        """Processed data should have base features and valid structure."""
        from data.processor import process

        df = _make_ohlcv(200)
        result = process(df, timeframe="M15", label="test")

        assert result is not None
        # Base features should be added
        assert "returns" in result.columns
        assert "log_returns" in result.columns
        assert "hl_range" in result.columns
        assert "body" in result.columns
        assert "is_bullish" in result.columns
        # OHLCV columns preserved
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns
        # Index is UTC DatetimeIndex
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None

    def test_process_removes_duplicates(self):
        """Duplicate timestamps should be removed."""
        from data.processor import process

        df = _make_ohlcv(200)
        # Add duplicate rows
        dup = df.iloc[:5].copy()
        df_with_dups = pd.concat([df, dup])

        result = process(df_with_dups, timeframe="M15", label="dup_test")
        assert result is not None
        assert not result.index.duplicated().any()

    def test_process_rejects_insufficient_data(self):
        """Too few candles after cleaning should return None."""
        from data.processor import process

        df = _make_ohlcv(10)
        result = process(df, timeframe="M15", label="small_test")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
