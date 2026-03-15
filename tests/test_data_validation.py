"""
Tests for data validation — database, validator, and processor.
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from data.validator import validate_ohlcv, DataValidationError
from database.db import (
    initialize_database, save_signal, save_trade,
    update_trade_result, has_recent_signal, DB_PATH,
)


def _make_ohlcv(n: int = 200, start_price: float = 2500.0) -> pd.DataFrame:
    """Create a valid synthetic OHLCV DataFrame."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [base + timedelta(minutes=15 * i) for i in range(n)]
    np.random.seed(42)
    prices = start_price + np.cumsum(np.random.randn(n) * 2)
    df = pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 3),
        "low": prices - np.abs(np.random.randn(n) * 3),
        "close": prices + np.random.randn(n) * 1,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=pd.DatetimeIndex(times, tz="UTC"))
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


class TestValidation:
    """Tests for data/validator.py."""

    def test_valid_data(self):
        df = _make_ohlcv()
        result = validate_ohlcv(df, min_bars=100)
        assert len(result) == 200

    def test_rejects_nan(self):
        df = _make_ohlcv()
        df.iloc[50, df.columns.get_loc("close")] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            validate_ohlcv(df)

    def test_rejects_high_lt_low(self):
        df = _make_ohlcv()
        df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["low"] - 10
        with pytest.raises(DataValidationError, match="high < low"):
            validate_ohlcv(df)

    def test_rejects_too_few_bars(self):
        df = _make_ohlcv(10)
        with pytest.raises(DataValidationError, match="Insufficient"):
            validate_ohlcv(df, min_bars=100)

    def test_rejects_negative_prices(self):
        df = _make_ohlcv()
        df.iloc[0, df.columns.get_loc("open")] = -1.0
        with pytest.raises(DataValidationError, match="non-positive"):
            validate_ohlcv(df)


class TestDatabase:
    """Tests for database/db.py."""

    @pytest.fixture(autouse=True)
    def _use_temp_db(self, tmp_path, monkeypatch):
        """Use a temp database for each test."""
        import database.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "test.db"))
        initialize_database()

    def test_initialize(self):
        """Database init should succeed."""
        assert initialize_database() is True

    def test_save_and_retrieve_signal(self):
        sig = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "BUY",
            "confidence_pct": 72.0,
            "entry_price": 2500.0,
            "bullish_count": 6,
            "bearish_count": 2,
        }
        row_id = save_signal(sig)
        assert row_id is not None
        assert row_id > 0

    def test_save_and_update_trade(self):
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "SELL",
            "entry_price": 2500.0,
            "stop_loss": 2515.0,
            "take_profit1": 2470.0,
            "lot_size": 0.1,
            "status": "open",
        }
        row_id = save_trade(trade)
        assert row_id is not None

        ok = update_trade_result(row_id, "closed_tp1", pnl_usd=150.0, pnl_pips=300.0)
        assert ok is True

    def test_duplicate_detection(self):
        """has_recent_signal should detect duplicates within 4 hours."""
        sig = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "BUY",
            "confidence_pct": 70.0,
        }
        save_signal(sig)

        assert has_recent_signal("BUY", hours=4.0) is True
        assert has_recent_signal("SELL", hours=4.0) is False

    def test_no_duplicate_for_old_signal(self):
        """Signals older than the window should not count."""
        old_time = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        sig = {
            "timestamp": old_time,
            "direction": "BUY",
            "confidence_pct": 70.0,
        }
        save_signal(sig)

        assert has_recent_signal("BUY", hours=4.0) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
