"""
Tests for alerts/chart_generator.py

Uses synthetic OHLCV data to avoid network calls.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from alerts.chart_generator import generate_chart, MPL_AVAILABLE, PLOTLY_AVAILABLE


def _make_ohlcv(n=60, base_price=2350.0):
    """Create synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="15min")
    close = base_price + np.cumsum(np.random.randn(n) * 0.5)
    high  = close + np.abs(np.random.randn(n) * 0.3)
    low   = close - np.abs(np.random.randn(n) * 0.3)
    opn   = close + np.random.randn(n) * 0.2

    df = pd.DataFrame({
        "open": opn, "high": high, "low": low, "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=dates)
    return df


# ── Test 1: Chart generates a file ──────────────────────────────────────────

def test_chart_generation():
    df = _make_ohlcv()
    path = generate_chart(
        df, direction="BUY", entry_price=2350.0,
        sl_price=2348.0, tp1_price=2354.0, tp2_price=2356.0,
    )
    assert path is not None
    assert os.path.isfile(path)
    assert path.endswith(".png")
    size = os.path.getsize(path)
    assert size > 1000  # should be a real image
    os.unlink(path)
    print(f"  ✓ Test 1 passed: Chart generated ({size:,} bytes)")


# ── Test 2: SELL chart ───────────────────────────────────────────────────────

def test_sell_chart():
    df = _make_ohlcv()
    path = generate_chart(
        df, direction="SELL", entry_price=2350.0,
        sl_price=2352.0, tp1_price=2346.0, tp2_price=2344.0,
    )
    assert path is not None
    assert os.path.isfile(path)
    os.unlink(path)
    print("  ✓ Test 2 passed: SELL chart generated")


# ── Test 3: Chart without SL/TP ─────────────────────────────────────────────

def test_chart_no_levels():
    df = _make_ohlcv()
    path = generate_chart(df, direction="WAIT", entry_price=2350.0)
    assert path is not None
    assert os.path.isfile(path)
    os.unlink(path)
    print("  ✓ Test 3 passed: WAIT chart without SL/TP generated")


# ── Test 4: Empty DataFrame returns None ─────────────────────────────────────

def test_empty_df():
    path = generate_chart(pd.DataFrame(), direction="BUY", entry_price=2350.0)
    assert path is None
    print("  ✓ Test 4 passed: Empty DataFrame returns None")


# ── Test 5: None DataFrame returns None ──────────────────────────────────────

def test_none_df():
    path = generate_chart(None, direction="BUY", entry_price=2350.0)
    assert path is None
    print("  ✓ Test 5 passed: None DataFrame returns None")


# ── Test 6: Custom title ────────────────────────────────────────────────────

def test_custom_title():
    df = _make_ohlcv(30)
    path = generate_chart(
        df, direction="BUY", entry_price=2350.0,
        title="Custom Test Chart",
    )
    assert path is not None
    os.unlink(path)
    print("  ✓ Test 6 passed: Custom title chart generated")


# ── Test 7: Backend availability ─────────────────────────────────────────────

def test_backend():
    assert MPL_AVAILABLE or PLOTLY_AVAILABLE, "Need at least one charting library"
    backends = []
    if PLOTLY_AVAILABLE:
        backends.append("plotly")
    if MPL_AVAILABLE:
        backends.append("matplotlib")
    print(f"  ✓ Test 7 passed: Available backends: {', '.join(backends)}")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_chart_generation,
        test_sell_chart,
        test_chart_no_levels,
        test_empty_df,
        test_none_df,
        test_custom_title,
        test_backend,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} FAILED: {e}")
    print(f"\n{'='*50}")
    print(f"Chart generator tests: {passed}/{len(tests)} passed")
