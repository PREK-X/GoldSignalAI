"""
Tests for signals/formatter.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from unittest.mock import MagicMock

from signals.generator import TradingSignal
from signals.risk_manager import RiskParameters
from analysis.multi_timeframe import MultiTimeframeResult, TimeframeAnalysis
from ml.predictor import MLPrediction
from signals.formatter import (
    format_signal,
    format_signal_short,
    format_signal_telegram,
    format_wait_reason,
)


def _make_risk(direction="BUY", entry=2350.00):
    """Create a realistic RiskParameters."""
    return RiskParameters(
        entry_price=entry,
        stop_loss=2348.00 if direction == "BUY" else 2352.00,
        sl_pips=20.0,
        sl_usd_per_lot=200.0,
        tp1_price=2354.00 if direction == "BUY" else 2346.00,
        tp1_pips=40.0,
        tp1_rr=2.0,
        tp2_price=2356.00 if direction == "BUY" else 2344.00,
        tp2_pips=60.0,
        tp2_rr=3.0,
        suggested_lot=0.25,
        risk_usd=500.0,
        potential_tp1_usd=1000.0,
        potential_tp2_usd=1500.0,
        direction=direction,
        atr_value=1.8,
        sl_method="ATR",
    )


def _make_mtf():
    """Minimal MTF result mock."""
    m15 = MagicMock(spec=TimeframeAnalysis)
    m15.score = None
    h1 = MagicMock(spec=TimeframeAnalysis)
    h1.score = None
    mtf = MagicMock(spec=MultiTimeframeResult)
    mtf.m15 = m15
    mtf.h1 = h1
    mtf.timeframes_agree = True
    return mtf


def _make_signal(direction="BUY", confidence=78.0, paused=False, pause_reason=""):
    """Build a complete TradingSignal for testing."""
    risk = _make_risk(direction) if direction != "WAIT" else None
    return TradingSignal(
        direction=direction,
        confidence_pct=confidence,
        timestamp=datetime(2025, 3, 11, 14, 30, tzinfo=timezone.utc),
        entry_price=2350.00,
        risk=risk,
        mtf_result=_make_mtf(),
        ml_prediction=MLPrediction(available=True, models_agree=True, direction="UP"),
        bullish_count=8,
        bearish_count=2,
        total_indicators=10,
        ml_confirms=True,
        reason="Strong technical consensus",
        is_paused=paused,
        pause_reason=pause_reason,
    )


# ── Test 1: BUY signal card has all required fields ─────────────────────────

def test_buy_signal_card():
    sig = _make_signal("BUY", 78.0)
    card = format_signal(sig)

    assert "GoldSignalAI" in card
    assert "BUY" in card
    assert "2,350.00" in card        # entry price
    assert "2,348.00" in card        # stop loss
    assert "78%" in card             # confidence
    assert "1:2" in card             # R/R TP1
    assert "1:3" in card             # R/R TP2
    assert "8/10 Bullish" in card    # indicators
    assert "2025-03-11 14:30 UTC" in card
    assert "YES" in card             # ML confirm
    assert "┌" in card and "┘" in card  # box chars
    print("  ✓ Test 1 passed: BUY signal card has all fields")


# ── Test 2: SELL signal card ─────────────────────────────────────────────────

def test_sell_signal_card():
    sig = _make_signal("SELL", 72.0)
    card = format_signal(sig)

    assert "SELL" in card
    assert "2,350.00" in card
    print("  ✓ Test 2 passed: SELL signal card renders correctly")


# ── Test 3: WAIT signal (no risk data) ──────────────────────────────────────

def test_wait_signal_card():
    sig = _make_signal("WAIT", 45.0)
    card = format_signal(sig)

    assert "WAIT" in card
    assert "—" in card  # no SL/TP/lot
    assert "N/A" in card  # R/R
    print("  ✓ Test 3 passed: WAIT signal shows dashes for risk fields")


# ── Test 4: News pause banner ───────────────────────────────────────────────

def test_paused_signal():
    sig = _make_signal("WAIT", 0.0, paused=True, pause_reason="NFP Release")
    card = format_signal(sig)

    assert "NEWS EVENT" in card
    assert "NFP Release" in card
    print("  ✓ Test 4 passed: Paused signal shows news banner")


# ── Test 5: Short format ────────────────────────────────────────────────────

def test_short_format():
    sig = _make_signal("BUY", 78.0)
    short = format_signal_short(sig)

    assert "BUY" in short
    assert "2,350.00" in short
    assert "78%" in short
    assert "SL=" in short
    assert "TP1=" in short
    print("  ✓ Test 5 passed: Short format contains key info")


# ── Test 6: Telegram format wraps in monospace ──────────────────────────────

def test_telegram_format():
    sig = _make_signal("BUY", 78.0)
    tg = format_signal_telegram(sig)

    assert tg.startswith("```")
    assert tg.endswith("```")
    assert "GoldSignalAI" in tg
    print("  ✓ Test 6 passed: Telegram format wraps in monospace block")


# ── Test 7: Wait reason ────────────────────────────────────────────────────

def test_wait_reason():
    sig = _make_signal("WAIT", 0.0)
    reason = format_wait_reason(sig)
    assert reason == "Strong technical consensus"

    sig_paused = _make_signal("WAIT", 0.0, paused=True, pause_reason="CPI")
    reason_p = format_wait_reason(sig_paused)
    assert "CPI" in reason_p
    assert "Paused" in reason_p
    print("  ✓ Test 7 passed: Wait reason returns correct text")


# ── Test 8: Card lines are consistent width ─────────────────────────────────

def test_card_line_width():
    sig = _make_signal("BUY", 78.0)
    card = format_signal(sig)
    lines = card.split("\n")

    # All box lines should start with │ or ┌ or ├ or └
    for line in lines:
        assert line[0] in "│┌├└", f"Unexpected line start: {line!r}"
        assert line[-1] in "│┐┤┘", f"Unexpected line end: {line!r}"
    print("  ✓ Test 8 passed: All card lines have proper box borders")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_buy_signal_card,
        test_sell_signal_card,
        test_wait_signal_card,
        test_paused_signal,
        test_short_format,
        test_telegram_format,
        test_wait_reason,
        test_card_line_width,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} FAILED: {e}")
    print(f"\n{'='*50}")
    print(f"Formatter tests: {passed}/{len(tests)} passed")
