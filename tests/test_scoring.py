"""
Tests for analysis/scoring.py — signal scoring engine.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from analysis.scoring import score_signal, SignalScore
from analysis.indicators import AllIndicators, IndicatorResult, BULLISH, BEARISH, NEUTRAL


def _make_indicators(bulls: int = 6, bears: int = 1, neutrals: int = 2) -> AllIndicators:
    """Build a mock AllIndicators with specified vote counts."""
    indicators = MagicMock(spec=AllIndicators)

    results = []
    for i in range(bulls):
        r = IndicatorResult(name=f"bull_{i}", signal=BULLISH, value=1.0, reason="test")
        results.append(r)
    for i in range(bears):
        r = IndicatorResult(name=f"bear_{i}", signal=BEARISH, value=-1.0, reason="test")
        results.append(r)
    for i in range(neutrals):
        r = IndicatorResult(name=f"neut_{i}", signal=NEUTRAL, value=0.0, reason="test")
        results.append(r)

    indicators.as_list.return_value = results
    indicators.bullish_count.return_value = bulls
    indicators.bearish_count.return_value = bears
    indicators.neutral_count.return_value = neutrals
    indicators.net_score.return_value = bulls - bears

    # Provide mock sub-indicators for bonus checks
    indicators.adx = MagicMock()
    indicators.adx.values = {"adx": 20}
    indicators.volume = MagicMock()
    indicators.volume.values = {"ratio": 1.0, "surge": 0}

    return indicators


class TestScoringEngine:
    """Tests for the score_signal function."""

    def test_strong_buy_signal(self):
        """3 bull / 1 bear (75% active ratio) should produce a BUY signal."""
        ind = _make_indicators(bulls=3, bears=1, neutrals=5)
        # Use bar_time in active session (14:00 UTC = NY session)
        bar_time = datetime(2025, 1, 6, 14, 0, tzinfo=timezone.utc)  # Monday
        result = score_signal(ind, bar_time=bar_time)

        assert result.direction == "BUY"
        assert result.confidence_pct >= 65
        assert result.bullish_count == 3
        assert result.bearish_count == 1

    def test_strong_sell_signal(self):
        """1 bull / 3 bear (75% active ratio) should produce a SELL signal."""
        ind = _make_indicators(bulls=1, bears=3, neutrals=5)
        bar_time = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)
        result = score_signal(ind, bar_time=bar_time)

        assert result.direction == "SELL"
        assert result.confidence_pct >= 65

    def test_wait_on_tie(self):
        """Equal bull/bear counts should produce WAIT."""
        ind = _make_indicators(bulls=3, bears=3, neutrals=3)
        bar_time = datetime(2025, 1, 6, 14, 0, tzinfo=timezone.utc)
        result = score_signal(ind, bar_time=bar_time)

        assert result.direction == "WAIT"

    def test_wait_below_confidence(self):
        """3 bull / 2 bear = 60% confidence — below 65% threshold."""
        ind = _make_indicators(bulls=3, bears=2, neutrals=4)
        bar_time = datetime(2025, 1, 6, 14, 0, tzinfo=timezone.utc)
        result = score_signal(ind, bar_time=bar_time)

        assert result.direction == "WAIT"
        assert result.raw_confidence < 65

    def test_session_gate_outside_hours(self):
        """Signals outside NY session (13-22 UTC) should be gated to WAIT."""
        ind = _make_indicators(bulls=3, bears=1, neutrals=5)
        bar_time = datetime(2025, 1, 6, 5, 0, tzinfo=timezone.utc)  # 05:00 UTC = Asian
        result = score_signal(ind, bar_time=bar_time)

        assert result.direction == "WAIT"
        assert any("session" in g.lower() for g in result.gates_triggered)

    def test_no_session_gate_without_bar_time(self):
        """Without bar_time, session gate should NOT trigger (safe for tests)."""
        ind = _make_indicators(bulls=3, bears=1, neutrals=5)
        result = score_signal(ind, bar_time=None)

        assert result.direction == "BUY"

    def test_score_returns_dataclass(self):
        """Result must be a SignalScore dataclass."""
        ind = _make_indicators()
        result = score_signal(ind)
        assert isinstance(result, SignalScore)
        assert hasattr(result, "direction")
        assert hasattr(result, "confidence_pct")
        assert hasattr(result, "bonuses")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
