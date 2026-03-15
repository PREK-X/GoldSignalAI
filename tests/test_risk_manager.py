"""
Tests for signals/risk_manager.py — SL/TP/lot sizing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from signals.risk_manager import calculate_risk, price_to_pips, pips_to_price, RiskParameters
from config import Config


class TestPipConversion:
    """Tests for price <-> pip conversion helpers."""

    def test_price_to_pips(self):
        """1.0 price distance = 10 pips for gold (PIP_SIZE=0.1)."""
        assert price_to_pips(1.0) == pytest.approx(10.0, abs=0.1)

    def test_pips_to_price(self):
        """10 pips = 1.0 price distance."""
        assert pips_to_price(10.0) == pytest.approx(1.0, abs=0.01)

    def test_always_positive(self):
        """Both functions should return positive values."""
        assert price_to_pips(-5.0) > 0
        assert pips_to_price(-50) > 0


class TestRiskCalculation:
    """Tests for the calculate_risk function."""

    def test_buy_sl_below_entry(self):
        """BUY stop loss must be below entry price."""
        risk = calculate_risk(
            entry_price=2500.0,
            direction="BUY",
            atr_value=5.0,
        )
        assert risk is not None
        assert risk.stop_loss < risk.entry_price

    def test_sell_sl_above_entry(self):
        """SELL stop loss must be above entry price."""
        risk = calculate_risk(
            entry_price=2500.0,
            direction="SELL",
            atr_value=5.0,
        )
        assert risk is not None
        assert risk.stop_loss > risk.entry_price

    def test_buy_tp_above_entry(self):
        """BUY take profits must be above entry."""
        risk = calculate_risk(
            entry_price=2500.0,
            direction="BUY",
            atr_value=5.0,
        )
        assert risk.tp1_price > risk.entry_price
        assert risk.tp2_price > risk.tp1_price

    def test_sell_tp_below_entry(self):
        """SELL take profits must be below entry."""
        risk = calculate_risk(
            entry_price=2500.0,
            direction="SELL",
            atr_value=5.0,
        )
        assert risk.tp1_price < risk.entry_price
        assert risk.tp2_price < risk.tp1_price

    def test_sl_within_limits(self):
        """SL must be clamped between MIN_SL_PIPS and MAX_SL_PIPS."""
        risk = calculate_risk(
            entry_price=2500.0,
            direction="BUY",
            atr_value=5.0,
        )
        assert risk.sl_pips >= Config.MIN_SL_PIPS
        assert risk.sl_pips <= Config.MAX_SL_PIPS

    def test_lot_size_positive(self):
        """Lot size must be positive."""
        risk = calculate_risk(
            entry_price=2500.0,
            direction="BUY",
            atr_value=5.0,
        )
        assert risk.suggested_lot > 0

    def test_risk_usd_within_budget(self):
        """Risk in USD should not exceed account balance * risk percentage."""
        risk = calculate_risk(
            entry_price=2500.0,
            direction="BUY",
            atr_value=5.0,
        )
        max_risk = Config.CHALLENGE_ACCOUNT_SIZE * (Config.RISK_PER_TRADE_PCT / 100)
        # Allow 10% tolerance for rounding
        assert risk.risk_usd <= max_risk * 1.1

    def test_returns_risk_parameters(self):
        """Result must be a RiskParameters dataclass."""
        risk = calculate_risk(
            entry_price=2500.0,
            direction="BUY",
            atr_value=5.0,
        )
        assert isinstance(risk, RiskParameters)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
