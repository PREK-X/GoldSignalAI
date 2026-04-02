"""
Tests for Stage 11: MT5 Bridge (simulation mode) + PositionMonitor + StateManager.
"""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# MT5Bridge — simulation mode
# ─────────────────────────────────────────────────────────────────────────────

class TestMT5BridgeSimulation:
    """Tests for MT5Bridge operating in simulation mode (Linux)."""

    def _make_bridge(self):
        from execution.mt5_bridge import MT5Bridge
        bridge = MT5Bridge()
        bridge.connect()
        return bridge

    def test_platform_detection(self):
        """On Linux, platform should be 'simulation'."""
        from execution.mt5_bridge import MT5Bridge
        bridge = MT5Bridge()
        assert bridge.platform == "simulation"
        assert bridge.is_simulation is True

    def test_connect_simulation(self):
        bridge = self._make_bridge()
        assert bridge._connected is True

    def test_place_order_simulation(self):
        """Placing an order in simulation returns a mock fill."""
        bridge = self._make_bridge()
        result = bridge.place_order(
            symbol="XAUUSD",
            direction="BUY",
            volume=0.01,
            sl_price=2300.0,
            tp_price=2350.0,
            comment="test",
        )
        assert result.success is True
        assert result.ticket > 0
        assert result.status == "simulated"
        assert result.fill_price > 0

    def test_close_order_simulation(self):
        """Closing a simulated order removes it from internal state."""
        bridge = self._make_bridge()
        order = bridge.place_order("XAUUSD", "SELL", 0.02, 2350.0, 2300.0)
        assert order.success

        close = bridge.close_order(order.ticket)
        assert close.success is True
        assert close.ticket == order.ticket

        # Position should be gone
        assert bridge.get_position(order.ticket) is None

    def test_close_nonexistent_order(self):
        bridge = self._make_bridge()
        result = bridge.close_order(999999)
        assert result.success is False

    def test_get_open_positions(self):
        bridge = self._make_bridge()
        bridge.place_order("XAUUSD", "BUY", 0.01, 2300.0, 2350.0)
        bridge.place_order("XAUUSD", "SELL", 0.02, 2350.0, 2300.0)
        positions = bridge.get_open_positions()
        assert len(positions) == 2

    def test_get_account_info_simulation(self):
        bridge = self._make_bridge()
        info = bridge.get_account_info()
        assert info is not None
        assert info.balance > 0

    def test_modify_sl_simulation(self):
        bridge = self._make_bridge()
        order = bridge.place_order("XAUUSD", "BUY", 0.01, 2300.0, 2350.0)
        assert bridge.modify_sl(order.ticket, 2310.0) is True

        pos = bridge.get_position(order.ticket)
        assert pos.sl == 2310.0

    def test_modify_sl_nonexistent(self):
        bridge = self._make_bridge()
        assert bridge.modify_sl(999999, 2310.0) is False

    def test_disconnect_simulation(self):
        bridge = self._make_bridge()
        bridge.disconnect()
        assert bridge._connected is False


# ─────────────────────────────────────────────────────────────────────────────
# PositionMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionMonitor:
    """Tests for PositionMonitor trailing stop and management rules."""

    def _make_monitor(self):
        from execution.mt5_bridge import MT5Bridge
        from execution.position_monitor import PositionMonitor
        bridge = MT5Bridge()
        bridge.connect()
        return bridge, PositionMonitor(bridge)

    def test_trailing_stop_buy(self):
        """SL moves to breakeven when profit >= 1R for BUY."""
        bridge, monitor = self._make_monitor()

        # Place a BUY: entry ~2325, SL=2300, TP=2350 → 1R = 25 units
        order = bridge.place_order("XAUUSD", "BUY", 0.01, 2300.0, 2350.0)
        pos = bridge.get_position(order.ticket)
        entry = pos.open_price  # midpoint = 2325

        # Price moves up by 1R (25 units) → should trigger trailing SL
        current_price = entry + abs(entry - 2300.0)
        monitor._check_trailing_stop(pos, current_price, current_atr=5.0)

        updated = bridge.get_position(order.ticket)
        # SL should have moved up from 2300 toward breakeven
        assert updated.sl > 2300.0

    def test_trailing_stop_sell(self):
        """SL moves to breakeven when profit >= 1R for SELL."""
        bridge, monitor = self._make_monitor()

        # Place a SELL: entry ~2325, SL=2350, TP=2300 → 1R = 25 units
        order = bridge.place_order("XAUUSD", "SELL", 0.01, 2350.0, 2300.0)
        pos = bridge.get_position(order.ticket)
        entry = pos.open_price

        # Price drops by 1R
        current_price = entry - abs(2350.0 - entry)
        monitor._check_trailing_stop(pos, current_price, current_atr=5.0)

        updated = bridge.get_position(order.ticket)
        assert updated.sl < 2350.0

    def test_friday_close(self):
        """Positions should be closed on Friday at 20:00 UTC."""
        from execution.position_monitor import PositionMonitor
        # Friday 20:00 UTC
        friday_20 = datetime(2026, 4, 3, 20, 0, 0, tzinfo=timezone.utc)
        assert PositionMonitor._should_friday_close(friday_20) is True

        # Friday 19:59 UTC
        friday_19 = datetime(2026, 4, 3, 19, 59, 0, tzinfo=timezone.utc)
        assert PositionMonitor._should_friday_close(friday_19) is False

        # Thursday 20:00 UTC
        thursday_20 = datetime(2026, 4, 2, 20, 0, 0, tzinfo=timezone.utc)
        assert PositionMonitor._should_friday_close(thursday_20) is False

    def test_time_exit_48_bars(self):
        """Positions older than 12 hours should be time-exited."""
        from execution.mt5_bridge import PositionInfo
        from execution.position_monitor import PositionMonitor

        old_time = datetime(2026, 4, 2, 0, 0, 0, tzinfo=timezone.utc)
        pos = PositionInfo(
            ticket=1, symbol="XAUUSD", direction="BUY",
            volume=0.01, open_price=2320.0, sl=2300.0, tp=2350.0,
            open_time=old_time,
        )

        # 13 hours later → should trigger exit
        now_13h = old_time + timedelta(hours=13)
        assert PositionMonitor._should_time_exit(pos, now_13h) is True

        # 11 hours later → should NOT trigger exit
        now_11h = old_time + timedelta(hours=11)
        assert PositionMonitor._should_time_exit(pos, now_11h) is False

    def test_manage_positions_friday_close(self):
        """Full integration: positions closed on Friday 20:00 UTC."""
        bridge, monitor = self._make_monitor()
        bridge.place_order("XAUUSD", "BUY", 0.01, 2300.0, 2350.0)

        friday_20 = datetime(2026, 4, 3, 20, 0, 0, tzinfo=timezone.utc)
        monitor.check_and_manage_positions(friday_20, 2320.0, 5.0)

        # Position should be closed
        assert len(bridge.get_open_positions()) == 0


# ─────────────────────────────────────────────────────────────────────────────
# StateManager
# ─────────────────────────────────────────────────────────────────────────────

class TestStateManager:
    """Tests for session loss tracking and JSON persistence."""

    def _make_manager(self, tmpdir):
        from state.state_manager import StateManager
        path = os.path.join(str(tmpdir), "state.json")
        return StateManager(state_file=path), path

    def test_initial_state(self, tmp_path):
        mgr, _ = self._make_manager(tmp_path)
        assert mgr.get_session_losses() == 0
        assert mgr.session_date == ""

    def test_increment_session_loss(self, tmp_path):
        mgr, _ = self._make_manager(tmp_path)
        now = datetime(2026, 4, 2, 14, 0, 0, tzinfo=timezone.utc)

        mgr.increment_session_loss(now)
        assert mgr.get_session_losses() == 1

        mgr.increment_session_loss(now)
        assert mgr.get_session_losses() == 2

    def test_session_reset_on_new_day(self, tmp_path):
        mgr, _ = self._make_manager(tmp_path)
        day1 = datetime(2026, 4, 2, 14, 0, 0, tzinfo=timezone.utc)
        day2 = datetime(2026, 4, 3, 14, 0, 0, tzinfo=timezone.utc)

        mgr.increment_session_loss(day1)
        mgr.increment_session_loss(day1)
        assert mgr.get_session_losses() == 2

        # New day resets counter before incrementing
        mgr.increment_session_loss(day2)
        assert mgr.get_session_losses() == 1

    def test_reset_session_losses(self, tmp_path):
        mgr, _ = self._make_manager(tmp_path)
        now = datetime(2026, 4, 2, 14, 0, 0, tzinfo=timezone.utc)
        mgr.increment_session_loss(now)
        mgr.reset_session_losses()
        assert mgr.get_session_losses() == 0

    def test_register_trade_outcome_loss(self, tmp_path):
        mgr, _ = self._make_manager(tmp_path)
        now = datetime(2026, 4, 2, 14, 0, 0, tzinfo=timezone.utc)

        mgr.register_trade_outcome("T1", "LOSS", now)
        assert mgr.get_session_losses() == 1

        mgr.register_trade_outcome("T2", "LOSS", now)
        assert mgr.get_session_losses() == 2

    def test_register_trade_outcome_win_resets(self, tmp_path):
        mgr, _ = self._make_manager(tmp_path)
        now = datetime(2026, 4, 2, 14, 0, 0, tzinfo=timezone.utc)

        mgr.register_trade_outcome("T1", "LOSS", now)
        mgr.register_trade_outcome("T2", "LOSS", now)
        assert mgr.get_session_losses() == 2

        mgr.register_trade_outcome("T3", "WIN", now)
        assert mgr.get_session_losses() == 0

    def test_persistence(self, tmp_path):
        """State survives across manager instances."""
        from state.state_manager import StateManager
        path = os.path.join(str(tmp_path), "state.json")

        mgr1 = StateManager(state_file=path)
        now = datetime(2026, 4, 2, 14, 0, 0, tzinfo=timezone.utc)
        mgr1.increment_session_loss(now)
        mgr1.increment_session_loss(now)
        mgr1.record_signal("BUY", now)

        # New instance loads from disk
        mgr2 = StateManager(state_file=path)
        assert mgr2.get_session_losses() == 2
        assert mgr2.session_date == "2026-04-02"
        assert mgr2.last_signal_direction == "BUY"
