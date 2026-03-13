"""
Tests for propfirm/tracker.py
"""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date
from propfirm.tracker import ComplianceTracker, TradeRecord, TrackerState


def _make_tracker(tmp_dir):
    """Create a tracker with a temp state file."""
    state_file = os.path.join(tmp_dir, "test_state.json")
    return ComplianceTracker(state_file=state_file)


# ── Test 1: Fresh tracker has correct defaults ──────────────────────────────

def test_fresh_state():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        assert t.state.current_balance == 10_000
        assert t.state.peak_balance == 10_000
        assert t.state.total_trades == 0
        assert t.state.trading_days == 0
        assert t.win_rate == 0.0
        print("  ✓ Test 1 passed: Fresh tracker has correct defaults")


# ── Test 2: Record a winning trade ──────────────────────────────────────────

def test_winning_trade():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        trade = TradeRecord(
            timestamp="2025-03-11T14:30:00Z",
            direction="BUY",
            entry_price=2350.00,
            exit_price=2354.00,
            pnl_usd=100.00,
            pnl_pips=40.0,
            lot_size=0.25,
            status="closed_tp1",
            date=date.today().isoformat(),
        )
        t.record_trade(trade)

        assert t.state.current_balance == 10_100
        assert t.state.peak_balance == 10_100
        assert t.state.winning_trades == 1
        assert t.state.total_trades == 1
        assert t.state.daily_pnl_usd == 100.0
        assert t.win_rate == 100.0
        print("  ✓ Test 2 passed: Winning trade updates balance correctly")


# ── Test 3: Record a losing trade ───────────────────────────────────────────

def test_losing_trade():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        trade = TradeRecord(
            timestamp="2025-03-11T15:00:00Z",
            direction="SELL",
            entry_price=2350.00,
            exit_price=2352.00,
            pnl_usd=-50.00,
            pnl_pips=-20.0,
            lot_size=0.25,
            status="closed_sl",
            date=date.today().isoformat(),
        )
        t.record_trade(trade)

        assert t.state.current_balance == 9_950
        assert t.state.losing_trades == 1
        assert t.state.daily_pnl_usd == -50.0
        print("  ✓ Test 3 passed: Losing trade updates balance correctly")


# ── Test 4: State persistence (save/load) ───────────────────────────────────

def test_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        state_file = os.path.join(tmp, "state.json")

        # Create and modify
        t1 = ComplianceTracker(state_file=state_file)
        trade = TradeRecord(
            timestamp="2025-03-11T14:30:00Z",
            direction="BUY", entry_price=2350.00,
            pnl_usd=200.0, status="closed_tp1",
            date=date.today().isoformat(),
        )
        t1.record_trade(trade)
        t1.save()

        # Reload from disk
        t2 = ComplianceTracker(state_file=state_file)
        assert t2.state.current_balance == 10_200
        assert t2.state.total_trades == 1
        print("  ✓ Test 4 passed: State persists across restarts")


# ── Test 5: Trading day counting ─────────────────────────────────────────────

def test_trading_days():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        # Two trades on same day = 1 trading day
        for i in range(2):
            trade = TradeRecord(
                timestamp=f"2025-03-11T{14+i}:00:00Z",
                direction="BUY", entry_price=2350.00,
                pnl_usd=50.0, status="closed_tp1",
                date="2025-03-11",
            )
            t.record_trade(trade)
        assert t.state.trading_days == 1

        # Trade on different day = 2 trading days
        trade = TradeRecord(
            timestamp="2025-03-12T14:00:00Z",
            direction="SELL", entry_price=2360.00,
            pnl_usd=-30.0, status="closed_sl",
            date="2025-03-12",
        )
        t.record_trade(trade)
        assert t.state.trading_days == 2
        print("  ✓ Test 5 passed: Trading days counted correctly")


# ── Test 6: Compliance check — all OK ───────────────────────────────────────

def test_compliance_ok():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        daily, dd = t.check_compliance()
        assert daily.ok
        assert dd.ok
        allowed, reason = t.is_trading_allowed()
        assert allowed
        assert reason == "OK"
        print("  ✓ Test 6 passed: Fresh tracker passes all compliance checks")


# ── Test 7: Daily loss breach stops trading ──────────────────────────────────

def test_daily_loss_stops_trading():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        # Simulate big loss today
        trade = TradeRecord(
            timestamp="2025-03-11T14:00:00Z",
            direction="SELL", entry_price=2350.00,
            pnl_usd=-600.0, status="closed_sl",
            date=date.today().isoformat(),
        )
        t.record_trade(trade)

        allowed, reason = t.is_trading_allowed()
        assert not allowed
        assert "Daily loss" in reason
        print("  ✓ Test 7 passed: Daily loss breach stops trading")


# ── Test 8: Peak balance tracking ────────────────────────────────────────────

def test_peak_balance():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        # Win then lose — peak should be after the win
        win = TradeRecord(
            timestamp="2025-03-11T14:00:00Z",
            direction="BUY", entry_price=2350.00,
            pnl_usd=300.0, status="closed_tp1",
            date="2025-03-11",
        )
        t.record_trade(win)
        assert t.state.peak_balance == 10_300

        loss = TradeRecord(
            timestamp="2025-03-11T15:00:00Z",
            direction="SELL", entry_price=2360.00,
            pnl_usd=-100.0, status="closed_sl",
            date="2025-03-11",
        )
        t.record_trade(loss)
        assert t.state.peak_balance == 10_300  # unchanged
        assert t.state.current_balance == 10_200
        print("  ✓ Test 8 passed: Peak balance tracked correctly")


# ── Test 9: Challenge progress ───────────────────────────────────────────────

def test_challenge_progress():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        prog = t.get_progress()
        assert prog.progress_pct == 0.0
        assert not prog.challenge_passed
        print("  ✓ Test 9 passed: Challenge progress starts at 0%")


# ── Test 10: Win rate calculation ────────────────────────────────────────────

def test_win_rate():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        today = date.today().isoformat()
        # 3 wins, 2 losses = 60%
        for i in range(3):
            t.record_trade(TradeRecord(
                timestamp=f"T{i}", direction="BUY", entry_price=2350,
                pnl_usd=100, status="closed_tp1", date=today,
            ))
        for i in range(2):
            t.record_trade(TradeRecord(
                timestamp=f"T{3+i}", direction="SELL", entry_price=2350,
                pnl_usd=-50, status="closed_sl", date=today,
            ))
        assert t.win_rate == 60.0
        print("  ✓ Test 10 passed: Win rate = 60% (3W/2L)")


# ── Test 11: Reset clears all state ─────────────────────────────────────────

def test_reset():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        t.record_trade(TradeRecord(
            timestamp="T", direction="BUY", entry_price=2350,
            pnl_usd=500, status="closed_tp1", date=date.today().isoformat(),
        ))
        assert t.state.total_trades == 1
        t.reset()
        assert t.state.total_trades == 0
        assert t.state.current_balance == 10_000
        print("  ✓ Test 11 passed: Reset clears all state")


# ── Test 12: Summary string ─────────────────────────────────────────────────

def test_summary():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        s = t.summary()
        assert "Balance=" in s
        assert "P/L=" in s
        assert "Win=" in s
        print("  ✓ Test 12 passed: Summary string contains key metrics")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_fresh_state,
        test_winning_trade,
        test_losing_trade,
        test_persistence,
        test_trading_days,
        test_compliance_ok,
        test_daily_loss_stops_trading,
        test_peak_balance,
        test_challenge_progress,
        test_win_rate,
        test_reset,
        test_summary,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} FAILED: {e}")
    print(f"\n{'='*50}")
    print(f"Tracker tests: {passed}/{len(tests)} passed")
