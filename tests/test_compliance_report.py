"""
Tests for propfirm/compliance_report.py
"""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date
from propfirm.tracker import ComplianceTracker, TradeRecord
from propfirm.compliance_report import (
    generate_daily_report,
    generate_report_data,
    generate_telegram_report,
    quick_status,
)


def _make_tracker(tmp_dir):
    state_file = os.path.join(tmp_dir, "test_state.json")
    return ComplianceTracker(state_file=state_file)


def _add_trades(tracker, wins=3, losses=1):
    today = date.today().isoformat()
    for i in range(wins):
        tracker.record_trade(TradeRecord(
            timestamp=f"T{i}", direction="BUY", entry_price=2350,
            pnl_usd=150, status="closed_tp1", date=today,
        ))
    for i in range(losses):
        tracker.record_trade(TradeRecord(
            timestamp=f"TL{i}", direction="SELL", entry_price=2360,
            pnl_usd=-80, status="closed_sl", date=today,
        ))


# ── Test 1: Daily report contains all sections ──────────────────────────────

def test_daily_report_sections():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        _add_trades(t)
        report = generate_daily_report(t)

        assert "Daily Compliance Report" in report
        assert "Account Status" in report
        assert "Challenge Progress" in report
        assert "Compliance" in report
        assert "Today Summary" in report
        assert "FundedNext" in report
        print("  ✓ Test 1 passed: Daily report has all sections")


# ── Test 2: Report shows correct balance ─────────────────────────────────────

def test_report_balance():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        _add_trades(t, wins=2, losses=0)  # +$300
        report = generate_daily_report(t)
        assert "10,300.00" in report
        print("  ✓ Test 2 passed: Report shows correct balance")


# ── Test 3: Report data dict ────────────────────────────────────────────────

def test_report_data():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        _add_trades(t, wins=3, losses=1)  # +$370
        data = generate_report_data(t)

        assert data["total_trades"] == 4
        assert data["winning_trades"] == 3
        assert data["losing_trades"] == 1
        assert data["win_rate"] == 75.0
        assert data["current_balance"] == 10_370
        assert "daily_loss" in data
        assert "drawdown" in data
        assert "progress" in data
        assert data["today_trade_count"] == 4
        print("  ✓ Test 3 passed: Report data dict is complete")


# ── Test 4: Telegram report wrapped in monospace ─────────────────────────────

def test_telegram_report():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        tg = generate_telegram_report(t)
        assert tg.startswith("```")
        assert tg.endswith("```")
        assert "Daily Compliance Report" in tg
        print("  ✓ Test 4 passed: Telegram report in monospace block")


# ── Test 5: Quick status ────────────────────────────────────────────────────

def test_quick_status():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        s = quick_status(t)
        assert "Balance=" in s
        assert "P/L=" in s
        print("  ✓ Test 5 passed: Quick status is a one-liner")


# ── Test 6: Breach warning in report ────────────────────────────────────────

def test_breach_warning():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        # Big loss to trigger daily breach
        t.record_trade(TradeRecord(
            timestamp="T", direction="SELL", entry_price=2350,
            pnl_usd=-600, status="closed_sl", date=date.today().isoformat(),
        ))
        report = generate_daily_report(t)
        assert "DAILY LOSS BREACHED" in report or "🔴" in report
        print("  ✓ Test 6 passed: Breach warning appears in report")


# ── Test 7: Empty tracker report ────────────────────────────────────────────

def test_empty_report():
    with tempfile.TemporaryDirectory() as tmp:
        t = _make_tracker(tmp)
        report = generate_daily_report(t)
        assert "10,000.00" in report
        data = generate_report_data(t)
        assert data["total_trades"] == 0
        print("  ✓ Test 7 passed: Empty tracker generates valid report")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_daily_report_sections,
        test_report_balance,
        test_report_data,
        test_telegram_report,
        test_quick_status,
        test_breach_warning,
        test_empty_report,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} FAILED: {e}")
    print(f"\n{'='*50}")
    print(f"Compliance report tests: {passed}/{len(tests)} passed")
