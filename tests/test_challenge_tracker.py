"""
tests/test_challenge_tracker.py
================================
Stage 12: Unit tests for ChallengeTracker compliance logic.

Tests cover:
  - Daily loss warning and breach thresholds
  - Total drawdown warning and breach thresholds
  - Trailing DD peak update and calculation
  - Session start balance reset at midnight UTC
  - Profit target detection
  - State persist/load cycle
"""

import json
import os
from datetime import datetime, timezone

import pytest

from propfirm.tracker import ChallengeTracker

INITIAL = 10_000.0
PROFILE = "FundedNext_1Step"

# FundedNext 1-Step thresholds (from config.py)
DAILY_WARN   = 2.5   # %
DAILY_LIMIT  = 3.0   # %
DD_WARN      = 5.0   # %
DD_LIMIT     = 6.0   # %
PROFIT_TARGET = 10.0  # %


def _tracker() -> ChallengeTracker:
    return ChallengeTracker(PROFILE, INITIAL)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _day(date_str: str) -> datetime:
    """Return a UTC datetime for the given YYYY-MM-DD at noon."""
    return datetime.fromisoformat(f"{date_str}T12:00:00+00:00")


# ─────────────────────────────────────────────────────────────────────────────
# Daily loss — warning threshold (2.5%)
# ─────────────────────────────────────────────────────────────────────────────

def test_daily_loss_warning_threshold():
    """2.5% daily loss triggers should_pause_trading()."""
    t = _tracker()
    # $250 loss = exactly 2.5% of $10,000
    t.update_balance(INITIAL - 250.0, _now())
    paused, reason = t.should_pause_trading()
    assert paused, "Expected pause at 2.5% daily loss"
    assert "daily" in reason.lower()


def test_daily_loss_below_warning_is_ok():
    """Daily loss just below 2.5% should NOT trigger pause."""
    t = _tracker()
    t.update_balance(INITIAL - 249.0, _now())
    paused, _ = t.should_pause_trading()
    assert not paused


# ─────────────────────────────────────────────────────────────────────────────
# Daily loss — hard breach (3.0%)
# ─────────────────────────────────────────────────────────────────────────────

def test_daily_loss_breach():
    """3.0% daily loss triggers is_breached()."""
    t = _tracker()
    # $300 = 3.0% of $10,000
    t.update_balance(INITIAL - 300.0, _now())
    breached, reason = t.is_breached()
    assert breached, "Expected breach at 3.0% daily loss"
    assert "daily" in reason.lower()


def test_daily_loss_just_below_breach_not_breached():
    """$299.99 loss should NOT trigger breach."""
    t = _tracker()
    t.update_balance(INITIAL - 299.99, _now())
    breached, _ = t.is_breached()
    assert not breached


# ─────────────────────────────────────────────────────────────────────────────
# Total DD — warning threshold (5.0%)
# ─────────────────────────────────────────────────────────────────────────────

def test_total_dd_warning():
    """5.0% trailing DD triggers should_pause_trading()."""
    t = _tracker()
    ts = _now()
    t.update_balance(10_500.0, ts)          # peak = 10,500
    t.update_balance(10_500.0 - 500.0, ts)  # DD = 500/10000 = 5.0%
    paused, reason = t.should_pause_trading()
    assert paused, "Expected pause at 5.0% total DD"
    assert any(kw in reason.lower() for kw in ("dd", "drawdown", "total"))


# ─────────────────────────────────────────────────────────────────────────────
# Total DD — hard breach (6.0%)
# ─────────────────────────────────────────────────────────────────────────────

def test_total_dd_breach():
    """6.0% trailing DD triggers is_breached()."""
    t = _tracker()
    ts = _now()
    t.update_balance(10_500.0, ts)          # peak = 10,500
    t.update_balance(10_500.0 - 600.0, ts)  # DD = 600/10000 = 6.0%
    breached, reason = t.is_breached()
    assert breached, "Expected breach at 6.0% total DD"


# ─────────────────────────────────────────────────────────────────────────────
# Trailing DD — peak update
# ─────────────────────────────────────────────────────────────────────────────

def test_trailing_dd_peak_update():
    """Peak balance updates on new highs and never decreases."""
    t = _tracker()
    ts = _now()
    t.update_balance(10_300.0, ts)
    assert t.peak_balance == 10_300.0

    t.update_balance(10_500.0, ts)
    assert t.peak_balance == 10_500.0

    # Drop — peak must not decrease
    t.update_balance(10_200.0, ts)
    assert t.peak_balance == 10_500.0


# ─────────────────────────────────────────────────────────────────────────────
# Trailing DD — calculated from peak, not initial
# ─────────────────────────────────────────────────────────────────────────────

def test_trailing_dd_calculation():
    """DD is (peak - current) / initial, not (initial - current) / initial."""
    t = _tracker()
    ts = _now()
    t.update_balance(10_500.0, ts)   # peak = 10,500
    t.update_balance(10_200.0, ts)   # drop $300 from peak

    s = t.get_status()
    # DD = (10500 - 10200) / 10000 = 3.0%
    assert abs(s["total_dd_pct"] - 3.0) < 0.01, (
        f"Expected total_dd_pct=3.0, got {s['total_dd_pct']}"
    )
    # Profit is still +2.0% (current > initial), not negative
    assert s["profit_pct"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Session reset — session_start_balance resets at midnight UTC
# ─────────────────────────────────────────────────────────────────────────────

def test_session_reset():
    """session_start_balance resets to yesterday's close at midnight UTC."""
    t = _tracker()

    day1 = _day("2026-04-02")
    t.update_balance(9_800.0, day1)           # $200 loss today
    assert t.session_start_balance == INITIAL  # set to initial on first call
    assert t.trading_day == "2026-04-02"

    day2 = _day("2026-04-03")
    t.update_balance(9_800.0, day2)            # same balance, new day
    # session_start_balance should carry yesterday's closing balance
    assert t.session_start_balance == 9_800.0, (
        f"Expected session_start=9800, got {t.session_start_balance}"
    )
    assert t.trading_day == "2026-04-03"

    # Daily loss for day2 is 0 (no loss yet today)
    s = t.get_status()
    assert s["daily_loss_pct"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Profit target
# ─────────────────────────────────────────────────────────────────────────────

def test_profit_target_met():
    """10% profit sets target_met=True in get_status()."""
    t = _tracker()
    t.update_balance(INITIAL * 1.10, _now())  # exactly $11,000
    s = t.get_status()
    assert s["target_met"] is True
    assert s["profit_progress_pct"] >= 100.0


def test_profit_target_not_met_below_ten_pct():
    """9.99% profit does NOT set target_met."""
    t = _tracker()
    t.update_balance(INITIAL * 1.0999, _now())
    s = t.get_status()
    assert s["target_met"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Persist and load
# ─────────────────────────────────────────────────────────────────────────────

def test_persist_and_load(tmp_path):
    """Tracker state survives a save/load cycle."""
    filepath = str(tmp_path / "challenge_state.json")

    t1 = _tracker()
    t1.update_balance(10_247.50, _day("2026-04-02"))
    t1.update_balance(10_100.00, _day("2026-04-02"))  # peak stays 10247.50
    t1.persist(filepath)

    assert os.path.isfile(filepath)

    t2 = ChallengeTracker(PROFILE, INITIAL)
    t2.load(filepath)

    assert abs(t2.current_balance - 10_100.00) < 0.01
    assert abs(t2.peak_balance - 10_247.50) < 0.01
    assert abs(t2.session_start_balance - INITIAL) < 0.01
    assert t2.trading_day == "2026-04-02"
    assert t2.breach_halted is False


def test_persist_and_load_breach_flag(tmp_path):
    """breach_halted flag persists correctly."""
    filepath = str(tmp_path / "challenge_state.json")

    t1 = _tracker()
    t1.update_balance(INITIAL - 300.0, _now())  # trigger breach
    t1.breach_halted = True
    t1.persist(filepath)

    t2 = ChallengeTracker(PROFILE, INITIAL)
    t2.load(filepath)
    assert t2.breach_halted is True


def test_load_profile_mismatch_ignored(tmp_path):
    """Loading a state file with wrong profile name is a no-op."""
    filepath = str(tmp_path / "wrong_profile.json")
    data = {
        "profile_name": "FTMO",
        "initial_balance": 10000.0,
        "peak_balance": 12000.0,
        "current_balance": 11000.0,
        "session_start_balance": 10000.0,
        "trading_day": "2026-04-02",
        "breach_halted": False,
    }
    with open(filepath, "w") as f:
        json.dump(data, f)

    t = _tracker()  # profile = FundedNext_1Step
    original_balance = t.current_balance
    t.load(filepath)
    # Should be unchanged since profile doesn't match
    assert t.current_balance == original_balance
