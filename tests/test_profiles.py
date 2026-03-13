"""
Tests for propfirm/profiles.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PropFirmProfile, PROP_FIRM_PROFILES
from propfirm.profiles import (
    get_profile,
    get_all_profiles,
    get_profile_names,
    daily_loss_check,
    drawdown_check,
    challenge_progress,
    format_profile_card,
)


# ── Test 1: Load active profile ─────────────────────────────────────────────

def test_active_profile():
    p = get_profile()
    assert isinstance(p, PropFirmProfile)
    assert p.daily_loss_limit > 0
    assert p.max_total_drawdown > 0
    print(f"  ✓ Test 1 passed: Active profile = {p.name}")


# ── Test 2: Load specific profile by name ────────────────────────────────────

def test_specific_profile():
    p = get_profile("FTMO")
    assert p.name == "FTMO"
    assert p.daily_loss_limit == 5.0
    assert p.profit_target == 10.0
    assert p.min_trading_days == 4
    print("  ✓ Test 2 passed: FTMO profile loaded correctly")


# ── Test 3: Invalid profile raises ValueError ───────────────────────────────

def test_invalid_profile():
    try:
        get_profile("NonExistentFirm")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "NonExistentFirm" in str(e)
    print("  ✓ Test 3 passed: Invalid profile raises ValueError")


# ── Test 4: All 8 presets exist ──────────────────────────────────────────────

def test_all_presets():
    names = get_profile_names()
    expected = ["FTMO", "FundedNext_1Step", "FundedNext_2Step", "The5ers",
                "E8_Funding", "MyForexFunds", "Apex", "Custom"]
    for e in expected:
        assert e in names, f"Missing preset: {e}"
    assert len(get_all_profiles()) >= 8
    print(f"  ✓ Test 4 passed: All {len(names)} presets found")


# ── Test 5: Daily loss check — OK ───────────────────────────────────────────

def test_daily_loss_ok():
    ftmo = get_profile("FTMO")
    # Lost $200 on $10,000 account = 2% — well within 5% limit
    status = daily_loss_check(-200, 10000, ftmo)
    assert status.ok
    assert not status.warning
    assert not status.breached
    assert status.current_pct == 2.0
    assert status.headroom_pct == 3.0
    print("  ✓ Test 5 passed: Daily loss OK (2% of 5% limit)")


# ── Test 6: Daily loss check — warning ───────────────────────────────────────

def test_daily_loss_warning():
    ftmo = get_profile("FTMO")
    # Lost $420 = 4.2% — above 4% warning, below 5% limit
    status = daily_loss_check(-420, 10000, ftmo)
    assert not status.ok
    assert status.warning
    assert not status.breached
    print("  ✓ Test 6 passed: Daily loss warning at 4.2%")


# ── Test 7: Daily loss check — breached ──────────────────────────────────────

def test_daily_loss_breached():
    ftmo = get_profile("FTMO")
    # Lost $550 = 5.5% — exceeds 5% limit
    status = daily_loss_check(-550, 10000, ftmo)
    assert not status.ok
    assert status.breached
    assert "BREACHED" in status.message
    assert status.status_icon == "🔴"
    print("  ✓ Test 7 passed: Daily loss breach detected at 5.5%")


# ── Test 8: Drawdown check — OK ─────────────────────────────────────────────

def test_drawdown_ok():
    ftmo = get_profile("FTMO")
    # Peak $10,500, current $10,200 → DD = $300 = 3% of $10,000 start
    status = drawdown_check(10500, 10200, 10000, ftmo)
    assert status.ok
    assert status.current_pct == 3.0
    print("  ✓ Test 8 passed: Drawdown OK at 3%")


# ── Test 9: Drawdown check — breached ───────────────────────────────────────

def test_drawdown_breached():
    ftmo = get_profile("FTMO")
    # Peak $10,800, current $9,700 → DD = $1,100 = 11% of $10,000 start
    status = drawdown_check(10800, 9700, 10000, ftmo)
    assert status.breached
    assert "FAILED" in status.message
    print("  ✓ Test 9 passed: Drawdown breach detected at 11%")


# ── Test 10: Challenge progress — in progress ───────────────────────────────

def test_challenge_progress_ongoing():
    ftmo = get_profile("FTMO")  # target = 10%
    # $10,000 → $10,500 = 5% profit, 3 trading days (need 4)
    prog = challenge_progress(10000, 10500, 3, ftmo)
    assert prog.progress_pct == 50.0
    assert not prog.target_met
    assert not prog.days_met
    assert not prog.challenge_passed
    print("  ✓ Test 10 passed: Challenge 50% progress, not yet passed")


# ── Test 11: Challenge progress — passed ─────────────────────────────────────

def test_challenge_passed():
    ftmo = get_profile("FTMO")  # target = 10%, min 4 days
    # $10,000 → $11,200 = 12% profit, 6 trading days
    prog = challenge_progress(10000, 11200, 6, ftmo)
    assert prog.target_met
    assert prog.days_met
    assert prog.challenge_passed
    assert "PASSED" in prog.message
    print("  ✓ Test 11 passed: Challenge correctly identified as passed")


# ── Test 12: Profile card format ─────────────────────────────────────────────

def test_profile_card():
    card = format_profile_card(get_profile("FTMO"))
    assert "FTMO" in card
    assert "5.0" in card   # daily loss
    assert "10.0" in card  # drawdown
    assert "┌" in card and "┘" in card
    print("  ✓ Test 12 passed: Profile card renders correctly")


# ── Test 13: Positive PnL doesn't trigger daily loss ────────────────────────

def test_positive_pnl_no_loss():
    status = daily_loss_check(500, 10000, get_profile("FTMO"))
    assert status.ok
    assert status.current_pct == 0.0
    assert status.status_icon == "🟢"
    print("  ✓ Test 13 passed: Positive PnL = no daily loss concern")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_active_profile,
        test_specific_profile,
        test_invalid_profile,
        test_all_presets,
        test_daily_loss_ok,
        test_daily_loss_warning,
        test_daily_loss_breached,
        test_drawdown_ok,
        test_drawdown_breached,
        test_challenge_progress_ongoing,
        test_challenge_passed,
        test_profile_card,
        test_positive_pnl_no_loss,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} FAILED: {e}")
    print(f"\n{'='*50}")
    print(f"Profile tests: {passed}/{len(tests)} passed")
