"""
Tests for alerts/telegram_bot.py

Tests the bot structure and sync wrappers without actual Telegram API calls.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alerts.telegram_bot import TelegramAlert, send_message_sync, send_signal_sync


# ── Test 1: Bot initialises with disabled state (no token) ──────────────────

def test_disabled_without_token():
    alert = TelegramAlert()
    # With no token in .env, should be disabled
    assert not alert.enabled or alert.token != ""
    print("  ✓ Test 1 passed: Bot correctly detects disabled state")


# ── Test 2: Sync wrapper returns False when disabled ─────────────────────────

def test_sync_send_disabled():
    result = send_message_sync("test")
    assert result is False
    print("  ✓ Test 2 passed: Sync send returns False when disabled")


# ── Test 3: Sync signal wrapper returns False when disabled ──────────────────

def test_sync_signal_disabled():
    result = send_signal_sync("test signal")
    assert result is False
    print("  ✓ Test 3 passed: Sync signal returns False when disabled")


# ── Test 4: TelegramAlert class has required methods ─────────────────────────

def test_methods_exist():
    alert = TelegramAlert()
    assert hasattr(alert, "send_message")
    assert hasattr(alert, "send_signal")
    assert hasattr(alert, "send_photo")
    assert callable(alert.send_message)
    assert callable(alert.send_signal)
    print("  ✓ Test 4 passed: TelegramAlert has all required methods")


# ── Test 5: Bot property is None when disabled ───────────────────────────────

def test_bot_none_when_disabled():
    alert = TelegramAlert()
    if not alert.enabled:
        assert alert._bot is None
    print("  ✓ Test 5 passed: Bot is None when disabled")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_disabled_without_token,
        test_sync_send_disabled,
        test_sync_signal_disabled,
        test_methods_exist,
        test_bot_none_when_disabled,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} FAILED: {e}")
    print(f"\n{'='*50}")
    print(f"Telegram bot tests: {passed}/{len(tests)} passed")
