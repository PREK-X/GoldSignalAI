"""
Tests for data/news_fetcher.py

Uses mock data to avoid network calls.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from data.news_fetcher import (
    NewsEvent,
    _NewsCache,
    _parse_ff_datetime,
    check_news_pause,
    fetch_news_events,
    get_upcoming_events,
)


def _make_event(title="Non-Farm Payroll", minutes_from_now=0, impact="High", currency="USD"):
    """Create a NewsEvent at a specific time relative to now."""
    now = datetime.now(timezone.utc)
    return NewsEvent(
        title=title,
        currency=currency,
        impact=impact,
        event_time=now + timedelta(minutes=minutes_from_now),
        source="test",
    )


# ── Test 1: NewsEvent properties ────────────────────────────────────────────

def test_event_properties():
    e = _make_event("Non-Farm Payroll", impact="High")
    assert e.is_high_impact
    assert e.matches_gold_keywords  # "Non-Farm Payroll" matches "NFP"... check

    e2 = _make_event("Random news", impact="Low")
    assert not e2.is_high_impact
    assert not e2.matches_gold_keywords

    e3 = _make_event("Federal Reserve Rate Decision", impact="High")
    assert e3.matches_gold_keywords
    print("  ✓ Test 1 passed: NewsEvent properties work correctly")


# ── Test 2: Parse ForexFactory datetime ──────────────────────────────────────

def test_parse_ff_datetime():
    dt = _parse_ff_datetime("03-11-2025", "8:30am")
    assert dt is not None
    # 8:30am EST = 13:30 UTC
    assert dt.hour == 13
    assert dt.minute == 30

    dt2 = _parse_ff_datetime("03-11-2025", "2:00pm")
    assert dt2 is not None
    # 2:00pm EST = 19:00 UTC
    assert dt2.hour == 19

    dt3 = _parse_ff_datetime("03-11-2025", "All Day")
    assert dt3 is not None  # defaults to noon

    dt4 = _parse_ff_datetime("invalid", "invalid")
    assert dt4 is None
    print("  ✓ Test 2 passed: FF datetime parsing works")


# ── Test 3: News pause — event happening NOW ────────────────────────────────

def test_pause_during_event():
    now = datetime.now(timezone.utc)
    events = [_make_event("NFP Release", minutes_from_now=0)]

    with patch("data.news_fetcher.fetch_news_events", return_value=events):
        paused, reason = check_news_pause(now)
        assert paused
        assert "NFP" in reason
    print("  ✓ Test 3 passed: Pause active during event window")


# ── Test 4: News pause — event 15 min away (within 30min window) ───────────

def test_pause_before_event():
    now = datetime.now(timezone.utc)
    events = [_make_event("CPI Report", minutes_from_now=15)]

    with patch("data.news_fetcher.fetch_news_events", return_value=events):
        paused, reason = check_news_pause(now)
        assert paused
        assert "CPI" in reason
    print("  ✓ Test 4 passed: Pause active 15min before event")


# ── Test 5: No pause — event 2 hours away ───────────────────────────────────

def test_no_pause_far_event():
    now = datetime.now(timezone.utc)
    events = [_make_event("FOMC Minutes", minutes_from_now=120)]

    with patch("data.news_fetcher.fetch_news_events", return_value=events):
        paused, reason = check_news_pause(now)
        assert not paused
        assert reason == ""
    print("  ✓ Test 5 passed: No pause when event is 2 hours away")


# ── Test 6: No pause — low impact event ─────────────────────────────────────

def test_no_pause_low_impact():
    now = datetime.now(timezone.utc)
    events = [_make_event("Building Permits", minutes_from_now=0, impact="Low")]

    with patch("data.news_fetcher.fetch_news_events", return_value=events):
        paused, reason = check_news_pause(now)
        assert not paused
    print("  ✓ Test 6 passed: Low impact event doesn't trigger pause")


# ── Test 7: News cache ──────────────────────────────────────────────────────

def test_cache():
    _NewsCache.events = []
    _NewsCache.last_fetch = None
    assert _NewsCache.is_stale()

    _NewsCache.update([_make_event("Test")])
    assert not _NewsCache.is_stale()
    assert len(_NewsCache.events) == 1

    # Reset for other tests
    _NewsCache.events = []
    _NewsCache.last_fetch = None
    print("  ✓ Test 7 passed: News cache works correctly")


# ── Test 8: Get upcoming events ─────────────────────────────────────────────

def test_upcoming_events():
    now = datetime.now(timezone.utc)
    events = [
        _make_event("NFP", minutes_from_now=60),      # 1h from now
        _make_event("GDP", minutes_from_now=180),      # 3h from now
        _make_event("Past Event", minutes_from_now=-60),  # already happened
        _make_event("Low Event", minutes_from_now=60, impact="Low"),  # low impact, no keyword
    ]

    with patch("data.news_fetcher.fetch_news_events", return_value=events):
        upcoming = get_upcoming_events(hours_ahead=4, now=now)
        # Should include NFP and GDP (high impact), not past or low
        titles = [e.title for e in upcoming]
        assert "NFP" in titles
        assert "GDP" in titles  # GDP matches keyword
        assert "Past Event" not in titles
    print("  ✓ Test 8 passed: Upcoming events filtered correctly")


# ── Test 9: Pause after event (within after-window) ─────────────────────────

def test_pause_after_event():
    now = datetime.now(timezone.utc)
    # Event was 20 minutes ago (within 30min after-window)
    events = [_make_event("Interest Rate Decision", minutes_from_now=-20)]

    with patch("data.news_fetcher.fetch_news_events", return_value=events):
        paused, reason = check_news_pause(now)
        assert paused
        assert "Interest Rate" in reason
    print("  ✓ Test 9 passed: Pause active 20min after event")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_event_properties,
        test_parse_ff_datetime,
        test_pause_during_event,
        test_pause_before_event,
        test_no_pause_far_event,
        test_no_pause_low_impact,
        test_cache,
        test_upcoming_events,
        test_pause_after_event,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} FAILED: {e}")
    print(f"\n{'='*50}")
    print(f"News fetcher tests: {passed}/{len(tests)} passed")
