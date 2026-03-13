"""
GoldSignalAI — data/news_fetcher.py
=====================================
Fetches high-impact economic news events and determines if the bot
should pause signal generation.

Sources (tried in order):
  1. ForexFactory RSS feed (free, no API key)
  2. NewsAPI.org (requires key, fallback)

Pause logic:
  - Fetch this week's high-impact events for USD/XAU
  - If current time is within [event_time - 30min, event_time + 30min]
    → signal generation is paused
  - Config.NEWS_PAUSE_MINUTES_BEFORE / AFTER controls the window

Used by:
  - main.py (checks before every signal cycle)
  - scheduler/tasks.py (periodic news refresh)
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NEWS EVENT TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NewsEvent:
    """A single economic news event."""
    title:      str
    currency:   str         # "USD", "EUR", etc.
    impact:     str         # "High", "Medium", "Low"
    event_time: datetime    # UTC
    source:     str = ""    # "ForexFactory" | "NewsAPI"

    @property
    def is_high_impact(self) -> bool:
        return self.impact.lower() == "high"

    @property
    def matches_gold_keywords(self) -> bool:
        """Check if this event matches Gold-relevant keywords."""
        title_lower = self.title.lower()
        return any(kw.lower() in title_lower for kw in Config.NEWS_HIGH_IMPACT_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# NEWS CACHE
# ─────────────────────────────────────────────────────────────────────────────

class _NewsCache:
    """In-memory cache for fetched news events."""
    events: list[NewsEvent] = []
    last_fetch: Optional[datetime] = None
    fetch_interval = timedelta(hours=1)  # re-fetch every hour

    @classmethod
    def is_stale(cls) -> bool:
        if cls.last_fetch is None:
            return True
        return datetime.now(timezone.utc) - cls.last_fetch > cls.fetch_interval

    @classmethod
    def update(cls, events: list[NewsEvent]) -> None:
        cls.events = events
        cls.last_fetch = datetime.now(timezone.utc)
        logger.info("News cache updated: %d events", len(events))


# ─────────────────────────────────────────────────────────────────────────────
# FOREXFACTORY RSS PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_forexfactory() -> list[NewsEvent]:
    """
    Fetch this week's economic calendar from ForexFactory RSS.

    The RSS feed returns XML with <event> elements containing:
      <title>, <country>, <impact>, <date>, <time>
    """
    events = []
    try:
        resp = requests.get(
            Config.FOREXFACTORY_RSS,
            timeout=10,
            headers={"User-Agent": "GoldSignalAI/1.0"},
        )
        resp.raise_for_status()

        root = ET.fromstring(resp.text)

        # The feed uses a simple structure: each <event> has child elements
        for event_elem in root.iter("event"):
            title   = _get_text(event_elem, "title")
            country = _get_text(event_elem, "country")
            impact  = _get_text(event_elem, "impact")
            date_s  = _get_text(event_elem, "date")
            time_s  = _get_text(event_elem, "time")

            if not title or not date_s:
                continue

            # Filter: only USD events or Gold-related
            if country not in Config.NEWS_CURRENCIES_TO_WATCH:
                continue

            # Parse datetime
            event_time = _parse_ff_datetime(date_s, time_s)
            if event_time is None:
                continue

            events.append(NewsEvent(
                title=title,
                currency=country,
                impact=impact or "Medium",
                event_time=event_time,
                source="ForexFactory",
            ))

        logger.info("ForexFactory: fetched %d relevant events", len(events))

    except requests.RequestException as exc:
        logger.warning("ForexFactory fetch failed: %s", exc)
    except ET.ParseError as exc:
        logger.warning("ForexFactory XML parse error: %s", exc)

    return events


def _get_text(parent, tag: str) -> str:
    """Safely get text content of a child element."""
    elem = parent.find(tag)
    return elem.text.strip() if elem is not None and elem.text else ""


def _est_utc_offset(dt_naive: datetime) -> timedelta:
    """
    Return the US Eastern offset for a given naive datetime.

    DST (EDT = UTC-4): Second Sunday of March → First Sunday of November.
    Standard (EST = UTC-5): otherwise.
    """
    year = dt_naive.year
    # Second Sunday of March
    mar1 = datetime(year, 3, 1)
    dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)  # 2nd Sunday
    # First Sunday of November
    nov1 = datetime(year, 11, 1)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)  # 1st Sunday

    # DST transitions at 2:00 AM local time
    dst_start = dst_start.replace(hour=2)
    dst_end = dst_end.replace(hour=2)

    if dst_start <= dt_naive < dst_end:
        return timedelta(hours=-4)  # EDT
    return timedelta(hours=-5)  # EST


def _parse_ff_datetime(date_s: str, time_s: str) -> Optional[datetime]:
    """
    Parse ForexFactory date/time strings into UTC datetime.

    Common formats:
      date: "03-11-2025" (MM-DD-YYYY)
      time: "8:30am" or "All Day" or "Tentative"

    ForexFactory uses US Eastern time (EST/EDT), so we detect DST.
    """
    if not time_s or time_s.lower() in ("all day", "tentative", ""):
        time_s = "12:00pm"  # default to noon

    try:
        dt_str = f"{date_s} {time_s}"
        dt = datetime.strptime(dt_str, "%m-%d-%Y %I:%M%p")
        offset = _est_utc_offset(dt)
        dt = dt.replace(tzinfo=timezone(offset))
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        try:
            dt = datetime.strptime(date_s, "%m-%d-%Y")
            dt = dt.replace(hour=12)
            offset = _est_utc_offset(dt)
            dt = dt.replace(tzinfo=timezone(offset))
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None


# ─────────────────────────────────────────────────────────────────────────────
# NEWSAPI.ORG FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_newsapi() -> list[NewsEvent]:
    """
    Fallback: search NewsAPI for Gold-relevant economic news.

    Requires NEWS_API_KEY in .env.
    """
    if not Config.NEWS_API_KEY:
        logger.debug("NewsAPI key not set — skipping fallback")
        return []

    events = []
    try:
        params = {
            "q": "Gold OR XAU OR Federal Reserve OR NFP OR CPI OR FOMC",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": Config.NEWS_API_KEY,
        }
        resp = requests.get(Config.NEWS_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for article in data.get("articles", []):
            title = article.get("title", "")
            published = article.get("publishedAt", "")

            # Check if title matches high-impact keywords
            is_high = any(
                kw.lower() in title.lower()
                for kw in Config.NEWS_HIGH_IMPACT_KEYWORDS
            )

            if not is_high:
                continue

            try:
                event_time = datetime.fromisoformat(
                    published.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                continue

            events.append(NewsEvent(
                title=title,
                currency="USD",
                impact="High" if is_high else "Medium",
                event_time=event_time,
                source="NewsAPI",
            ))

        logger.info("NewsAPI: fetched %d high-impact articles", len(events))

    except requests.RequestException as exc:
        logger.warning("NewsAPI fetch failed: %s", exc)

    return events


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news_events(force_refresh: bool = False) -> list[NewsEvent]:
    """
    Fetch news events (uses cache if fresh).

    Tries ForexFactory first, falls back to NewsAPI.

    Args:
        force_refresh: Bypass cache and fetch fresh data.

    Returns:
        List of NewsEvent objects.
    """
    if not force_refresh and not _NewsCache.is_stale():
        return _NewsCache.events

    events = _fetch_forexfactory()
    if not events:
        events = _fetch_newsapi()

    _NewsCache.update(events)
    return events


def check_news_pause(
    now: Optional[datetime] = None,
) -> tuple[bool, str]:
    """
    Check if signal generation should be paused due to upcoming news.

    This is the primary function called by main.py before every signal.

    Args:
        now: Current time (UTC). None = use current time.

    Returns:
        (should_pause, reason)
        reason is empty string if not paused.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    events = fetch_news_events()

    before_window = timedelta(minutes=Config.NEWS_PAUSE_MINUTES_BEFORE)
    after_window  = timedelta(minutes=Config.NEWS_PAUSE_MINUTES_AFTER)

    for event in events:
        if not event.is_high_impact and not event.matches_gold_keywords:
            continue

        window_start = event.event_time - before_window
        window_end   = event.event_time + after_window

        if window_start <= now <= window_end:
            reason = f"{event.title} ({event.currency}) at {event.event_time.strftime('%H:%M UTC')}"
            logger.info("News pause active: %s", reason)
            return True, reason

    return False, ""


def get_upcoming_events(
    hours_ahead: int = 24,
    now: Optional[datetime] = None,
) -> list[NewsEvent]:
    """
    Get high-impact events in the next N hours.
    Used by the dashboard for the "upcoming events" widget.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    cutoff = now + timedelta(hours=hours_ahead)
    events = fetch_news_events()

    return [
        e for e in events
        if e.event_time >= now
        and e.event_time <= cutoff
        and (e.is_high_impact or e.matches_gold_keywords)
    ]
