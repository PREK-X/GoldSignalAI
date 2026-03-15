"""
GoldSignalAI — data/polygon_fetcher.py
======================================
Fetch historical XAUUSD OHLCV data from Polygon.io REST API.

Polygon provides 2+ years of intraday forex data, solving the
yfinance 60-day M15 limitation. This module handles:
  - Pagination (Polygon caps results per request)
  - Rate-limit safety (5 req/min on free tier)
  - Retry logic with exponential backoff
  - Clean DataFrame output matching the project schema

DataFrame schema returned:
    Index  : DatetimeIndex (UTC, timezone-aware)
    Columns: open, high, low, close, volume  (all float64)
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import requests

from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_BASE_URL = "https://api.polygon.io"

# Polygon multiplier/timespan mapping for our timeframe strings
_TF_MAP: dict[str, tuple[int, str]] = {
    "M1":  (1,  "minute"),
    "M5":  (5,  "minute"),
    "M15": (15, "minute"),
    "M30": (30, "minute"),
    "H1":  (1,  "hour"),
    "H4":  (4,  "hour"),
    "D1":  (1,  "day"),
}

# Polygon free tier: 5 requests/minute — we pace conservatively
_REQUEST_DELAY = 12.5  # seconds between requests (< 5/min)
_MAX_RETRIES = 3
_RESULTS_PER_PAGE = 50000  # Polygon max limit per request


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """Return the Polygon API key or raise with a clear message."""
    key = Config.POLYGON_API_KEY
    if not key:
        raise ValueError(
            "POLYGON_API_KEY is not set. "
            "Add it to your .env file: POLYGON_API_KEY=your_key_here\n"
            "Get a free key at: https://polygon.io"
        )
    return key


def _request_with_retry(
    url: str,
    params: dict,
    max_retries: int = _MAX_RETRIES,
) -> Optional[dict]:
    """
    Make a GET request with exponential backoff retry.

    Returns the JSON response dict, or None on total failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                # Rate limited — back off
                wait = min(60, 2 ** attempt * 5)
                logger.warning(
                    "Polygon rate limited (429). Waiting %ds before retry %d/%d…",
                    wait, attempt, max_retries,
                )
                time.sleep(wait)
                continue

            if resp.status_code == 403:
                logger.error(
                    "Polygon API key invalid or insufficient permissions (403). "
                    "Check your POLYGON_API_KEY in .env."
                )
                return None

            logger.warning(
                "Polygon returned HTTP %d on attempt %d/%d: %s",
                resp.status_code, attempt, max_retries,
                resp.text[:200],
            )

        except requests.exceptions.Timeout:
            logger.warning(
                "Polygon request timed out (attempt %d/%d).",
                attempt, max_retries,
            )
        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "Polygon connection error (attempt %d/%d): %s",
                attempt, max_retries, exc,
            )
        except Exception as exc:
            logger.exception(
                "Unexpected error fetching from Polygon (attempt %d/%d): %s",
                attempt, max_retries, exc,
            )

        if attempt < max_retries:
            wait = 2 ** attempt
            logger.debug("Retrying in %ds…", wait)
            time.sleep(wait)

    logger.error("All %d Polygon request attempts failed.", max_retries)
    return None


def _results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert Polygon aggregates results to a normalised DataFrame."""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Polygon field names: t=timestamp(ms), o=open, h=high, l=low, c=close, v=volume
    rename_map = {"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=rename_map)

    # Convert millisecond timestamp to UTC datetime index
    df.index = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.drop(columns=["time"], errors="ignore")

    # Keep only OHLCV columns
    keep = ["open", "high", "low", "close", "volume"]
    for col in keep:
        if col not in df.columns:
            df[col] = 0.0
    df = df[keep].copy()

    # Ensure numeric types
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0.0)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_index()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_polygon_data(
    symbol: str = "C:XAUUSD",
    timeframe: str = "M15",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data from Polygon.io.

    Args:
        symbol:     Polygon forex pair format, e.g. "C:XAUUSD"
        timeframe:  Our timeframe string: "M1", "M5", "M15", "M30", "H1", "H4", "D1"
        start_date: Start date as "YYYY-MM-DD" string.
                    Defaults to 2 years ago.
        end_date:   End date as "YYYY-MM-DD" string.
                    Defaults to today.

    Returns:
        DataFrame with columns [open, high, low, close, volume] and
        UTC DatetimeIndex, or None on failure.

    Example:
        >>> df = fetch_polygon_data("C:XAUUSD", "M15", "2024-01-01", "2026-03-15")
        >>> len(df)
        35000
    """
    api_key = _get_api_key()

    # Resolve timeframe
    tf_config = _TF_MAP.get(timeframe)
    if tf_config is None:
        logger.error("Unknown timeframe '%s' for Polygon. Valid: %s", timeframe, list(_TF_MAP.keys()))
        return None
    multiplier, timespan = tf_config

    # Default date range: 2 years ago → today
    now = datetime.now(timezone.utc)
    if end_date is None:
        end_date = now.strftime("%Y-%m-%d")
    if start_date is None:
        start_dt = now - timedelta(days=365 * Config.HISTORICAL_YEARS)
        start_date = start_dt.strftime("%Y-%m-%d")

    logger.info(
        "Fetching Polygon data: %s %s from %s to %s…",
        symbol, timeframe, start_date, end_date,
    )

    # Polygon endpoint for aggregates (bars)
    url = f"{_BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

    all_results: list[dict] = []
    page = 0

    while True:
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": _RESULTS_PER_PAGE,
            "apiKey": api_key,
        }

        data = _request_with_retry(url, params)
        if data is None:
            break

        results = data.get("results", [])
        if not results:
            if page == 0:
                logger.warning(
                    "Polygon returned no results for %s %s (%s → %s). "
                    "Check your subscription tier — forex may require a paid plan.",
                    symbol, timeframe, start_date, end_date,
                )
            break

        all_results.extend(results)
        page += 1

        logger.debug(
            "Polygon page %d: %d results (total so far: %d)",
            page, len(results), len(all_results),
        )

        # Check for next page URL
        next_url = data.get("next_url")
        if next_url:
            # Polygon next_url doesn't include apiKey
            url = next_url
            # Rate limit pause between pages
            time.sleep(_REQUEST_DELAY)
        else:
            break

    if not all_results:
        logger.error("Polygon fetch returned 0 results total for %s %s.", symbol, timeframe)
        return None

    df = _results_to_dataframe(all_results)

    if df.empty:
        logger.error("Polygon results could not be converted to DataFrame.")
        return None

    logger.info(
        "Polygon fetched %d candles for %s %s (%s → %s)",
        len(df), symbol, timeframe,
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
    )

    return df


def is_polygon_available() -> bool:
    """
    Quick check: is the Polygon API key configured and does a basic
    request succeed? Used by the fallback system to decide source priority.
    """
    try:
        key = _get_api_key()
    except ValueError:
        return False

    # Test with a lightweight request
    url = f"{_BASE_URL}/v2/aggs/ticker/C:XAUUSD/prev"
    params = {"apiKey": key}

    try:
        resp = requests.get(url, params=params, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False
