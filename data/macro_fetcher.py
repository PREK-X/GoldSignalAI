"""
GoldSignalAI -- data/macro_fetcher.py
======================================
Fetch daily macro data (DXY, VIX, US10Y) via yfinance and cache
to a local SQLite database.

These provide *independent* features for the ML models -- unlike
the technical indicators which are derived from gold price itself,
macro context captures the broader environment that drives gold moves.

Key relationships:
  - DXY (US Dollar Index): ~-0.80 correlation with gold
  - VIX (volatility index): spikes often coincide with gold rallies
  - US10Y (10-year yield): rising yields = headwind for gold

Public API:
  fetch_and_cache_macro()         -- pull latest data from yfinance, store in SQLite
  get_macro_context(timestamp)    -- return most recent macro values for a given bar
  get_macro_series(start, end)    -- return DataFrame of daily macro values

Usage:
    from data.macro_fetcher import fetch_and_cache_macro, get_macro_context
    fetch_and_cache_macro()
    ctx = get_macro_context(some_utc_timestamp)
    # ctx = {"dxy": 104.2, "dxy_1d_return": -0.003, ...}
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------

_TICKERS = {
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "us10y": "^TNX",
}

_DB_PATH = os.path.join(Config.BASE_DIR, "database", "macro_data.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS macro_daily (
    date        TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    PRIMARY KEY (date, ticker)
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_macro_ticker_date
ON macro_daily(ticker, date)
"""


# -------------------------------------------------------------------------
# DATABASE HELPERS
# -------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db() -> None:
    conn = _get_conn()
    conn.execute(_CREATE_TABLE)
    conn.executescript(_CREATE_INDEX)
    conn.commit()
    conn.close()


def _latest_cached_date(ticker: str) -> Optional[str]:
    """Return the most recent cached date for a ticker, or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT MAX(date) as max_date FROM macro_daily WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    conn.close()
    if row and row["max_date"]:
        return row["max_date"]
    return None


def _upsert_rows(rows: list[tuple]) -> int:
    """Insert or replace rows into macro_daily. Returns count."""
    if not rows:
        return 0
    conn = _get_conn()
    conn.executemany(
        """INSERT OR REPLACE INTO macro_daily
           (date, ticker, open, high, low, close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


# -------------------------------------------------------------------------
# YFINANCE FETCH
# -------------------------------------------------------------------------

def _fetch_yf(
    yf_ticker: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV from yfinance for a single ticker.
    Returns a DataFrame with columns [open, high, low, close, volume]
    and a DatetimeIndex, or None on failure.
    """
    try:
        import yfinance as yf

        raw = yf.Ticker(yf_ticker).history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            prepost=False,
        )
        if raw is None or raw.empty:
            logger.warning("yfinance returned no data for %s", yf_ticker)
            return None

        # Normalise columns
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        keep = []
        for col in ("open", "high", "low", "close", "volume"):
            if col in raw.columns:
                keep.append(col)
        if "close" not in keep:
            logger.warning("yfinance data for %s missing 'close' column", yf_ticker)
            return None

        df = raw[keep].copy()
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = 0.0
        df = df.dropna(subset=["close"])
        df = df.sort_index()
        return df

    except Exception as exc:
        logger.error("Failed to fetch %s from yfinance: %s", yf_ticker, exc)
        return None


# -------------------------------------------------------------------------
# PUBLIC API: FETCH + CACHE
# -------------------------------------------------------------------------

def fetch_and_cache_macro(years: int = 3) -> dict[str, int]:
    """
    Fetch DXY, VIX, US10Y daily data from yfinance and cache to SQLite.

    Only fetches data newer than what's already cached (incremental).
    On first run, fetches `years` years of history.

    Returns dict mapping ticker name to number of rows upserted.
    """
    _init_db()

    now = datetime.now(timezone.utc)
    end_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    result = {}

    for name, yf_ticker in _TICKERS.items():
        latest = _latest_cached_date(name)
        if latest:
            # Fetch from 1 day before latest (overlap for safety)
            start_date = (
                datetime.strptime(latest, "%Y-%m-%d") - timedelta(days=1)
            ).strftime("%Y-%m-%d")
        else:
            start_date = (now - timedelta(days=365 * years)).strftime("%Y-%m-%d")

        logger.info("Fetching %s (%s) from %s to %s...", name, yf_ticker, start_date, end_date)
        df = _fetch_yf(yf_ticker, start_date, end_date)

        if df is None or df.empty:
            logger.warning("No data fetched for %s — skipping", name)
            result[name] = 0
            continue

        # Convert to rows for SQLite
        rows = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            rows.append((
                date_str,
                name,
                float(row.get("open", 0)),
                float(row.get("high", 0)),
                float(row.get("low", 0)),
                float(row["close"]),
                float(row.get("volume", 0)),
            ))

        count = _upsert_rows(rows)
        result[name] = count
        logger.info("Cached %d rows for %s", count, name)

    return result


# -------------------------------------------------------------------------
# PUBLIC API: GET MACRO CONTEXT FOR A TIMESTAMP
# -------------------------------------------------------------------------

def get_macro_context(timestamp: datetime) -> dict:
    """
    Return the most recent macro values for a given bar timestamp.

    Looks up the most recent daily close for DXY, VIX, US10Y that
    is on or before the given timestamp's date.

    Returns a dict with keys:
      dxy, dxy_1d_return, dxy_5d_return, dxy_trend_flag,
      vix_level, vix_1d_change, vix_regime,
      us10y_level, us10y_1d_change

    All values are None if data is unavailable.
    """
    date_str = timestamp.strftime("%Y-%m-%d") if hasattr(timestamp, "strftime") else str(timestamp)[:10]

    ctx = {
        "dxy": None, "dxy_1d_return": None, "dxy_5d_return": None, "dxy_trend_flag": None,
        "vix_level": None, "vix_1d_change": None, "vix_regime": None,
        "us10y_level": None, "us10y_1d_change": None,
    }

    try:
        conn = _get_conn()

        for name in ("dxy", "vix", "us10y"):
            # Get last 25 trading days up to and including this date
            rows = conn.execute(
                """SELECT date, close FROM macro_daily
                   WHERE ticker = ? AND date <= ?
                   ORDER BY date DESC LIMIT 25""",
                (name, date_str),
            ).fetchall()

            if not rows:
                continue

            closes = [r["close"] for r in rows]  # most recent first

            if name == "dxy":
                ctx["dxy"] = closes[0]
                if len(closes) >= 2:
                    ctx["dxy_1d_return"] = (closes[0] - closes[1]) / closes[1] if closes[1] != 0 else 0.0
                if len(closes) >= 6:
                    ctx["dxy_5d_return"] = (closes[0] - closes[5]) / closes[5] if closes[5] != 0 else 0.0
                if len(closes) >= 21:
                    ma_20 = np.mean(closes[:21])
                    ctx["dxy_trend_flag"] = 1.0 if closes[0] > ma_20 else -1.0
                elif len(closes) >= 2:
                    ctx["dxy_trend_flag"] = 1.0 if closes[0] > np.mean(closes) else -1.0

            elif name == "vix":
                ctx["vix_level"] = closes[0]
                if len(closes) >= 2:
                    ctx["vix_1d_change"] = closes[0] - closes[1]
                # Regime: low < 15, medium 15-25, high > 25
                if closes[0] < 15:
                    ctx["vix_regime"] = 0.0  # low
                elif closes[0] <= 25:
                    ctx["vix_regime"] = 1.0  # medium
                else:
                    ctx["vix_regime"] = 2.0  # high

            elif name == "us10y":
                ctx["us10y_level"] = closes[0]
                if len(closes) >= 2:
                    ctx["us10y_1d_change"] = closes[0] - closes[1]

        conn.close()

    except Exception as exc:
        logger.warning("Failed to get macro context: %s", exc)

    return ctx


# -------------------------------------------------------------------------
# PUBLIC API: GET MACRO SERIES (for backtest)
# -------------------------------------------------------------------------

def get_macro_series(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily macro values pivoted wide, indexed by date.

    Columns: dxy, vix, us10y (daily close values)
    Plus derived columns: dxy_1d_return, dxy_5d_return, dxy_trend_flag,
                          vix_1d_change, vix_regime, us10y_1d_change

    This is used by the backtest engine to efficiently look up macro
    context for each bar without repeated SQLite queries.
    """
    _init_db()

    conn = _get_conn()

    query = "SELECT date, ticker, close FROM macro_daily"
    params: list = []
    conditions = []
    if start_date:
        conditions.append("date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("date <= ?")
        params.append(end_date)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY date"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    # Build a long-form DataFrame, then pivot
    data = [(r["date"], r["ticker"], r["close"]) for r in rows]
    long_df = pd.DataFrame(data, columns=["date", "ticker", "close"])
    wide = long_df.pivot(index="date", columns="ticker", values="close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()

    # Ensure all columns exist
    for col in ("dxy", "vix", "us10y"):
        if col not in wide.columns:
            wide[col] = np.nan

    # Forward-fill gaps (weekends, holidays)
    wide = wide.ffill()

    # Derived features
    wide["dxy_1d_return"] = wide["dxy"].pct_change()
    wide["dxy_5d_return"] = wide["dxy"].pct_change(periods=5)
    wide["dxy_ma20"] = wide["dxy"].rolling(20).mean()
    wide["dxy_trend_flag"] = np.where(wide["dxy"] > wide["dxy_ma20"], 1.0, -1.0)
    wide["vix_1d_change"] = wide["vix"].diff()
    wide["vix_regime"] = np.where(
        wide["vix"] < 15, 0.0,
        np.where(wide["vix"] <= 25, 1.0, 2.0)
    )
    wide["us10y_1d_change"] = wide["us10y"].diff()

    # Drop helper column
    wide = wide.drop(columns=["dxy_ma20"], errors="ignore")

    return wide


def is_macro_data_available() -> tuple[bool, str]:
    """
    Check if macro data is cached and reasonably fresh.
    Returns (ok, detail_message).
    """
    try:
        _init_db()
        conn = _get_conn()

        counts = {}
        latest_dates = {}
        for name in ("dxy", "vix", "us10y"):
            row = conn.execute(
                "SELECT COUNT(*) as cnt, MAX(date) as latest FROM macro_daily WHERE ticker = ?",
                (name,),
            ).fetchone()
            counts[name] = row["cnt"] if row else 0
            latest_dates[name] = row["latest"] if row else None

        conn.close()

        if all(c == 0 for c in counts.values()):
            return False, "No macro data cached. Run fetch_and_cache_macro() first."

        missing = [n for n, c in counts.items() if c == 0]
        if missing:
            return False, f"Missing macro data for: {', '.join(missing)}"

        # Check freshness: latest data should be within ~5 trading days
        now = datetime.now(timezone.utc)
        for name, latest in latest_dates.items():
            if latest:
                latest_dt = datetime.strptime(latest, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                age_days = (now - latest_dt).days
                if age_days > 7:
                    return True, (
                        f"Macro data stale: {name} last updated {latest} "
                        f"({age_days} days ago). Consider running fetch_and_cache_macro()."
                    )

        detail_parts = []
        for name in ("dxy", "vix", "us10y"):
            detail_parts.append(f"{name}={counts[name]} rows (latest: {latest_dates[name]})")
        return True, "; ".join(detail_parts)

    except Exception as exc:
        return False, f"Error checking macro data: {exc}"
