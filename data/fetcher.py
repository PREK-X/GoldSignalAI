"""
GoldSignalAI — data/fetcher.py
==============================
Responsible for fetching OHLCV candle data from two sources:

  PRIMARY:  MetaTrader5 (live XM broker connection)
  FALLBACK: yfinance   (if MT5 is unavailable or not installed)

The rest of the application calls only `get_candles()` and never
needs to know which source was used — the return format is always
an identical pandas DataFrame.

DataFrame schema returned:
    Index  : DatetimeIndex (UTC, timezone-aware)
    Columns: open, high, low, close, volume  (all float64)
"""

import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import Config
from data.polygon_fetcher import fetch_polygon_data, is_polygon_available

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MT5 IMPORT — optional (not available on Linux without Wine/MT5 gateway)
# We import lazily so the rest of the app works even without the MT5 package.
# ─────────────────────────────────────────────────────────────────────────────

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    logger.debug("MetaTrader5 package imported successfully.")
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logger.warning(
        "MetaTrader5 package not found — will use yfinance fallback. "
        "Install via: pip install MetaTrader5"
    )

# ─────────────────────────────────────────────────────────────────────────────
# MT5 TIMEFRAME MAP
# Maps our string names from Config to MT5 integer constants.
# ─────────────────────────────────────────────────────────────────────────────

_MT5_TF_MAP: dict[str, int] = {}   # populated after MT5 import succeeds

def _build_tf_map() -> None:
    """Build the timeframe mapping once MT5 is confirmed available."""
    global _MT5_TF_MAP
    if MT5_AVAILABLE and mt5 is not None:
        _MT5_TF_MAP = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
        }


# yfinance interval names that correspond to our timeframe strings
_YF_INTERVAL_MAP: dict[str, str] = {
    "M1":  "1m",
    "M5":  "5m",
    "M15": "15m",
    "M30": "30m",
    "H1":  "1h",
    "H4":  "4h",   # yfinance doesn't have 4h; will use 1h and resample
    "D1":  "1d",
}

# yfinance maximum look-back limits (in days) per interval
# Going beyond these causes yfinance to silently return less data
_YF_MAX_DAYS: dict[str, int] = {
    "1m":  7,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "1h":  730,
    "1d":  3650,
}


# ─────────────────────────────────────────────────────────────────────────────
# MT5 CONNECTION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class MT5Connection:
    """
    Manages the lifecycle of the MetaTrader5 connection.
    Keeps track of whether we're currently connected so callers
    don't attempt to use a dead connection.
    """

    _connected: bool = False

    @classmethod
    def connect(cls) -> bool:
        """
        Initialise and log in to the MT5 terminal.
        Returns True if connection is established, False otherwise.
        The terminal must already be running on the machine.
        """
        if not MT5_AVAILABLE:
            logger.debug("MT5 package not installed — skipping connect.")
            return False

        if cls._connected:
            # Verify the existing connection is still alive
            if mt5.terminal_info() is not None:
                return True
            # Connection silently dropped — reset and reconnect
            cls._connected = False
            logger.warning("MT5 connection lost — attempting reconnect.")

        try:
            # Initialize the MT5 terminal
            init_ok = mt5.initialize(
                login=Config.MT5_LOGIN,
                password=Config.MT5_PASSWORD,
                server=Config.MT5_SERVER,
            )

            if not init_ok:
                error = mt5.last_error()
                logger.error("MT5 initialize() failed: %s", error)
                return False

            # Confirm the account is accessible
            account = mt5.account_info()
            if account is None:
                logger.error("MT5 login succeeded but account_info() returned None.")
                mt5.shutdown()
                return False

            cls._connected = True
            logger.info(
                "MT5 connected — Account: %s | Server: %s | Balance: %.2f %s",
                account.login,
                account.server,
                account.balance,
                account.currency,
            )
            _build_tf_map()
            return True

        except Exception as exc:
            logger.exception("Unexpected error during MT5 connect: %s", exc)
            return False

    @classmethod
    def disconnect(cls) -> None:
        """Cleanly shut down the MT5 connection."""
        if MT5_AVAILABLE and cls._connected:
            mt5.shutdown()
            cls._connected = False
            logger.info("MT5 connection closed.")

    @classmethod
    def is_connected(cls) -> bool:
        """Return True if MT5 is currently connected and responsive."""
        if not MT5_AVAILABLE or not cls._connected:
            return False
        return mt5.terminal_info() is not None

    @classmethod
    def ensure_connected(cls) -> bool:
        """Connect if not already connected. Returns connection status."""
        if cls.is_connected():
            return True
        return cls.connect()


# ─────────────────────────────────────────────────────────────────────────────
# DATA NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names to lowercase and ensure the index is a
    UTC-aware DatetimeIndex. Called by both MT5 and yfinance paths so
    the rest of the app always sees the same schema.
    """
    # Lower-case all columns
    df.columns = [c.lower() for c in df.columns]

    # Keep only the columns we need
    needed = ["open", "high", "low", "close", "volume"]
    for col in needed:
        if col not in df.columns:
            # Volume is sometimes missing from Gold on MT5 — fill with 0
            df[col] = 0.0
    df = df[needed].copy()

    # Ensure numeric types
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with any NaN in OHLC (volume NaN → fill 0)
    df["volume"] = df["volume"].fillna(0.0)
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Ensure UTC-aware DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()
    return df


def _validate_ohlcv(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Sanity-check OHLCV data after fetching.
    Removes clearly corrupt rows (e.g. high < low, zero prices).
    Logs a warning for each removed row.
    """
    initial_len = len(df)

    # Remove rows with non-positive prices
    df = df[df["close"] > 0]
    df = df[df["open"] > 0]

    # High must be >= Low
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        logger.warning(
            "[%s] Dropped %d rows where high < low.", label, bad_hl.sum()
        )
        df = df[~bad_hl]

    # High must be >= Open and Close
    df = df[(df["high"] >= df["open"]) & (df["high"] >= df["close"])]
    # Low must be <= Open and Close
    df = df[(df["low"] <= df["open"]) & (df["low"] <= df["close"])]

    removed = initial_len - len(df)
    if removed > 0:
        logger.warning("[%s] Removed %d corrupt OHLCV rows total.", label, removed)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MT5 FETCH
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_mt5(
    symbol: str,
    timeframe: str,
    n_candles: int,
) -> Optional[pd.DataFrame]:
    """
    Fetch the last `n_candles` OHLCV bars from MT5 for `symbol`.

    Args:
        symbol:    MT5 symbol string, e.g. "XAUUSD"
        timeframe: Our string key, e.g. "M15" or "H1"
        n_candles: Number of bars to fetch (most recent first)

    Returns:
        Normalised DataFrame or None on failure.
    """
    if not MT5Connection.ensure_connected():
        logger.debug("MT5 not connected — _fetch_mt5 returning None.")
        return None

    tf_const = _MT5_TF_MAP.get(timeframe)
    if tf_const is None:
        logger.error("Unknown timeframe '%s' for MT5.", timeframe)
        return None

    try:
        # copy_rates_from_pos(symbol, timeframe, start_pos, count)
        # start_pos=0 means the most recent completed bar
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n_candles)
    except Exception as exc:
        logger.exception("MT5 copy_rates_from_pos raised: %s", exc)
        return None

    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        logger.warning(
            "MT5 returned no data for %s %s. MT5 error: %s", symbol, timeframe, err
        )
        return None

    # MT5 returns a numpy structured array — convert to DataFrame
    df = pd.DataFrame(rates)

    # MT5 timestamp column is called 'time' (Unix seconds, UTC)
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop(columns=["time"], errors="ignore")

    # MT5 uses 'tick_volume' instead of 'volume' for Forex/Gold
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})

    df = _normalise_columns(df)
    df = _validate_ohlcv(df, f"MT5:{symbol}:{timeframe}")

    logger.debug(
        "MT5 fetched %d candles for %s %s (latest: %s)",
        len(df), symbol, timeframe, df.index[-1] if len(df) > 0 else "N/A"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# YFINANCE FETCH (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_yfinance(
    symbol: str,
    timeframe: str,
    n_candles: int,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from yfinance as a fallback when MT5 is unavailable.

    Args:
        symbol:    yfinance ticker, e.g. "GC=F" (Gold futures)
        timeframe: Our string key, e.g. "M15" or "H1"
        n_candles: Approximate number of candles needed

    Returns:
        Normalised DataFrame (last n_candles rows) or None on failure.
    """
    yf_interval = _YF_INTERVAL_MAP.get(timeframe)
    if yf_interval is None:
        logger.error("No yfinance interval mapping for timeframe '%s'.", timeframe)
        return None

    # Calculate how many days back we need to cover n_candles
    minutes_per_candle = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440,
    }
    mins = minutes_per_candle.get(yf_interval, 15)
    # Add 40% buffer for weekends and market closures
    days_needed = max(2, int((n_candles * mins / (60 * 16)) * 1.4) + 1)

    # Clamp to yfinance's hard limits
    max_days = _YF_MAX_DAYS.get(yf_interval, 60)
    days_needed = min(days_needed, max_days)

    now_dt   = datetime.now(timezone.utc)
    # yfinance end is exclusive — use tomorrow so today's candles are included
    end_dt   = now_dt + timedelta(days=1)
    start_dt = now_dt - timedelta(days=days_needed)

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval=yf_interval,
            auto_adjust=True,
            prepost=False,
        )
    except Exception as exc:
        logger.exception("yfinance download raised: %s", exc)
        return None

    if df is None or df.empty:
        logger.warning(
            "yfinance returned empty data for %s %s.", symbol, timeframe
        )
        return None

    df = _normalise_columns(df)
    df = _validate_ohlcv(df, f"yfinance:{symbol}:{timeframe}")

    # Trim to exactly the last n_candles rows
    if len(df) > n_candles:
        df = df.iloc[-n_candles:]

    logger.info(
        "yfinance (fallback) fetched %d candles for %s %s (latest: %s)",
        len(df), symbol, timeframe, df.index[-1] if len(df) > 0 else "N/A"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# POLYGON FETCH (primary for historical data)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_polygon(
    symbol: str,
    timeframe: str,
    n_candles: int,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from Polygon.io.

    Args:
        symbol:    Ignored — always uses Config.POLYGON_SYMBOL for forex
        timeframe: Our string key, e.g. "M15" or "H1"
        n_candles: Approximate number of candles needed

    Returns:
        Normalised DataFrame or None on failure.
    """
    if not Config.POLYGON_API_KEY:
        logger.debug("Polygon API key not configured — skipping.")
        return None

    # Calculate date range from n_candles
    minutes_per_candle = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
                          "H1": 60, "H4": 240, "D1": 1440}
    mins = minutes_per_candle.get(timeframe, 15)
    # Gold trades ~20h/day, 5 days/week → ~100 candles/day for M15
    trading_hours_per_day = 20
    candles_per_day = int(trading_hours_per_day * 60 / mins)
    days_needed = max(2, int(n_candles / max(candles_per_day, 1) * 1.5))

    now_dt = datetime.now(timezone.utc)
    start_dt = now_dt - timedelta(days=days_needed)

    try:
        df = fetch_polygon_data(
            symbol=Config.POLYGON_SYMBOL,
            timeframe=timeframe,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=now_dt.strftime("%Y-%m-%d"),
        )
    except Exception as exc:
        logger.warning("Polygon fetch failed: %s", exc)
        return None

    if df is None or df.empty:
        return None

    df = _normalise_columns(df)
    df = _validate_ohlcv(df, f"Polygon:{timeframe}")

    # Trim to requested candle count
    if len(df) > n_candles:
        df = df.iloc[-n_candles:]

    logger.info(
        "Polygon fetched %d candles for %s (latest: %s)",
        len(df), timeframe, df.index[-1] if len(df) > 0 else "N/A"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED DATA ACCESS (with fallback hierarchy)
# ─────────────────────────────────────────────────────────────────────────────

def get_market_data(
    symbol: str = Config.SYMBOL,
    timeframe: str = Config.PRIMARY_TIMEFRAME,
    bars: int = Config.LOOKBACK_CANDLES,
) -> Optional[pd.DataFrame]:
    """
    Fetch market data using the configured fallback hierarchy:
        1. Polygon.io (primary — 2+ years of data)
        2. MT5 (if available and configured)
        3. yfinance (last resort — limited to ~60 days for M15)

    The system automatically chooses the best available source.

    Args:
        symbol:    Asset symbol (e.g. "XAUUSD")
        timeframe: "M15", "H1", etc.
        bars:      Number of candles to return

    Returns:
        DataFrame with [open, high, low, close, volume] and UTC DatetimeIndex,
        or None if all sources fail.
    """
    yf_symbol = Config.YFINANCE_SYMBOL if symbol == Config.SYMBOL else symbol
    sources_tried = []

    for source in Config.DATA_SOURCE_PRIORITY:
        if source == "polygon":
            df = _fetch_polygon(symbol, timeframe, bars)
            if df is not None and len(df) > 0:
                logger.info("Data source: Polygon (%d bars)", len(df))
                return df
            sources_tried.append("polygon")

        elif source == "mt5":
            if MT5_AVAILABLE and Config.MT5_LOGIN != 0:
                df = _fetch_mt5(symbol, timeframe, bars)
                if df is not None and len(df) > 0:
                    logger.info("Data source: MT5 (%d bars)", len(df))
                    return df
            sources_tried.append("mt5")

        elif source == "yfinance":
            df = _fetch_yfinance(yf_symbol, timeframe, bars)
            if df is not None and len(df) > 0:
                logger.info("Data source: yfinance (%d bars)", len(df))
                return df
            sources_tried.append("yfinance")

    logger.error(
        "All data sources failed for %s %s. Tried: %s",
        symbol, timeframe, sources_tried,
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL DATA FOR ML TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_historical(
    symbol: str,
    timeframe: str,
    years: int = Config.HISTORICAL_YEARS,
    use_mt5: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch a long historical dataset for ML model training.

    For M15 data, 2 years ≈ 35,000 candles (24h × 4 per hour × 365 × 2,
    minus weekends). We chunk the download to work within yfinance limits.

    Args:
        symbol:    MT5 or yfinance symbol
        timeframe: e.g. "M15"
        years:     How many years of history to fetch
        use_mt5:   Try MT5 first if True

    Returns:
        Combined DataFrame sorted by time, or None on complete failure.
    """
    logger.info("Fetching %d years of %s %s historical data…", years, symbol, timeframe)

    # ── Polygon path: best for long historical data ───────────────────────
    if Config.POLYGON_API_KEY:
        try:
            now = datetime.now(timezone.utc)
            start_dt = now - timedelta(days=365 * years)
            df = fetch_polygon_data(
                symbol=Config.POLYGON_SYMBOL,
                timeframe=timeframe,
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=now.strftime("%Y-%m-%d"),
            )
            if df is not None and len(df) > 100:
                df = _normalise_columns(df)
                df = _validate_ohlcv(df, f"Polygon_hist:{timeframe}")
                logger.info("Polygon historical: %d candles fetched.", len(df))
                return df
        except Exception as exc:
            logger.warning("Polygon historical fetch failed: %s — trying other sources.", exc)

    # ── MT5 path: can request large amounts directly ──────────────────────
    if use_mt5 and MT5Connection.ensure_connected():
        # Estimate candle count: 24h × (60/period_mins) × ~250 trading days/year
        period_map = {"M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
        mins = period_map.get(timeframe, 15)
        candles_per_day = int(24 * 60 / mins)
        n = candles_per_day * 365 * years  # Gold trades ~24/5

        df = _fetch_mt5(symbol, timeframe, n_candles=n)
        if df is not None and len(df) > 100:
            logger.info("MT5 historical: %d candles fetched.", len(df))
            return df
        logger.warning("MT5 historical fetch failed — falling back to yfinance.")

    # ── yfinance path: chunk requests to stay within 60-day limit ────────
    yf_interval = _YF_INTERVAL_MAP.get(timeframe, "15m")
    max_days     = _YF_MAX_DAYS.get(yf_interval, 60)
    chunk_days   = min(max_days - 2, 55)   # conservative chunk size
    total_days   = 365 * years

    all_chunks: list[pd.DataFrame] = []
    end_dt = datetime.now(timezone.utc)

    # Map yfinance symbol (Gold futures) for fallback
    yf_symbol = Config.YFINANCE_SYMBOL if symbol == Config.SYMBOL else symbol

    days_fetched = 0
    while days_fetched < total_days:
        start_dt = end_dt - timedelta(days=chunk_days)
        try:
            ticker = yf.Ticker(yf_symbol)
            chunk = ticker.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
            )
            if chunk is not None and not chunk.empty:
                chunk = _normalise_columns(chunk)
                chunk = _validate_ohlcv(chunk, f"yf_hist_chunk:{yf_symbol}")
                all_chunks.append(chunk)
                logger.debug(
                    "Chunk fetched: %d rows (%s → %s)",
                    len(chunk),
                    start_dt.strftime("%Y-%m-%d"),
                    end_dt.strftime("%Y-%m-%d"),
                )
        except Exception as exc:
            logger.warning("yfinance chunk failed (%s → %s): %s",
                           start_dt.date(), end_dt.date(), exc)

        days_fetched += chunk_days
        end_dt = start_dt - timedelta(days=1)

        # Respect yfinance rate limits
        time.sleep(0.5)

    if not all_chunks:
        logger.error("All historical data fetches failed for %s %s.", symbol, timeframe)
        return None

    # Concatenate, deduplicate, and sort
    df_full = pd.concat(all_chunks)
    df_full = df_full[~df_full.index.duplicated(keep="last")]
    df_full = df_full.sort_index()

    logger.info(
        "Historical fetch complete: %d candles | %s → %s",
        len(df_full),
        df_full.index[0].strftime("%Y-%m-%d"),
        df_full.index[-1].strftime("%Y-%m-%d"),
    )
    return df_full


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_candles(
    timeframe: str = Config.PRIMARY_TIMEFRAME,
    n_candles: int = Config.LOOKBACK_CANDLES,
    symbol: str = Config.SYMBOL,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> Optional[pd.DataFrame]:
    """
    Primary data-fetching function used by all analysis modules.

    Tries MT5 first; silently falls back to yfinance on any failure.
    Retries up to `retries` times before giving up.

    Args:
        timeframe:   "M15", "H1", etc.
        n_candles:   Number of bars to return
        symbol:      Asset symbol (MT5 format, e.g. "XAUUSD")
        retries:     Number of retry attempts on transient failures
        retry_delay: Seconds to wait between retries

    Returns:
        DataFrame with columns [open, high, low, close, volume] and a
        UTC DatetimeIndex, or None if all sources fail.

    Example:
        >>> df = get_candles("M15", 300)
        >>> df.tail()
    """
    for attempt in range(1, retries + 1):
        # Use the unified fallback system
        df = get_market_data(symbol, timeframe, n_candles)
        if df is not None and len(df) > 0:
            return df

        if attempt < retries:
            logger.warning(
                "Fetch attempt %d/%d failed — retrying in %.1fs…",
                attempt, retries, retry_delay,
            )
            time.sleep(retry_delay)

    logger.error(
        "All %d fetch attempts failed for %s %s.", retries, symbol, timeframe
    )
    return None


def get_candles_both_timeframes(
    symbol: str = Config.SYMBOL,
    n_candles: int = Config.LOOKBACK_CANDLES,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetch candles for both the primary (M15) and confirmation (H1) timeframes
    in a single call. Used by the multi-timeframe analysis module.

    Returns:
        (m15_df, h1_df) — either may be None if fetching failed.
    """
    logger.info("Fetching M15 + H1 candles for %s…", symbol)
    df_m15 = get_candles(Config.PRIMARY_TIMEFRAME, n_candles, symbol)
    df_h1  = get_candles(Config.CONFIRMATION_TIMEFRAME, n_candles, symbol)
    return df_m15, df_h1


def is_market_open() -> bool:
    """
    Heuristic check for whether the Gold market is likely open.
    Gold (XAU/USD) trades Sunday 23:00 UTC through Friday 22:00 UTC.

    This is used to suppress "no data" warnings during weekends.
    For a definitive check, MT5 terminal_info().trade_allowed is preferred.
    """
    now = datetime.now(timezone.utc)
    weekday = now.weekday()   # Monday=0, Sunday=6

    # Hard closed: Saturday all day
    if weekday == 5:
        return False
    # Hard closed: Sunday before 23:00 UTC
    if weekday == 6 and now.hour < 23:
        return False
    # Hard closed: Friday after 22:00 UTC
    if weekday == 4 and now.hour >= 22:
        return False

    return True


def get_data_source_status() -> dict:
    """
    Return a status dictionary showing which data source is active.
    Used by the dashboard and Telegram /status command.
    """
    mt5_ok = MT5Connection.is_connected()
    polygon_ok = bool(Config.POLYGON_API_KEY)

    if polygon_ok:
        active = "Polygon.io (primary)"
    elif mt5_ok:
        active = "MT5"
    else:
        active = "yfinance (fallback)"

    return {
        "polygon_configured": polygon_ok,
        "mt5_available":  MT5_AVAILABLE,
        "mt5_connected":  mt5_ok,
        "active_source":  active,
        "source_priority": Config.DATA_SOURCE_PRIORITY,
        "symbol":         Config.SYMBOL,
        "yf_symbol":      Config.YFINANCE_SYMBOL,
        "market_open":    is_market_open(),
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
    }
