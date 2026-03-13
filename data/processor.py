"""
GoldSignalAI — data/processor.py
=================================
Cleans, validates, and prepares raw OHLCV DataFrames for the
analysis pipeline. Called immediately after fetcher.py returns data
and before any indicator calculations.

Responsibilities:
  1. Remove duplicates and sort by time
  2. Detect and handle market gaps (weekends / holidays)
  3. Remove outlier candles (price spikes > N× ATR)
  4. Forward-fill short gaps (≤ 3 missing bars) — common in Gold data
  5. Add derived base columns used by multiple indicator modules:
       - returns      : percentage close-to-close return
       - log_returns  : log(close / prev_close)
       - hl_range     : high − low (raw candle range in price)
       - body         : abs(close − open)
       - upper_shadow : high − max(open, close)
       - lower_shadow : min(open, close) − low
       - is_bullish   : 1 if close > open else 0
  6. Validate the final DataFrame is safe for pandas-ta

All functions return a new DataFrame — input is never mutated.
"""

import logging
from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Maximum number of consecutive missing bars to fill forward.
# Larger gaps are left as-is so indicators don't use stale data over
# an extended period (e.g. Friday close → Sunday open = 48h gap on Gold).
MAX_FORWARD_FILL_BARS = 3

# A candle whose range (high−low) exceeds this multiple of the rolling
# ATR is considered a data spike and removed.
SPIKE_ATR_MULTIPLIER = 10.0

# Minimum number of candles required after cleaning before we trust the data.
MIN_CLEAN_CANDLES = 50

# Expected frequency strings for each timeframe (used by pd.date_range checks)
_TF_FREQ: dict[str, str] = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "M30": "30min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DEDUPLICATION & SORT
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate timestamps, keeping the last occurrence (most recent
    data from broker). Sort ascending by time.

    Why keep last: MT5 sometimes returns the same bar twice if it was
    updated mid-fetch; the later copy has the final settled price.
    """
    before = len(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    removed = before - len(df)
    if removed:
        logger.debug("Removed %d duplicate rows.", removed)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — PRICE SPIKE REMOVAL
# ─────────────────────────────────────────────────────────────────────────────

def remove_spikes(df: pd.DataFrame, multiplier: float = SPIKE_ATR_MULTIPLIER) -> pd.DataFrame:
    """
    Remove candles whose high−low range exceeds `multiplier` × rolling ATR.

    Why: Broker data sometimes contains erroneous "ghost" candles with
    ranges of hundreds of dollars — these corrupt every indicator that
    follows. Removing them is safer than capping values.

    The rolling ATR here is a simplified true-range average (not
    pandas-ta ATR, which requires clean data first).

    Args:
        df:         OHLCV DataFrame
        multiplier: Candles with range > multiplier × ATR are dropped

    Returns:
        Cleaned DataFrame
    """
    if len(df) < 20:
        return df

    # Simplified true range: max of (H-L, |H-prevC|, |L-prevC|)
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Rolling ATR over 14 periods
    rolling_atr = true_range.rolling(window=14, min_periods=5).mean()

    candle_range = df["high"] - df["low"]
    spike_mask   = candle_range > (multiplier * rolling_atr)

    # Never remove the first 14 rows (ATR not yet stable)
    spike_mask.iloc[:14] = False

    n_spikes = spike_mask.sum()
    if n_spikes:
        logger.warning("Removed %d spike candles (range > %g× ATR).", n_spikes, multiplier)
        df = df[~spike_mask].copy()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — GAP ANALYSIS & FORWARD-FILL
# ─────────────────────────────────────────────────────────────────────────────

def _expected_freq(timeframe: str) -> Optional[pd.Timedelta]:
    """Return the expected bar interval as a Timedelta, or None if unknown."""
    freq_str = _TF_FREQ.get(timeframe)
    if freq_str is None:
        return None
    try:
        return pd.tseries.frequencies.to_offset(freq_str).nanos / 1e9
    except Exception:
        # Manual fallback
        minutes_map = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
                       "H1": 60, "H4": 240, "D1": 1440}
        mins = minutes_map.get(timeframe)
        return pd.Timedelta(minutes=mins) if mins else None


def fill_gaps(
    df: pd.DataFrame,
    timeframe: str = Config.PRIMARY_TIMEFRAME,
    max_fill: int = MAX_FORWARD_FILL_BARS,
) -> pd.DataFrame:
    """
    Forward-fill short gaps in the time series caused by momentary
    data feed interruptions (not weekends or holidays).

    Logic:
      - Calculate the most common bar interval (mode of time deltas).
      - Any gap of 2–max_fill missing bars is forward-filled.
      - Gaps > max_fill bars are left as genuine market closures.
      - Weekend / Friday-close gaps are always ignored.

    Why forward-fill and not interpolate: Forward-fill replicates
    the last known candle rather than inventing synthetic prices.
    Indicators that need continuity (e.g. EMA) benefit from this;
    the introduced bias is minimal for short gaps.

    Args:
        df:        OHLCV DataFrame with UTC DatetimeIndex
        timeframe: Used to determine expected bar frequency
        max_fill:  Maximum consecutive missing bars to fill

    Returns:
        DataFrame with short gaps filled
    """
    if len(df) < 2:
        return df

    # Infer bar interval from the data itself (robust to irregular feeds)
    deltas = df.index.to_series().diff().dropna()
    if deltas.empty:
        return df

    mode_delta = deltas.mode().iloc[0]

    # Build a complete regular index between first and last bar
    try:
        full_index = pd.date_range(
            start=df.index[0],
            end=df.index[-1],
            freq=mode_delta,
            tz="UTC",
        )
    except Exception as exc:
        logger.debug("fill_gaps: could not build full index (%s) — skipping.", exc)
        return df

    # Identify missing bars
    missing = full_index.difference(df.index)
    if missing.empty:
        return df

    # Group consecutive missing bars to find gap lengths
    # Only fill gaps that are short (≤ max_fill) AND not weekend-spanning
    missing_series = pd.Series(missing, index=missing)
    gap_groups: list[list[pd.Timestamp]] = []
    current_group: list[pd.Timestamp] = []

    for ts in missing:
        if current_group and (ts - current_group[-1]) > mode_delta * 1.5:
            gap_groups.append(current_group)
            current_group = [ts]
        else:
            current_group.append(ts)
    if current_group:
        gap_groups.append(current_group)

    fill_timestamps: list[pd.Timestamp] = []
    for group in gap_groups:
        gap_len = len(group)
        # Skip large gaps (weekends = ~48h on Gold, holidays = ~24h)
        gap_hours = (group[-1] - group[0]).total_seconds() / 3600
        if gap_hours >= 20:    # Likely a weekend/holiday closure — skip
            continue
        if gap_len <= max_fill:
            fill_timestamps.extend(group)

    if not fill_timestamps:
        return df

    # Create synthetic rows for the missing timestamps via ffill
    fill_df = pd.DataFrame(index=fill_timestamps, columns=df.columns, dtype=float)
    combined = pd.concat([df, fill_df]).sort_index()
    combined = combined.ffill()          # forward fill OHLCV from preceding bar
    combined = combined.loc[df.index[0]:df.index[-1]]  # trim to original range

    n_filled = len(fill_timestamps)
    logger.debug("Forward-filled %d missing bars (gap ≤ %d bars).", n_filled, max_fill)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — DERIVED BASE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight derived columns that multiple downstream modules use.
    These are cheap to compute and avoid repeated recalculation.

    New columns added:
        returns       — % close-to-close return (e.g. 0.0025 = +0.25%)
        log_returns   — natural log of (close / prev_close)
        hl_range      — high − low  (candle price range)
        body          — abs(close − open)  (candle body size)
        upper_shadow  — distance from body top to high
        lower_shadow  — distance from body bottom to low
        is_bullish    — 1 if close > open, 0 otherwise (−1 if equal)

    Args:
        df: Clean OHLCV DataFrame

    Returns:
        DataFrame with added columns (copy, original unchanged)
    """
    df = df.copy()

    close      = df["close"]
    open_      = df["open"]
    high       = df["high"]
    low        = df["low"]
    prev_close = close.shift(1)

    df["returns"]      = close.pct_change()
    df["log_returns"]  = np.log(close / prev_close.replace(0, np.nan))
    df["hl_range"]     = high - low
    df["body"]         = (close - open_).abs()
    df["upper_shadow"]  = high - pd.concat([close, open_], axis=1).max(axis=1)
    df["lower_shadow"]  = pd.concat([close, open_], axis=1).min(axis=1) - low
    df["is_bullish"]   = np.where(close > open_, 1, np.where(close < open_, -1, 0))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FINAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_for_indicators(df: pd.DataFrame, label: str = "") -> bool:
    """
    Final gate-check before the DataFrame enters the indicator pipeline.
    Logs specific failure reasons so they're easy to diagnose.

    Checks:
      - Minimum row count
      - Required columns present
      - No NaN in core OHLCV columns
      - Index is UTC-aware DatetimeIndex
      - Prices are positive and OHLCV-logically consistent

    Returns:
        True if data is clean, False otherwise
    """
    tag = f"[{label}] " if label else ""

    if df is None or df.empty:
        logger.error("%sDataFrame is None or empty.", tag)
        return False

    if len(df) < MIN_CLEAN_CANDLES:
        logger.error(
            "%sOnly %d rows after cleaning — need at least %d.",
            tag, len(df), MIN_CLEAN_CANDLES
        )
        return False

    required = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        logger.error("%sMissing columns: %s", tag, missing_cols)
        return False

    nan_counts = df[required].isnull().sum()
    if nan_counts.any():
        logger.error("%sNaN values found: %s", tag, nan_counts[nan_counts > 0].to_dict())
        return False

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("%sIndex is not a DatetimeIndex.", tag)
        return False

    if df.index.tz is None:
        logger.error("%sIndex is not timezone-aware (expected UTC).", tag)
        return False

    if (df["close"] <= 0).any():
        logger.error("%sNon-positive close prices found.", tag)
        return False

    if (df["high"] < df["low"]).any():
        logger.error("%shigh < low found in validated data.", tag)
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def process(
    df: pd.DataFrame,
    timeframe: str = Config.PRIMARY_TIMEFRAME,
    label: str = "",
    fill_small_gaps: bool = True,
    add_features: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Run the full processing pipeline on a raw OHLCV DataFrame.

    Pipeline order:
        1. remove_duplicates
        2. remove_spikes
        3. fill_gaps          (optional, default on)
        4. add_base_features  (optional, default on)
        5. validate_for_indicators  (always runs — returns None on failure)

    Args:
        df:              Raw DataFrame from fetcher.py
        timeframe:       Used for gap detection frequency
        label:           Identifier for log messages (e.g. "M15", "H1")
        fill_small_gaps: Whether to forward-fill short gaps
        add_features:    Whether to add derived base columns

    Returns:
        Clean, validated DataFrame ready for the analysis pipeline,
        or None if the data fails final validation.
    """
    if df is None or df.empty:
        logger.error("process() received None or empty DataFrame [%s].", label)
        return None

    tag = label or timeframe

    # Step 1 — deduplicate & sort
    df = remove_duplicates(df)

    # Step 2 — spike removal
    df = remove_spikes(df)

    # Step 3 — gap filling
    if fill_small_gaps:
        df = fill_gaps(df, timeframe=timeframe)

    # Step 4 — derived base columns
    if add_features:
        df = add_base_features(df)

    # Step 5 — final validation gate
    if not validate_for_indicators(df, label=tag):
        logger.error("process() pipeline failed validation for [%s].", tag)
        return None

    logger.info(
        "Processed [%s]: %d candles | %s → %s",
        tag, len(df),
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
    )
    return df


def process_both_timeframes(
    df_m15: Optional[pd.DataFrame],
    df_h1: Optional[pd.DataFrame],
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Convenience wrapper: process M15 and H1 DataFrames together.
    Returns (processed_m15, processed_h1) — either may be None on failure.
    """
    out_m15 = process(df_m15, timeframe="M15", label="M15") if df_m15 is not None else None
    out_h1  = process(df_h1,  timeframe="H1",  label="H1")  if df_h1  is not None else None
    return out_m15, out_h1


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES USED BY DOWNSTREAM MODULES
# ─────────────────────────────────────────────────────────────────────────────

def get_latest_close(df: pd.DataFrame) -> float:
    """Return the most recent close price. Raises ValueError if df is empty."""
    if df is None or df.empty:
        raise ValueError("Cannot get latest close — DataFrame is empty.")
    return float(df["close"].iloc[-1])


def get_latest_candle(df: pd.DataFrame) -> pd.Series:
    """Return the last row of the DataFrame as a Series."""
    if df is None or df.empty:
        raise ValueError("Cannot get latest candle — DataFrame is empty.")
    return df.iloc[-1]


def trim_to_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the last `n` rows. Safe: returns full df if len < n."""
    return df.iloc[-n:] if len(df) > n else df
