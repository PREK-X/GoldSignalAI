"""
GoldSignalAI — data/validator.py
================================
Strict data validation for OHLCV DataFrames.

Called after fetching and before processing to ensure data integrity.
Raises DataValidationError with clear messages on failure so upstream
callers can log, alert, or fall back to another source.

Validation checks:
  1. No NaN values in OHLC columns
  2. high >= low for all rows
  3. Timestamps are strictly increasing
  4. Minimum bar count threshold met
  5. Prices are positive
  6. Required columns present
"""

import logging
from typing import Optional

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTION
# ─────────────────────────────────────────────────────────────────────────────

class DataValidationError(Exception):
    """Raised when OHLCV data fails validation checks."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame) -> None:
    """Ensure required OHLCV columns are present."""
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}. "
            f"DataFrame has: {list(df.columns)}"
        )


def validate_no_nans(df: pd.DataFrame) -> None:
    """Ensure no NaN values in OHLC columns (volume NaN is tolerated)."""
    ohlc = ["open", "high", "low", "close"]
    nan_counts = df[ohlc].isnull().sum()
    has_nans = nan_counts[nan_counts > 0]
    if not has_nans.empty:
        raise DataValidationError(
            f"NaN values found in OHLC columns: {has_nans.to_dict()}"
        )


def validate_high_low(df: pd.DataFrame) -> None:
    """Ensure high >= low for every row."""
    bad = df["high"] < df["low"]
    n_bad = bad.sum()
    if n_bad > 0:
        first_bad = df[bad].index[0]
        raise DataValidationError(
            f"{n_bad} rows have high < low. "
            f"First occurrence at: {first_bad}"
        )


def validate_timestamps(df: pd.DataFrame) -> None:
    """Ensure the index is a DatetimeIndex in strictly increasing order."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"Index must be DatetimeIndex, got {type(df.index).__name__}"
        )

    diffs = df.index.to_series().diff().dropna()
    non_positive = diffs <= pd.Timedelta(0)
    n_bad = non_positive.sum()
    if n_bad > 0:
        first_bad = diffs[non_positive].index[0]
        raise DataValidationError(
            f"{n_bad} timestamps are not strictly increasing. "
            f"First violation at: {first_bad}"
        )


def validate_min_bars(df: pd.DataFrame, min_bars: Optional[int] = None) -> None:
    """Ensure the DataFrame has at least min_bars rows."""
    if min_bars is None:
        min_bars = Config.MIN_BARS_REQUIRED
    if len(df) < min_bars:
        raise DataValidationError(
            f"Insufficient data: {len(df)} bars, minimum required: {min_bars}"
        )


def validate_positive_prices(df: pd.DataFrame) -> None:
    """Ensure all OHLC prices are positive."""
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            n_bad = (df[col] <= 0).sum()
            raise DataValidationError(
                f"{n_bad} non-positive values in '{col}' column"
            )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def validate_ohlcv(
    df: pd.DataFrame,
    min_bars: Optional[int] = None,
    label: str = "",
) -> pd.DataFrame:
    """
    Run all validation checks on an OHLCV DataFrame.

    Args:
        df:       DataFrame to validate
        min_bars: Override minimum bar count (default: Config.MIN_BARS_REQUIRED)
        label:    Identifier for log messages

    Returns:
        The same DataFrame if all checks pass.

    Raises:
        DataValidationError: If any check fails, with a clear message.
    """
    tag = f"[{label}] " if label else ""

    if df is None or df.empty:
        raise DataValidationError(f"{tag}DataFrame is None or empty")

    logger.debug("%sValidating %d rows…", tag, len(df))

    validate_columns(df)
    validate_no_nans(df)
    validate_positive_prices(df)
    validate_high_low(df)
    validate_timestamps(df)
    validate_min_bars(df, min_bars)

    logger.debug("%sValidation passed: %d rows OK.", tag, len(df))
    return df
