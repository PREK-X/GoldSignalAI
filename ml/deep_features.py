"""
GoldSignalAI — ml/deep_features.py
====================================
Feature engineering for the CNN-BiLSTM deep learning model (Stage 7).

Builds 60-bar sliding windows of 15 normalized features for each bar.
Features are independent of technical indicator outputs — they capture
price dynamics, volatility, volume, time encoding, and macro context.

Input:  M15 OHLCV DataFrame + optional macro DataFrame
Output: 3D array (N_samples, 60, 15) + labels array

Features per bar (15 total):
  1-5:   Log returns at multiple lookbacks (1,3,5,15,30 bars)
  6:     ATR ratio (ATR7 / ATR28) — normalized volatility
  7:     Volume ratio — volume / 20-bar rolling mean
  8:     High-low ratio — (high-low)/close — bar range
  9:     Close position — (close-low)/(high-low) — bar position
  10-11: Hour encoding (sin, cos)
  12-13: Day-of-week encoding (sin, cos)
  14:    DXY 1-day return (from macro data, forward-filled)
  15:    VIX level normalized by 30-day rolling mean
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEEP_FEATURE_NAMES = [
    "ret_1", "ret_3", "ret_5", "ret_15", "ret_30",
    "atr_ratio", "volume_ratio", "high_low_ratio", "close_position",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "dxy_1d_return", "vix_level_norm",
]

N_FEATURES = len(DEEP_FEATURE_NAMES)  # 15


def build_deep_features(
    df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build per-bar feature columns for the deep model.

    Args:
        df: M15 OHLCV DataFrame with datetime index.
        macro_df: Daily macro series from get_macro_series().
                  If None, macro features are filled with 0.

    Returns:
        DataFrame with DEEP_FEATURE_NAMES columns, same index as df.
        Rows where features can't be computed are NaN.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)

    feat = pd.DataFrame(index=df.index)

    # 1-5: Log returns at multiple lookbacks
    log_close = np.log(close.replace(0, np.nan))
    for n in [1, 3, 5, 15, 30]:
        feat[f"ret_{n}"] = log_close.diff(n)

    # 6: ATR ratio (ATR7 / ATR28)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr7 = tr.ewm(span=7, adjust=False).mean()
    atr28 = tr.ewm(span=28, adjust=False).mean()
    feat["atr_ratio"] = atr7 / atr28.replace(0, np.nan)

    # 7: Volume ratio
    vol_ma20 = volume.rolling(20, min_periods=1).mean().replace(0, np.nan)
    feat["volume_ratio"] = volume / vol_ma20
    feat["volume_ratio"] = feat["volume_ratio"].clip(0, 10)  # cap outliers

    # 8: High-low ratio
    feat["high_low_ratio"] = (high - low) / close.replace(0, np.nan)

    # 9: Close position within bar (0 = at low, 1 = at high)
    bar_range = (high - low).replace(0, np.nan)
    feat["close_position"] = (close - low) / bar_range
    feat["close_position"] = feat["close_position"].clip(0, 1)

    # 10-13: Time encoding
    if hasattr(df.index, "hour"):
        hours = df.index.hour
        weekdays = df.index.weekday
    else:
        hours = pd.to_datetime(df.index).hour
        weekdays = pd.to_datetime(df.index).weekday

    feat["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    feat["day_sin"] = np.sin(2 * np.pi * weekdays / 5)
    feat["day_cos"] = np.cos(2 * np.pi * weekdays / 5)

    # 14-15: Macro features
    feat["dxy_1d_return"] = 0.0
    feat["vix_level_norm"] = 1.0  # default: at 30-day mean

    if macro_df is not None and not macro_df.empty:
        _merge_macro(feat, df, macro_df)

    return feat


def _merge_macro(feat: pd.DataFrame, df: pd.DataFrame, macro_df: pd.DataFrame) -> None:
    """Merge macro data into feat DataFrame in place via as-of date join."""
    # Normalize bar dates (timezone-naive) for merge
    bar_dates = df.index.normalize()
    if bar_dates.tz is not None:
        bar_dates = bar_dates.tz_localize(None)

    macro = macro_df.copy()
    if macro.index.tz is not None:
        macro.index = macro.index.tz_localize(None)
    macro.index = macro.index.normalize()
    macro = macro.sort_index()

    # Build a date-keyed lookup
    if "dxy_1d_return" in macro.columns:
        dxy_series = macro["dxy_1d_return"].dropna()
    else:
        dxy_series = pd.Series(dtype=float)

    if "vix" in macro.columns:
        vix_raw = macro["vix"].dropna()
        vix_ma30 = vix_raw.rolling(30, min_periods=1).mean().replace(0, np.nan)
        vix_norm = vix_raw / vix_ma30
    else:
        vix_norm = pd.Series(dtype=float)

    # For each bar, look up the most recent macro value by date
    for i, bar_date in enumerate(bar_dates):
        if len(dxy_series) > 0:
            mask = dxy_series.index <= bar_date
            if mask.any():
                feat.iloc[i, feat.columns.get_loc("dxy_1d_return")] = dxy_series.loc[mask].iloc[-1]

        if len(vix_norm) > 0:
            mask = vix_norm.index <= bar_date
            if mask.any():
                feat.iloc[i, feat.columns.get_loc("vix_level_norm")] = vix_norm.loc[mask].iloc[-1]

    # Forward fill any remaining gaps
    feat["dxy_1d_return"] = feat["dxy_1d_return"].ffill()
    feat["vix_level_norm"] = feat["vix_level_norm"].ffill()


def build_sequences(
    feat_df: pd.DataFrame,
    lookback: int = 60,
    future_bars: int = 3,
    close_series: Optional[pd.Series] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 3D sequences for the CNN-BiLSTM model.

    Args:
        feat_df: DataFrame with DEEP_FEATURE_NAMES columns.
        lookback: Number of bars per window (default 60).
        future_bars: Bars ahead for label (default 3).
        close_series: Close prices for labeling. If None, no labels.

    Returns:
        X: shape (N, lookback, N_FEATURES) — float32
        y: shape (N,) — 1 if close[t+future_bars] > close[t] else 0
        indices: shape (N,) — integer indices into feat_df for each sample
    """
    # Use only the defined feature columns in order
    cols = [c for c in DEEP_FEATURE_NAMES if c in feat_df.columns]
    values = feat_df[cols].values.astype(np.float32)
    n_bars = len(values)

    # Build labels if close_series provided
    if close_series is not None:
        close_vals = close_series.values.astype(np.float64)
    else:
        close_vals = None

    # Determine valid range: need lookback bars before and future_bars after
    start = lookback
    if close_vals is not None:
        end = n_bars - future_bars
    else:
        end = n_bars

    X_list = []
    y_list = []
    idx_list = []

    for i in range(start, end):
        window = values[i - lookback:i]
        # Skip if any NaN in window
        if np.isnan(window).any():
            continue
        X_list.append(window)
        idx_list.append(i)
        if close_vals is not None:
            label = 1.0 if close_vals[i + future_bars] > close_vals[i] else 0.0
            y_list.append(label)

    if not X_list:
        return np.empty((0, lookback, len(cols))), np.empty(0), np.empty(0, dtype=int)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32) if y_list else np.empty(0)
    indices = np.array(idx_list, dtype=int)

    return X, y, indices


def build_inference_window(
    feat_df: pd.DataFrame,
    bar_index: int,
    lookback: int = 60,
) -> Optional[np.ndarray]:
    """
    Build a single inference window at the given bar index.

    Returns:
        Array of shape (1, lookback, N_FEATURES) or None if insufficient data.
    """
    if bar_index < lookback:
        return None

    cols = [c for c in DEEP_FEATURE_NAMES if c in feat_df.columns]
    window = feat_df[cols].iloc[bar_index - lookback:bar_index].values.astype(np.float32)

    if np.isnan(window).any():
        return None

    return window[np.newaxis, :]  # shape (1, lookback, N_FEATURES)
