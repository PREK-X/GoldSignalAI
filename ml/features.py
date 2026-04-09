"""
GoldSignalAI — ml/features.py
===============================
Feature engineering for the XGBoost and Random Forest classifiers.

Builds a feature matrix from raw OHLCV data by computing:
  1. All technical indicator values (EMA, RSI, MACD, etc.)
  2. Previous N candle price changes (momentum memory)
  3. Volume ratio (current vs average)
  4. Time features (hour of day, day of week, market session)
  5. Distance from nearest support/resistance
  6. ATR (market volatility context)
  7. Candlestick body/shadow ratios

Target variable:
  Will price go UP or DOWN after Config.ML_FUTURE_CANDLES (15) bars?
  Binary: 1 = price went up, 0 = price went down

All features are returned as a single pandas DataFrame with one row
per candle. The target column is "target". NaN rows at the edges
(where indicators haven't stabilised or future price is unknown) are
dropped automatically.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMN NAMES (for reference and model inspection)
# ─────────────────────────────────────────────────────────────────────────────

# These are populated dynamically, but listing them helps with debugging.
# The actual feature list is whatever build_features() returns minus "target".


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _add_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicator values as continuous features.
    Unlike indicators.py which produces categorical signals, here we
    feed the RAW numeric values to the ML model so it can learn its
    own thresholds.
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── EMAs ──────────────────────────────────────────────────────────────
    df["ema_20"]  = close.ewm(span=Config.EMA_FAST, adjust=False).mean()
    df["ema_50"]  = close.ewm(span=Config.EMA_MID,  adjust=False).mean()
    df["ema_200"] = close.ewm(span=Config.EMA_SLOW, adjust=False).mean()

    # Normalised distances from EMAs (% of price)
    df["dist_ema20"]  = (close - df["ema_20"])  / close * 100
    df["dist_ema50"]  = (close - df["ema_50"])  / close * 100
    df["dist_ema200"] = (close - df["ema_200"]) / close * 100

    # EMA stack alignment: +1 if perfectly bullish, -1 if perfectly bearish
    df["ema_stack"] = np.where(
        (close > df["ema_20"]) & (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]),
        1.0,
        np.where(
            (close < df["ema_20"]) & (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"]),
            -1.0, 0.0
        )
    )

    # ── ADX ───────────────────────────────────────────────────────────────
    period = Config.ADX_PERIOD
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm  = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    atr_s    = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    df["adx"]      = dx.ewm(alpha=1/period, adjust=False).mean()
    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di
    df["di_diff"]  = plus_di - minus_di   # positive = bullish trend

    # ── RSI ───────────────────────────────────────────────────────────────
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/Config.RSI_PERIOD, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1/Config.RSI_PERIOD, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────────────────────────────────
    ema_fast = close.ewm(span=Config.MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=Config.MACD_SLOW, adjust=False).mean()
    df["macd_line"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=Config.MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]

    # ── Stochastic ────────────────────────────────────────────────────────
    lowest  = low.rolling(Config.STOCH_K).min()
    highest = high.rolling(Config.STOCH_K).max()
    denom   = (highest - lowest).replace(0, np.nan)
    raw_k   = 100 * (close - lowest) / denom
    df["stoch_k"] = raw_k.rolling(Config.STOCH_SMOOTH).mean()
    df["stoch_d"] = df["stoch_k"].rolling(Config.STOCH_D).mean()

    # ── CCI ───────────────────────────────────────────────────────────────
    tp      = (high + low + close) / 3
    tp_mean = tp.rolling(Config.CCI_PERIOD).mean()
    tp_mad  = tp.rolling(Config.CCI_PERIOD).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    df["cci"] = (tp - tp_mean) / (0.015 * tp_mad.replace(0, np.nan))

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_mean  = close.rolling(Config.BB_PERIOD).mean()
    bb_std   = close.rolling(Config.BB_PERIOD).std()
    bb_upper = bb_mean + Config.BB_STDDEV * bb_std
    bb_lower = bb_mean - Config.BB_STDDEV * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mean.replace(0, np.nan)

    df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_width"]    = bb_width

    # ── ATR ───────────────────────────────────────────────────────────────
    df["atr"]     = atr_s  # reuse from ADX calculation
    df["atr_pct"] = atr_s / close * 100   # normalised ATR

    # ── Ichimoku (simplified — tenkan/kijun distance) ─────────────────────
    tenkan = (high.rolling(Config.ICHIMOKU_TENKAN).max() + low.rolling(Config.ICHIMOKU_TENKAN).min()) / 2
    kijun  = (high.rolling(Config.ICHIMOKU_KIJUN).max()  + low.rolling(Config.ICHIMOKU_KIJUN).min())  / 2
    df["tk_diff"] = (tenkan - kijun) / close * 100   # TK cross as % of price

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PRICE CHANGE MEMORY
# ─────────────────────────────────────────────────────────────────────────────

def _add_price_changes(df: pd.DataFrame, n: int = Config.ML_FEATURES_CANDLE_HISTORY) -> pd.DataFrame:
    """
    Add the previous N candle returns as features, giving the model
    short-term momentum memory.

    Also adds cumulative return over the window and max drawdown.
    """
    close = df["close"]
    returns = close.pct_change()

    for i in range(1, n + 1):
        df[f"return_lag_{i}"] = returns.shift(i)

    # Cumulative return over last N candles
    df[f"return_sum_{n}"] = returns.rolling(n).sum()

    # Max single-bar move in last N candles
    df[f"return_max_{n}"] = returns.rolling(n).max()
    df[f"return_min_{n}"] = returns.rolling(n).min()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume ratio and rolling volume trend.
    """
    vol = df["volume"]
    vol_avg = vol.rolling(Config.VOLUME_LOOKBACK).mean()

    df["vol_ratio"]    = vol / vol_avg.replace(0, np.nan)
    df["vol_change"]   = vol.pct_change()
    df["vol_trend_5"]  = vol.rolling(5).mean() / vol_avg.replace(0, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# TIME FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal features — market behaviour varies by session and day.

    Gold is most active during London-NY overlap (13:00–16:00 UTC) and
    quieter during Asian session. Day of week matters (lower volume on
    Mondays/Fridays, news-heavy Wednesdays/Thursdays).

    We use sine/cosine encoding for cyclical features so 23:00 is
    close to 00:00 in feature space.
    """
    idx = df.index

    # Hour of day (0–23) as sin/cos
    hour = idx.hour + idx.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day of week (0=Mon, 4=Fri) as sin/cos
    dow = idx.dayofweek.astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # Market session flags (binary)
    h = idx.hour
    df["session_london"] = ((h >= Config.LONDON_OPEN_UTC) & (h < Config.LONDON_CLOSE_UTC)).astype(int)
    df["session_ny"]     = ((h >= Config.NEW_YORK_OPEN_UTC) & (h < Config.NEW_YORK_CLOSE_UTC)).astype(int)
    df["session_overlap"] = (df["session_london"] & df["session_ny"]).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SUPPORT/RESISTANCE DISTANCE (simplified for ML)
# ─────────────────────────────────────────────────────────────────────────────

def _add_sr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified S/R features using rolling highs/lows as proxy levels.
    Full pivot-based S/R from sr_levels.py is too slow to run per-row
    over 35k+ training rows, so we use rolling max/min as a fast proxy.
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    for window in [20, 50, 100]:
        rolling_high = high.rolling(window).max()
        rolling_low  = low.rolling(window).min()
        rng = (rolling_high - rolling_low).replace(0, np.nan)

        # Position within the range (0 = at support, 1 = at resistance)
        df[f"sr_position_{window}"] = (close - rolling_low) / rng

        # Distance from highs/lows as % of price
        df[f"dist_high_{window}"] = (rolling_high - close) / close * 100
        df[f"dist_low_{window}"]  = (close - rolling_low) / close * 100

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CANDLE SHAPE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candle body and shadow ratios as continuous ML features.
    """
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    rng = (h - l).replace(0, np.nan)

    df["body_ratio"]    = (c - o).abs() / rng    # body as % of range
    df["upper_shadow_ratio"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / rng
    df["lower_shadow_ratio"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / rng
    df["is_bullish_candle"]  = (c > o).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED ML FEATURES (volatility regime, market context, momentum)
# ─────────────────────────────────────────────────────────────────────────────

def _add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additional features to improve ML accuracy beyond random:
      - Volatility regime (ATR ratio)
      - Market session (categorical)
      - Day of week (raw)
      - Price vs VWAP
      - Consecutive candle direction
      - Distance from 20-day high/low
      - Volume vs 20-period average
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # ── 1. ATR ratio: current volatility vs recent average ──────────────
    # Tells the model if we're in a high/low volatility regime
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_20 = tr.rolling(20).mean()
    df["atr_ratio"] = atr_14 / atr_20.replace(0, np.nan)

    # ── 2. Market session as single categorical feature ─────────────────
    # Asia=0, London=1, NY=2, London-NY overlap=3
    h = df.index.hour
    london = (h >= Config.LONDON_OPEN_UTC) & (h < Config.LONDON_CLOSE_UTC)
    ny = (h >= Config.NEW_YORK_OPEN_UTC) & (h < Config.NEW_YORK_CLOSE_UTC)
    overlap = london & ny
    df["session_cat"] = np.where(
        overlap, 3,
        np.where(ny, 2, np.where(london, 1, 0))
    )

    # ── 3. Day of week (0=Mon, 4=Fri) ──────────────────────────────────
    df["day_of_week"] = df.index.dayofweek.astype(float)

    # ── 4. Price vs VWAP ───────────────────────────────────────────────
    # VWAP proxy: rolling volume-weighted average price
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * vol).rolling(20).sum()
    cum_vol = vol.rolling(20).sum().replace(0, np.nan)
    vwap = cum_tp_vol / cum_vol
    df["price_vs_vwap"] = (close - vwap) / vwap.replace(0, np.nan) * 100

    # ── 5. Consecutive candle direction count ───────────────────────────
    # How many candles in a row have been bullish/bearish
    bullish = (close > df["open"]).astype(int)
    # Count consecutive same-direction candles
    groups = (bullish != bullish.shift(1)).cumsum()
    streak = bullish.groupby(groups).cumcount() + 1
    # Positive for bullish streaks, negative for bearish
    df["candle_streak"] = np.where(bullish == 1, streak, -streak).astype(float)

    # ── 6. Distance from 20-day high/low as percentage ──────────────────
    rolling_h20 = high.rolling(20 * 4).max()  # ~20 trading periods
    rolling_l20 = low.rolling(20 * 4).min()
    df["dist_20d_high_pct"] = (rolling_h20 - close) / close * 100
    df["dist_20d_low_pct"] = (close - rolling_l20) / close * 100

    # ── 7. Volume vs 20-period average ratio ────────────────────────────
    vol_ma20 = vol.rolling(20).mean().replace(0, np.nan)
    df["vol_ratio_20"] = vol / vol_ma20

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MACRO FEATURES (DXY, VIX, US10Y — independent of gold price)
# ─────────────────────────────────────────────────────────────────────────────

def _add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily macro context (DXY, VIX, US10Y) into the feature matrix.

    These are *independent* features — not derived from gold price — which
    is critical for ML edge. DXY has ~-0.80 correlation with gold.

    The macro series is daily, so we align by date: each intraday bar
    gets the most recent daily macro values (as-of join).

    Features added:
      dxy_1d_return, dxy_5d_return, dxy_trend_flag,
      vix_level, vix_1d_change, vix_regime,
      us10y_level, us10y_1d_change
    """
    try:
        from data.macro_fetcher import get_macro_series

        # Get macro series covering the DataFrame's date range
        start = df.index.min().strftime("%Y-%m-%d") if hasattr(df.index.min(), "strftime") else None
        end = df.index.max().strftime("%Y-%m-%d") if hasattr(df.index.max(), "strftime") else None
        macro = get_macro_series(start_date=start, end_date=end)

        if macro.empty:
            logger.warning("No macro data available — macro features will be NaN")
            for col in ("dxy_1d_return", "dxy_5d_return", "dxy_trend_flag",
                         "vix_level", "vix_1d_change", "vix_regime",
                         "us10y_level", "us10y_1d_change"):
                df[col] = np.nan
            return df

        # Select only the feature columns we want
        macro_cols = [
            "dxy_1d_return", "dxy_5d_return", "dxy_trend_flag",
            "vix_1d_change", "vix_regime",
            "us10y_1d_change",
        ]
        # Use raw values for level features
        level_cols = {"vix": "vix_level", "us10y": "us10y_level"}

        # Build the macro feature DataFrame
        macro_feat = pd.DataFrame(index=macro.index)
        for col in macro_cols:
            if col in macro.columns:
                macro_feat[col] = macro[col]
        for raw_col, feat_col in level_cols.items():
            if raw_col in macro.columns:
                macro_feat[feat_col] = macro[raw_col]

        # As-of merge: for each intraday bar, get the most recent daily macro row
        # Normalise the bar index to date for merging (timezone-naive dates)
        bar_dates = df.index.normalize().tz_localize(None) if df.index.tz else df.index.normalize()
        macro_feat_daily = macro_feat.copy()
        macro_idx = macro_feat_daily.index.normalize()
        macro_feat_daily.index = macro_idx.tz_localize(None) if macro_idx.tz else macro_idx

        # Use merge_asof for efficient alignment
        idx_name = df.index.name or "datetime"
        df.index.name = idx_name          # ensure reset_index creates the right column name
        df_reset = df.reset_index()
        df_reset["_merge_date"] = bar_dates.values

        macro_reset = macro_feat_daily.reset_index()
        macro_reset = macro_reset.rename(columns={macro_reset.columns[0]: "_merge_date"})
        # Drop duplicate dates (keep last)
        macro_reset = macro_reset.drop_duplicates(subset="_merge_date", keep="last")
        macro_reset = macro_reset.sort_values("_merge_date")

        merged = pd.merge_asof(
            df_reset.sort_values("_merge_date"),
            macro_reset,
            on="_merge_date",
            direction="backward",
        )

        # Restore original index
        merged = merged.set_index(idx_name)
        merged = merged.sort_index()

        # Copy macro features back to df
        for col in macro_feat.columns:
            if col in merged.columns:
                df[col] = merged[col].values

        logger.info("Added %d macro features (DXY/VIX/US10Y)", len(macro_feat.columns))

    except ImportError:
        logger.warning("macro_fetcher not available — skipping macro features")
        for col in ("dxy_1d_return", "dxy_5d_return", "dxy_trend_flag",
                     "vix_level", "vix_1d_change", "vix_regime",
                     "us10y_level", "us10y_1d_change"):
            df[col] = np.nan
    except Exception as exc:
        logger.warning("Failed to add macro features: %s — columns will be NaN", exc)
        for col in ("dxy_1d_return", "dxy_5d_return", "dxy_trend_flag",
                     "vix_level", "vix_1d_change", "vix_regime",
                     "us10y_level", "us10y_1d_change"):
            if col not in df.columns:
                df[col] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# TARGET VARIABLE
# ─────────────────────────────────────────────────────────────────────────────

def _add_target(df: pd.DataFrame, future_candles: int = Config.ML_FUTURE_CANDLES) -> pd.DataFrame:
    """
    Binary target: will the close price be HIGHER after `future_candles` bars?

    target = 1 if close[t + future_candles] > close[t]
    target = 0 if close[t + future_candles] <= close[t]

    The last `future_candles` rows will have NaN target (unknown future)
    and must be dropped before training.
    """
    future_close = df["close"].shift(-future_candles)
    df["target"] = (future_close > df["close"]).astype(float)

    # Mark the last rows as NaN (no future data yet)
    df.loc[df.index[-future_candles:], "target"] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

# Feature columns returned by build_features (minus "target")
# This list is populated at runtime by get_feature_columns().
_FEATURE_COLUMNS: Optional[list[str]] = None


def build_features(
    df: pd.DataFrame,
    include_target: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Build the complete feature matrix from a processed OHLCV DataFrame.

    This is the single function called by trainer.py and predictor.py.
    It takes clean OHLCV data and returns a DataFrame ready for
    sklearn/xgboost with all features as columns and optionally a
    "target" column.

    Args:
        df:             Processed OHLCV DataFrame (from processor.py)
        include_target: If True, add the binary target column.
                        Set to False for live prediction (no future data).
        dropna:         If True, drop rows with any NaN.

    Returns:
        DataFrame with feature columns + optional "target" column.
        Index preserved for alignment with original data.
    """
    if df is None or len(df) < Config.EMA_SLOW + 50:
        logger.error(
            "build_features: Need at least %d rows, got %d.",
            Config.EMA_SLOW + 50, len(df) if df is not None else 0
        )
        return pd.DataFrame()

    logger.debug("Building features on %d rows…", len(df))

    # Work on a copy to avoid mutating the caller's DataFrame
    feat = df[["open", "high", "low", "close", "volume"]].copy()

    # ── Add all feature groups ────────────────────────────────────────────
    feat = _add_indicator_features(feat)
    feat = _add_price_changes(feat)
    feat = _add_volume_features(feat)
    feat = _add_time_features(feat)
    feat = _add_sr_features(feat)
    feat = _add_candle_features(feat)
    feat = _add_enhanced_features(feat)

    feat = _add_macro_features(feat)

    if include_target:
        feat = _add_target(feat)

    # ── Drop raw OHLCV columns (model shouldn't see absolute prices) ─────
    # Keep only engineered features + target
    drop_cols = ["open", "high", "low", "close", "volume"]
    feat = feat.drop(columns=drop_cols, errors="ignore")

    # ── Drop NaN rows (indicator warm-up + future target) ─────────────────
    if dropna:
        before = len(feat)
        feat = feat.dropna()
        dropped = before - len(feat)
        if dropped > 0:
            logger.debug("Dropped %d NaN rows (warm-up + future). %d rows remain.", dropped, len(feat))

    # ── Replace any remaining infinities ──────────────────────────────────
    feat = feat.replace([np.inf, -np.inf], np.nan)
    if dropna:
        feat = feat.dropna()

    logger.info("Feature matrix: %d rows × %d columns", len(feat), len(feat.columns))
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: INDEPENDENT FEATURES FOR LIGHTGBM
# These features are INDEPENDENT of the scoring engine indicators.
# They use only raw OHLCV, macro data, and statistical measures.
# ─────────────────────────────────────────────────────────────────────────────

def _hurst_exponent(series: pd.Series, window: int = 50) -> pd.Series:
    """
    Rolling Hurst exponent using rescaled range (R/S) analysis.
    H > 0.5 → trending, H < 0.5 → mean-reverting, H ≈ 0.5 → random walk.
    """
    def _hurst_single(x):
        if len(x) < 20:
            return np.nan
        n = len(x)
        mean = np.mean(x)
        deviations = np.cumsum(x - mean)
        R = np.max(deviations) - np.min(deviations)
        S = np.std(x, ddof=1)
        if S == 0 or R == 0:
            return 0.5
        # Simple R/S ratio → estimate H via log(R/S) / log(n)
        return np.log(R / S) / np.log(n)

    return series.rolling(window, min_periods=20).apply(_hurst_single, raw=True)


def _add_lgbm_independent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features that are INDEPENDENT of the scoring engine.
    No indicator outputs (RSI, MACD, EMA, etc.) — only raw OHLCV
    and statistical/structural measures.

    Feature groups:
      1. Multi-lookback returns (5)
      2. Volatility features (3)
      3. Session & temporal encoding (4)
      4. Price structure (3)
      5. Hurst exponent (1)
      6. Macro features (8) — reused from _add_macro_features

    Total: ~24 independent features + 8 macro = ~32
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    opn = df["open"]

    # ── 1. Multi-lookback returns ─────────────────────────────────────────
    returns = close.pct_change()
    for n in [5, 15, 30, 60, 120]:
        df[f"ret_{n}"] = close.pct_change(n)

    # ── 2. Volatility features ────────────────────────────────────────────
    # ATR ratio: short-term vs long-term volatility expansion
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_7 = tr.ewm(span=7, adjust=False).mean()
    atr_28 = tr.ewm(span=28, adjust=False).mean()
    df["lgbm_atr_ratio"] = atr_7 / atr_28.replace(0, np.nan)

    # Realized vol ratio: 5-bar vs 20-bar
    log_ret = np.log(close / close.shift(1))
    vol_5 = log_ret.rolling(5).std()
    vol_20 = log_ret.rolling(20).std()
    df["lgbm_vol_ratio"] = vol_5 / vol_20.replace(0, np.nan)

    # High-low range as % of close (rolling 14-bar avg)
    bar_range_pct = (high - low) / close
    df["lgbm_hl_range_pct"] = bar_range_pct.rolling(14).mean()

    # ── 3. Session & temporal encoding ────────────────────────────────────
    hour = df.index.hour + df.index.minute / 60.0
    df["lgbm_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["lgbm_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["lgbm_day_of_week"] = df.index.dayofweek.astype(float)
    # London-NY overlap: 13-17 UTC
    h = df.index.hour
    df["lgbm_is_overlap"] = ((h >= 13) & (h < 17)).astype(float)

    # ── 4. Price structure ────────────────────────────────────────────────
    rolling_high_20 = high.rolling(20).max()
    rolling_low_20 = low.rolling(20).min()
    df["lgbm_dist_to_high_20"] = (rolling_high_20 - close) / close
    df["lgbm_dist_to_low_20"] = (close - rolling_low_20) / close
    df["lgbm_close_vs_open"] = (close - opn) / opn

    # ── 5. Hurst exponent (rolling 50-bar window) ─────────────────────────
    df["lgbm_hurst_50"] = _hurst_exponent(log_ret, window=50)

    return df


def build_lgbm_features(
    df: pd.DataFrame,
    include_target: bool = True,
    dropna: bool = True,
    target_source: str = "future_price",
    trades_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build feature matrix for the LightGBM direction classifier.

    Uses ONLY independent features (no indicator outputs).
    Target variable depends on target_source:
      - "future_price": binary, 1 if close[t+15] > close[t]
      - "trade_outcome": binary, 1 if trade was profitable (from backtest CSV)

    Args:
        df:             Processed M15 OHLCV DataFrame
        include_target: If True, add target column
        dropna:         If True, drop rows with any NaN
        target_source:  "future_price" or "trade_outcome"
        trades_df:      Backtest trades DataFrame (required if target_source="trade_outcome")

    Returns:
        DataFrame with independent features + optional "lgbm_target" column.
    """
    if df is None or len(df) < 250:
        logger.error(
            "build_lgbm_features: Need at least 250 rows, got %d.",
            len(df) if df is not None else 0
        )
        return pd.DataFrame()

    logger.debug("Building LGBM independent features on %d rows…", len(df))

    # Work on a copy — keep OHLCV for feature computation, drop at end
    feat = df[["open", "high", "low", "close", "volume"]].copy()

    # ── Add independent features ──────────────────────────────────────────
    feat = _add_lgbm_independent_features(feat)

    # ── Add macro features (DXY, VIX, US10Y — independent of gold price) ─
    feat = _add_macro_features(feat)

    # ── Target variable ───────────────────────────────────────────────────
    if include_target:
        if target_source == "trade_outcome" and trades_df is not None:
            # Label from backtest trade results: 1 = profitable, 0 = not
            feat["lgbm_target"] = np.nan
            for _, trade in trades_df.iterrows():
                entry_time = pd.Timestamp(trade["entry_time"])
                if entry_time in feat.index:
                    feat.loc[entry_time, "lgbm_target"] = (
                        1.0 if float(trade["pnl_pips"]) > 0 else 0.0
                    )
            # Only keep rows that have a trade label
            if dropna:
                feat = feat.dropna(subset=["lgbm_target"])
        else:
            # Default: future price direction
            future_close = feat["close"].shift(-Config.ML_FUTURE_CANDLES)
            feat["lgbm_target"] = (future_close > feat["close"]).astype(float)
            feat.loc[feat.index[-Config.ML_FUTURE_CANDLES:], "lgbm_target"] = np.nan

    # ── Drop raw OHLCV columns ────────────────────────────────────────────
    drop_cols = ["open", "high", "low", "close", "volume"]
    feat = feat.drop(columns=drop_cols, errors="ignore")

    # ── Replace inf values (always, not just when dropping NaN) ─────────
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # ── Drop NaN rows ─────────────────────────────────────────────────────
    if dropna:
        before = len(feat)
        feat = feat.dropna()
        dropped = before - len(feat)
        if dropped > 0:
            logger.debug(
                "LGBM features: Dropped %d NaN rows. %d rows remain.",
                dropped, len(feat)
            )

    logger.info(
        "LGBM feature matrix: %d rows × %d columns",
        len(feat), len(feat.columns)
    )
    return feat


def get_lgbm_feature_columns(df_features: pd.DataFrame) -> list[str]:
    """Return feature column names for LGBM (everything except lgbm_target)."""
    return [c for c in df_features.columns if c != "lgbm_target"]


def split_lgbm_xy(df_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split LGBM feature DataFrame into X (features) and y (target)."""
    if "lgbm_target" not in df_features.columns:
        raise ValueError("DataFrame has no 'lgbm_target' column.")
    feature_cols = get_lgbm_feature_columns(df_features)
    return df_features[feature_cols], df_features["lgbm_target"]


def get_feature_columns(df_features: pd.DataFrame) -> list[str]:
    """
    Return the list of feature column names (everything except 'target').
    Used by trainer.py and predictor.py to split X and y.
    """
    return [c for c in df_features.columns if c != "target"]


def split_xy(df_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a feature DataFrame into X (features) and y (target).

    Args:
        df_features: Output of build_features() with include_target=True

    Returns:
        (X, y) where X has feature columns only, y is the binary target.

    Raises:
        ValueError: If "target" column is not present.
    """
    if "target" not in df_features.columns:
        raise ValueError("DataFrame has no 'target' column. Call build_features(include_target=True).")

    feature_cols = get_feature_columns(df_features)
    X = df_features[feature_cols]
    y = df_features["target"]
    return X, y
