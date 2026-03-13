"""
GoldSignalAI — analysis/fibonacci.py
======================================
Auto-calculates Fibonacci retracement levels from the most recent
significant swing high and swing low.

Algorithm:
  1. Find the most recent swing high and swing low within the
     lookback window using a pivot-detection approach.
  2. Determine trend direction from their relative positions:
       - Swing low occurred MORE RECENTLY than swing high → price
         has been falling → measure retracement of the DOWN move
         (bearish retracement, expecting bounce UP from key levels)
       - Swing high occurred MORE RECENTLY than swing low → price
         has been rising → measure retracement of the UP move
         (bullish retracement, expecting pullback to find support)
  3. Calculate Fibonacci price levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
  4. Detect which level (if any) the current price is near.
  5. The 61.8% level is the "golden ratio" — highest-probability entry.

Scoring rule:
  - Price within tolerance of 61.8% level → highest confidence signal
  - Price within tolerance of 38.2% or 50% → moderate signal
  - Price within tolerance of 23.6% or 78.6% → noted but low weight
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Standard Fibonacci ratios in ascending order
FIBO_RATIOS: list[float] = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]

# Proximity tolerance to call a price "at" a Fib level (in pips)
FIBO_PROXIMITY_PIPS = 8.0

# Level weights for confidence scoring (higher = more significant)
FIBO_LEVEL_WEIGHT: dict[float, float] = {
    0.236: 0.5,
    0.382: 0.8,
    0.500: 0.9,
    0.618: 1.0,   # golden ratio — highest weight
    0.786: 0.7,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FibLevel:
    """A single Fibonacci price level."""
    ratio:     float    # e.g. 0.618
    price:     float    # actual price at this level
    label:     str      # e.g. "61.8%"
    is_key:    bool     # True for 38.2%, 50%, 61.8%
    pips_from_price: float   # current distance in pips
    near:      bool     # within FIBO_PROXIMITY_PIPS


@dataclass
class FibonacciLevels:
    """
    Complete Fibonacci analysis for one timeframe.

    Attributes:
        levels:         All calculated Fib levels (0% to 100%)
        swing_high:     Price of the detected swing high
        swing_low:      Price of the detected swing low
        swing_high_idx: Bar index (from end) of the swing high
        swing_low_idx:  Bar index (from end) of the swing low
        trend:          "bullish_retracement" | "bearish_retracement" | "unknown"
        nearest_level:  The Fib level closest to current price
        at_golden:      True if price is within tolerance of 61.8%
        signal:         "bullish" | "bearish" | "neutral"
        reason:         Human-readable explanation
        current_price:  Latest close price
    """
    levels:          list[FibLevel]
    swing_high:      float
    swing_low:       float
    swing_high_idx:  int         # bars from the end (0 = last bar)
    swing_low_idx:   int
    trend:           str
    nearest_level:   Optional[FibLevel]
    at_golden:       bool
    signal:          str         # "bullish" | "bearish" | "neutral"
    reason:          str
    current_price:   float

    def get_level(self, ratio: float) -> Optional[FibLevel]:
        """Return the FibLevel for the given ratio, or None."""
        for lv in self.levels:
            if abs(lv.ratio - ratio) < 0.001:
                return lv
        return None

    def key_levels_summary(self) -> str:
        key = [lv for lv in self.levels if lv.is_key]
        parts = [f"{lv.label}={lv.price:.2f}" for lv in key]
        return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SWING HIGH / LOW DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _find_swing_points(
    df:         pd.DataFrame,
    lookback:   int = Config.FIBO_LOOKBACK,
    left:       int = 8,
    right:      int = 8,
) -> tuple[float, int, float, int]:
    """
    Find the most significant swing high and swing low within the
    last `lookback` candles.

    Uses a stricter pivot window (left=8, right=8) than S/R detection
    to catch only major turning points, not every minor wiggle.

    Returns:
        (swing_high_price, swing_high_bars_from_end,
         swing_low_price,  swing_low_bars_from_end)

        bars_from_end: 0 = last bar, 1 = second-to-last, etc.
        If no pivot found, falls back to the absolute high/low of the window.
    """
    data = df.iloc[-lookback:].copy() if len(df) > lookback else df.copy()
    n    = len(data)
    high = data["high"].values
    low  = data["low"].values

    # ── Pivot detection ───────────────────────────────────────────────────
    ph_price, ph_idx = np.nan, -1   # pivot high
    pl_price, pl_idx = np.nan, -1   # pivot low

    # Scan from most recent backward so we find the LATEST pivots first
    for i in range(n - right - 1, left - 1, -1):
        window_h = high[i - left : i + right + 1]
        window_l = low[i  - left : i + right + 1]

        if np.isnan(ph_price) and high[i] == np.max(window_h):
            ph_price = float(high[i])
            ph_idx   = n - 1 - i   # convert to bars-from-end

        if np.isnan(pl_price) and low[i] == np.min(window_l):
            pl_price = float(low[i])
            pl_idx   = n - 1 - i

        # Stop once we have both
        if not np.isnan(ph_price) and not np.isnan(pl_price):
            break

    # Fallback: use absolute high/low of the window if no pivot found
    if np.isnan(ph_price):
        abs_hi_i = int(np.argmax(high))
        ph_price = float(high[abs_hi_i])
        ph_idx   = n - 1 - abs_hi_i
        logger.debug("Fib: no pivot high found — using window maximum.")

    if np.isnan(pl_price):
        abs_lo_i = int(np.argmin(low))
        pl_price = float(low[abs_lo_i])
        pl_idx   = n - 1 - abs_lo_i
        logger.debug("Fib: no pivot low found — using window minimum.")

    return ph_price, ph_idx, pl_price, pl_idx


# ─────────────────────────────────────────────────────────────────────────────
# FIBONACCI LEVEL CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def _calc_levels(
    swing_high:    float,
    swing_low:     float,
    trend:         str,
    current_price: float,
    ratios:        list[float] = FIBO_RATIOS,
    proximity_pips: float      = FIBO_PROXIMITY_PIPS,
) -> list[FibLevel]:
    """
    Compute price for each Fibonacci ratio between swing_high and swing_low.

    For a BULLISH retracement (uptrend pulling back):
      price_at_ratio = swing_high - ratio × (swing_high - swing_low)
      → 0% = swing_high (top), 100% = swing_low (bottom)
      → 61.8% level is a support zone to BUY into

    For a BEARISH retracement (downtrend bouncing):
      price_at_ratio = swing_low + ratio × (swing_high - swing_low)
      → 0% = swing_low (bottom), 100% = swing_high (top)
      → 61.8% level is a resistance zone to SELL into
    """
    rng    = swing_high - swing_low
    levels = []

    for ratio in ratios:
        if trend == "bullish_retracement":
            # Measure down from the high (pullback finding support)
            price = swing_high - ratio * rng
        else:
            # Measure up from the low (bounce finding resistance)
            price = swing_low + ratio * rng

        pips_away   = abs(price - current_price) / Config.PIP_SIZE
        near        = pips_away <= proximity_pips
        label       = f"{ratio * 100:.1f}%"
        is_key      = ratio in (0.382, 0.500, 0.618)

        levels.append(FibLevel(
            ratio=ratio, price=price, label=label,
            is_key=is_key, pips_from_price=pips_away, near=near,
        ))

    return levels


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DERIVATION
# ─────────────────────────────────────────────────────────────────────────────

def _derive_signal(
    levels:        list[FibLevel],
    trend:         str,
    current_price: float,
    at_golden:     bool,
    nearest:       Optional[FibLevel],
) -> tuple[str, str]:
    """
    Translate Fibonacci proximity into a directional signal.

    Logic:
      Bullish retracement + price near a key Fib support → BUY signal
        (price has pulled back to a key level, expect continuation up)
      Bearish retracement + price near a key Fib resistance → SELL signal
        (price has bounced to a key level, expect continuation down)
      Otherwise → NEUTRAL

    The 61.8% level generates the strongest signal. 38.2% and 50%
    generate moderate signals. Other levels are informational only.
    """
    if nearest is None or not nearest.near:
        trend_str = "uptrend" if trend == "bullish_retracement" else "downtrend"
        if nearest:
            return "neutral", (
                f"No Fib level nearby | nearest {nearest.label} "
                f"at {nearest.price:.2f} ({nearest.pips_from_price:.0f} pips)"
            )
        return "neutral", "No Fib levels calculable"

    weight = FIBO_LEVEL_WEIGHT.get(nearest.ratio, 0.5)
    strength = "golden ratio — highest probability" if nearest.ratio == 0.618 else \
               "key level" if nearest.is_key else "minor level"

    if trend == "bullish_retracement":
        # Price pulled back to Fib support → expect bounce UP
        signal = "bullish"
        reason = (
            f"Price({current_price:.2f}) at {nearest.label} Fib support "
            f"({nearest.price:.2f}, {nearest.pips_from_price:.1f} pips) — "
            f"{strength} in uptrend retracement"
        )
    else:
        # Price bounced to Fib resistance → expect reversal DOWN
        signal = "bearish"
        reason = (
            f"Price({current_price:.2f}) at {nearest.label} Fib resistance "
            f"({nearest.price:.2f}, {nearest.pips_from_price:.1f} pips) — "
            f"{strength} in downtrend retracement"
        )

    return signal, reason


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_fibonacci(
    df:             pd.DataFrame,
    lookback:       int   = Config.FIBO_LOOKBACK,
    proximity_pips: float = FIBO_PROXIMITY_PIPS,
) -> FibonacciLevels:
    """
    Auto-calculate Fibonacci retracement levels for the most recent
    swing high/low and determine if current price is at a key level.

    Args:
        df:             Processed OHLCV DataFrame (from processor.py)
        lookback:       Number of recent candles to search for swings
        proximity_pips: Pips tolerance to consider price "at" a level

    Returns:
        FibonacciLevels with all levels, signal, and metadata.
    """
    min_needed = 20   # need at least this many bars for meaningful pivots
    if df is None or len(df) < min_needed:
        logger.warning("Fib: insufficient data (%d candles).", len(df) if df is not None else 0)
        return FibonacciLevels(
            levels=[], swing_high=0.0, swing_low=0.0,
            swing_high_idx=0, swing_low_idx=0,
            trend="unknown", nearest_level=None,
            at_golden=False, signal="neutral",
            reason="Insufficient data for Fibonacci",
            current_price=0.0,
        )

    current_price = float(df["close"].iloc[-1])

    # ── Step 1: find swing points ─────────────────────────────────────────
    sh_price, sh_idx, sl_price, sl_idx = _find_swing_points(df, lookback=lookback)

    if sh_price <= sl_price:
        logger.warning("Fib: swing_high(%.2f) <= swing_low(%.2f) — skipping.", sh_price, sl_price)
        return FibonacciLevels(
            levels=[], swing_high=sh_price, swing_low=sl_price,
            swing_high_idx=sh_idx, swing_low_idx=sl_idx,
            trend="unknown", nearest_level=None,
            at_golden=False, signal="neutral",
            reason=f"Swing high ≈ swing low (range too small)",
            current_price=current_price,
        )

    # ── Step 2: determine trend direction ────────────────────────────────
    # The swing that occurred MORE RECENTLY (smaller bars_from_end index)
    # tells us what direction the market just moved INTO.
    if sl_idx < sh_idx:
        # Swing low is more recent → price fell recently → we're in a
        # downtrend or at its low → measure bearish retracement upward
        trend = "bearish_retracement"
    else:
        # Swing high is more recent → price rose recently → we're in an
        # uptrend or near its high → measure bullish retracement downward
        trend = "bullish_retracement"

    # ── Step 3: calculate Fibonacci price levels ─────────────────────────
    levels = _calc_levels(
        swing_high=sh_price,
        swing_low=sl_price,
        trend=trend,
        current_price=current_price,
        proximity_pips=proximity_pips,
    )

    # ── Step 4: find nearest level ────────────────────────────────────────
    # Only consider the meaningful middle levels (exclude 0% and 100%)
    middle_levels = [lv for lv in levels if 0.0 < lv.ratio < 1.0]
    nearest = min(middle_levels, key=lambda lv: lv.pips_from_price) if middle_levels else None

    at_golden = any(
        lv.ratio == Config.FIBO_KEY_LEVEL and lv.near
        for lv in levels
    )

    # ── Step 5: derive signal ─────────────────────────────────────────────
    signal, reason = _derive_signal(levels, trend, current_price, at_golden, nearest)

    result = FibonacciLevels(
        levels=levels,
        swing_high=sh_price,
        swing_low=sl_price,
        swing_high_idx=sh_idx,
        swing_low_idx=sl_idx,
        trend=trend,
        nearest_level=nearest,
        at_golden=at_golden,
        signal=signal,
        reason=reason,
        current_price=current_price,
    )

    logger.info(
        "Fib [%s] swing %.2f–%.2f | nearest %s @ %.2f (%.1f pips) | %s",
        trend, sl_price, sh_price,
        nearest.label if nearest else "N/A",
        nearest.price if nearest else 0.0,
        nearest.pips_from_price if nearest else 0.0,
        signal,
    )
    return result
