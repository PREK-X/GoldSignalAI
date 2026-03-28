"""
GoldSignalAI — analysis/sr_levels.py
======================================
Auto-detects support and resistance zones from the last N candles
using pivot point clustering.

Algorithm:
  1. Find pivot highs (local maxima) and pivot lows (local minima)
     using a configurable left/right neighbour window.
  2. Collect all pivot prices into a pool.
  3. Cluster prices that are within SR_TOLERANCE_PIPS of each other
     into a single zone (take the mean as the zone price).
  4. Count how many pivots touched each zone — zones touched 3+ times
     are "strong" zones.
  5. Classify each zone as Support (below current price) or
     Resistance (above current price).
  6. Return the nearest support, nearest resistance, and their
     distance from current price (in pips and %).

Scoring hook used by scoring.py:
  - Price within 5 pips of a strong support zone  → +1 BUY signal
  - Price within 5 pips of a strong resistance zone → +1 SELL signal
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SRZone:
    """A single support or resistance price zone."""
    price:    float          # Zone centre price
    zone_type: str           # "support" or "resistance"
    strength:  int           # Number of pivot touches (higher = stronger)
    strong:    bool          # True if strength >= SR_MIN_BOUNCES
    pips_from_price: float   # Distance from current price in pips
    pct_from_price:  float   # Distance as % of current price


@dataclass
class SRLevels:
    """
    Complete support/resistance analysis for one timeframe.
    Consumed by scoring.py and risk_manager.py (SL placement).
    """
    zones:              list[SRZone]      # All detected zones, sorted by price
    nearest_support:    Optional[SRZone]  # Closest strong zone below price
    nearest_resistance: Optional[SRZone]  # Closest strong zone above price
    current_price:      float
    at_support:         bool              # Price within tolerance of support
    at_resistance:      bool              # Price within tolerance of resistance

    def all_support_prices(self) -> list[float]:
        return [z.price for z in self.zones if z.zone_type == "support"]

    def all_resistance_prices(self) -> list[float]:
        return [z.price for z in self.zones if z.zone_type == "resistance"]

    def strong_support_prices(self) -> list[float]:
        return [z.price for z in self.zones if z.zone_type == "support" and z.strong]

    def strong_resistance_prices(self) -> list[float]:
        return [z.price for z in self.zones if z.zone_type == "resistance" and z.strong]

    def summary(self) -> str:
        ns  = self.nearest_support
        nr  = self.nearest_resistance
        s_str = f"{ns.price:.2f} ({ns.pips_from_price:.1f} pips, str={ns.strength})" if ns else "None"
        r_str = f"{nr.price:.2f} ({nr.pips_from_price:.1f} pips, str={nr.strength})" if nr else "None"
        return (
            f"Support: {s_str} | Resistance: {r_str} | "
            f"At support: {self.at_support} | At resistance: {self.at_resistance}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PIVOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _find_pivots(
    df: pd.DataFrame,
    left:  int = 5,
    right: int = 5,
) -> tuple[list[float], list[float]]:
    """
    Find pivot highs and pivot lows using a symmetric neighbourhood window.

    A bar at index i is a pivot high if its high is the maximum of the
    window [i-left … i+right]. Similarly for pivot low with minimum.

    A larger window (e.g. left=10, right=10) finds fewer but more
    significant pivots. left=5, right=5 balances sensitivity with noise.

    Args:
        df:    OHLCV DataFrame
        left:  Number of bars to the left of the pivot candidate
        right: Number of bars to the right of the pivot candidate

    Returns:
        (pivot_highs, pivot_lows) — lists of price levels
    """
    high  = df["high"].values
    low   = df["low"].values
    n     = len(high)

    pivot_highs: list[float] = []
    pivot_lows:  list[float] = []

    for i in range(left, n - right):
        window_h = high[i - left : i + right + 1]
        window_l = low[i  - left : i + right + 1]

        if high[i] == np.max(window_h):
            pivot_highs.append(float(high[i]))
        if low[i] == np.min(window_l):
            pivot_lows.append(float(low[i]))

    return pivot_highs, pivot_lows


# ─────────────────────────────────────────────────────────────────────────────
# ZONE CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_pivots(
    prices:     list[float],
    tolerance:  float,
) -> list[tuple[float, int]]:
    """
    Merge pivot prices that are within `tolerance` of each other into zones.
    Returns a list of (zone_price, touch_count) tuples.

    Algorithm (single-pass greedy):
      - Sort all prices ascending.
      - Scan left to right; if the next price is within `tolerance` of the
        current cluster's running mean, add it to the cluster.
      - When a price exceeds the tolerance, finalise the cluster and start
        a new one.
      - Zone price = mean of all prices in the cluster.

    This is O(n log n) due to the sort and simpler than DBSCAN for this
    one-dimensional case.

    Args:
        prices:    List of raw pivot price values
        tolerance: Max price distance to merge two pivots (in price units)

    Returns:
        List of (zone_price, count) sorted by price
    """
    if not prices:
        return []

    sorted_prices = sorted(prices)
    clusters: list[list[float]] = []
    current: list[float] = [sorted_prices[0]]

    for price in sorted_prices[1:]:
        cluster_mean = float(np.mean(current))
        if abs(price - cluster_mean) <= tolerance:
            current.append(price)
        else:
            clusters.append(current)
            current = [price]
    clusters.append(current)

    return [(float(np.mean(c)), len(c)) for c in clusters]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN S/R DETECTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _resample_to_h4(df: pd.DataFrame, m15_bars_per_h4: int = 16) -> pd.DataFrame:
    """
    Resample M15 data to H4 equivalent by taking every Nth bar group.
    Returns a DataFrame with OHLCV columns at H4 resolution.
    """
    n = len(df)
    if n < m15_bars_per_h4:
        return df.copy()

    # Group into chunks of m15_bars_per_h4 bars
    # Trim the beginning so the last group aligns with the most recent bar
    trim = n % m15_bars_per_h4
    trimmed = df.iloc[trim:].copy()
    groups = len(trimmed) // m15_bars_per_h4

    records = []
    for i in range(groups):
        chunk = trimmed.iloc[i * m15_bars_per_h4 : (i + 1) * m15_bars_per_h4]
        records.append({
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum() if "volume" in chunk.columns else 0,
        })

    return pd.DataFrame(records)


def _daily_pivot_zones(
    df: pd.DataFrame,
    current_price: float,
    tolerance_pips: float,
) -> list[SRZone]:
    """
    Calculate daily pivot points from the most recent complete daily bar.

    Pivot = (High + Low + Close) / 3
    R1 = 2 × Pivot − Low
    S1 = 2 × Pivot − High

    For M15 data, a "day" ≈ 96 bars. We use the last complete 96-bar block.
    """
    bars_per_day = 96  # 24h × 4 bars/hour
    if len(df) < bars_per_day + 1:
        return []

    # Use the previous complete day (skip the current incomplete day)
    day_data = df.iloc[-(bars_per_day * 2):-bars_per_day]
    if len(day_data) < bars_per_day:
        day_data = df.iloc[:-(bars_per_day)]
    if len(day_data) == 0:
        return []

    h = float(day_data["high"].max())
    l = float(day_data["low"].min())
    c = float(day_data["close"].iloc[-1])

    pivot = (h + l + c) / 3
    r1 = 2 * pivot - l
    s1 = 2 * pivot - h

    zones = []
    for level, label in [(pivot, "Daily Pivot"), (r1, "Daily R1"), (s1, "Daily S1")]:
        pips_away = abs(level - current_price) / Config.PIP_SIZE
        pct_away  = abs(level - current_price) / current_price * 100 if current_price > 0 else 0
        zone_type = "support" if level < current_price else "resistance"
        zones.append(SRZone(
            price=level,
            zone_type=zone_type,
            strength=3,  # Pivot points are inherently strong institutional levels
            strong=True,
            pips_from_price=pips_away,
            pct_from_price=pct_away,
        ))

    return zones


def detect_sr_levels(
    df:          pd.DataFrame,
    lookback:    int   = Config.SR_LOOKBACK,
    min_bounces: int   = Config.SR_MIN_BOUNCES,
    tolerance_pips: float = Config.SR_TOLERANCE_PIPS,
    pivot_left:  int   = 5,
    pivot_right: int   = 5,
) -> SRLevels:
    """
    Detect support and resistance zones from recent price history.

    Upgraded approach:
      - Resample M15 bars to H4 equivalent (every 16 bars) for stronger
        institutional-grade pivot detection
      - Add daily pivot points (Pivot, R1, S1) as strong zones
      - Merge all zones with standard clustering

    Args:
        df:             Processed OHLCV DataFrame
        lookback:       Number of recent candles to analyse (default 200)
        min_bounces:    Minimum pivot touches for a "strong" zone (default 3)
        tolerance_pips: Zone width — pivots within this many pips are merged
        pivot_left:     Left window for pivot detection
        pivot_right:    Right window for pivot detection

    Returns:
        SRLevels with all zones classified and nearest S/R identified.
    """
    if df is None or len(df) < pivot_left + pivot_right + 10:
        logger.warning("SR: Insufficient data (%d candles).", len(df) if df is not None else 0)
        return SRLevels(
            zones=[], nearest_support=None, nearest_resistance=None,
            current_price=0.0, at_support=False, at_resistance=False,
        )

    # Use only the last `lookback` candles
    data = df.iloc[-lookback:].copy() if len(df) > lookback else df.copy()

    current_price = float(data["close"].iloc[-1])

    # Convert tolerance from pips to price units
    tolerance_price = tolerance_pips * Config.PIP_SIZE

    # ── Step 1a: Detect pivots on H4 resampled data (institutional levels) ──
    h4_data = _resample_to_h4(data, m15_bars_per_h4=16)
    h4_pivot_left  = max(2, pivot_left // 3)
    h4_pivot_right = max(2, pivot_right // 3)

    if len(h4_data) >= h4_pivot_left + h4_pivot_right + 5:
        h4_highs, h4_lows = _find_pivots(h4_data, left=h4_pivot_left, right=h4_pivot_right)
    else:
        h4_highs, h4_lows = [], []

    # ── Step 1b: Also detect pivots on M15 data (fine-grained levels) ────
    m15_highs, m15_lows = _find_pivots(data, left=pivot_left, right=pivot_right)

    # Combine M15 and H4 pivots (H4 pivots are weighted by adding them twice)
    all_pivots = m15_highs + m15_lows + h4_highs * 2 + h4_lows * 2

    if not all_pivots:
        logger.warning("SR: No pivots detected in %d candles.", len(data))
        return SRLevels(
            zones=[], nearest_support=None, nearest_resistance=None,
            current_price=current_price, at_support=False, at_resistance=False,
        )

    # ── Step 2: Cluster into zones ────────────────────────────────────────
    raw_zones  = _cluster_pivots(all_pivots, tolerance=tolerance_price)

    # ── Step 3: Build SRZone objects ─────────────────────────────────────
    zones: list[SRZone] = []
    for zone_price, count in raw_zones:
        pips_away = abs(zone_price - current_price) / Config.PIP_SIZE
        pct_away  = abs(zone_price - current_price) / current_price * 100
        zone_type = "support" if zone_price < current_price else "resistance"
        strong    = count >= min_bounces

        zones.append(SRZone(
            price=zone_price,
            zone_type=zone_type,
            strength=count,
            strong=strong,
            pips_from_price=pips_away,
            pct_from_price=pct_away,
        ))

    # ── Step 3b: Add daily pivot point zones ──────────────────────────────
    pivot_zones = _daily_pivot_zones(df, current_price, tolerance_pips)
    zones.extend(pivot_zones)

    # Sort by price ascending
    zones.sort(key=lambda z: z.price)

    # ── Step 4: Identify nearest strong support and resistance ────────────
    strong_supports    = [z for z in zones if z.zone_type == "support"    and z.strong]
    strong_resistances = [z for z in zones if z.zone_type == "resistance" and z.strong]

    all_supports    = [z for z in zones if z.zone_type == "support"]
    all_resistances = [z for z in zones if z.zone_type == "resistance"]

    nearest_support = (
        min(strong_supports,    key=lambda z: z.pips_from_price) if strong_supports
        else min(all_supports,  key=lambda z: z.pips_from_price) if all_supports
        else None
    )
    nearest_resistance = (
        min(strong_resistances,    key=lambda z: z.pips_from_price) if strong_resistances
        else min(all_resistances,  key=lambda z: z.pips_from_price) if all_resistances
        else None
    )

    # ── Step 5: At-level detection ────────────────────────────────────────
    at_support    = any(
        z.pips_from_price <= tolerance_pips
        for z in strong_supports
    )
    at_resistance = any(
        z.pips_from_price <= tolerance_pips
        for z in strong_resistances
    )

    sr = SRLevels(
        zones=zones,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        current_price=current_price,
        at_support=at_support,
        at_resistance=at_resistance,
    )

    n_strong = sum(1 for z in zones if z.strong)
    logger.info(
        "SR detected %d zones (%d strong, incl H4+pivots) | %s",
        len(zones), n_strong, sr.summary()
    )
    return sr


# ─────────────────────────────────────────────────────────────────────────────
# SCORING HOOK (used by analysis/scoring.py)
# ─────────────────────────────────────────────────────────────────────────────

def sr_signal(sr: SRLevels) -> tuple[str, str]:
    """
    Convert SRLevels into a directional signal for the scoring engine.

    Returns:
        (signal, reason) where signal ∈ {"bullish", "bearish", "neutral"}
    """
    if sr.at_support:
        ns = sr.nearest_support
        reason = (
            f"Price near strong support {ns.price:.2f} "
            f"({ns.pips_from_price:.1f} pips, {ns.strength} touches)"
        )
        return "bullish", reason

    if sr.at_resistance:
        nr = sr.nearest_resistance
        reason = (
            f"Price near strong resistance {nr.price:.2f} "
            f"({nr.pips_from_price:.1f} pips, {nr.strength} touches)"
        )
        return "bearish", reason

    # Not at a key level — report distances as context
    parts = []
    if sr.nearest_support:
        parts.append(f"Support {sr.nearest_support.price:.2f} ({sr.nearest_support.pips_from_price:.0f} pips)")
    if sr.nearest_resistance:
        parts.append(f"Resistance {sr.nearest_resistance.price:.2f} ({sr.nearest_resistance.pips_from_price:.0f} pips)")
    reason = " | ".join(parts) if parts else "No strong S/R zones detected nearby"
    return "neutral", reason
