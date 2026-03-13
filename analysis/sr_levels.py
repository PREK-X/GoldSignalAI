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
    # Gold: 1 pip = Config.PIP_SIZE (0.1) → e.g. 5 pips = 0.5 price units
    tolerance_price = tolerance_pips * Config.PIP_SIZE

    # ── Step 1: Detect pivot highs and lows ──────────────────────────────
    pivot_highs, pivot_lows = _find_pivots(data, left=pivot_left, right=pivot_right)

    if not pivot_highs and not pivot_lows:
        logger.warning("SR: No pivots detected in %d candles.", len(data))
        return SRLevels(
            zones=[], nearest_support=None, nearest_resistance=None,
            current_price=current_price, at_support=False, at_resistance=False,
        )

    # ── Step 2: Cluster into zones ────────────────────────────────────────
    all_pivots = pivot_highs + pivot_lows
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

    # Sort by price ascending
    zones.sort(key=lambda z: z.price)

    # ── Step 4: Identify nearest strong support and resistance ────────────
    strong_supports    = [z for z in zones if z.zone_type == "support"    and z.strong]
    strong_resistances = [z for z in zones if z.zone_type == "resistance" and z.strong]

    all_supports    = [z for z in zones if z.zone_type == "support"]
    all_resistances = [z for z in zones if z.zone_type == "resistance"]

    # Prefer nearest STRONG zone; fall back to nearest weak zone so that
    # nearest_support/nearest_resistance are never None when zones exist.
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
    # "At" = within tolerance_pips of any strong zone
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
        "SR detected %d zones (%d strong) | %s",
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
