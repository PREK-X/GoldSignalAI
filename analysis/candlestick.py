"""
GoldSignalAI — analysis/candlestick.py
========================================
Detects high-probability candlestick reversal and continuation patterns
on the most recent 1–3 candles of an OHLCV DataFrame.

Patterns detected (all manual calculation — no external library needed):
  Reversal bullish  : Hammer, Bullish Engulfing, Morning Star, Pin Bar (bull)
  Reversal bearish  : Shooting Star, Hanging Man, Bearish Engulfing,
                      Evening Star, Pin Bar (bear), Inverted Hammer (bear ctx)
  Indecision        : Doji, Spinning Top

Scoring rule (used by scoring.py):
  - Each confirmed pattern adds +1 to the bullish or bearish score
  - A Doji / Spinning Top does NOT add a score but sets `has_doji=True`
    which the scoring engine uses to reduce overall confidence

Body ratio thresholds (expressed as fraction of hl_range):
  - Doji        : body < 5% of range
  - Small body  : body < 30% of range  (Spinning Top)
  - Normal body : 30–70%
  - Large body  : > 70%  (Marubozu-like — strong momentum)

Shadow ratio thresholds:
  - Long shadow  : shadow > 2× body
  - Very long    : shadow > 3× body  (Pin Bar / strong Hammer)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS (tuned for Gold's typical volatility profile)
# ─────────────────────────────────────────────────────────────────────────────

DOJI_BODY_RATIO       = 0.05   # body / range < this → doji
SMALL_BODY_RATIO      = 0.30   # body / range < this → small body
LARGE_BODY_RATIO      = 0.70   # body / range > this → strong body
LONG_SHADOW_RATIO     = 2.0    # shadow > this × body → long shadow
PIN_BAR_SHADOW_RATIO  = 2.5    # shadow > this × body → pin bar quality
ENGULF_TOLERANCE      = 0.001  # 0.1% tolerance for engulfing comparison


# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandlePattern:
    """A single detected candlestick pattern."""
    name:      str
    signal:    str        # "bullish" | "bearish" | "neutral"
    strength:  str        # "strong" | "moderate" | "weak"
    bars_used: int        # 1, 2, or 3 candles involved
    reason:    str        # Why this pattern was triggered


@dataclass
class CandlestickAnalysis:
    """
    All patterns detected on the last 1–3 candles.

    Attributes:
        patterns    : All confirmed patterns (may be empty)
        has_doji    : True if latest candle is a Doji — reduces confidence
        bullish_count : Number of bullish patterns detected
        bearish_count : Number of bearish patterns detected
        signal      : "bullish" | "bearish" | "neutral"
        reason      : Best pattern explanation (strongest signal)
        score_delta : Net score contribution (+N for bull, -N for bear)
    """
    patterns:      list[CandlePattern]
    has_doji:      bool
    bullish_count: int
    bearish_count: int
    signal:        str
    reason:        str
    score_delta:   int   # net contribution to the scoring engine

    def top_pattern(self) -> Optional[CandlePattern]:
        """Return the highest-strength pattern, or None."""
        order = {"strong": 3, "moderate": 2, "weak": 1}
        if not self.patterns:
            return None
        return max(self.patterns, key=lambda p: order.get(p.strength, 0))


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-CANDLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class _Candle:
    """Lightweight wrapper for a single OHLC row with precomputed properties."""

    __slots__ = ("o", "h", "l", "c", "rng", "body", "upper", "lower",
                 "bullish", "bearish", "doji", "small_body", "large_body")

    def __init__(self, row: pd.Series):
        self.o = float(row["open"])
        self.h = float(row["high"])
        self.l = float(row["low"])
        self.c = float(row["close"])

        self.rng   = self.h - self.l
        self.body  = abs(self.c - self.o)
        self.upper = self.h - max(self.c, self.o)
        self.lower = min(self.c, self.o) - self.l

        self.bullish = self.c > self.o
        self.bearish = self.c < self.o

        body_ratio       = self.body / self.rng if self.rng > 0 else 0
        self.doji        = body_ratio < DOJI_BODY_RATIO
        self.small_body  = body_ratio < SMALL_BODY_RATIO
        self.large_body  = body_ratio > LARGE_BODY_RATIO


def _last_candles(df: pd.DataFrame, n: int) -> list[_Candle]:
    """Return the last `n` candles as _Candle objects, most recent last."""
    rows = df.iloc[-n:] if len(df) >= n else df.iloc[:]
    return [_Candle(rows.iloc[i]) for i in range(len(rows))]


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-CANDLE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_doji(c: _Candle) -> Optional[CandlePattern]:
    """
    Doji: open ≈ close, body < 5% of range.
    Signals indecision. Does NOT contribute a directional score.
    """
    if c.doji and c.rng > 0:
        return CandlePattern(
            name="Doji", signal="neutral", strength="moderate",
            bars_used=1,
            reason=f"Body={c.body:.2f} < 5% of range={c.rng:.2f} — buyer/seller equilibrium"
        )
    return None


def _detect_hammer(c: _Candle, prev_trend: str) -> Optional[CandlePattern]:
    """
    Hammer: small body at TOP of range, long lower shadow (≥ 2× body),
    tiny upper shadow. Appears after a downtrend → bullish reversal.

    Hanging Man is the same shape but after an uptrend → bearish.
    We use `prev_trend` to distinguish them.
    """
    if c.doji or c.rng == 0:
        return None

    # Long lower shadow, small upper shadow
    lower_ok = c.lower >= LONG_SHADOW_RATIO * c.body if c.body > 0 else c.lower > c.rng * 0.55
    upper_ok = c.upper <= c.body * 0.3 or c.upper <= c.rng * 0.1

    if not (lower_ok and upper_ok):
        return None

    if prev_trend == "down":
        strength = "strong" if c.lower >= PIN_BAR_SHADOW_RATIO * max(c.body, 0.001) else "moderate"
        return CandlePattern(
            name="Hammer", signal="bullish", strength=strength,
            bars_used=1,
            reason=f"Long lower shadow ({c.lower:.2f}) after downtrend — bullish reversal"
        )
    elif prev_trend == "up":
        return CandlePattern(
            name="Hanging Man", signal="bearish", strength="moderate",
            bars_used=1,
            reason=f"Hammer shape after uptrend — bearish reversal warning"
        )
    return None


def _detect_shooting_star(c: _Candle, prev_trend: str) -> Optional[CandlePattern]:
    """
    Shooting Star: small body at BOTTOM of range, long upper shadow (≥ 2× body),
    tiny lower shadow. After uptrend → bearish reversal.

    Inverted Hammer is the same shape after a downtrend → mildly bullish.
    """
    if c.doji or c.rng == 0:
        return None

    upper_ok = c.upper >= LONG_SHADOW_RATIO * c.body if c.body > 0 else c.upper > c.rng * 0.55
    lower_ok = c.lower <= c.body * 0.3 or c.lower <= c.rng * 0.1

    if not (upper_ok and lower_ok):
        return None

    if prev_trend == "up":
        strength = "strong" if c.upper >= PIN_BAR_SHADOW_RATIO * max(c.body, 0.001) else "moderate"
        return CandlePattern(
            name="Shooting Star", signal="bearish", strength=strength,
            bars_used=1,
            reason=f"Long upper shadow ({c.upper:.2f}) after uptrend — bearish reversal"
        )
    elif prev_trend == "down":
        return CandlePattern(
            name="Inverted Hammer", signal="bullish", strength="weak",
            bars_used=1,
            reason=f"Inverted Hammer after downtrend — weak bullish signal, needs confirmation"
        )
    return None


def _detect_pin_bar(c: _Candle, prev_trend: str) -> Optional[CandlePattern]:
    """
    Pin Bar: extreme shadow (≥ 2.5× body) on one side with a tiny opposing shadow.
    Stronger than a Hammer/Shooting Star — indicates aggressive rejection of a level.

    Bullish pin: very long lower shadow (price rejected a low level)
    Bearish pin: very long upper shadow (price rejected a high level)
    """
    if c.doji or c.rng == 0 or c.body == 0:
        return None

    bull_pin = (
        c.lower >= PIN_BAR_SHADOW_RATIO * c.body and
        c.upper <= c.lower * 0.33
    )
    bear_pin = (
        c.upper >= PIN_BAR_SHADOW_RATIO * c.body and
        c.lower <= c.upper * 0.33
    )

    if bull_pin:
        return CandlePattern(
            name="Bullish Pin Bar", signal="bullish", strength="strong",
            bars_used=1,
            reason=f"Lower wick ({c.lower:.2f}) = {c.lower/c.body:.1f}× body — strong support rejection"
        )
    if bear_pin:
        return CandlePattern(
            name="Bearish Pin Bar", signal="bearish", strength="strong",
            bars_used=1,
            reason=f"Upper wick ({c.upper:.2f}) = {c.upper/c.body:.1f}× body — strong resistance rejection"
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TWO-CANDLE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_engulfing(prev: _Candle, curr: _Candle) -> Optional[CandlePattern]:
    """
    Engulfing pattern: current candle's body completely engulfs the previous
    candle's body. One of the most reliable reversal signals.

    Bullish Engulfing:
      - Previous candle is bearish (c < o)
      - Current candle is bullish (c > o)
      - Current body engulfs previous body (open ≤ prev.close, close ≥ prev.open)

    Bearish Engulfing: mirror image.

    A larger engulfing candle (e.g. 2× previous body) is stronger.
    """
    if prev.doji or curr.doji:
        return None

    tol = ENGULF_TOLERANCE

    # Bullish: prev bearish, curr bullish, curr body wraps prev body
    if (prev.bearish and curr.bullish and
            curr.o <= prev.c * (1 + tol) and
            curr.c >= prev.o * (1 - tol)):
        ratio = curr.body / prev.body if prev.body > 0 else 1
        strength = "strong" if ratio >= 1.5 else "moderate"
        return CandlePattern(
            name="Bullish Engulfing", signal="bullish", strength=strength,
            bars_used=2,
            reason=f"Bull candle ({curr.body:.2f}) engulfs bear candle ({prev.body:.2f}) — {ratio:.1f}× size"
        )

    # Bearish: prev bullish, curr bearish, curr body wraps prev body
    if (prev.bullish and curr.bearish and
            curr.o >= prev.c * (1 - tol) and
            curr.c <= prev.o * (1 + tol)):
        ratio = curr.body / prev.body if prev.body > 0 else 1
        strength = "strong" if ratio >= 1.5 else "moderate"
        return CandlePattern(
            name="Bearish Engulfing", signal="bearish", strength=strength,
            bars_used=2,
            reason=f"Bear candle ({curr.body:.2f}) engulfs bull candle ({prev.body:.2f}) — {ratio:.1f}× size"
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# THREE-CANDLE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def _detect_morning_star(c1: _Candle, c2: _Candle, c3: _Candle) -> Optional[CandlePattern]:
    """
    Morning Star (3-candle bullish reversal):
      1. First candle: large bearish body
      2. Middle candle: small body or doji (gap down from c1 close)
      3. Third candle: large bullish body closing above c1's midpoint

    One of the most reliable 3-bar reversal patterns.
    """
    # c1: large bear, c2: small/doji, c3: large bull
    if not (c1.bearish and c1.large_body):
        return None
    if not (c2.small_body or c2.doji):
        return None
    if not (c3.bullish and c3.large_body):
        return None

    # c3 must close above the midpoint of c1's body
    c1_midpoint = (c1.o + c1.c) / 2
    if c3.c < c1_midpoint:
        return None

    return CandlePattern(
        name="Morning Star", signal="bullish", strength="strong",
        bars_used=3,
        reason=f"Morning Star: bear({c1.body:.2f}) → doji/small → bull({c3.body:.2f}) above c1 mid"
    )


def _detect_evening_star(c1: _Candle, c2: _Candle, c3: _Candle) -> Optional[CandlePattern]:
    """
    Evening Star (3-candle bearish reversal) — mirror of Morning Star:
      1. First candle: large bullish body
      2. Middle candle: small body or doji (gap up from c1 close)
      3. Third candle: large bearish body closing below c1's midpoint
    """
    if not (c1.bullish and c1.large_body):
        return None
    if not (c2.small_body or c2.doji):
        return None
    if not (c3.bearish and c3.large_body):
        return None

    c1_midpoint = (c1.o + c1.c) / 2
    if c3.c > c1_midpoint:
        return None

    return CandlePattern(
        name="Evening Star", signal="bearish", strength="strong",
        bars_used=3,
        reason=f"Evening Star: bull({c1.body:.2f}) → doji/small → bear({c3.body:.2f}) below c1 mid"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TREND CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

def _prior_trend(df: pd.DataFrame, window: int = 10) -> str:
    """
    Determine whether the market has been trending up or down over the
    last `window` candles. Used to contextualise single-candle patterns.

    Returns "up", "down", or "sideways".
    """
    if len(df) < window + 1:
        return "sideways"
    closes = df["close"].iloc[-(window + 1):-1]   # exclude current candle
    net_change = float(closes.iloc[-1]) - float(closes.iloc[0])
    avg_range  = float((df["high"].iloc[-window:] - df["low"].iloc[-window:]).mean())
    if avg_range == 0:
        return "sideways"
    if net_change > avg_range * 0.3:
        return "up"
    if net_change < -avg_range * 0.3:
        return "down"
    return "sideways"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DETECTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> CandlestickAnalysis:
    """
    Run all candlestick pattern detectors on the last 3 candles of `df`.

    Detection order (to avoid double-counting):
      3-candle: Morning Star, Evening Star
      2-candle: Bullish/Bearish Engulfing
      1-candle: Pin Bar, Hammer/Hanging Man, Shooting Star/Inv. Hammer, Doji

    A pattern found at the 3-candle level means we still report 1-candle
    patterns on the latest bar — they provide independent confirmation.

    Args:
        df: Processed OHLCV DataFrame (minimum 3 rows)

    Returns:
        CandlestickAnalysis with all detected patterns and net signal.
    """
    patterns: list[CandlePattern] = []
    has_doji  = False

    if df is None or len(df) < 1:
        return CandlestickAnalysis(
            patterns=[], has_doji=False, bullish_count=0, bearish_count=0,
            signal="neutral", reason="Insufficient data", score_delta=0,
        )

    prior_trend = _prior_trend(df)

    # ── 1-candle patterns (on the latest bar) ────────────────────────────
    candles_1 = _last_candles(df, 1)
    c0 = candles_1[0]

    doji_p = _detect_doji(c0)
    if doji_p:
        has_doji = True
        patterns.append(doji_p)

    if not c0.doji:
        pin_p = _detect_pin_bar(c0, prior_trend)
        if pin_p:
            patterns.append(pin_p)

        hammer_p = _detect_hammer(c0, prior_trend)
        if hammer_p:
            patterns.append(hammer_p)

        star_p = _detect_shooting_star(c0, prior_trend)
        if star_p:
            patterns.append(star_p)

    # ── 2-candle patterns ─────────────────────────────────────────────────
    if len(df) >= 2:
        candles_2 = _last_candles(df, 2)
        eng_p = _detect_engulfing(candles_2[0], candles_2[1])
        if eng_p:
            patterns.append(eng_p)

    # ── 3-candle patterns ─────────────────────────────────────────────────
    if len(df) >= 3:
        candles_3 = _last_candles(df, 3)
        c1, c2, c3 = candles_3[0], candles_3[1], candles_3[2]

        ms_p = _detect_morning_star(c1, c2, c3)
        if ms_p:
            patterns.append(ms_p)

        es_p = _detect_evening_star(c1, c2, c3)
        if es_p:
            patterns.append(es_p)

    # ── Aggregate ────────────────────────────────────────────────────────
    bull_count = sum(1 for p in patterns if p.signal == "bullish")
    bear_count = sum(1 for p in patterns if p.signal == "bearish")
    score_delta = bull_count - bear_count

    # Overall signal: whichever direction dominates, or neutral if tied/doji-only
    if has_doji and bull_count == 0 and bear_count == 0:
        signal  = "neutral"
        reason  = "Doji — market indecision, reduces confidence"
    elif bull_count > bear_count:
        signal  = "bullish"
        top = max((p for p in patterns if p.signal == "bullish"),
                  key=lambda p: {"strong": 3, "moderate": 2, "weak": 1}.get(p.strength, 0))
        reason  = top.reason
    elif bear_count > bull_count:
        signal  = "bearish"
        top = max((p for p in patterns if p.signal == "bearish"),
                  key=lambda p: {"strong": 3, "moderate": 2, "weak": 1}.get(p.strength, 0))
        reason  = top.reason
    else:
        signal  = "neutral"
        reason  = (
            f"Mixed patterns ({bull_count} bull, {bear_count} bear)"
            if patterns else f"No patterns detected (prior trend: {prior_trend})"
        )

    if patterns:
        logger.info(
            "Candlestick patterns: %s | %s",
            ", ".join(p.name for p in patterns), signal
        )

    return CandlestickAnalysis(
        patterns=patterns,
        has_doji=has_doji,
        bullish_count=bull_count,
        bearish_count=bear_count,
        signal=signal,
        reason=reason,
        score_delta=score_delta,
    )
