"""
GoldSignalAI — analysis/scoring.py
====================================
The Signal Scoring Engine — combines all analysis layers into a single
confidence score and BUY / SELL / WAIT decision.

Scoring methodology:
  10 indicators are monitored. Each votes +1 (bullish), -1 (bearish),
  or 0 (neutral).

  Confidence % = (dominant_count / active_count) × 100
  where active_count = bullish + bearish (only indicators with an opinion).
  Neutral indicators are excluded from the ratio so that a 4-bull / 1-bear
  result scores 80% (4/5) instead of 40% (4/10).

  Decision rules:
    - BUY  : confidence ≥ 65%  AND  dominant == BUY  AND  bearish ≤ 2
    - SELL : confidence ≥ 65%  AND  dominant == SELL  AND  bullish ≤ 2
    - WAIT : everything else

  Modifiers that adjust the raw score:
    - Candlestick patterns: +2% per confirming pattern (capped +6%)
    - S/R proximity: +3%
    - Fibonacci golden ratio proximity: +3%
    - ADX > 40 (very strong trend): +3%
    - Volume surge: +2%
    - Doji detected: -5%

  Hard gates (override score to WAIT regardless):
    - Both timeframes must agree (handled by multi_timeframe.py)
    - Doji present and confidence ≤ MIN_CONFIDENCE + 8% → force WAIT

Output: a SignalScore dataclass consumed by generator.py and formatter.py.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import Config
from analysis.indicators import AllIndicators, IndicatorResult, BULLISH, BEARISH, NEUTRAL
from analysis.sr_levels import SRLevels, sr_signal
from analysis.fibonacci import FibonacciLevels
from analysis.candlestick import CandlestickAnalysis

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SCORING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_INDICATORS = Config.TOTAL_INDICATORS    # 9 (BBands removed)
MIN_CONFIDENCE   = Config.MIN_CONFIDENCE_PCT  # 65
MAX_CONFIDENCE   = Config.MAX_CONFIDENCE_PCT  # 75 — above this = over-consensus, lagging

# Trading sessions (UTC hours). Diagnostic: NY=63.3%, Overlap=53.6%, London=33.9%.
# Only trade during NY session and London-NY overlap.
SESSION_ACTIVE_HOURS = frozenset(range(13, 22))  # 13:00–21:59 UTC (NY + Overlap)

# Bonus/penalty values (added to the raw percentage)
BONUS_VERY_STRONG_TREND  =  3.0   # ADX > 40
BONUS_VOLUME_SURGE       =  2.0   # volume ≥ 2× average
PENALTY_DOJI             = -5.0   # Doji detected
BONUS_AT_KEY_SR          =  3.0   # price at strong S/R zone
BONUS_FIBO_GOLDEN        =  3.0   # price at 61.8% Fib level
BONUS_CANDLESTICK        =  2.0   # per confirming candlestick pattern


# ─────────────────────────────────────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalScore:
    """
    Final scored signal for one timeframe.

    Attributes:
        direction       : "BUY" | "SELL" | "WAIT"
        confidence_pct  : 0–100 (may exceed 100 after bonuses, capped to 99)
        raw_confidence  : confidence before bonuses/penalties
        bullish_count   : how many of the 10 indicators voted bullish
        bearish_count   : how many voted bearish
        neutral_count   : how many voted neutral
        net_score       : bullish_count - bearish_count
        indicator_detail: per-indicator breakdown for display
        bonuses         : list of (label, value) applied
        gates_triggered : list of gate reasons that forced WAIT
        reason          : one-sentence explanation of the decision
    """
    direction:        str
    confidence_pct:   float
    raw_confidence:   float
    bullish_count:    int
    bearish_count:    int
    neutral_count:    int
    net_score:        int
    indicator_detail: list[dict] = field(default_factory=list)
    bonuses:          list[tuple[str, float]] = field(default_factory=list)
    gates_triggered:  list[str] = field(default_factory=list)
    reason:           str = ""

    @property
    def is_actionable(self) -> bool:
        """True if this is a BUY or SELL (not WAIT)."""
        return self.direction in ("BUY", "SELL")

    def summary(self) -> str:
        b = self.bullish_count
        br = self.bearish_count
        active = b + br
        return (
            f"{self.direction} | {self.confidence_pct:.0f}% confidence | "
            f"{b} bull, {br} bear, {self.neutral_count} neutral "
            f"(active={active})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def score_signal(
    indicators:  AllIndicators,
    sr_levels:   Optional[SRLevels]            = None,
    fib_levels:  Optional[FibonacciLevels]     = None,
    candlestick: Optional[CandlestickAnalysis] = None,
    bar_time:    Optional[datetime]            = None,
) -> SignalScore:
    """
    Score a single timeframe's analysis into a BUY/SELL/WAIT decision.

    This function takes the raw outputs from all analysis layers and
    produces a single SignalScore. It does NOT enforce the multi-timeframe
    agreement rule — that's handled by multi_timeframe.py which calls
    this function twice (M15 and H1) and then compares.

    Args:
        indicators:  AllIndicators from indicators.py
        sr_levels:   SRLevels from sr_levels.py (optional)
        fib_levels:  FibonacciLevels from fibonacci.py (optional)
        candlestick: CandlestickAnalysis from candlestick.py (optional)
        bar_time:    UTC timestamp of the bar — used for session gate (optional).
                     If None, session gate is skipped (safe for unit tests).

    Returns:
        SignalScore with the final decision and full breakdown.
    """
    bonuses: list[tuple[str, float]] = []
    gates:   list[str] = []

    # ── Step 1: Count raw indicator votes ─────────────────────────────────
    ind_list = indicators.as_list()
    bullish_count = indicators.bullish_count()
    bearish_count = indicators.bearish_count()
    neutral_count = indicators.neutral_count()
    net_score     = indicators.net_score()

    # Build per-indicator detail for display
    indicator_detail = []
    for r in ind_list:
        indicator_detail.append({
            "name":   r.name,
            "signal": r.signal,
            "value":  r.value,
            "reason": r.reason,
            "score":  r.score(),
        })

    # ── Step 2: Determine dominant direction ──────────────────────────────
    if bullish_count > bearish_count:
        dominant   = "BUY"
        dom_count  = bullish_count
    elif bearish_count > bullish_count:
        dominant   = "SELL"
        dom_count  = bearish_count
    else:
        dominant   = "WAIT"
        dom_count  = max(bullish_count, bearish_count)

    # Raw confidence: dominant count as % of ACTIVE indicators only.
    # Active = indicators that voted bullish or bearish (not neutral).
    # This prevents neutral indicators from diluting the score.
    active_count = bullish_count + bearish_count
    if active_count > 0:
        raw_confidence = (dom_count / active_count) * 100
    else:
        raw_confidence = 0.0

    # ── Step 3: Apply bonuses and penalties ────────────────────────────────
    adjusted = raw_confidence

    # ADX very strong trend bonus
    adx_val = indicators.adx.values.get("adx", 0)
    if adx_val > Config.ADX_STRONG_THRESHOLD:
        bonuses.append(("ADX very strong trend", BONUS_VERY_STRONG_TREND))
        adjusted += BONUS_VERY_STRONG_TREND

    # Volume surge bonus
    vol_ratio = indicators.volume.values.get("ratio", 0)
    vol_surge = indicators.volume.values.get("surge", 0)
    if vol_surge:
        bonuses.append(("Volume surge", BONUS_VOLUME_SURGE))
        adjusted += BONUS_VOLUME_SURGE

    # S/R proximity bonus
    if sr_levels is not None:
        sr_sig, sr_reason = sr_signal(sr_levels)
        if sr_sig == "bullish" and dominant == "BUY":
            bonuses.append(("At strong support", BONUS_AT_KEY_SR))
            adjusted += BONUS_AT_KEY_SR
        elif sr_sig == "bearish" and dominant == "SELL":
            bonuses.append(("At strong resistance", BONUS_AT_KEY_SR))
            adjusted += BONUS_AT_KEY_SR

    # Fibonacci golden ratio bonus
    if fib_levels is not None and fib_levels.at_golden:
        if (fib_levels.signal == "bullish" and dominant == "BUY") or \
           (fib_levels.signal == "bearish" and dominant == "SELL"):
            bonuses.append(("At Fib 61.8% (golden ratio)", BONUS_FIBO_GOLDEN))
            adjusted += BONUS_FIBO_GOLDEN

    # Candlestick pattern bonuses
    if candlestick is not None:
        if candlestick.signal == "bullish" and dominant == "BUY":
            n = candlestick.bullish_count
            bonus = min(n * BONUS_CANDLESTICK, 6.0)  # cap at +6%
            bonuses.append((f"Candlestick patterns ({n} bullish)", bonus))
            adjusted += bonus
        elif candlestick.signal == "bearish" and dominant == "SELL":
            n = candlestick.bearish_count
            bonus = min(n * BONUS_CANDLESTICK, 6.0)
            bonuses.append((f"Candlestick patterns ({n} bearish)", bonus))
            adjusted += bonus

        # Doji penalty
        if candlestick.has_doji:
            bonuses.append(("Doji — indecision", PENALTY_DOJI))
            adjusted += PENALTY_DOJI

    # ── Step 4: Gate checks (force WAIT) ──────────────────────────────────

    # Gate 0: Session filter — only trade NY session + Overlap (UTC 13:00–21:59).
    # Diagnostic result: NY=63.3%, Overlap=53.6%, London=33.9% win rate.
    # London-only session actively hurts PnL; outside sessions have no edge.
    if bar_time is not None:
        bar_hour = bar_time.hour
        if bar_hour not in SESSION_ACTIVE_HOURS:
            gates.append(
                f"Outside active session (bar hour {bar_hour:02d}:xx UTC) "
                f"— only trading 13:00–21:59 UTC (NY + Overlap)"
            )

    # Gate 1: Minimum active and dominant counts.
    # Diagnostic (10-indicator system): 4 dominant → 56.8%, 3 dominant → 44.2%.
    # With 9 indicators (BBands removed), require >= 3 dominant; the session gate
    # does the heavy lifting since London=33.9% and NY=63.3%.
    MIN_ACTIVE = 4
    MIN_DOMINANT = 3
    if active_count < MIN_ACTIVE:
        gates.append(
            f"Too few active indicators ({active_count}/{TOTAL_INDICATORS}) "
            f"— need at least {MIN_ACTIVE}"
        )
    elif dom_count < MIN_DOMINANT:
        gates.append(
            f"Dominant count too low ({dom_count} {dominant}) "
            f"— need at least {MIN_DOMINANT} in same direction"
        )

    # Gate 2: Confidence ceiling removed — was blocking high-confidence signals
    # (e.g. 88% adj → WAIT) because MAX_CONFIDENCE=75% gate fired. The diagnostic
    # showed 75–80% had 38.3% accuracy, but that does not justify blocking 85–99%.
    # Direction assignment now relies on MIN_CONFIDENCE (65%) as the sole lower bound.

    # Gate 3: Opposing indicators too high (unreliable signal).
    if dominant == "BUY" and bearish_count > 2:
        gates.append(f"Too many bearish indicators ({bearish_count}) for a clean BUY")
    elif dominant == "SELL" and bullish_count > 2:
        gates.append(f"Too many bullish indicators ({bullish_count}) for a clean SELL")

    # Gate 4: Doji present with borderline confidence
    if candlestick is not None and candlestick.has_doji and adjusted <= (MIN_CONFIDENCE + 8):
        gates.append(f"Doji present with borderline confidence ({adjusted:.0f}%)")

    # ── Step 5: Final decision ────────────────────────────────────────────
    # Clamp confidence to 0–99 range
    confidence = max(0.0, min(99.0, adjusted))

    if gates:
        direction = "WAIT"
        reason = f"WAIT — {'; '.join(gates)}"
    elif confidence >= MIN_CONFIDENCE and dominant in ("BUY", "SELL"):
        direction = dominant
        reason = (
            f"{dominant} — {dom_count}/{active_count} active indicators agree "
            f"({confidence:.0f}% confidence, {neutral_count} neutral)"
        )
    else:
        direction = "WAIT"
        if confidence < MIN_CONFIDENCE:
            reason = (
                f"WAIT — confidence {confidence:.0f}% < {MIN_CONFIDENCE}% threshold "
                f"({bullish_count} bull, {bearish_count} bear, {neutral_count} neutral)"
            )
        else:
            reason = f"WAIT — no clear direction"

    result = SignalScore(
        direction=direction,
        confidence_pct=confidence,
        raw_confidence=raw_confidence,
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        neutral_count=neutral_count,
        net_score=net_score,
        indicator_detail=indicator_detail,
        bonuses=bonuses,
        gates_triggered=gates,
        reason=reason,
    )

    logger.info(
        "Score: %s | raw=%d%% adj=%.0f%% | %d bull %d bear %d neutral | bonuses=%d gates=%d",
        direction, raw_confidence, confidence,
        bullish_count, bearish_count, neutral_count,
        len(bonuses), len(gates),
    )
    return result
