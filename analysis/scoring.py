"""
GoldSignalAI — analysis/scoring.py
====================================
The Signal Scoring Engine — combines all analysis layers into a single
confidence score and BUY / SELL / WAIT decision.

Scoring methodology:
  10 indicators are monitored. Each votes +1 (bullish), -1 (bearish),
  or 0 (neutral).

  Confidence % = (count_in_dominant_direction / 10) × 100

  Decision rules:
    - BUY  : bullish count ≥ 7 (70%+)   AND  bearish count ≤ 2
    - SELL : bearish count ≥ 7 (70%+)   AND  bullish count ≤ 2
    - WAIT : everything else

  Modifiers that adjust the raw score:
    - Candlestick patterns add ±1 per pattern to the score
    - S/R proximity adds ±1 to the score
    - Fibonacci golden ratio proximity adds ±1 to the score
    - ADX > 40 (very strong trend) adds +3% confidence bonus
    - Volume surge adds +2% confidence bonus
    - Doji detected subtracts -5% confidence penalty

  Hard gates (override score to WAIT regardless):
    - Both timeframes must agree (handled by multi_timeframe.py)
    - ADX < 25 and score is borderline → force WAIT
    - Doji present and score ≤ 73% → force WAIT

Output: a SignalScore dataclass consumed by generator.py and formatter.py.
"""

import logging
from dataclasses import dataclass, field
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

TOTAL_INDICATORS = Config.TOTAL_INDICATORS   # 10
MIN_CONFIDENCE   = Config.MIN_CONFIDENCE_PCT  # 70

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
        return (
            f"{self.direction} | {self.confidence_pct:.0f}% confidence | "
            f"{b}/{TOTAL_INDICATORS} bull, {br}/{TOTAL_INDICATORS} bear"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def score_signal(
    indicators:  AllIndicators,
    sr_levels:   Optional[SRLevels]         = None,
    fib_levels:  Optional[FibonacciLevels]  = None,
    candlestick: Optional[CandlestickAnalysis] = None,
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

    # Raw confidence: dominant count as % of total indicators
    raw_confidence = (dom_count / TOTAL_INDICATORS) * 100

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

    # Gate 1: Opposing indicators too high (unreliable)
    if dominant == "BUY" and bearish_count > 2:
        gates.append(f"Too many bearish indicators ({bearish_count}) for a clean BUY")
    elif dominant == "SELL" and bullish_count > 2:
        gates.append(f"Too many bullish indicators ({bullish_count}) for a clean SELL")

    # Gate 2: Doji present with borderline confidence
    if candlestick is not None and candlestick.has_doji and adjusted <= 73:
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
            f"{dominant} — {dom_count}/{TOTAL_INDICATORS} indicators agree "
            f"({confidence:.0f}% confidence)"
        )
    else:
        direction = "WAIT"
        if confidence < MIN_CONFIDENCE:
            reason = (
                f"WAIT — confidence {confidence:.0f}% < {MIN_CONFIDENCE}% threshold "
                f"({bullish_count} bull, {bearish_count} bear)"
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
