"""
GoldSignalAI -- signals/meta_decision.py
==========================================
Stage 8: Meta-Decision Layer (extended in Stage 10)

Combines the scoring engine, HMM regime detector, LightGBM classifier,
and news/volatility filter into a single cascading gate system.

Cascade order:
  Rule 1: HMM Hard Gate      — CRISIS blocks all; RANGING halves size
  Rule 2: LGBM Soft Vote     — strong disagreement blocks signal
  Rule 3: Confidence Boost   — HMM+LGBM agreement boosts/penalises confidence
  Rule 4: Session Loss Limit — 2 consecutive losses skip rest of session
  Rule 5: News/Volatility    — ATR spike or high-impact calendar event blocks

The class is stateless between calls — session_consecutive_losses is managed
by the caller (backtest engine or signal generator).
signal_time / current_atr / rolling_atr_mean are optional; if omitted Rule 5
is skipped (backward-compatible with existing tests and callers).
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class MetaResult:
    """Outcome of the meta-decision cascade."""
    allowed: bool                  # True = signal passes, False = blocked
    block_reason: str              # "" if allowed, else reason string
    position_size_mult: float      # 1.0 normal, 0.5 ranging, 0.0 blocked
    adjusted_confidence: float     # confidence after boost/penalty
    lgbm_prob: float               # raw LGBM probability (-1.0 if unavailable)
    hmm_state: str                 # TRENDING / RANGING / CRISIS
    news_blocked: bool = False     # True if Rule 5 triggered


class MetaDecision:
    """
    Stateless meta-decision engine (5-rule cascade as of Stage 10).

    Call decide() for every candidate signal. The caller tracks
    session_consecutive_losses externally.
    """

    def __init__(self):
        # Lazy-init NewsFilter to avoid import at module load time
        self._news_filter = None

    def _get_news_filter(self):
        if self._news_filter is None:
            from signals.news_filter import NewsFilter
            self._news_filter = NewsFilter()
        return self._news_filter

    def decide(
        self,
        direction: str,
        base_confidence: float,
        lgbm_prob: float,
        hmm_state: str,
        session_consecutive_losses: int,
        signal_time: Optional[datetime] = None,
        current_atr: float = 0.0,
        rolling_atr_mean: float = 0.0,
    ) -> MetaResult:
        """
        Run the 5-rule cascade and return a MetaResult.

        Args:
            direction:  "BUY" or "SELL" from the scoring engine.
            base_confidence:  Scoring engine confidence (0-100).
            lgbm_prob:  LGBM P(UP) for this bar, or -1.0 if unavailable.
            hmm_state:  "TRENDING", "RANGING", or "CRISIS".
            session_consecutive_losses:  Losses so far this NY session.
            signal_time:  UTC datetime of signal bar (for news calendar check).
            current_atr:  Current ATR-14 value (for volatility spike check).
            rolling_atr_mean:  28-bar rolling ATR mean.

        Returns:
            MetaResult with allowed/blocked, adjusted confidence, and
            position size multiplier.
        """
        # Defaults
        position_mult = 1.0
        adjusted_conf = base_confidence

        # ── Rule 1: HMM Hard Gate ────────────────────────────────────────
        if hmm_state == "CRISIS":
            return MetaResult(
                allowed=False,
                block_reason="HMM CRISIS regime — all signals blocked",
                position_size_mult=0.0,
                adjusted_confidence=base_confidence,
                lgbm_prob=lgbm_prob,
                hmm_state=hmm_state,
            )

        if hmm_state == "RANGING":
            position_mult = 0.5

        # ── Rule 2: LGBM Soft Vote ──────────────────────────────────────
        lgbm_available = lgbm_prob >= 0.0
        if lgbm_available:
            if direction == "BUY" and lgbm_prob < Config.META_LGBM_BLOCK_LOW:
                return MetaResult(
                    allowed=False,
                    block_reason=f"LGBM soft vote: P(UP)={lgbm_prob:.2f} < {Config.META_LGBM_BLOCK_LOW} blocks BUY",
                    position_size_mult=0.0,
                    adjusted_confidence=base_confidence,
                    lgbm_prob=lgbm_prob,
                    hmm_state=hmm_state,
                )
            if direction == "SELL" and lgbm_prob > Config.META_LGBM_BLOCK_HIGH:
                return MetaResult(
                    allowed=False,
                    block_reason=f"LGBM soft vote: P(UP)={lgbm_prob:.2f} > {Config.META_LGBM_BLOCK_HIGH} blocks SELL",
                    position_size_mult=0.0,
                    adjusted_confidence=base_confidence,
                    lgbm_prob=lgbm_prob,
                    hmm_state=hmm_state,
                )

        # ── Rule 3: Confidence Boost / Penalty ───────────────────────────
        if lgbm_available:
            lgbm_agrees_buy = direction == "BUY" and lgbm_prob > 0.55
            lgbm_agrees_sell = direction == "SELL" and lgbm_prob < 0.45
            lgbm_agrees = lgbm_agrees_buy or lgbm_agrees_sell

            if hmm_state == "TRENDING" and lgbm_agrees:
                adjusted_conf += Config.META_CONFIDENCE_BOOST

        if hmm_state == "RANGING":
            adjusted_conf -= Config.META_CONFIDENCE_PEN

        # Clamp to 0-100
        adjusted_conf = max(0.0, min(100.0, adjusted_conf))

        # Check against MIN_CONFIDENCE_PCT after adjustment
        if adjusted_conf < Config.MIN_CONFIDENCE_PCT:
            return MetaResult(
                allowed=False,
                block_reason=f"Adjusted confidence {adjusted_conf:.1f}% < {Config.MIN_CONFIDENCE_PCT}% gate",
                position_size_mult=0.0,
                adjusted_confidence=adjusted_conf,
                lgbm_prob=lgbm_prob,
                hmm_state=hmm_state,
            )

        # ── Rule 4: Consecutive Session Loss Circuit ─────────────────────
        if session_consecutive_losses >= Config.META_MAX_SESSION_LOSS:
            return MetaResult(
                allowed=False,
                block_reason=f"Session loss limit: {session_consecutive_losses} consecutive losses >= {Config.META_MAX_SESSION_LOSS}",
                position_size_mult=0.0,
                adjusted_confidence=adjusted_conf,
                lgbm_prob=lgbm_prob,
                hmm_state=hmm_state,
            )

        # ── Rule 5: News & Volatility Filter ────────────────────────────
        if signal_time is not None and Config.NEWS_FILTER_ENABLED:
            try:
                news_result = self._get_news_filter().check(
                    signal_time=signal_time,
                    current_atr=current_atr,
                    rolling_atr_mean=rolling_atr_mean,
                )
                if not news_result.allowed:
                    return MetaResult(
                        allowed=False,
                        block_reason=news_result.block_reason,
                        position_size_mult=0.0,
                        adjusted_confidence=adjusted_conf,
                        lgbm_prob=lgbm_prob,
                        hmm_state=hmm_state,
                        news_blocked=True,
                    )
                # Apply news size multiplier (take the more conservative value)
                if news_result.position_size_mult < position_mult:
                    position_mult = news_result.position_size_mult
            except Exception as exc:
                logger.warning("Rule 5 (news filter) failed (continuing): %s", exc)

        # ── All rules passed ─────────────────────────────────────────────
        return MetaResult(
            allowed=True,
            block_reason="",
            position_size_mult=position_mult,
            adjusted_confidence=adjusted_conf,
            lgbm_prob=lgbm_prob,
            hmm_state=hmm_state,
        )
