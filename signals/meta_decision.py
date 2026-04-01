"""
GoldSignalAI -- signals/meta_decision.py
==========================================
Stage 8: Meta-Decision Layer

Combines the scoring engine, HMM regime detector, and LightGBM classifier
into a single cascading gate system. Each rule can block or modify a signal.

Cascade order:
  Rule 1: HMM Hard Gate      — CRISIS blocks all; RANGING halves size
  Rule 2: LGBM Soft Vote     — strong disagreement blocks signal
  Rule 3: Confidence Boost   — HMM+LGBM agreement boosts/penalises confidence
  Rule 4: Session Loss Limit — 2 consecutive losses skip rest of session

The class is stateless between calls — session_consecutive_losses is managed
by the caller (backtest engine or signal generator).
"""

import logging
from dataclasses import dataclass

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


class MetaDecision:
    """
    Stateless meta-decision engine.

    Call decide() for every candidate signal. The caller tracks
    session_consecutive_losses externally.
    """

    def decide(
        self,
        direction: str,
        base_confidence: float,
        lgbm_prob: float,
        hmm_state: str,
        session_consecutive_losses: int,
    ) -> MetaResult:
        """
        Run the 4-rule cascade and return a MetaResult.

        Args:
            direction:  "BUY" or "SELL" from the scoring engine.
            base_confidence:  Scoring engine confidence (0-100).
            lgbm_prob:  LGBM P(UP) for this bar, or -1.0 if unavailable.
            hmm_state:  "TRENDING", "RANGING", or "CRISIS".
            session_consecutive_losses:  Losses so far this NY session.

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

        # ── All rules passed ─────────────────────────────────────────────
        return MetaResult(
            allowed=True,
            block_reason="",
            position_size_mult=position_mult,
            adjusted_confidence=adjusted_conf,
            lgbm_prob=lgbm_prob,
            hmm_state=hmm_state,
        )
