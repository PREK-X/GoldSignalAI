"""
GoldSignalAI — signals/generator.py
======================================
Final signal generation — the top-level orchestrator that combines
ALL analysis layers into a complete, actionable trading signal.

Pipeline (runs every 15 minutes on candle close):
  1. Multi-timeframe analysis (M15 + H1) → direction + confidence
  2. ML prediction → confirmation or override to WAIT
  3. Risk calculation → SL, TP1, TP2, lot size
  4. Package everything into a TradingSignal dataclass

Filtering rules applied:
  - Confidence must be >= 70%
  - Both timeframes must agree
  - ML must confirm (or at least not contradict)
  - If ML is unavailable, technical signal still fires (with a note)
  - WAIT signals are generated but NOT alerted

The TradingSignal dataclass contains everything the formatter,
Telegram bot, and dashboard need to display a complete signal.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import Config
from analysis.multi_timeframe import analyse, MultiTimeframeResult
from analysis.scoring import SignalScore
from ml.predictor import predict, MLPrediction, is_model_ready
from signals.risk_manager import calculate_risk, RiskParameters

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradingSignal:
    """
    Complete trading signal with all data needed for display and execution.

    This is the final output of the entire analysis pipeline, consumed by:
      - signals/formatter.py  (text formatting)
      - alerts/telegram_bot.py (alert delivery)
      - alerts/chart_generator.py (chart image)
      - dashboard/app.py (web dashboard)
      - propfirm/tracker.py (compliance tracking)
    """
    # Core signal
    direction:       str          # "BUY" | "SELL" | "WAIT"
    confidence_pct:  float        # 0–99
    timestamp:       datetime

    # Price levels
    entry_price:     float
    risk:            Optional[RiskParameters]  # None for WAIT signals

    # Analysis breakdown
    mtf_result:      MultiTimeframeResult
    ml_prediction:   MLPrediction

    # Display helpers
    bullish_count:   int
    bearish_count:   int
    total_indicators: int
    ml_confirms:     bool
    reason:          str
    is_paused:       bool = False  # True during news events
    pause_reason:    str  = ""

    @property
    def is_actionable(self) -> bool:
        """True if this is a BUY or SELL that should be alerted."""
        return self.direction in ("BUY", "SELL") and not self.is_paused

    @property
    def symbol(self) -> str:
        return Config.SYMBOL_DISPLAY

    @property
    def timeframe_label(self) -> str:
        agree = self.mtf_result.timeframes_agree
        if agree:
            return f"{Config.PRIMARY_TIMEFRAME} + {Config.CONFIRMATION_TIMEFRAME} ✅"
        # Show each timeframe's direction so the user knows the actual state.
        m15_dir = self.mtf_result.m15.score.direction if self.mtf_result.m15.score else "?"
        h1_dir  = self.mtf_result.h1.score.direction  if self.mtf_result.h1.score  else "?"
        return f"M15:{m15_dir} H1:{h1_dir} ❌"

    @property
    def indicator_label(self) -> str:
        # Show the dominant direction so the card is never misleadingly zero.
        # e.g. "4/10 Bearish" when market is bearish, "7/10 Bullish" when bullish.
        if self.bearish_count > self.bullish_count:
            return f"{self.bearish_count}/{self.total_indicators} Bearish"
        return f"{self.bullish_count}/{self.total_indicators} Bullish"

    @property
    def ml_label(self) -> str:
        # For WAIT signals ML confirmation doesn't apply.
        if self.direction == "WAIT":
            return "N/A (WAIT)"
        if not self.ml_prediction.available:
            return "N/A (model not trained)"
        return f"{'YES ✅' if self.ml_confirms else 'NO ❌'}"

    @property
    def rr_label(self) -> str:
        if self.risk is None:
            return "N/A"
        return f"1:{self.risk.tp1_rr:.0f} / 1:{self.risk.tp2_rr:.0f}"

    def summary(self) -> str:
        return (
            f"{self.direction} | {self.confidence_pct:.0f}% | "
            f"Entry={self.entry_price:.2f} | "
            f"{self.indicator_label} | ML={self.ml_label}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(
    news_paused:  bool = False,
    pause_reason: str  = "",
) -> TradingSignal:
    """
    Run the full analysis pipeline and produce a TradingSignal.

    This is the primary function called by main.py on every new
    M15 candle close.

    Args:
        news_paused:  If True, force WAIT regardless of analysis
        pause_reason: Reason for the news pause (displayed in signal)

    Returns:
        TradingSignal with all data populated.
    """
    now = datetime.now(timezone.utc)

    # ── Step 1: Multi-timeframe analysis ──────────────────────────────────
    logger.info("Generating signal…")
    mtf = analyse()

    entry_price = mtf.latest_close
    direction   = mtf.direction
    confidence  = mtf.confidence_pct

    # Get indicator counts from M15 (primary timeframe)
    if mtf.m15.score:
        bull_count = mtf.m15.score.bullish_count
        bear_count = mtf.m15.score.bearish_count
    else:
        bull_count = bear_count = 0

    # ── Step 2: ML prediction ─────────────────────────────────────────────
    ml_pred = MLPrediction(available=False, reason="ML not run")

    if mtf.m15.df is not None:
        try:
            ml_pred = predict(mtf.m15.df)
        except Exception as exc:
            logger.warning("ML prediction failed: %s", exc)
            ml_pred = MLPrediction(available=False, reason=f"Error: {exc}")

    # ML confirmation check
    ml_confirms = ml_pred.confirms(direction)

    # If ML is available and actively contradicts, downgrade to WAIT
    if ml_pred.available and ml_pred.models_agree and direction in ("BUY", "SELL"):
        if not ml_pred.confirms(direction):
            logger.info(
                "ML contradicts technical signal (%s vs ML=%s) → forcing WAIT",
                direction, ml_pred.direction
            )
            direction = "WAIT"
            confidence = 0.0

    # ── Step 3: News pause override ───────────────────────────────────────
    is_paused = news_paused
    if is_paused:
        direction = "WAIT"
        confidence = 0.0

    # ── Step 4: Risk calculation (only for actionable signals) ────────────
    risk = None
    if direction in ("BUY", "SELL") and mtf.m15.indicators:
        atr_val = mtf.m15.indicators.atr.value
        sr      = mtf.m15.sr_levels
        risk = calculate_risk(
            entry_price=entry_price,
            direction=direction,
            atr_value=atr_val,
            sr_levels=sr,
        )

    # ── Step 5: Build reason string ───────────────────────────────────────
    if is_paused:
        reason = f"WAIT — {pause_reason}"
    elif direction == "WAIT" and mtf.direction in ("BUY", "SELL"):
        reason = f"WAIT — ML contradicts ({ml_pred.direction} vs {mtf.direction})"
    else:
        reason = mtf.reason

    # ── Step 6: Package result ────────────────────────────────────────────
    signal = TradingSignal(
        direction=direction,
        confidence_pct=confidence,
        timestamp=now,
        entry_price=entry_price,
        risk=risk,
        mtf_result=mtf,
        ml_prediction=ml_pred,
        bullish_count=bull_count,
        bearish_count=bear_count,
        total_indicators=Config.TOTAL_INDICATORS,
        ml_confirms=ml_confirms,
        reason=reason,
        is_paused=is_paused,
        pause_reason=pause_reason,
    )

    logger.info("Signal: %s", signal.summary())
    return signal


def generate_from_data(
    m15_df,
    h1_df,
    news_paused: bool = False,
    pause_reason: str = "",
) -> TradingSignal:
    """
    Generate a signal from pre-fetched DataFrames (for backtesting).

    Same as generate_signal() but skips the data fetch step.
    """
    now = datetime.now(timezone.utc)

    mtf = analyse(df_m15=m15_df, df_h1=h1_df)
    entry_price = mtf.latest_close
    direction   = mtf.direction
    confidence  = mtf.confidence_pct

    if mtf.m15.score:
        bull_count = mtf.m15.score.bullish_count
        bear_count = mtf.m15.score.bearish_count
    else:
        bull_count = bear_count = 0

    ml_pred = MLPrediction(available=False, reason="Skipped in backtest mode")
    ml_confirms = False

    is_paused = news_paused

    risk = None
    if direction in ("BUY", "SELL") and mtf.m15.indicators:
        atr_val = mtf.m15.indicators.atr.value
        risk = calculate_risk(
            entry_price=entry_price,
            direction=direction,
            atr_value=atr_val,
            sr_levels=mtf.m15.sr_levels,
        )

    reason = f"WAIT — {pause_reason}" if is_paused else mtf.reason

    return TradingSignal(
        direction=direction,
        confidence_pct=confidence,
        timestamp=now,
        entry_price=entry_price,
        risk=risk,
        mtf_result=mtf,
        ml_prediction=ml_pred,
        bullish_count=bull_count,
        bearish_count=bear_count,
        total_indicators=Config.TOTAL_INDICATORS,
        ml_confirms=ml_confirms,
        reason=reason,
        is_paused=is_paused,
        pause_reason=pause_reason,
    )
