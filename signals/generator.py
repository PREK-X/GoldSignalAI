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
from ml.predictor import predict_lgbm, LGBMPrediction, is_lgbm_ready
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
    ml_pred = MLPrediction(available=False, reason="ML filter disabled (USE_ML_FILTER=False)")
    ml_confirms = False

    if Config.USE_ML_FILTER and mtf.m15.df is not None:
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

    # ── Step 3: HMM regime filter ───────────────────────────────────────
    regime_label = None
    regime_multiplier = 1.0
    if direction in ("BUY", "SELL") and mtf.h1.df is not None:
        try:
            from analysis.regime_filter import get_current_regime
            _state, regime_label, regime_multiplier = get_current_regime(mtf.h1.df)
            if regime_multiplier == 0.0:
                logger.info("CRISIS regime detected — suppressing %s signal", direction)
                direction = "WAIT"
                confidence = 0.0
            elif regime_multiplier < 1.0:
                logger.info("RANGING regime — position size will be scaled to %.1fx", regime_multiplier)
        except Exception as exc:
            logger.warning("Regime filter failed (continuing without): %s", exc)

    # ── Step 3b: Deep model direction filter ─────────────────────────────
    if Config.USE_DEEP_FILTER and direction in ("BUY", "SELL") and mtf.m15.df is not None:
        try:
            from ml.deep_predictor import predict_deep, is_deep_ready
            from ml.deep_features import build_deep_features
            if is_deep_ready():
                deep_feat = build_deep_features(mtf.m15.df)
                deep_pred = predict_deep(deep_feat, len(deep_feat) - 1)
                if deep_pred.available and not deep_pred.confirms(direction):
                    logger.info(
                        "Deep filter: P(up)=%.1f%% does not confirm %s → forcing WAIT",
                        deep_pred.probability * 100, direction,
                    )
                    direction = "WAIT"
                    confidence = 0.0
        except Exception as exc:
            logger.warning("Deep model prediction failed (continuing without): %s", exc)

    # ── Step 3c: LGBM profitability filter ────────────────────────────────
    lgbm_pred = LGBMPrediction(available=False, reason="LGBM filter disabled")
    if Config.USE_LGBM_FILTER and direction in ("BUY", "SELL") and mtf.m15.df is not None:
        try:
            lgbm_pred = predict_lgbm(mtf.m15.df)
            if lgbm_pred.available and not lgbm_pred.confirms(direction):
                logger.info(
                    "LGBM filter: P(profitable)=%.1f%% < %.0f%% → forcing WAIT",
                    lgbm_pred.probability * 100, Config.LGBM_MIN_PROBABILITY * 100,
                )
                direction = "WAIT"
                confidence = 0.0
        except Exception as exc:
            logger.warning("LGBM prediction failed (continuing without): %s", exc)

    # ── Step 4: News pause override ───────────────────────────────────────
    is_paused = news_paused
    if is_paused:
        direction = "WAIT"
        confidence = 0.0

    # ── Step 5: Risk calculation (only for actionable signals) ────────────
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
        # Apply regime multiplier to position size
        if risk is not None and regime_multiplier < 1.0:
            risk.suggested_lot = round(max(0.01, risk.suggested_lot * regime_multiplier), 2)

    # ── Step 6: Build reason string ───────────────────────────────────────
    if is_paused:
        reason = f"WAIT — {pause_reason}"
    elif direction == "WAIT" and regime_label == "CRISIS":
        reason = f"WAIT — CRISIS regime detected (high volatility)"
    elif direction == "WAIT" and mtf.direction in ("BUY", "SELL"):
        reason = f"WAIT — ML contradicts ({ml_pred.direction} vs {mtf.direction})"
    else:
        reason = mtf.reason
        if regime_label and regime_multiplier < 1.0:
            reason += f" [RANGING regime: {regime_multiplier:.0%} size]"

    # ── Step 7: Package result ────────────────────────────────────────────
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
