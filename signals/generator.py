"""
GoldSignalAI -- signals/generator.py
======================================
Final signal generation -- the top-level orchestrator that combines
ALL analysis layers into a complete, actionable trading signal.

Pipeline (runs every 15 minutes on candle close):
  1. Multi-timeframe analysis (M15 + H1) -> direction + confidence
  2. ML prediction (XGBoost, if enabled) -> confirmation or override
  3. MetaDecision cascade (HMM + LGBM + confidence adj + session loss + news)
  4. Risk calculation -> SL, TP1, TP2, lot size
  5. Package everything into a TradingSignal dataclass

Stage 11: The separate HMM / Deep / LGBM filter blocks have been replaced
by a single MetaDecision.decide() call, mirroring the backtest engine
(backtest/engine.py). This ensures live trading behaviour matches backtest.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from config import Config
from analysis.multi_timeframe import analyse, MultiTimeframeResult
from analysis.scoring import SignalScore
from ml.predictor import predict, MLPrediction, is_model_ready
from ml.predictor import predict_lgbm, LGBMPrediction, is_lgbm_ready
from signals.meta_decision import MetaDecision, MetaResult
from signals.risk_manager import calculate_risk, RiskParameters
from state.state_manager import StateManager

logger = logging.getLogger(__name__)

# ── Module-level singletons (initialised once, reused across calls) ──────
_meta_decision: Optional[MetaDecision] = None
_state_manager: Optional[StateManager] = None


def _get_meta_decision() -> MetaDecision:
    global _meta_decision
    if _meta_decision is None:
        _meta_decision = MetaDecision()
    return _meta_decision


def _get_state_manager() -> StateManager:
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


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
    confidence_pct:  float        # 0-99
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

    # Meta-decision details (Stage 11)
    meta_result:     Optional[MetaResult] = None
    position_size_multiplier: float = 1.0

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
            return f"{Config.PRIMARY_TIMEFRAME} + {Config.CONFIRMATION_TIMEFRAME}"
        # Show each timeframe's direction so the user knows the actual state.
        m15_dir = self.mtf_result.m15.score.direction if self.mtf_result.m15.score else "?"
        h1_dir  = self.mtf_result.h1.score.direction  if self.mtf_result.h1.score  else "?"
        return f"M15:{m15_dir} H1:{h1_dir}"

    @property
    def indicator_label(self) -> str:
        if self.bearish_count > self.bullish_count:
            return f"{self.bearish_count}/{self.total_indicators} Bearish"
        return f"{self.bullish_count}/{self.total_indicators} Bullish"

    @property
    def ml_label(self) -> str:
        if self.direction == "WAIT":
            return "N/A (WAIT)"
        if not self.ml_prediction.available:
            return "N/A (model not trained)"
        return f"{'YES' if self.ml_confirms else 'NO'}"

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
# HELPER: compute rolling ATR mean from M15 DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rolling_atr_mean(df, current_atr: float, window: int = 28) -> float:
    """
    Compute the rolling mean of Wilder's ATR-14 over the last `window` bars.

    Uses EWM(span=14) smoothing to match the backtest engine's ATR calculation
    (backtest/engine.py lines 1352-1355), then returns the mean of the last
    `window` ATR values.

    Falls back to current_atr if the DataFrame doesn't have enough data.
    """
    try:
        if df is None or len(df) < window + 14:
            return current_atr

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range series
        prev_close = close.shift(1)
        tr = np.maximum(
            high - low,
            np.maximum(
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ),
        )

        # Wilder's ATR-14 (EWM smoothing, matching backtest engine)
        atr14 = tr.ewm(span=14, adjust=False).mean()

        # Rolling mean of the last `window` ATR-14 values
        return float(atr14.iloc[-window:].mean())
    except Exception:
        return current_atr


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
    meta = _get_meta_decision()
    state_mgr = _get_state_manager()

    # ── Step 1: Multi-timeframe analysis ──────────────────────────────────
    logger.info("Generating signal...")
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

    # ── Step 2: ML prediction (XGBoost — legacy, disabled) ────────────────
    ml_pred = MLPrediction(available=False, reason="ML filter disabled (USE_ML_FILTER=False)")
    ml_confirms = False

    if Config.USE_ML_FILTER and mtf.m15.df is not None:
        try:
            ml_pred = predict(mtf.m15.df)
        except Exception as exc:
            logger.warning("ML prediction failed: %s", exc)
            ml_pred = MLPrediction(available=False, reason=f"Error: {exc}")

        ml_confirms = ml_pred.confirms(direction)

        if ml_pred.available and ml_pred.models_agree and direction in ("BUY", "SELL"):
            if not ml_pred.confirms(direction):
                logger.info(
                    "ML contradicts technical signal (%s vs ML=%s) -> forcing WAIT",
                    direction, ml_pred.direction
                )
                direction = "WAIT"
                confidence = 0.0

    # ── Step 3: Legacy news pause override ────────────────────────────────
    is_paused = news_paused
    if is_paused:
        direction = "WAIT"
        confidence = 0.0

    # ── Step 4: MetaDecision cascade (replaces separate HMM/LGBM/Deep) ───
    meta_result = None
    position_size_mult = 1.0

    if direction in ("BUY", "SELL"):
        # 4a. HMM regime state
        hmm_state = "RANGING"  # safe default
        if mtf.h1.df is not None:
            try:
                from analysis.regime_filter import get_current_regime
                _state, hmm_state, _mult = get_current_regime(mtf.h1.df)
            except Exception as exc:
                logger.warning("Regime filter failed (defaulting RANGING): %s", exc)

        # 4b. LGBM probability (always fetch for meta soft vote)
        lgbm_prob = -1.0  # unavailable sentinel
        if mtf.m15.df is not None:
            try:
                if is_lgbm_ready():
                    lgbm_pred = predict_lgbm(mtf.m15.df)
                    if lgbm_pred.available:
                        lgbm_prob = lgbm_pred.probability
            except Exception as exc:
                logger.warning("LGBM prediction failed (continuing): %s", exc)

        # 4c. ATR values for news/volatility filter
        current_atr = 0.0
        rolling_atr_mean = 0.0
        if mtf.m15.indicators:
            current_atr = mtf.m15.indicators.atr.value
            rolling_atr_mean = _compute_rolling_atr_mean(
                mtf.m15.df, current_atr, window=28
            )

        # 4d. Session consecutive losses
        session_losses = state_mgr.get_session_losses()

        # 4e. Run the 5-rule cascade
        meta_result = meta.decide(
            direction=direction,
            base_confidence=confidence,
            lgbm_prob=lgbm_prob,
            hmm_state=hmm_state,
            session_consecutive_losses=session_losses,
            signal_time=now,
            current_atr=current_atr,
            rolling_atr_mean=rolling_atr_mean,
        )

        if not meta_result.allowed:
            logger.info("MetaDecision blocked: %s", meta_result.block_reason)
            direction = "WAIT"
            confidence = 0.0
        else:
            # Apply adjusted confidence and position size multiplier
            old_confidence = confidence
            confidence = meta_result.adjusted_confidence
            position_size_mult = meta_result.position_size_mult

            if confidence > old_confidence:
                logger.info("MetaDecision boosted confidence %.1f%% → %.1f%%", old_confidence, confidence)

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
        # Apply meta-decision position size multiplier
        if risk is not None and position_size_mult < 1.0:
            risk.suggested_lot = round(max(0.01, risk.suggested_lot * position_size_mult), 2)

    # ── Step 6: Build reason string ───────────────────────────────────────
    if is_paused:
        reason = f"WAIT -- {pause_reason}"
    elif direction == "WAIT" and meta_result is not None and meta_result.block_reason:
        reason = f"WAIT -- {meta_result.block_reason}"
    elif direction == "WAIT" and mtf.direction in ("BUY", "SELL"):
        reason = f"WAIT -- ML contradicts ({ml_pred.direction} vs {mtf.direction})"
    else:
        reason = mtf.reason
        if meta_result is not None and position_size_mult < 1.0:
            reason += f" [{meta_result.hmm_state}: {position_size_mult:.0%} size]"

    # ── Step 7: Record signal state ───────────────────────────────────────
    if direction in ("BUY", "SELL"):
        state_mgr.record_signal(direction, now)

    # ── Step 8: Package result ────────────────────────────────────────────
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
        meta_result=meta_result,
        position_size_multiplier=position_size_mult,
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

    reason = f"WAIT -- {pause_reason}" if is_paused else mtf.reason

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
