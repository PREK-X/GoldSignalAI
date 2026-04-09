"""
GoldSignalAI — analysis/multi_timeframe.py
============================================
Enforces the multi-timeframe agreement rule: both M15 and H1 must
produce the same directional signal (both BUY or both SELL) before
the system outputs a tradeable signal.

Pipeline:
  1. Fetch and process candles for both M15 and H1
  2. Run all analysis layers on each timeframe independently
  3. Score each timeframe via scoring.py
  4. Compare directions:
       M15=BUY  + H1=BUY  → BUY  (confirmed)
       M15=SELL + H1=SELL → SELL (confirmed)
       Anything else      → WAIT (disagreement)
  5. Final confidence = min(m15_confidence, h1_confidence)
     (the weaker timeframe limits the overall signal quality)

This module is the single entry point called by signals/generator.py.
It returns a MultiTimeframeResult containing both individual scores
plus the merged final decision.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import Config
from data.fetcher import get_candles
from data.processor import process
from analysis.indicators import calculate_all, AllIndicators
from analysis.sr_levels import detect_sr_levels, SRLevels
from analysis.fibonacci import calculate_fibonacci, FibonacciLevels
from analysis.candlestick import detect_patterns, CandlestickAnalysis
from analysis.scoring import score_signal, SignalScore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TimeframeAnalysis:
    """Full analysis result for one timeframe."""
    timeframe:   str
    df:          Optional[pd.DataFrame]
    indicators:  Optional[AllIndicators]
    sr_levels:   Optional[SRLevels]
    fib_levels:  Optional[FibonacciLevels]
    candlestick: Optional[CandlestickAnalysis]
    score:       Optional[SignalScore]
    valid:       bool  # True if all layers computed successfully


@dataclass
class MultiTimeframeResult:
    """
    Combined M15 + H1 analysis with the final merged decision.

    Attributes:
        m15             : Full M15 analysis
        h1              : Full H1 analysis
        direction       : "BUY" | "SELL" | "WAIT" (merged)
        confidence_pct  : min(m15, h1) confidence
        timeframes_agree: True if both point the same direction
        reason          : Explanation of the merged decision
        latest_close    : Most recent M15 close price
    """
    m15:              TimeframeAnalysis
    h1:               TimeframeAnalysis
    direction:        str
    confidence_pct:   float
    timeframes_agree: bool
    reason:           str
    latest_close:     float = 0.0

    @property
    def is_actionable(self) -> bool:
        return self.direction in ("BUY", "SELL")

    def summary(self) -> str:
        m = self.m15.score.direction if self.m15.score else "N/A"
        h = self.h1.score.direction  if self.h1.score  else "N/A"
        return (
            f"{self.direction} | {self.confidence_pct:.0f}% | "
            f"M15={m} H1={h} | agree={self.timeframes_agree}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-TIMEFRAME ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _analyse_timeframe(
    timeframe: str,
    n_candles: int = Config.LOOKBACK_CANDLES,
    symbol:    str = Config.SYMBOL,
    df_override: Optional[pd.DataFrame] = None,
    bar_time:  Optional[pd.Timestamp] = None,
) -> TimeframeAnalysis:
    """
    Run the full analysis pipeline on a single timeframe.

    Fetches data, processes it, calculates all indicators, S/R, Fib,
    candlestick patterns, and scores the result.

    Args:
        timeframe:   "M15" or "H1"
        n_candles:   How many bars to fetch
        symbol:      Trading symbol
        df_override: If provided, skip fetching and use this DataFrame
                     (useful for testing and backtesting)

    Returns:
        TimeframeAnalysis with all layers populated (or valid=False on failure)
    """
    label = timeframe

    # ── Step 1: Fetch ─────────────────────────────────────────────────────
    if df_override is not None:
        df = df_override
    else:
        raw = get_candles(timeframe=timeframe, n_candles=n_candles, symbol=symbol)
        if raw is None:
            logger.error("[%s] Data fetch failed.", label)
            return TimeframeAnalysis(
                timeframe=timeframe, df=None, indicators=None,
                sr_levels=None, fib_levels=None, candlestick=None,
                score=None, valid=False,
            )
        df = process(raw, timeframe=timeframe, label=label)
        if df is None:
            logger.error("[%s] Data processing failed.", label)
            return TimeframeAnalysis(
                timeframe=timeframe, df=None, indicators=None,
                sr_levels=None, fib_levels=None, candlestick=None,
                score=None, valid=False,
            )

    # ── Step 2: Indicators ────────────────────────────────────────────────
    try:
        indicators  = calculate_all(df)
        sr_levels   = detect_sr_levels(df)
        fib_levels  = calculate_fibonacci(df)
        candlestick = detect_patterns(df)
    except Exception as exc:
        logger.exception("[%s] Analysis layer error: %s", label, exc)
        return TimeframeAnalysis(
            timeframe=timeframe, df=df, indicators=None,
            sr_levels=None, fib_levels=None, candlestick=None,
            score=None, valid=False,
        )

    # ── Step 3: Score ─────────────────────────────────────────────────────
    try:
        score = score_signal(indicators, sr_levels, fib_levels, candlestick, bar_time=bar_time)
    except Exception as exc:
        logger.exception("[%s] Scoring error: %s", label, exc)
        return TimeframeAnalysis(
            timeframe=timeframe, df=df, indicators=indicators,
            sr_levels=sr_levels, fib_levels=fib_levels, candlestick=candlestick,
            score=None, valid=False,
        )

    logger.info("[%s] → %s", label, score.summary())

    return TimeframeAnalysis(
        timeframe=timeframe, df=df, indicators=indicators,
        sr_levels=sr_levels, fib_levels=fib_levels, candlestick=candlestick,
        score=score, valid=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME MERGE
# ─────────────────────────────────────────────────────────────────────────────

def _merge_decisions(
    m15: TimeframeAnalysis,
    h1:  TimeframeAnalysis,
) -> MultiTimeframeResult:
    """
    Merge M15 and H1 scores into a single decision.

    Agreement rule:
      Both must be BUY  → final = BUY   (confidence = min of both)
      Both must be SELL → final = SELL  (confidence = min of both)
      Anything else     → final = WAIT

    If either timeframe failed to produce a score, the result is WAIT.
    """
    latest_close = 0.0

    # Handle failures
    if not m15.valid or m15.score is None:
        return MultiTimeframeResult(
            m15=m15, h1=h1, direction="WAIT", confidence_pct=0.0,
            timeframes_agree=False,
            reason="WAIT — M15 analysis failed, cannot confirm",
            latest_close=latest_close,
        )
    if not h1.valid or h1.score is None:
        return MultiTimeframeResult(
            m15=m15, h1=h1, direction="WAIT", confidence_pct=0.0,
            timeframes_agree=False,
            reason="WAIT — H1 analysis failed, cannot confirm",
            latest_close=latest_close,
        )

    m15_dir = m15.score.direction
    h1_dir  = h1.score.direction
    m15_conf = m15.score.confidence_pct
    h1_conf  = h1.score.confidence_pct

    if m15.indicators:
        latest_close = m15.indicators.latest_close

    # ── Agreement check ───────────────────────────────────────────────────
    agree = (m15_dir == h1_dir) and m15_dir in ("BUY", "SELL")

    if agree:
        # Both timeframes confirm — use the weaker confidence
        direction  = m15_dir
        confidence = min(m15_conf, h1_conf)
        reason = (
            f"{direction} confirmed on M15 ({m15_conf:.0f}%) + H1 ({h1_conf:.0f}%) "
            f"— confidence {confidence:.0f}%"
        )
    else:
        direction  = "WAIT"
        confidence = 0.0
        reason = (
            f"WAIT — timeframes disagree: M15={m15_dir}({m15_conf:.0f}%) "
            f"vs H1={h1_dir}({h1_conf:.0f}%)"
        )

    result = MultiTimeframeResult(
        m15=m15, h1=h1,
        direction=direction,
        confidence_pct=confidence,
        timeframes_agree=agree,
        reason=reason,
        latest_close=latest_close,
    )

    logger.info("Multi-TF: %s", result.summary())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    symbol:     str = Config.SYMBOL,
    n_candles:  int = Config.LOOKBACK_CANDLES,
    df_m15:     Optional[pd.DataFrame] = None,
    df_h1:      Optional[pd.DataFrame] = None,
) -> MultiTimeframeResult:
    """
    Run the full multi-timeframe analysis pipeline.

    This is the primary function called by signals/generator.py on
    every new candle close. It orchestrates:
      M15 full analysis → score
      H1  full analysis → score
      Merge decisions   → final BUY/SELL/WAIT

    Args:
        symbol:    Trading symbol (default from Config)
        n_candles: Candles to fetch per timeframe
        df_m15:    Pre-fetched M15 DataFrame (skips fetch if provided)
        df_h1:     Pre-fetched H1 DataFrame (skips fetch if provided)

    Returns:
        MultiTimeframeResult with all analysis data and the final decision.
    """
    logger.info("Running multi-timeframe analysis for %s…", symbol)

    # Extract M15 bar_time for session gate (only applies to primary TF).
    # The session gate blocks signals outside NY hours (13:00-21:59 UTC).
    m15_bar_time = None
    m15_src = df_m15 if df_m15 is not None else None
    if m15_src is not None and len(m15_src) > 0:
        m15_bar_time = m15_src.index[-1]
    elif df_m15 is None:
        # Will be fetched inside _analyse_timeframe; extract after
        pass

    m15 = _analyse_timeframe(
        Config.PRIMARY_TIMEFRAME, n_candles, symbol, df_override=df_m15,
        bar_time=m15_bar_time,
    )

    # If bar_time wasn't available before fetch, extract from the fetched df
    if m15_bar_time is None and m15.valid and m15.df is not None and len(m15.df) > 0:
        m15_bar_time = m15.df.index[-1]
        # Re-score M15 with the session gate active
        if m15.score is not None:
            from analysis.scoring import score_signal as _rescore
            m15 = TimeframeAnalysis(
                timeframe=m15.timeframe, df=m15.df, indicators=m15.indicators,
                sr_levels=m15.sr_levels, fib_levels=m15.fib_levels,
                candlestick=m15.candlestick,
                score=_rescore(m15.indicators, m15.sr_levels, m15.fib_levels,
                               m15.candlestick, bar_time=m15_bar_time),
                valid=True,
            )

    h1 = _analyse_timeframe(
        Config.CONFIRMATION_TIMEFRAME, n_candles, symbol, df_override=df_h1
    )

    return _merge_decisions(m15, h1)
