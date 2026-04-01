"""
GoldSignalAI — signals/news_filter.py
=======================================
Stage 10: News & Volatility Filter

Two independent checks before a signal is allowed:

A. Economic Calendar Gate (live only — calendar empty in backtest)
   - Block 30 min BEFORE a high-impact USD/XAU event
   - Block 15 min AFTER a high-impact event
   - Reduce to 50% size for medium-impact events

B. ATR Volatility Spike Detection (works in backtest and live)
   - Current ATR > 2.0× 28-bar rolling mean → block signal entirely
   - Current ATR > 1.5× 28-bar rolling mean → reduce size to 50%

C. Spread Monitor (optional, live only)
   - spread_pips > NEWS_MAX_SPREAD_PIPS → block signal

In backtest: calendar is always empty (ForexFactory only has ~2 weeks forward).
Only the ATR spike check runs on historical data. This is correct and expected —
document it here so the discrepancy isn't confusing when reviewing results.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class NewsFilterResult:
    """Result from the news/volatility filter check."""
    allowed: bool
    block_reason: str          # "" if allowed
    position_size_mult: float  # 1.0 = full, 0.5 = half, 0.0 = blocked
    next_clear_time: Optional[datetime] = None  # when block lifts (if known)


class NewsFilter:
    """
    Stateless news + volatility filter.

    Can be instantiated once and reused. Calendar data is fetched from
    the module-level cache in data/news_fetcher.py (refreshed hourly).
    """

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────

    def check(
        self,
        signal_time: datetime,
        current_atr: float,
        rolling_atr_mean: float,
        spread_pips: Optional[float] = None,
    ) -> NewsFilterResult:
        """
        Run all news/volatility checks for a candidate signal.

        Args:
            signal_time:       UTC datetime of the signal bar.
            current_atr:       Current ATR-14 value.
            rolling_atr_mean:  28-bar rolling ATR mean (baseline volatility).
            spread_pips:       Current broker spread in pips (optional).

        Returns:
            NewsFilterResult with allowed/blocked and position size multiplier.
        """
        if not Config.NEWS_FILTER_ENABLED:
            return NewsFilterResult(allowed=True, block_reason="", position_size_mult=1.0)

        # Check A: Volatility spike (runs in backtest + live)
        vol_result = self._check_atr_spike(current_atr, rolling_atr_mean)
        if not vol_result.allowed:
            return vol_result

        # Check B: Economic calendar (live only — empty in backtest)
        cal_result = self._check_economic_calendar(signal_time)
        if not cal_result.allowed:
            return cal_result

        # Check C: Spread monitor (optional)
        if spread_pips is not None:
            spread_result = self._check_spread(spread_pips)
            if not spread_result.allowed:
                return spread_result

        # All checks passed — take the most conservative size multiplier
        final_mult = min(vol_result.position_size_mult, cal_result.position_size_mult)
        return NewsFilterResult(
            allowed=True,
            block_reason="",
            position_size_mult=final_mult,
        )

    # ─────────────────────────────────────────────────────────────────────
    # CHECK A: ATR SPIKE DETECTION
    # ─────────────────────────────────────────────────────────────────────

    def _check_atr_spike(
        self,
        current_atr: float,
        rolling_atr_mean: float,
    ) -> NewsFilterResult:
        """
        Detect abnormal ATR spikes indicating news-driven volatility.

        Thresholds (from config):
          ATR > SPIKE_BLOCK × mean → block entirely (default: 2.0×)
          ATR > SPIKE_REDUCE × mean → halve position (default: 1.5×)
        """
        if rolling_atr_mean <= 0:
            return NewsFilterResult(allowed=True, block_reason="", position_size_mult=1.0)

        atr_ratio = current_atr / rolling_atr_mean

        if atr_ratio >= Config.NEWS_ATR_SPIKE_BLOCK:
            return NewsFilterResult(
                allowed=False,
                block_reason=(
                    f"ATR spike: current={current_atr:.4f} is {atr_ratio:.1f}× "
                    f"28-bar mean ({rolling_atr_mean:.4f}) — volatility too high"
                ),
                position_size_mult=0.0,
            )

        if atr_ratio >= Config.NEWS_ATR_SPIKE_REDUCE:
            logger.debug(
                "Elevated volatility: ATR ratio %.2f >= %.1f — reducing to 50%% size",
                atr_ratio, Config.NEWS_ATR_SPIKE_REDUCE,
            )
            return NewsFilterResult(
                allowed=True,
                block_reason="",
                position_size_mult=0.5,
            )

        return NewsFilterResult(allowed=True, block_reason="", position_size_mult=1.0)

    # ─────────────────────────────────────────────────────────────────────
    # CHECK B: ECONOMIC CALENDAR
    # ─────────────────────────────────────────────────────────────────────

    def _check_economic_calendar(self, signal_time: datetime) -> NewsFilterResult:
        """
        Check if signal_time falls within a news blackout window.

        For high-impact events:
          Block from (event_time - 30min) to (event_time + 15min)
        For medium-impact events:
          Reduce size to 50% during event minute ± 5 minutes

        The calendar is fetched from the module-level cache in news_fetcher.py.
        In backtest, the RSS feed only has ~2 weeks forward so historical
        bars will always see an empty calendar — only the ATR check runs.
        """
        try:
            from data.news_fetcher import fetch_news_events
            events = fetch_news_events()
        except Exception as exc:
            logger.debug("News calendar fetch failed (continuing without): %s", exc)
            return NewsFilterResult(allowed=True, block_reason="", position_size_mult=1.0)

        if not events:
            return NewsFilterResult(allowed=True, block_reason="", position_size_mult=1.0)

        pre_high  = timedelta(minutes=Config.NEWS_HIGH_IMPACT_PRE_MIN)
        post_high = timedelta(minutes=Config.NEWS_HIGH_IMPACT_POST_MIN)
        med_window = timedelta(minutes=5)

        best_mult = 1.0

        for event in events:
            # Only check USD / XAU events
            if event.currency not in Config.NEWS_CURRENCIES_TO_WATCH:
                continue

            impact_lower = event.impact.lower()

            if impact_lower == "high" or event.is_high_impact or event.matches_gold_keywords:
                window_start = event.event_time - pre_high
                window_end   = event.event_time + post_high

                if window_start <= signal_time <= window_end:
                    clear_time = event.event_time + post_high
                    return NewsFilterResult(
                        allowed=False,
                        block_reason=(
                            f"High-impact event: {event.title} ({event.currency}) "
                            f"at {event.event_time.strftime('%H:%M UTC')}"
                        ),
                        position_size_mult=0.0,
                        next_clear_time=clear_time,
                    )

            elif impact_lower == "medium":
                window_start = event.event_time - med_window
                window_end   = event.event_time + med_window

                if window_start <= signal_time <= window_end:
                    best_mult = min(best_mult, Config.NEWS_MED_IMPACT_SIZE_MULT)

        return NewsFilterResult(
            allowed=True,
            block_reason="",
            position_size_mult=best_mult,
        )

    # ─────────────────────────────────────────────────────────────────────
    # CHECK C: SPREAD MONITOR
    # ─────────────────────────────────────────────────────────────────────

    def _check_spread(self, spread_pips: float) -> NewsFilterResult:
        """Block if spread is too wide (indicates low liquidity / news spike)."""
        if spread_pips > Config.NEWS_MAX_SPREAD_PIPS:
            return NewsFilterResult(
                allowed=False,
                block_reason=(
                    f"Wide spread: {spread_pips:.1f} pips > {Config.NEWS_MAX_SPREAD_PIPS:.0f} pip limit"
                ),
                position_size_mult=0.0,
            )
        return NewsFilterResult(allowed=True, block_reason="", position_size_mult=1.0)
