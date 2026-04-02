"""
GoldSignalAI -- execution/position_monitor.py
===============================================
Stage 11: Position management — runs alongside the signal loop.

Responsibilities (checked every 15 minutes):
  1. Trailing stop: if unrealised profit >= 1R, move SL to breakeven + buffer
  2. Time exit:     if position age > 48 bars (12 hours M15), close it
  3. Friday close:  if Friday >= 20:00 UTC, close all positions (gap protection)
  4. Discord alert on any automated close
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import Config
from execution.mt5_bridge import MT5Bridge, PositionInfo

logger = logging.getLogger(__name__)

# 48 bars of M15 = 12 hours
MAX_POSITION_AGE = timedelta(hours=12)

# Buffer in price units added to breakeven SL
BREAKEVEN_BUFFER_PIPS = 5  # 5 pips = 0.5 price units on Gold


class PositionMonitor:
    """
    Periodic position management.

    Call check_and_manage_positions() on every signal loop iteration.
    """

    def __init__(self, bridge: MT5Bridge, notifier=None):
        self._bridge = bridge
        self._notifier = notifier  # DiscordNotifier or None

    def check_and_manage_positions(
        self,
        current_time: datetime,
        current_price: float,
        current_atr: float,
    ) -> None:
        """
        Run all position management checks.

        Args:
            current_time:   UTC datetime of the current bar.
            current_price:  Latest close price.
            current_atr:    Current ATR-14 value (for trailing stop calc).
        """
        positions = self._bridge.get_open_positions()
        if not positions:
            return

        for pos in positions:
            # 1. Friday close (highest priority)
            if self._should_friday_close(current_time):
                self._close_position(pos, current_price, "FRIDAY_CLOSE")
                continue

            # 2. Time exit (48 bars / 12 hours)
            if self._should_time_exit(pos, current_time):
                self._close_position(pos, current_price, "TIME_EXIT_48")
                continue

            # 3. Trailing stop to breakeven
            self._check_trailing_stop(pos, current_price, current_atr)

    # ── Friday close ──────────────────────────────────────────────────────

    @staticmethod
    def _should_friday_close(current_time: datetime) -> bool:
        """Close all positions on Friday at or after 20:00 UTC."""
        return current_time.weekday() == 4 and current_time.hour >= 20

    # ── Time exit ─────────────────────────────────────────────────────────

    @staticmethod
    def _should_time_exit(pos: PositionInfo, current_time: datetime) -> bool:
        """Close if position has been open > 12 hours (48 M15 bars)."""
        age = current_time - pos.open_time
        return age >= MAX_POSITION_AGE

    # ── Trailing stop to breakeven ────────────────────────────────────────

    def _check_trailing_stop(
        self,
        pos: PositionInfo,
        current_price: float,
        current_atr: float,
    ) -> None:
        """
        If unrealised profit >= 1R (distance from entry to original SL),
        move SL to breakeven + small buffer.
        """
        if pos.sl == 0.0:
            return

        sl_distance = abs(pos.open_price - pos.sl)
        if sl_distance == 0:
            return

        buffer = BREAKEVEN_BUFFER_PIPS * Config.PIP_SIZE

        if pos.direction == "BUY":
            unrealised_pips = current_price - pos.open_price
            breakeven_sl = pos.open_price + buffer

            # Only move SL up (never widen it)
            if unrealised_pips >= sl_distance and breakeven_sl > pos.sl:
                logger.info(
                    "Trailing SL: ticket=%d moving SL %.2f -> %.2f (breakeven + buffer)",
                    pos.ticket, pos.sl, breakeven_sl,
                )
                self._bridge.modify_sl(pos.ticket, breakeven_sl)

        elif pos.direction == "SELL":
            unrealised_pips = pos.open_price - current_price
            breakeven_sl = pos.open_price - buffer

            # Only move SL down (never widen it)
            if unrealised_pips >= sl_distance and breakeven_sl < pos.sl:
                logger.info(
                    "Trailing SL: ticket=%d moving SL %.2f -> %.2f (breakeven + buffer)",
                    pos.ticket, pos.sl, breakeven_sl,
                )
                self._bridge.modify_sl(pos.ticket, breakeven_sl)

    # ── Close helper ──────────────────────────────────────────────────────

    def _close_position(
        self,
        pos: PositionInfo,
        current_price: float,
        reason: str,
    ) -> None:
        """Close a position and send Discord alert."""
        result = self._bridge.close_order(pos.ticket)

        if result.success:
            logger.info(
                "Position closed [%s]: ticket=%d %s %s @ %.2f (opened %.2f)",
                reason, pos.ticket, pos.direction, pos.symbol,
                current_price, pos.open_price,
            )
        else:
            logger.warning(
                "Failed to close position [%s]: ticket=%d — %s",
                reason, pos.ticket, result.message,
            )

        # Discord notification
        if self._notifier is not None:
            try:
                pnl_pips = (current_price - pos.open_price) if pos.direction == "BUY" \
                    else (pos.open_price - current_price)
                pnl_pips_display = pnl_pips / Config.PIP_SIZE
                self._notifier(
                    f"Position auto-closed [{reason}]\n"
                    f"  Ticket: {pos.ticket}\n"
                    f"  {pos.direction} {pos.symbol} {pos.volume} lots\n"
                    f"  PnL: {pnl_pips_display:+.1f} pips"
                )
            except Exception as exc:
                logger.warning("Discord notification failed: %s", exc)
