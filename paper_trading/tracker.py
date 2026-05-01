"""
GoldSignalAI -- paper_trading/tracker.py
=========================================
Stage 4: Per-cycle wrapper around PaperTradingEngine.

Called once per signal-loop iteration (M15 candle close) from main.py BEFORE
generate_signal(). Walks every open paper trade and closes any whose price
crossed SL / TP1 (full close) or aged past 48 M15 bars (TIME exit). Discord-
notifies on each close. tp2_price is logged in DB observationally only.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from config import Config
from database import db
from paper_trading.engine import PaperTradingEngine, get_paper_engine

logger = logging.getLogger(__name__)

_M15_SECONDS = 900


def _format_paper_exit(t: dict) -> str:
    """Discord message for a closed paper trade."""
    pnl_pips = t.get("pnl_pips") or 0.0
    pnl_dollar = t.get("pnl_dollar") or 0.0
    return (
        f"📝 Paper trade closed [{t.get('exit_reason', '?')}]\n"
        f"  {t.get('direction')} {Config.SYMBOL} {t.get('lot_size'):.2f} lots\n"
        f"  Entry {t.get('entry_price'):.2f} → Exit {t.get('exit_price'):.2f}\n"
        f"  PnL: {pnl_pips:+.1f} pips / ${pnl_dollar:+.2f} ({t.get('outcome')})"
    )


class PaperTradingTracker:
    """One run_cycle() call per M15 candle close."""

    def __init__(self, engine: Optional[PaperTradingEngine] = None,
                 notifier=None):
        self._engine = engine or get_paper_engine()
        # Lazy-load Discord notifier so import-time failures don't kill the bot.
        if notifier is None:
            try:
                from alerts.discord_notifier import send_message as discord_send
                notifier = discord_send
            except Exception as exc:
                logger.debug("Discord notifier unavailable for paper tracker: %s",
                             exc)
                notifier = None
        self._notifier = notifier

    def run_cycle(self, latest_bar: dict) -> dict:
        """
        latest_bar keys:
          - close:     float, latest M15 close price
          - timestamp: datetime (UTC), bar close time

        Returns: {"closed": [...closed trade dicts...], "open_count": int}
        """
        price = latest_bar["close"]
        now = latest_bar["timestamp"]

        bars_elapsed_fn = lambda entry_time: int(
            (now - entry_time).total_seconds() // _M15_SECONDS
        )

        closed = self._engine.update_trades(price, now, bars_elapsed_fn)

        if closed and self._notifier is not None:
            for t in closed:
                try:
                    self._notifier(_format_paper_exit(t))
                except Exception as exc:
                    logger.warning("Discord paper-exit alert failed: %s", exc)

        return {
            "closed": closed,
            "open_count": len(db.get_open_paper_trades()),
        }


# Module-level singleton
_INSTANCE: Optional[PaperTradingTracker] = None


def get_paper_tracker() -> PaperTradingTracker:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = PaperTradingTracker()
    return _INSTANCE


def reset_paper_tracker_for_tests() -> None:
    global _INSTANCE
    _INSTANCE = None
