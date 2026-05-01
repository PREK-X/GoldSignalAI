"""
GoldSignalAI -- paper_trading/engine.py
========================================
Stage 4: Automated paper-trading forward-test engine.

Mirrors backtest/engine.py exit semantics on a single observation per cycle
(close price at M15 candle close):

  - 48-bar TIME exit (12 hours M15)
  - SL touch (conservative: close crosses sl_price)
  - TP1 touch (conservative: close crosses tp_price)

Trades exit fully at TP1. tp2_price is recorded observationally only — no
sizing, risk, or downstream decision in this stage may consume it.

PnL formula (mirrors backtest/engine.py:708-718 time-exit case):
    direction_sign = +1 (BUY) | -1 (SELL)
    pnl_pips   = direction_sign * (exit_price - entry_price) / Config.PIP_SIZE
    pnl_dollar = pnl_pips * Config.GOLD_PIP_VALUE * lot_size
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

from config import Config
from database import db

logger = logging.getLogger(__name__)

# 48 M15 bars = 12 hours; matches PositionMonitor.MAX_POSITION_AGE.
MAX_BARS_OPEN = 48

# Outcome thresholds in pips (avoid float-noise BREAKEVEN flips).
_BREAKEVEN_PIPS = 0.5


class PaperTradingEngine:
    """
    Open and update paper trades. Persists every state transition to SQLite
    via the helpers in database/db.py.
    """

    def __init__(
        self,
        account_balance: float = Config.CHALLENGE_ACCOUNT_SIZE,
        risk_pct: float = Config.RISK_PER_TRADE_PCT,
    ):
        self._account_balance = account_balance
        self._risk_pct = risk_pct

    # ── Open ──────────────────────────────────────────────────────────────

    def open_trade(
        self,
        signal,
        entry_price: float,
        signal_id: Optional[int] = None,
    ) -> Optional[int]:
        """
        Persist a new open paper trade derived from a TradingSignal.
        Returns the paper_trades.id, or None on failure.
        """
        if signal.risk is None:
            logger.warning("paper open_trade: signal has no risk params -- skipped")
            return None
        if signal.direction not in ("BUY", "SELL"):
            logger.warning("paper open_trade: non-actionable direction %r",
                           signal.direction)
            return None

        return db.open_paper_trade(
            signal_id=signal_id,
            direction=signal.direction,
            entry_price=entry_price,
            sl_price=signal.risk.stop_loss,
            tp_price=signal.risk.tp1_price,
            tp2_price=signal.risk.tp2_price,  # OBSERVATIONAL ONLY (Stage 4)
            lot_size=signal.risk.suggested_lot,
            entry_time=signal.timestamp.isoformat(),
        )

    # ── Update ────────────────────────────────────────────────────────────

    def update_trades(
        self,
        current_price: float,
        current_time: datetime,
        bars_elapsed_fn: Callable[[datetime], int],
    ) -> list[dict]:
        """
        Walk every open paper trade, close any that hit SL / TP1 / 48-bar TIME
        on this bar's close. Returns the list of closed trade dicts (post-close
        view, including pnl + outcome).
        """
        closed: list[dict] = []
        for row in db.get_open_paper_trades():
            try:
                exit_info = self._evaluate_exit(row, current_price, current_time,
                                                bars_elapsed_fn)
            except Exception as exc:
                logger.warning("paper update: eval failed for id=%s: %s",
                               row.get("id"), exc)
                continue

            if exit_info is None:
                continue

            exit_price, exit_reason = exit_info
            pnl = self._compute_pnl(row, exit_price)
            ok = db.close_paper_trade(
                trade_id=row["id"],
                exit_price=exit_price,
                exit_time=current_time.isoformat(),
                exit_reason=exit_reason,
                pnl_pct=pnl["pnl_pct"],
                pnl_dollar=pnl["pnl_dollar"],
                outcome=pnl["outcome"],
            )
            if ok:
                closed.append({
                    **row,
                    "exit_price": exit_price,
                    "exit_time": current_time.isoformat(),
                    "exit_reason": exit_reason,
                    **pnl,
                })
        return closed

    # ── Stats ─────────────────────────────────────────────────────────────

    @staticmethod
    def get_stats() -> dict:
        return db.get_paper_trade_stats()

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _evaluate_exit(
        row: dict,
        current_price: float,
        current_time: datetime,
        bars_elapsed_fn: Callable[[datetime], int],
    ) -> Optional[tuple[float, str]]:
        """
        Decide whether the open trade exits this bar. Returns (exit_price, reason)
        or None if the trade stays open.

        Order of checks: SL → TP → TIME (matches backtest/engine.py for the
        single-observation case).
        """
        direction = row["direction"]
        sl = row["sl_price"]
        tp = row["tp_price"]

        if direction == "BUY":
            if current_price <= sl:
                return sl, "SL"
            if current_price >= tp:
                return tp, "TP"
        else:  # SELL
            if current_price >= sl:
                return sl, "SL"
            if current_price <= tp:
                return tp, "TP"

        try:
            entry_dt = datetime.fromisoformat(row["entry_time"])
        except Exception:
            entry_dt = current_time
        bars_open = bars_elapsed_fn(entry_dt)
        if bars_open >= MAX_BARS_OPEN:
            return current_price, "TIME"

        return None

    @staticmethod
    def _compute_pnl(row: dict, exit_price: float) -> dict:
        """
        Mirror backtest/engine.py full-lot PnL math.
        See module docstring for the exact formula and references.
        """
        entry = row["entry_price"]
        lot = row["lot_size"] or 0.0
        direction_sign = 1 if row["direction"] == "BUY" else -1

        pnl_pips = direction_sign * (exit_price - entry) / Config.PIP_SIZE
        pnl_dollar = pnl_pips * Config.GOLD_PIP_VALUE * lot
        pnl_pct = (direction_sign * (exit_price - entry) / entry * 100.0) \
            if entry > 0 else 0.0

        if pnl_pips > _BREAKEVEN_PIPS:
            outcome = "WIN"
        elif pnl_pips < -_BREAKEVEN_PIPS:
            outcome = "LOSS"
        else:
            outcome = "BREAKEVEN"

        return {
            "pnl_pips": pnl_pips,
            "pnl_dollar": pnl_dollar,
            "pnl_pct": pnl_pct,
            "outcome": outcome,
        }


# Module-level singleton: live bot uses one engine instance.
_INSTANCE: Optional[PaperTradingEngine] = None


def get_paper_engine() -> PaperTradingEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = PaperTradingEngine()
    return _INSTANCE


def reset_paper_engine_for_tests() -> None:
    global _INSTANCE
    _INSTANCE = None
