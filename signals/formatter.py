"""
GoldSignalAI — signals/formatter.py
=====================================
Formats a TradingSignal into the exact display format specified in
the requirements, for both terminal output and Telegram messages.

Output format:
┌─────────────────────────────────────┐
│         GoldSignalAI                │
├─────────────────────────────────────┤
│ Asset:        XAU/USD (Gold)        │
│ Signal:       BUY / SELL / WAIT     │
│ Entry Price:  2,312.50              │
│ Stop Loss:    2,298.00 (-14.5 pips) │
│ Take Profit 1: 2,341.00 (+28.5p)   │
│ Take Profit 2: 2,370.00 (+57 p)    │
│ Confidence:   73%                   │
│ Risk/Reward:  1:2 / 1:3             │
│ Timeframe:    M15 + H1              │
│ ML Confirm:   YES / NO              │
│ Indicators:   8/10 Bullish          │
│ Timestamp:    2025-03-11 14:30 UTC  │
└─────────────────────────────────────┘
"""

import logging
from datetime import datetime, timezone

from signals.generator import TradingSignal
from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ICONS
# ─────────────────────────────────────────────────────────────────────────────

_DIRECTION_ICON = {
    "BUY":  "BUY 🟢",
    "SELL": "SELL 🔴",
    "WAIT": "WAIT ⚪",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def format_signal(sig: TradingSignal) -> str:
    """
    Format a TradingSignal into the full display card.

    Works for both terminal (monospace) and Telegram (monospace block).
    """
    d = sig.direction
    icon = _DIRECTION_ICON.get(d, d)

    # Price formatting helper
    def _f(price: float) -> str:
        return f"{price:,.2f}"

    # Entry
    entry_str = _f(sig.entry_price)

    # SL / TP lines
    if sig.risk:
        r = sig.risk
        sl_sign = "-" if d == "BUY" else "+"
        tp_sign = "+" if d == "BUY" else "-"
        sl_line  = f"{_f(r.stop_loss)} ({sl_sign}{r.sl_pips:.1f} pips)"
        tp1_line = f"{_f(r.tp1_price)} ({tp_sign}{r.tp1_pips:.1f} pips)"
        tp2_line = f"{_f(r.tp2_price)} ({tp_sign}{r.tp2_pips:.1f} pips)"
        rr_line  = sig.rr_label
        lot_line = f"{r.suggested_lot:.2f} lot ({Config.RISK_PER_TRADE_PCT}% risk = ${r.risk_usd:.2f})"
    else:
        sl_line = tp1_line = tp2_line = "—"
        rr_line  = "N/A"
        lot_line = "—"

    # ML
    ml_line = sig.ml_label

    # Indicators
    ind_line = sig.indicator_label

    # Timeframe
    tf_line = sig.timeframe_label

    # Confidence — for WAIT signals show the underlying M15 raw score so the
    # user can see how close the market is to a tradeable signal.
    if d == "WAIT":
        m15_score = sig.mtf_result.m15.score
        h1_score  = sig.mtf_result.h1.score
        m15_raw = m15_score.raw_confidence if m15_score else 0.0
        h1_raw  = h1_score.raw_confidence  if h1_score  else 0.0
        conf_line = f"0% (M15:{m15_raw:.0f}% H1:{h1_raw:.0f}% — need {Config.MIN_CONFIDENCE_PCT}%)"
    else:
        conf_line = f"{sig.confidence_pct:.0f}%"

    # Timestamp
    ts = sig.timestamp.strftime("%Y-%m-%d %H:%M UTC")

    # ── Build the card ────────────────────────────────────────────────────
    w = 41   # inner width
    sep = "─" * w

    lines = [
        f"┌{sep}┐",
        f"│{'GoldSignalAI 🤖':^{w}}│",
        f"├{sep}┤",
        f"│ {'Asset:':<14}{Config.SYMBOL_DISPLAY:<{w-15}}│",
        f"│ {'Signal:':<14}{icon:<{w-15}}│",
        f"│ {'Entry Price:':<14}{entry_str:<{w-15}}│",
        f"│ {'Stop Loss:':<14}{sl_line:<{w-15}}│",
        f"│ {'Take Profit 1:':<15}{tp1_line:<{w-16}}│",
        f"│ {'Take Profit 2:':<15}{tp2_line:<{w-16}}│",
        f"│ {'Confidence:':<14}{conf_line:<{w-15}}│",
        f"│ {'Risk/Reward:':<14}{rr_line:<{w-15}}│",
        f"│ {'Lot Size:':<14}{lot_line:<{w-15}}│",
        f"│ {'Timeframe:':<14}{tf_line:<{w-15}}│",
        f"│ {'ML Confirm:':<14}{ml_line:<{w-15}}│",
        f"│ {'Indicators:':<14}{ind_line:<{w-15}}│",
        f"│ {'Timestamp:':<14}{ts:<{w-15}}│",
        f"└{sep}┘",
    ]

    # News pause banner
    if sig.is_paused:
        pause_str = f"⚠️  NEWS EVENT — Bot paused: {sig.pause_reason}"
        lines.insert(3, f"│ {pause_str:<{w-1}}│")

    return "\n".join(lines)


def format_signal_short(sig: TradingSignal) -> str:
    """
    Short one-line format for logs and compact displays.
    """
    icon = _DIRECTION_ICON.get(sig.direction, sig.direction)
    sl = f"SL={sig.risk.stop_loss:.2f}" if sig.risk else ""
    tp = f"TP1={sig.risk.tp1_price:.2f}" if sig.risk else ""
    return (
        f"{icon} @ {sig.entry_price:,.2f} | "
        f"{sig.confidence_pct:.0f}% | {sig.indicator_label} | "
        f"{sl} {tp}"
    ).strip()


def format_signal_telegram(sig: TradingSignal) -> str:
    """
    Telegram-optimised format (wrapped in monospace block).
    """
    card = format_signal(sig)
    return f"```\n{card}\n```"


def format_wait_reason(sig: TradingSignal) -> str:
    """
    Short explanation for why a WAIT signal was generated.
    Used in dashboard status updates (not alerted to Telegram).
    """
    if sig.is_paused:
        return f"⚠️ Paused: {sig.pause_reason}"
    return sig.reason
