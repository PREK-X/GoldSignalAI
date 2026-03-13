"""
GoldSignalAI — propfirm/compliance_report.py
==============================================
Generates daily and weekly compliance reports for prop firm tracking.

Reports include:
  - Account status (balance, P/L, drawdown)
  - Daily trade summary
  - Challenge progress bar
  - Compliance warnings/breaches
  - Trade log for the period

Output formats:
  - Text (for Telegram / terminal)
  - Dict (for dashboard rendering)
"""

import logging
from datetime import date, datetime, timezone
from typing import Optional

from config import Config
from propfirm.profiles import (
    get_profile,
    format_profile_card,
    ComplianceStatus,
    ChallengeProgress,
)
from propfirm.tracker import ComplianceTracker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DAILY REPORT (TEXT)
# ─────────────────────────────────────────────────────────────────────────────

def generate_daily_report(tracker: ComplianceTracker) -> str:
    """
    Generate a text-based daily compliance report.

    Used by:
      - Telegram daily summary (5 PM EST)
      - Terminal output
      - Dashboard

    Args:
        tracker: The active ComplianceTracker

    Returns:
        Formatted report string.
    """
    profile = tracker.profile
    state   = tracker.state
    daily, dd = tracker.check_compliance()
    progress  = tracker.get_progress()

    w = 45
    sep = "─" * w

    # Progress bar (20 chars wide)
    bar_len = 20
    filled = int(progress.progress_pct / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    # Today's trades
    today_trades = tracker.get_daily_trades()
    today_wins  = sum(1 for t in today_trades if t.get("pnl_usd", 0) > 0)
    today_losses = sum(1 for t in today_trades if t.get("pnl_usd", 0) < 0)

    lines = [
        f"┌{sep}┐",
        f"│{'📊 Daily Compliance Report':^{w}}│",
        f"├{sep}┤",
        f"│ {'Firm:':<18}{profile.name:<{w-19}}│",
        f"│ {'Date:':<18}{date.today().isoformat():<{w-19}}│",
        f"├{sep}┤",
        f"│{'💰 Account Status':^{w}}│",
        f"├{sep}┤",
        f"│ {'Starting Balance:':<20}${state.starting_balance:>10,.2f}{' '*(w-32)}│",
        f"│ {'Current Balance:':<20}${state.current_balance:>10,.2f}{' '*(w-32)}│",
        f"│ {'Peak Balance:':<20}${state.peak_balance:>10,.2f}{' '*(w-32)}│",
        f"│ {'P/L:':<20}{tracker.profit_pct:>+10.2f}%{' '*(w-32)}│",
        f"├{sep}┤",
        f"│{'📈 Challenge Progress':^{w}}│",
        f"├{sep}┤",
        f"│ [{bar}] {progress.progress_pct:>5.1f}%{' '*(w-30)}│",
        f"│ {'Target:':<18}{progress.target_pct:.1f}%{' '*(w-24)}│",
        f"│ {'Trading Days:':<18}{state.trading_days}/{profile.min_trading_days}{' '*(w-24)}│",
        f"├{sep}┤",
        f"│{'⚠️  Compliance':^{w}}│",
        f"├{sep}┤",
        f"│ {daily.status_icon} {'Daily Loss:':<16}{daily.current_pct:.2f}% / {daily.limit_pct:.1f}%{' '*(w-32)}│",
        f"│ {dd.status_icon} {'Max Drawdown:':<16}{dd.current_pct:.2f}% / {dd.limit_pct:.1f}%{' '*(w-32)}│",
        f"├{sep}┤",
        f"│{'📋 Today Summary':^{w}}│",
        f"├{sep}┤",
        f"│ {'Trades Today:':<18}{len(today_trades):<{w-19}}│",
        f"│ {'Wins/Losses:':<18}{today_wins}W / {today_losses}L{' '*(w-26)}│",
        f"│ {'Daily P/L:':<18}${state.daily_pnl_usd:>+10,.2f}{' '*(w-30)}│",
        f"│ {'Win Rate (All):':<18}{tracker.win_rate:.0f}% ({state.winning_trades}W/{state.losing_trades}L){' '*(w-36)}│",
        f"└{sep}┘",
    ]

    # Add breach warnings at the bottom
    warnings = []
    if daily.breached:
        warnings.append(f"🔴 DAILY LOSS BREACHED — STOP TRADING")
    if dd.breached:
        warnings.append(f"🔴 MAX DRAWDOWN BREACHED — CHALLENGE FAILED")
    if daily.warning:
        warnings.append(f"🟡 Daily loss approaching limit ({daily.headroom_pct:.2f}% headroom)")
    if dd.warning:
        warnings.append(f"🟡 Drawdown approaching limit ({dd.headroom_pct:.2f}% headroom)")
    if progress.challenge_passed:
        warnings.append(f"🟢 CHALLENGE PASSED!")

    if warnings:
        lines.append("")
        for w_msg in warnings:
            lines.append(w_msg)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT AS DICT (for dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def generate_report_data(tracker: ComplianceTracker) -> dict:
    """
    Generate report data as a dict for the Streamlit dashboard.

    Returns:
        Dict with all report fields for easy rendering.
    """
    state   = tracker.state
    daily, dd = tracker.check_compliance()
    progress  = tracker.get_progress()
    today_trades = tracker.get_daily_trades()

    return {
        "firm_name":          tracker.profile.name,
        "date":               date.today().isoformat(),
        "starting_balance":   state.starting_balance,
        "current_balance":    state.current_balance,
        "peak_balance":       state.peak_balance,
        "profit_pct":         tracker.profit_pct,
        "drawdown_pct":       tracker.drawdown_pct,
        "daily_pnl_usd":     state.daily_pnl_usd,
        "total_trades":       state.total_trades,
        "winning_trades":     state.winning_trades,
        "losing_trades":      state.losing_trades,
        "win_rate":           tracker.win_rate,
        "trading_days":       state.trading_days,
        "daily_loss": {
            "current_pct":    daily.current_pct,
            "limit_pct":      daily.limit_pct,
            "headroom_pct":   daily.headroom_pct,
            "ok":             daily.ok,
            "warning":        daily.warning,
            "breached":       daily.breached,
            "icon":           daily.status_icon,
        },
        "drawdown": {
            "current_pct":    dd.current_pct,
            "limit_pct":      dd.limit_pct,
            "headroom_pct":   dd.headroom_pct,
            "ok":             dd.ok,
            "warning":        dd.warning,
            "breached":       dd.breached,
            "icon":           dd.status_icon,
        },
        "progress": {
            "current_pct":    progress.current_profit_pct,
            "target_pct":     progress.target_pct,
            "progress_pct":   progress.progress_pct,
            "target_met":     progress.target_met,
            "days_met":       progress.days_met,
            "passed":         progress.challenge_passed,
        },
        "today_trades":       today_trades,
        "today_trade_count":  len(today_trades),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM-OPTIMISED REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_telegram_report(tracker: ComplianceTracker) -> str:
    """
    Generate a Telegram-friendly compliance report.
    Wrapped in monospace block for proper alignment.
    """
    report = generate_daily_report(tracker)
    return f"```\n{report}\n```"


# ─────────────────────────────────────────────────────────────────────────────
# QUICK STATUS LINE (for inline messages)
# ─────────────────────────────────────────────────────────────────────────────

def quick_status(tracker: ComplianceTracker) -> str:
    """
    One-line status for quick Telegram updates or log output.
    """
    return tracker.summary()
