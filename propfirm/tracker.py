"""
GoldSignalAI — propfirm/tracker.py
====================================
Universal prop firm compliance tracker.

Tracks:
  - Daily PnL (realised + unrealised)
  - Peak balance and total drawdown
  - Trading days count
  - Trade history (for daily summary and reports)
  - Challenge progress vs. profit target

State is persisted to JSON (Config.PROP_STATE_FILE) so the bot can
survive restarts without losing progress.

Called by:
  - main.py — after each signal / position update
  - alerts/telegram_bot.py — /status command
  - dashboard/app.py — live compliance display
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, date, timedelta
from typing import Optional

from config import Config
from propfirm.profiles import (
    get_profile,
    daily_loss_check,
    drawdown_check,
    challenge_progress,
    ComplianceStatus,
    ChallengeProgress,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TRADE RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """A single completed or open trade."""
    timestamp:     str          # ISO format
    direction:     str          # "BUY" or "SELL"
    entry_price:   float
    exit_price:    Optional[float] = None
    sl_price:      float = 0.0
    tp1_price:     float = 0.0
    tp2_price:     float = 0.0
    lot_size:      float = 0.0
    pnl_usd:       float = 0.0
    pnl_pips:      float = 0.0
    status:        str   = "open"  # "open" | "closed_tp1" | "closed_tp2" | "closed_sl" | "closed_manual"
    date:          str   = ""      # YYYY-MM-DD for daily grouping


# ─────────────────────────────────────────────────────────────────────────────
# TRACKER STATE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackerState:
    """Persistent state for the compliance tracker."""
    # Balance tracking
    starting_balance:   float = Config.CHALLENGE_ACCOUNT_SIZE
    current_balance:    float = Config.CHALLENGE_ACCOUNT_SIZE
    peak_balance:       float = Config.CHALLENGE_ACCOUNT_SIZE

    # Daily tracking
    daily_pnl_usd:     float = 0.0
    daily_date:         str   = ""     # YYYY-MM-DD of current trading day

    # Challenge tracking
    trading_days:       int   = 0
    trading_days_list:  list  = field(default_factory=list)  # list of date strings
    total_trades:       int   = 0
    winning_trades:     int   = 0
    losing_trades:      int   = 0

    # Trade history
    trades:             list  = field(default_factory=list)  # list of dicts

    # Status flags
    daily_loss_breached:    bool = False
    drawdown_breached:      bool = False
    challenge_passed:       bool = False

    # Metadata
    firm_name:          str   = ""
    last_updated:       str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLIANCE TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class ComplianceTracker:
    """
    Tracks all prop firm compliance metrics and persists state to disk.

    Usage:
        tracker = ComplianceTracker()      # loads state from disk
        tracker.record_trade(trade)        # after a trade closes
        tracker.check_compliance()         # returns (daily_status, dd_status)
        tracker.get_progress()             # returns ChallengeProgress
        tracker.save()                     # persist to disk
    """

    def __init__(self, state_file: Optional[str] = None):
        self.state_file = state_file or Config.PROP_STATE_FILE
        self.profile = get_profile()
        self.state = self._load_state()
        self._check_day_rollover()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_state(self) -> TrackerState:
        """Load state from disk, or create fresh state."""
        if os.path.isfile(self.state_file):
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                state = TrackerState(**{
                    k: v for k, v in data.items()
                    if k in TrackerState.__dataclass_fields__
                })
                logger.info("Loaded tracker state from %s", self.state_file)
                return state
            except Exception as exc:
                logger.warning("Failed to load tracker state: %s — starting fresh", exc)

        state = TrackerState(
            firm_name=self.profile.name,
            daily_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )
        return state

    def save(self) -> None:
        """Persist current state to disk."""
        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        try:
            with open(self.state_file, "w") as f:
                json.dump(asdict(self.state), f, indent=2)
            logger.debug("Tracker state saved to %s", self.state_file)
        except Exception as exc:
            logger.error("Failed to save tracker state: %s", exc)

    def reset(self) -> None:
        """Reset all state (start a new challenge)."""
        self.state = TrackerState(
            firm_name=self.profile.name,
            daily_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )
        self.save()
        logger.info("Tracker state reset for new challenge")

    # ── Day Rollover ─────────────────────────────────────────────────────────

    def _check_day_rollover(self) -> None:
        """Reset daily PnL if the date has changed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.daily_date != today:
            if self.state.daily_pnl_usd != 0:
                logger.info(
                    "Day rollover: %s → %s | Yesterday PnL: $%.2f",
                    self.state.daily_date, today, self.state.daily_pnl_usd
                )
            self.state.daily_pnl_usd = 0.0
            self.state.daily_loss_breached = False
            self.state.daily_date = today

    # ── Trade Recording ──────────────────────────────────────────────────────

    def record_trade(self, trade: TradeRecord) -> None:
        """
        Record a completed trade and update all metrics.

        Args:
            trade: TradeRecord with pnl_usd populated.
        """
        self._check_day_rollover()

        # Add to history
        self.state.trades.append(asdict(trade))
        self.state.total_trades += 1

        # Win/loss count
        if trade.pnl_usd > 0:
            self.state.winning_trades += 1
        elif trade.pnl_usd < 0:
            self.state.losing_trades += 1

        # Update balance
        self.state.current_balance += trade.pnl_usd
        self.state.daily_pnl_usd += trade.pnl_usd

        # Update peak balance
        if self.state.current_balance > self.state.peak_balance:
            self.state.peak_balance = self.state.current_balance

        # Track trading day
        trade_date = trade.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if trade_date not in self.state.trading_days_list:
            self.state.trading_days_list.append(trade_date)
            self.state.trading_days = len(self.state.trading_days_list)

        # Auto-save after each trade
        self.save()

        logger.info(
            "Trade recorded: %s %s PnL=$%.2f | Balance=$%.2f | Daily=$%.2f",
            trade.direction, trade.status, trade.pnl_usd,
            self.state.current_balance, self.state.daily_pnl_usd
        )

    # ── Compliance Checks ────────────────────────────────────────────────────

    def check_compliance(self) -> tuple[ComplianceStatus, ComplianceStatus]:
        """
        Run daily loss and drawdown checks against the active profile.

        Returns:
            (daily_loss_status, drawdown_status)
        """
        self._check_day_rollover()

        daily = daily_loss_check(
            self.state.daily_pnl_usd,
            self.state.starting_balance,
            self.profile,
        )

        dd = drawdown_check(
            self.state.peak_balance,
            self.state.current_balance,
            self.state.starting_balance,
            self.profile,
        )

        # Update breach flags
        if daily.breached:
            self.state.daily_loss_breached = True
        if dd.breached:
            self.state.drawdown_breached = True

        return daily, dd

    def is_trading_allowed(self) -> tuple[bool, str]:
        """
        Check if the bot should continue trading.

        Returns:
            (allowed, reason)
        """
        daily, dd = self.check_compliance()

        if dd.breached:
            return False, f"Max drawdown breached ({dd.current_pct:.1f}%)"
        if daily.breached:
            return False, f"Daily loss limit breached ({daily.current_pct:.1f}%)"
        if self.state.challenge_passed:
            return False, "Challenge already passed — no more trading needed"

        # Warning: still allowed but reduced risk
        if daily.warning:
            logger.warning("Daily loss approaching limit: %s", daily.message)
        if dd.warning:
            logger.warning("Drawdown approaching limit: %s", dd.message)

        return True, "OK"

    def get_progress(self) -> ChallengeProgress:
        """Get current challenge progress."""
        return challenge_progress(
            self.state.starting_balance,
            self.state.current_balance,
            self.state.trading_days,
            self.profile,
        )

    # ── Metrics ──────────────────────────────────────────────────────────────

    @property
    def win_rate(self) -> float:
        """Win rate as a percentage (0–100)."""
        total = self.state.winning_trades + self.state.losing_trades
        if total == 0:
            return 0.0
        return self.state.winning_trades / total * 100

    @property
    def profit_pct(self) -> float:
        """Current profit as % of starting balance."""
        return (self.state.current_balance - self.state.starting_balance) / self.state.starting_balance * 100

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak as % of starting balance."""
        dd = max(0, self.state.peak_balance - self.state.current_balance)
        return dd / self.state.starting_balance * 100

    def get_daily_trades(self) -> list[dict]:
        """Return all trades from today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return [t for t in self.state.trades if t.get("date") == today]

    def summary(self) -> str:
        """One-line status summary."""
        daily, dd = self.check_compliance()
        return (
            f"{self.profile.name} | "
            f"Balance=${self.state.current_balance:,.2f} | "
            f"P/L={self.profit_pct:+.2f}% | "
            f"DD={self.drawdown_pct:.1f}% | "
            f"Win={self.win_rate:.0f}% ({self.state.winning_trades}W/{self.state.losing_trades}L) | "
            f"Daily: {daily.status_icon} | DD: {dd.status_icon}"
        )
