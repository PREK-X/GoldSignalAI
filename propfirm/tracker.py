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

        # FundedNext 1-Step pre-emptive daily ceiling (2.8% < 3.0% hard limit)
        if (Config.CHALLENGE_MODE_ENABLED
                and Config.ACTIVE_PROP_FIRM == "FundedNext_1Step"):
            daily_loss_pct = abs(min(0.0, self.state.daily_pnl_usd)) / self.state.starting_balance * 100
            if daily_loss_pct >= Config.FUNDEDNEXT_DAILY_CEILING_PCT:
                return False, (
                    f"FundedNext daily ceiling hit ({daily_loss_pct:.2f}% >= "
                    f"{Config.FUNDEDNEXT_DAILY_CEILING_PCT}%)"
                )

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


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 12: CHALLENGE TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class ChallengeTracker:
    """
    Real-time FundedNext challenge compliance tracker (Stage 12).

    Tracks:
      - Daily loss vs 3% hard limit (2.5% auto-pause warning)
      - Trailing total drawdown vs 6% hard limit (5% auto-pause warning)
      - Session start balance (resets at midnight UTC each trading day)
      - Peak balance high-watermark (for trailing DD calculation)

    Usage:
        tracker = ChallengeTracker("FundedNext_1Step", 10000.0)
        tracker.load("state/challenge_state.json")   # restore after restart
        tracker.update_balance(new_balance, datetime.now(timezone.utc))
        paused, reason = tracker.should_pause_trading()
        breached, reason = tracker.is_breached()
        tracker.persist("state/challenge_state.json")
    """

    def __init__(self, profile_name: str, initial_balance: float):
        self.profile_name    = profile_name
        self.profile         = get_profile(profile_name)
        self.initial_balance = initial_balance
        self.peak_balance    = initial_balance       # trailing high-watermark
        self.current_balance = initial_balance
        self.session_start_balance = initial_balance # resets at midnight UTC
        self.trading_day: Optional[str] = None       # YYYY-MM-DD of current session
        self.breach_halted:  bool = False            # True after hard breach

    # ── Balance update ────────────────────────────────────────────────────────

    def update_balance(self, new_balance: float, timestamp: datetime) -> dict:
        """
        Update current balance, roll over session if new day, update peak.

        Args:
            new_balance: Current account balance in USD.
            timestamp:   UTC datetime of the update.

        Returns:
            Current status dict from get_status().
        """
        today = timestamp.strftime("%Y-%m-%d")
        if self.trading_day != today:
            # New session: carry yesterday's closing balance as today's start
            self.session_start_balance = self.current_balance
            self.trading_day = today

        self.current_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        return self.get_status()

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        Return full compliance status as a dict.

        All dollar amounts are positive (loss/DD are reported as positive numbers).
        Percentage values are 0–100 floats (e.g. 2.5 = 2.5%).
        """
        p = self.profile

        daily_loss_dollars   = max(0.0, self.session_start_balance - self.current_balance)
        daily_loss_pct       = daily_loss_dollars / self.initial_balance * 100

        total_dd_dollars     = max(0.0, self.peak_balance - self.current_balance)
        total_dd_pct         = total_dd_dollars / self.initial_balance * 100

        daily_limit_dollars  = self.initial_balance * p.daily_loss_limit / 100
        total_dd_limit_dollars = self.initial_balance * p.max_total_drawdown / 100

        daily_remaining_dollars   = max(0.0, daily_limit_dollars - daily_loss_dollars)
        total_dd_remaining_dollars = max(0.0, total_dd_limit_dollars - total_dd_dollars)

        profit_pct = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        profit_target_remaining_pct = max(0.0, p.profit_target - profit_pct)
        profit_progress_pct = (
            min(100.0, max(0.0, profit_pct / p.profit_target * 100))
            if p.profit_target > 0 else 0.0
        )
        target_met  = profit_pct >= p.profit_target
        target_amount = self.initial_balance * (1.0 + p.profit_target / 100)

        # Determine compliance status
        if self.breach_halted or daily_loss_pct >= p.daily_loss_limit or total_dd_pct >= p.max_total_drawdown:
            compliance_status = "BREACHED"
        elif daily_loss_pct >= p.daily_loss_warning or total_dd_pct >= p.total_drawdown_warning:
            compliance_status = "PAUSED"
        else:
            compliance_status = "OK"

        pause_reason: Optional[str] = None
        if compliance_status == "BREACHED":
            if daily_loss_pct >= p.daily_loss_limit:
                pause_reason = (
                    f"Daily loss limit exceeded "
                    f"({daily_loss_pct:.2f}% vs {p.daily_loss_limit:.1f}%)"
                )
            else:
                pause_reason = (
                    f"Total drawdown limit exceeded "
                    f"({total_dd_pct:.2f}% vs {p.max_total_drawdown:.1f}%)"
                )
        elif compliance_status == "PAUSED":
            if daily_loss_pct >= p.daily_loss_warning:
                pause_reason = (
                    f"Daily loss approaching limit "
                    f"({daily_loss_pct:.2f}% of {p.daily_loss_limit:.1f}%)"
                )
            else:
                pause_reason = (
                    f"Total DD approaching limit "
                    f"({total_dd_pct:.2f}% of {p.max_total_drawdown:.1f}%)"
                )

        return {
            "current_balance":            self.current_balance,
            "peak_balance":               self.peak_balance,
            "initial_balance":            self.initial_balance,
            "profit_pct":                 profit_pct,
            "daily_loss_pct":             daily_loss_pct,
            "total_dd_pct":               total_dd_pct,
            "daily_loss_dollars":         daily_loss_dollars,
            "total_dd_dollars":           total_dd_dollars,
            "daily_limit_dollars":        daily_limit_dollars,
            "total_dd_limit_dollars":     total_dd_limit_dollars,
            "daily_remaining_dollars":    daily_remaining_dollars,
            "total_dd_remaining_dollars": total_dd_remaining_dollars,
            "daily_limit_pct":            p.daily_loss_limit,
            "total_dd_limit_pct":         p.max_total_drawdown,
            "daily_warning_pct":          p.daily_loss_warning,
            "total_dd_warning_pct":       p.total_drawdown_warning,
            "profit_target_pct":          p.profit_target,
            "profit_target_remaining_pct": profit_target_remaining_pct,
            "profit_progress_pct":        profit_progress_pct,
            "target_amount":              target_amount,
            "target_met":                 target_met,
            "compliance_status":          compliance_status,
            "pause_reason":               pause_reason,
        }

    # ── Trading gates ─────────────────────────────────────────────────────────

    def should_pause_trading(self) -> tuple[bool, str]:
        """
        Returns (True, reason) when daily loss or total DD hits the warning
        threshold (2.5% / 5.0% for FundedNext 1-Step).
        Trading is paused but the bot keeps running (shows WAIT signals).
        """
        s = self.get_status()
        p = self.profile
        if s["daily_loss_pct"] >= p.daily_loss_warning:
            return True, (
                f"Daily loss {s['daily_loss_pct']:.2f}% >= "
                f"{p.daily_loss_warning:.1f}% warning threshold"
            )
        if s["total_dd_pct"] >= p.total_drawdown_warning:
            return True, (
                f"Total DD {s['total_dd_pct']:.2f}% >= "
                f"{p.total_drawdown_warning:.1f}% warning threshold"
            )
        return False, ""

    def is_breached(self) -> tuple[bool, str]:
        """
        Returns (True, reason) when daily loss or total DD hits the hard
        limit (3% / 6% for FundedNext 1-Step).
        Trading is halted permanently for the session.
        """
        s = self.get_status()
        p = self.profile
        if s["daily_loss_pct"] >= p.daily_loss_limit:
            return True, (
                f"Daily loss {s['daily_loss_pct']:.2f}% exceeded "
                f"{p.daily_loss_limit:.1f}% limit"
            )
        if s["total_dd_pct"] >= p.max_total_drawdown:
            return True, (
                f"Total DD {s['total_dd_pct']:.2f}% exceeded "
                f"{p.max_total_drawdown:.1f}% limit"
            )
        return False, ""

    # ── Discord summary ───────────────────────────────────────────────────────

    def get_daily_summary(self) -> str:
        """Return formatted string for Discord daily challenge report."""
        s = self.get_status()
        p = self.profile
        sep = "━" * 36

        lines = [
            "📊 GoldSignalAI — Daily Challenge Report",
            sep,
            f"💰 Balance:     ${s['current_balance']:,.2f}  ({s['profit_pct']:+.2f}%)",
            f"🎯 Target:      ${s['target_amount']:,.2f}  ({s['profit_progress_pct']:.1f}% there)",
            sep,
            f"📉 Daily Loss:  -${s['daily_loss_dollars']:.2f}  "
            f"({s['daily_loss_pct']:.2f}% of ${s['daily_limit_dollars']:.0f} limit)",
            f"   Remaining:   ${s['daily_remaining_dollars']:.2f} today",
            f"📉 Total DD:    {s['total_dd_pct']:.2f}% of {p.max_total_drawdown:.2f}% limit",
            f"   Remaining:   ${s['total_dd_remaining_dollars']:.2f} buffer",
            sep,
        ]

        if s["target_met"]:
            lines.append("🏆 Status: TARGET MET — Challenge complete!")
        elif s["compliance_status"] == "BREACHED":
            lines.append("🔴 Status: BREACHED — Trading HALTED")
        elif s["compliance_status"] == "PAUSED":
            if s["daily_loss_pct"] >= p.daily_loss_warning:
                lines.append("⚠️ Status: WARNING — approaching daily limit")
            else:
                lines.append("⚠️ Status: WARNING — approaching DD limit")
        else:
            lines.append("✅ Status: ON TRACK")

        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def persist(self, filepath: str) -> None:
        """Save tracker state to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "profile_name":          self.profile_name,
            "initial_balance":       self.initial_balance,
            "peak_balance":          self.peak_balance,
            "current_balance":       self.current_balance,
            "session_start_balance": self.session_start_balance,
            "trading_day":           self.trading_day,
            "breach_halted":         self.breach_halted,
        }
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("ChallengeTracker state saved to %s", filepath)
        except Exception as exc:
            logger.error("ChallengeTracker persist failed: %s", exc)

    def load(self, filepath: str) -> None:
        """Restore tracker state from JSON file."""
        if not os.path.isfile(filepath):
            return
        try:
            with open(filepath) as f:
                data = json.load(f)
            if data.get("profile_name") != self.profile_name:
                logger.warning(
                    "ChallengeTracker: saved profile '%s' != current '%s' — ignoring",
                    data.get("profile_name"), self.profile_name,
                )
                return
            self.initial_balance       = data.get("initial_balance",       self.initial_balance)
            self.peak_balance          = data.get("peak_balance",          self.initial_balance)
            self.current_balance       = data.get("current_balance",       self.initial_balance)
            self.session_start_balance = data.get("session_start_balance", self.initial_balance)
            self.trading_day           = data.get("trading_day")
            self.breach_halted         = data.get("breach_halted",         False)
            logger.info(
                "ChallengeTracker loaded from %s — balance=$%.2f peak=$%.2f",
                filepath, self.current_balance, self.peak_balance,
            )
        except Exception as exc:
            logger.error("ChallengeTracker load failed: %s", exc)
