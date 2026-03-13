"""
GoldSignalAI — propfirm/profiles.py
====================================
Prop firm profile management — loads the active firm's rules and provides
helpers for the compliance tracker and dashboard.

The actual profile data lives in config.py (PROP_FIRM_PROFILES dict).
This module adds:
  - get_profile() — load active profile or any by name
  - get_all_profiles() — list all presets
  - daily_loss_check() — is today's PnL within the daily limit?
  - drawdown_check() — is total drawdown within the max allowed?
  - challenge_progress() — how close to passing the challenge?
  - format_profile_card() — display card for dashboard / Telegram
"""

import logging
from dataclasses import dataclass
from typing import Optional

from config import Config, PropFirmProfile, PROP_FIRM_PROFILES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE ACCESSORS
# ─────────────────────────────────────────────────────────────────────────────

def get_profile(name: Optional[str] = None) -> PropFirmProfile:
    """
    Get a prop firm profile by name, or the active one if name is None.

    Args:
        name: Profile key (e.g. "FTMO", "FundedNext_2Step"). None = active.

    Returns:
        PropFirmProfile

    Raises:
        ValueError if the name doesn't match any preset.
    """
    if name is None:
        return Config.get_active_prop_firm()

    profile = PROP_FIRM_PROFILES.get(name)
    if profile is None:
        raise ValueError(
            f"Unknown prop firm '{name}'. "
            f"Available: {list(PROP_FIRM_PROFILES.keys())}"
        )
    return profile


def get_all_profiles() -> dict[str, PropFirmProfile]:
    """Return all available prop firm presets."""
    return PROP_FIRM_PROFILES


def get_profile_names() -> list[str]:
    """Return list of all preset names."""
    return list(PROP_FIRM_PROFILES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# COMPLIANCE CHECK RESULTS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComplianceStatus:
    """Result of a compliance check against prop firm rules."""
    ok:           bool    # True = within limits
    warning:      bool    # True = approaching limit
    breached:     bool    # True = rule violated (challenge failed)
    current_pct:  float   # Current value as % of account
    limit_pct:    float   # The rule limit %
    warning_pct:  float   # Warning threshold %
    headroom_pct: float   # How much room left before breach
    label:        str     # "daily_loss" | "total_drawdown"
    message:      str     # Human-readable status

    @property
    def status_icon(self) -> str:
        if self.breached:
            return "🔴"
        if self.warning:
            return "🟡"
        return "🟢"


@dataclass
class ChallengeProgress:
    """Progress towards passing the prop firm challenge."""
    current_profit_pct:   float   # Current profit as % of starting balance
    target_pct:           float   # Required profit target %
    progress_pct:         float   # 0–100 how close to target
    trading_days:         int     # Days traded so far
    min_trading_days:     int     # Minimum required
    days_met:             bool    # True if min days requirement met
    target_met:           bool    # True if profit target reached
    challenge_passed:     bool    # True if ALL conditions met
    message:              str


# ─────────────────────────────────────────────────────────────────────────────
# COMPLIANCE CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def daily_loss_check(
    daily_pnl_usd:     float,
    account_balance:    float = Config.CHALLENGE_ACCOUNT_SIZE,
    profile:            Optional[PropFirmProfile] = None,
) -> ComplianceStatus:
    """
    Check if today's PnL is within the daily loss limit.

    Args:
        daily_pnl_usd:   Today's realised + unrealised PnL (negative = loss)
        account_balance:  Starting account balance
        profile:          Prop firm profile (None = active)

    Returns:
        ComplianceStatus
    """
    if profile is None:
        profile = get_profile()

    # Daily loss is checked as a percentage of the starting balance
    daily_loss_pct = abs(min(0, daily_pnl_usd)) / account_balance * 100

    breached = daily_loss_pct >= profile.daily_loss_limit
    warning  = daily_loss_pct >= profile.daily_loss_warning and not breached
    headroom = max(0, profile.daily_loss_limit - daily_loss_pct)

    if breached:
        msg = (f"DAILY LOSS BREACHED: -{daily_loss_pct:.2f}% "
               f"(limit: {profile.daily_loss_limit}%) — STOP TRADING")
    elif warning:
        msg = (f"Daily loss WARNING: -{daily_loss_pct:.2f}% "
               f"(limit: {profile.daily_loss_limit}%, "
               f"headroom: {headroom:.2f}%)")
    else:
        msg = (f"Daily loss OK: -{daily_loss_pct:.2f}% "
               f"(limit: {profile.daily_loss_limit}%, "
               f"headroom: {headroom:.2f}%)")

    return ComplianceStatus(
        ok=not breached and not warning,
        warning=warning,
        breached=breached,
        current_pct=daily_loss_pct,
        limit_pct=profile.daily_loss_limit,
        warning_pct=profile.daily_loss_warning,
        headroom_pct=headroom,
        label="daily_loss",
        message=msg,
    )


def drawdown_check(
    peak_balance:       float,
    current_balance:    float,
    account_balance:    float = Config.CHALLENGE_ACCOUNT_SIZE,
    profile:            Optional[PropFirmProfile] = None,
) -> ComplianceStatus:
    """
    Check if total drawdown from peak is within the maximum allowed.

    Args:
        peak_balance:     Highest account balance achieved
        current_balance:  Current account balance
        account_balance:  Starting account balance (for % calculation)
        profile:          Prop firm profile (None = active)

    Returns:
        ComplianceStatus
    """
    if profile is None:
        profile = get_profile()

    drawdown_usd = max(0, peak_balance - current_balance)
    drawdown_pct = drawdown_usd / account_balance * 100

    breached = drawdown_pct >= profile.max_total_drawdown
    warning  = drawdown_pct >= profile.total_drawdown_warning and not breached
    headroom = max(0, profile.max_total_drawdown - drawdown_pct)

    if breached:
        msg = (f"MAX DRAWDOWN BREACHED: -{drawdown_pct:.2f}% "
               f"(limit: {profile.max_total_drawdown}%) — CHALLENGE FAILED")
    elif warning:
        msg = (f"Drawdown WARNING: -{drawdown_pct:.2f}% "
               f"(limit: {profile.max_total_drawdown}%, "
               f"headroom: {headroom:.2f}%)")
    else:
        msg = (f"Drawdown OK: -{drawdown_pct:.2f}% "
               f"(limit: {profile.max_total_drawdown}%, "
               f"headroom: {headroom:.2f}%)")

    return ComplianceStatus(
        ok=not breached and not warning,
        warning=warning,
        breached=breached,
        current_pct=drawdown_pct,
        limit_pct=profile.max_total_drawdown,
        warning_pct=profile.total_drawdown_warning,
        headroom_pct=headroom,
        label="total_drawdown",
        message=msg,
    )


def challenge_progress(
    starting_balance:   float = Config.CHALLENGE_ACCOUNT_SIZE,
    current_balance:    float = Config.CHALLENGE_ACCOUNT_SIZE,
    trading_days:       int   = 0,
    profile:            Optional[PropFirmProfile] = None,
) -> ChallengeProgress:
    """
    Calculate how close the trader is to passing the challenge.

    Args:
        starting_balance: Initial account size
        current_balance:  Current account balance
        trading_days:     Number of unique trading days so far
        profile:          Prop firm profile (None = active)

    Returns:
        ChallengeProgress
    """
    if profile is None:
        profile = get_profile()

    profit_usd = current_balance - starting_balance
    profit_pct = profit_usd / starting_balance * 100

    target_pct = profile.profit_target
    progress = min(100.0, max(0.0, profit_pct / target_pct * 100)) if target_pct > 0 else 0.0

    days_met    = trading_days >= profile.min_trading_days
    target_met  = profit_pct >= target_pct
    passed      = target_met and days_met

    if passed:
        msg = f"CHALLENGE PASSED! +{profit_pct:.2f}% (target: {target_pct}%) in {trading_days} days"
    elif target_met and not days_met:
        msg = (f"Profit target met (+{profit_pct:.2f}%), "
               f"but need {profile.min_trading_days - trading_days} more trading days")
    else:
        remaining_pct = max(0, target_pct - profit_pct)
        remaining_usd = remaining_pct / 100 * starting_balance
        msg = (f"Progress: {progress:.0f}% | "
               f"+{profit_pct:.2f}% of {target_pct}% target | "
               f"${remaining_usd:,.2f} to go | "
               f"{trading_days}/{profile.min_trading_days} min days")

    return ChallengeProgress(
        current_profit_pct=profit_pct,
        target_pct=target_pct,
        progress_pct=progress,
        trading_days=trading_days,
        min_trading_days=profile.min_trading_days,
        days_met=days_met,
        target_met=target_met,
        challenge_passed=passed,
        message=msg,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def format_profile_card(profile: Optional[PropFirmProfile] = None) -> str:
    """
    Format a prop firm profile into a display card for dashboard/Telegram.

    Returns a text card like:
    ┌─────────────────────────────────────┐
    │   FundedNext (2-Step)               │
    ├─────────────────────────────────────┤
    │ Daily Loss Limit: 5.0%             │
    │ Max Drawdown:     10.0%            │
    │ Profit Target:    8.0%             │
    │ Min Trading Days: 5                │
    │ News Filter:      ON               │
    └─────────────────────────────────────┘
    """
    if profile is None:
        profile = get_profile()

    w = 39
    sep = "─" * w

    news = "ON" if profile.news_filter_enabled else "OFF"

    lines = [
        f"┌{sep}┐",
        f"│ {profile.name:<{w-2}} │",
        f"├{sep}┤",
        f"│ {'Daily Loss Limit:':<20}{profile.daily_loss_limit:<{w-21}.1f}%│",
        f"│ {'Max Drawdown:':<20}{profile.max_total_drawdown:<{w-21}.1f}%│",
        f"│ {'Profit Target:':<20}{profile.profit_target:<{w-21}.1f}%│",
        f"│ {'Min Trading Days:':<20}{profile.min_trading_days:<{w-21}} │",
        f"│ {'News Filter:':<20}{news:<{w-21}} │",
        f"└{sep}┘",
    ]

    return "\n".join(lines)
