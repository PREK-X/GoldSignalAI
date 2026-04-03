"""
alerts/discord_notifier.py
==========================
Sends alerts to a Discord channel via an incoming webhook.

Environment variable:
    DISCORD_WEBHOOK_URL — the full webhook URL from Discord channel settings.
    If not set, all sends are silently skipped with a warning log.
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger("GoldSignalAI.discord")

_WEBHOOK_URL: Optional[str] = os.getenv("DISCORD_WEBHOOK_URL")


def send_message(text: str) -> bool:
    """
    POST a plain-text message to the Discord webhook.

    Args:
        text: Message content (max 2000 chars; longer strings are truncated).

    Returns:
        True on success, False on failure or if webhook not configured.
    """
    if not _WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL not set — skipping Discord alert.")
        return False

    # Discord message limit is 2000 characters
    payload = {"content": text[:2000]}

    try:
        resp = requests.post(_WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.error("Discord webhook failed: %s", exc)
        return False


def send_signal(signal_data: dict) -> bool:
    """
    Format and send a trading signal to Discord.

    Accepts a dict with the same shape as what _process_signal() builds,
    or the pre-formatted text string stored in signal history.

    If the caller already has a formatted string, pass it as
    signal_data["formatted_text"].  Otherwise a compact summary is built
    from the raw fields.

    Args:
        signal_data: Dict with signal fields, or {"formatted_text": "..."}.

    Returns:
        True on success, False otherwise.
    """
    if not _WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL not set — skipping Discord alert.")
        return False

    # Prefer pre-formatted text if provided
    if "formatted_text" in signal_data:
        return send_message(signal_data["formatted_text"])

    # Build a compact summary from raw fields
    direction = signal_data.get("direction", "?")
    confidence = signal_data.get("confidence_pct", 0)
    entry = signal_data.get("entry_price", 0)
    sl = signal_data.get("stop_loss")
    tp1 = signal_data.get("tp1")
    tp2 = signal_data.get("tp2")
    lot = signal_data.get("lot_size")
    reason = signal_data.get("reason", "")

    lines = [
        f"**GoldSignalAI Signal**",
        f"Direction: **{direction}**  |  Confidence: **{confidence:.0f}%**",
        f"Entry: `{entry:.2f}`",
    ]
    if sl:
        lines.append(f"SL: `{sl:.2f}`")
    if tp1:
        lines.append(f"TP1: `{tp1:.2f}`" + (f"  TP2: `{tp2:.2f}`" if tp2 else ""))
    if lot:
        lines.append(f"Lot: `{lot}`")
    if reason:
        lines.append(f"Note: {reason}")

    return send_message("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 12: CHALLENGE COMPLIANCE NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────

def send_daily_challenge_report(status: dict) -> bool:
    """
    Send the daily FundedNext challenge progress report to Discord.

    Args:
        status: Dict from ChallengeTracker.get_status()

    Returns:
        True on success, False otherwise.
    """
    sep = "━" * 36

    daily_loss_pct   = status.get("daily_loss_pct", 0.0)
    daily_limit_pct  = status.get("daily_limit_pct", 3.0)
    daily_warning_pct = status.get("daily_warning_pct", 2.5)
    total_dd_pct     = status.get("total_dd_pct", 0.0)
    total_dd_limit_pct = status.get("total_dd_limit_pct", 6.0)
    total_dd_warning_pct = status.get("total_dd_warning_pct", 5.0)
    compliance_status = status.get("compliance_status", "OK")

    # Status line
    if status.get("target_met"):
        status_line = "🏆 Status: TARGET MET — Challenge complete!"
    elif compliance_status == "BREACHED":
        status_line = "🔴 Status: BREACHED — Trading HALTED"
    elif compliance_status == "PAUSED":
        if daily_loss_pct >= daily_warning_pct:
            status_line = "⚠️ Status: WARNING — approaching daily limit"
        else:
            status_line = "⚠️ Status: WARNING — approaching DD limit"
    else:
        status_line = "✅ Status: ON TRACK"

    msg = "\n".join([
        "📊 GoldSignalAI — Daily Challenge Report",
        sep,
        f"💰 Balance:     ${status.get('current_balance', 0):,.2f}  ({status.get('profit_pct', 0):+.2f}%)",
        f"🎯 Target:      ${status.get('target_amount', 0):,.2f}  ({status.get('profit_progress_pct', 0):.1f}% there)",
        sep,
        f"📉 Daily Loss:  -${status.get('daily_loss_dollars', 0):.2f}  "
        f"({daily_loss_pct:.2f}% of ${status.get('daily_limit_dollars', 0):.0f} limit)",
        f"   Remaining:   ${status.get('daily_remaining_dollars', 0):.2f} today",
        f"📉 Total DD:    {total_dd_pct:.2f}% of {total_dd_limit_pct:.2f}% limit",
        f"   Remaining:   ${status.get('total_dd_remaining_dollars', 0):.2f} buffer",
        sep,
        status_line,
    ])
    return send_message(msg)


def send_challenge_breach_alert(reason: str, status: dict) -> bool:
    """
    Send an immediate breach alert to Discord (fires on hard limit violation).

    Args:
        reason: Human-readable breach reason.
        status: Dict from ChallengeTracker.get_status()

    Returns:
        True on success, False otherwise.
    """
    sep = "━" * 26
    msg = "\n".join([
        "🚨 CHALLENGE BREACH ALERT",
        sep,
        f"Reason:  {reason}",
        f"Balance: ${status.get('current_balance', 0):,.2f}",
        f"Action:  Trading HALTED",
        sep,
        "Review required before resuming.",
    ])
    return send_message(msg)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 13: ML RETRAIN NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────

def send_retrain_report(model_name: str, result: dict) -> bool:
    """
    Send ML retrain outcome report to Discord.

    Args:
        model_name: "LightGBM" or "CNN-BiLSTM"
        result:     Dict from ModelRetrainer.retrain_lgbm() / retrain_deep_if_ready()

    Returns:
        True on success, False otherwise.
    """
    from datetime import datetime, timezone, timedelta
    from config import Config

    sep = "━" * 36
    old_acc = result.get("old_accuracy", 0.0)
    new_acc = result.get("new_accuracy", 0.0)
    deployed = result.get("deployed", False)
    reason = result.get("reason", "")
    timestamp = result.get("timestamp", "")

    # Next retrain date
    try:
        next_dt = (datetime.now(timezone.utc) + timedelta(days=Config.RETRAIN_LGBM_INTERVAL_DAYS))
        next_str = next_dt.strftime("%Y-%m-%d")
    except Exception:
        next_str = "N/A"

    if deployed:
        acc_line = f"📈 New Accuracy:  {new_acc:.1%}"
        deploy_line = "✅ Deployed:      Yes"
        backup_file = ""
        backups = result.get("backup_paths", {})
        if backups:
            backup_file = "\n📁 Backup saved:  " + os.path.basename(list(backups.values())[0])
        footer = f"Next retrain: {next_str}"
        msg = "\n".join(filter(None, [
            f"🔄 ML Retrain Report — {model_name}",
            sep,
            f"📊 Old Accuracy:  {old_acc:.1%}",
            acc_line,
            deploy_line,
            backup_file,
            sep,
            footer,
        ]))
    else:
        acc_line = f"📉 New Accuracy:  {new_acc:.1%}"
        deploy_line = f"❌ Deployed:      No — {reason}"
        msg = "\n".join([
            f"🔄 ML Retrain Report — {model_name}",
            sep,
            f"📊 Old Accuracy:  {old_acc:.1%}",
            acc_line,
            deploy_line,
            "📁 Old model retained",
            sep,
        ])

    return send_message(msg)


def send_deep_retrain_waiting(count: int, required: int) -> bool:
    """
    Notify Discord that CNN-BiLSTM retrain is pending more live trade outcomes.

    Args:
        count:    Current number of completed live trades.
        required: Minimum required (RETRAIN_DEEP_MIN_TRADES).

    Returns:
        True on success, False otherwise.
    """
    msg = "\n".join([
        "⏳ CNN-BiLSTM Retrain Pending",
        f"Trade outcomes: {count} / {required} required",
        "Collect more live trades to trigger retrain.",
    ])
    return send_message(msg)


def send_challenge_warning(reason: str, status: dict) -> bool:
    """
    Send a warning to Discord when the bot auto-pauses near a limit.

    Args:
        reason: Human-readable pause reason.
        status: Dict from ChallengeTracker.get_status()

    Returns:
        True on success, False otherwise.
    """
    sep = "━" * 26
    msg = "\n".join([
        "⚠️ GoldSignalAI — Trading Paused",
        sep,
        f"Reason:        {reason}",
        f"Balance:       ${status.get('current_balance', 0):,.2f}",
        f"Daily Loss:    {status.get('daily_loss_pct', 0):.2f}% of {status.get('daily_limit_pct', 3):.1f}% limit",
        f"Total DD:      {status.get('total_dd_pct', 0):.2f}% of {status.get('total_dd_limit_pct', 6):.1f}% limit",
        sep,
        "Trading will resume automatically tomorrow (midnight UTC).",
    ])
    return send_message(msg)
