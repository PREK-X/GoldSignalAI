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
