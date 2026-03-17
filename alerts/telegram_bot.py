"""
GoldSignalAI — alerts/telegram_bot.py
=======================================
Telegram bot for:
  - Sending BUY/SELL signal alerts with chart images
  - /status command — current compliance status
  - /signal command — latest signal info
  - /report command — daily compliance report
  - /help command — list available commands

Uses python-telegram-bot library (async).

Integration:
  - Called by main.py to send alerts when actionable signals fire
  - Called by scheduler for daily summary at 5 PM EST
  - Runs command handlers for interactive use
"""

import asyncio
import logging
import os
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)

# Try to import telegram library
try:
    from telegram import Bot, Update
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.info("python-telegram-bot not installed — Telegram features disabled")


# ─────────────────────────────────────────────────────────────────────────────
# BOT SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

class TelegramAlert:
    """
    Manages Telegram bot connections and message sending.

    Usage:
        alert = TelegramAlert()
        await alert.send_signal(formatted_text, chart_path)
        await alert.send_message("Hello")
    """

    def __init__(self):
        self.token   = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id and TELEGRAM_AVAILABLE)
        self._bot: Optional[object] = None

        if not self.enabled:
            reasons = []
            if not TELEGRAM_AVAILABLE:
                reasons.append("library not installed")
            if not self.token:
                reasons.append("TELEGRAM_BOT_TOKEN not set")
            if not self.chat_id:
                reasons.append("TELEGRAM_CHAT_ID not set")
            logger.info("Telegram alerts disabled: %s", ", ".join(reasons))

    @property
    def bot(self):
        if self._bot is None and self.enabled:
            self._bot = Bot(token=self.token)
        return self._bot

    # ── Send Methods ─────────────────────────────────────────────────────────

    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a text message to the configured chat.

        Args:
            text: Message text (supports Markdown)
            parse_mode: "Markdown" or "HTML"

        Returns:
            True if sent successfully.
        """
        if not self.enabled:
            logger.debug("Telegram disabled — message not sent")
            return False

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
            )
            logger.info("Telegram message sent (%d chars)", len(text))
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    async def send_signal(
        self,
        signal_text: str,
        chart_path: Optional[str] = None,
    ) -> bool:
        """
        Send a trading signal alert with optional chart image.

        Args:
            signal_text: Formatted signal text (from formatter.py)
            chart_path:  Path to chart PNG (from chart_generator.py)

        Returns:
            True if sent successfully.
        """
        if not self.enabled:
            logger.debug("Telegram disabled — signal not sent")
            return False

        success = True

        # Send chart image first (if available)
        if chart_path and os.path.isfile(chart_path):
            try:
                with open(chart_path, "rb") as photo:
                    await self.bot.send_photo(
                        chat_id=self.chat_id,
                        photo=photo,
                    )
                logger.info("Chart image sent to Telegram")
            except Exception as exc:
                logger.error("Failed to send chart: %s", exc)
                success = False

        # Send signal text
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=signal_text,
                parse_mode="Markdown",
            )
            logger.info("Signal alert sent to Telegram")
        except Exception as exc:
            logger.error("Failed to send signal text: %s", exc)
            success = False

        return success

    async def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Send a photo with optional caption."""
        if not self.enabled:
            return False

        try:
            with open(photo_path, "rb") as photo:
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=photo,
                    caption=caption,
                )
            return True
        except Exception as exc:
            logger.error("Telegram photo send failed: %s", exc)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# SYNC WRAPPERS (for non-async callers — safe to call from any thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_async(coro) -> any:
    """
    Run an async coroutine from synchronous code, safe for any thread.

    asyncio.run() fails if called from a thread that already has a running
    event loop (e.g., the scheduler thread). This helper creates a fresh
    loop for the current thread to avoid that RuntimeError.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already inside an event loop — create a new loop in a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=30)
    else:
        return asyncio.run(coro)


def send_signal_sync(signal_text: str, chart_path: Optional[str] = None) -> bool:
    """
    Synchronous wrapper for sending signals.
    Safe to call from any thread.
    """
    alert = TelegramAlert()
    if not alert.enabled:
        return False
    try:
        return _run_async(alert.send_signal(signal_text, chart_path))
    except Exception as exc:
        logger.error("send_signal_sync failed: %s", exc)
        return False


def send_message_sync(text: str) -> bool:
    """
    Synchronous wrapper for sending messages.
    Safe to call from any thread.
    """
    alert = TelegramAlert()
    if not alert.enabled:
        return False
    try:
        return _run_async(alert.send_message(text))
    except Exception as exc:
        logger.error("send_message_sync failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS & BOT RUNNER (require telegram library)
# ─────────────────────────────────────────────────────────────────────────────

def run_bot():
    """
    Start the Telegram bot command handler in a background thread.

    Uses the async context-manager API instead of app.run_polling() to avoid
    RuntimeError: set_wakeup_fd only works in main thread of the main interpreter.

    run_polling() internally calls signal.set_wakeup_fd() which is restricted
    to the main thread. The fix is to create a fresh event loop in this thread
    and drive the Application manually — no signal handler registration needed.
    """
    if not TELEGRAM_AVAILABLE:
        logger.error("Cannot run bot: python-telegram-bot not installed")
        return

    if not Config.TELEGRAM_BOT_TOKEN:
        logger.error("Cannot run bot: TELEGRAM_BOT_TOKEN not set")
        return

    # ── Command handler coroutines ────────────────────────────────────────────

    async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "*GoldSignalAI Bot Active*\n\n"
            "Commands:\n"
            "/status  — Account compliance status\n"
            "/signal  — Latest signal info\n"
            "/report  — Daily compliance report\n"
            "/help    — Show this message\n",
            parse_mode="Markdown",
        )

    async def _cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await _cmd_start(update, context)

    async def _cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            from propfirm.tracker import ComplianceTracker
            from propfirm.compliance_report import generate_daily_report
            tracker = ComplianceTracker()
            status = generate_daily_report(tracker)
            await update.message.reply_text(f"```\n{status}\n```", parse_mode="Markdown")
        except Exception as exc:
            await update.message.reply_text(f"Error getting status: {exc}")

    async def _cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            from propfirm.tracker import ComplianceTracker
            from propfirm.compliance_report import generate_daily_report
            tracker = ComplianceTracker()
            report = generate_daily_report(tracker)
            await update.message.reply_text(report, parse_mode="Markdown")
        except Exception as exc:
            await update.message.reply_text(f"Error generating report: {exc}")

    async def _cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Signal generation runs automatically every 15 minutes.\n"
            "Use the dashboard for real-time signal info."
        )

    # ── Async bot runner — no signal handlers ─────────────────────────────────

    async def _run_polling_no_signals():
        """
        Drive the Application polling loop without installing OS signal handlers.

        The key difference from app.run_polling():
          - We manually call initialize/start/stop/shutdown
          - We pass close_loop=False so PTB doesn't try to close our loop
          - No signal.set_wakeup_fd() is called
        """
        from telegram.request import HTTPXRequest

        request = HTTPXRequest(
            connect_timeout=30,
            read_timeout=30,
            write_timeout=30,
        )
        app = (
            Application.builder()
            .token(Config.TELEGRAM_BOT_TOKEN)
            .request(request)
            .build()
        )

        app.add_handler(CommandHandler("start",  _cmd_start))
        app.add_handler(CommandHandler("help",   _cmd_help))
        app.add_handler(CommandHandler("status", _cmd_status))
        app.add_handler(CommandHandler("report", _cmd_report))
        app.add_handler(CommandHandler("signal", _cmd_signal))

        logger.info("Telegram bot initialising…")

        # Retry once on startup if connection fails
        for attempt in range(2):
            try:
                await app.initialize()
                await app.updater.start_polling(
                    allowed_updates=Update.ALL_TYPES,
                    drop_pending_updates=True,
                )
                await app.start()
                break
            except Exception as exc:
                if attempt == 0:
                    logger.warning("Telegram bot startup failed (%s) — retrying in 5s…", exc)
                    await asyncio.sleep(5)
                else:
                    raise

        logger.info("Telegram bot polling for commands.")

        # Block indefinitely until the task is cancelled (by loop.stop()).
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Telegram bot shutting down…")
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
            logger.info("Telegram bot stopped.")

    # ── Create a dedicated event loop for this thread ──────────────────────────
    # Each background thread must own its event loop — never share loops across
    # threads. asyncio.new_event_loop() gives us a clean, thread-local loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(_run_polling_no_signals())
    except Exception as exc:
        logger.error("Telegram bot crashed: %s", exc)
    finally:
        try:
            # Cancel all remaining tasks cleanly before closing the loop.
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        finally:
            loop.close()
            logger.info("Telegram bot event loop closed.")
