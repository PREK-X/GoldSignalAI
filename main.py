"""
GoldSignalAI — main.py
========================
The master entry point that ties every module together and runs
the bot 24/7 as a single process.

Architecture:
  main thread  →  signal loop (every 15 minutes on candle close)
  thread 1     →  scheduler (retrain, daily summary, prop firm checks)
  thread 2     →  Telegram bot polling (optional, if credentials set)
  on demand    →  Streamlit dashboard (launched as subprocess)

Signal Loop (every M15 candle close):
  1. Check if market is open (skip weekends)
  2. Check news filter (pause if high-impact event)
  3. Check prop firm compliance (stop if limits breached)
  4. Generate signal (full pipeline: indicators → scoring → ML → risk)
  5. If actionable (BUY/SELL >= 70%):
       a. Format signal text
       b. Generate chart image
       c. Send Telegram alert
       d. Log to signal history
       e. Record trade in compliance tracker
  6. Display signal on terminal
  7. Sleep until next candle close

Graceful shutdown:
  Ctrl+C or SIGTERM → saves all state, stops scheduler, exits cleanly.

Usage:
    venv/bin/python main.py                # Run the bot
    venv/bin/python main.py --dashboard    # Also launch Streamlit dashboard
    venv/bin/python main.py --backtest     # Run backtest instead of live
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import Config
from infrastructure.logger import setup_logging
from infrastructure.monitoring import init_sentry
from infrastructure.health import run_health_check
from database.db import initialize_database, save_signal, save_trade, has_recent_signal


logger = logging.getLogger("GoldSignalAI")


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL HISTORY LOGGER
# ─────────────────────────────────────────────────────────────────────────────

def _log_signal(signal_data: dict) -> None:
    """Append a signal record to JSON history file + SQLite database."""
    # ── JSON file (existing behaviour) ────────────────────────────────
    try:
        history = []
        if os.path.isfile(Config.SIGNAL_HISTORY_FILE):
            with open(Config.SIGNAL_HISTORY_FILE) as f:
                history = json.load(f)

        history.append(signal_data)

        # Keep last 500 signals
        history = history[-500:]

        os.makedirs(os.path.dirname(Config.SIGNAL_HISTORY_FILE), exist_ok=True)
        with open(Config.SIGNAL_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)

    except Exception as exc:
        logger.warning("Failed to write signal history: %s", exc)

    # ── SQLite database ───────────────────────────────────────────────
    try:
        save_signal(signal_data)
    except Exception as exc:
        logger.warning("Failed to save signal to database: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOURS CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _is_market_open(now: Optional[datetime] = None) -> tuple[bool, str]:
    """
    Check if Gold markets are open.

    Gold trades ~23 hours/day Mon-Fri.
    Closed: Saturday full day, Sunday until ~22:00 UTC.
    We also skip the daily maintenance window (22:00-23:00 UTC weekdays).

    Returns:
        (is_open, reason)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    weekday = now.weekday()  # Mon=0, Sun=6
    hour = now.hour

    # Saturday: fully closed
    if weekday == 5:
        return False, "Market closed (Saturday)"

    # Sunday: closed until 22:00 UTC
    if weekday == 6 and hour < 22:
        return False, "Market closed (Sunday — opens 22:00 UTC)"

    # Friday: closes at 22:00 UTC
    if weekday == 4 and hour >= 22:
        return False, "Market closed (Friday close)"

    # Daily maintenance: 22:00-23:00 UTC (Mon-Thu)
    if weekday < 4 and hour == 22:
        return False, "Daily maintenance window (22:00-23:00 UTC)"

    return True, "Market open"


# ─────────────────────────────────────────────────────────────────────────────
# NEXT CANDLE CLOSE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def _seconds_until_next_candle(now: Optional[datetime] = None) -> float:
    """
    Calculate seconds until the next M15 candle close.

    M15 candles close at :00, :15, :30, :45 of each hour.
    We add a small buffer (5 seconds) to ensure the candle has fully closed
    and data providers have updated.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    interval = Config.CANDLE_INTERVAL_SECONDS  # 900 = 15 minutes
    buffer = 5  # seconds after candle close

    # Seconds since midnight
    seconds_today = now.hour * 3600 + now.minute * 60 + now.second

    # Seconds into current candle
    seconds_into_candle = seconds_today % interval

    # Time until next close
    remaining = interval - seconds_into_candle + buffer

    # If buffer already passed the close window, wait for next one
    if remaining <= buffer:
        remaining += interval

    return remaining


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _process_signal(
    sig,
    tracker,
    signal_count: int,
) -> int:
    """
    Process a generated signal: format, chart, alert, log, track.

    Args:
        sig:          TradingSignal from generator
        tracker:      ComplianceTracker instance
        signal_count: Running count of signals (for display)

    Returns:
        Updated signal_count
    """
    from signals.formatter import format_signal, format_signal_short
    from alerts.telegram_bot import send_signal_sync
    from alerts.chart_generator import generate_signal_chart
    from propfirm.tracker import TradeRecord

    signal_count += 1

    # Always display the signal on terminal
    formatted = format_signal(sig)
    print(f"\n{'=' * 50}")
    print(f" Signal #{signal_count}")
    print(f"{'=' * 50}")
    print(formatted)
    print()

    # Log every signal (including WAIT) to history
    _log_signal({
        "timestamp": sig.timestamp.isoformat(),
        "direction": sig.direction,
        "confidence_pct": sig.confidence_pct,
        "entry_price": sig.entry_price,
        "stop_loss": sig.risk.stop_loss if sig.risk else None,
        "tp1": sig.risk.tp1_price if sig.risk else None,
        "tp2": sig.risk.tp2_price if sig.risk else None,
        "sl_pips": sig.risk.sl_pips if sig.risk else None,
        "lot_size": sig.risk.suggested_lot if sig.risk else None,
        "bullish_count": sig.bullish_count,
        "bearish_count": sig.bearish_count,
        "ml_confirms": sig.ml_confirms,
        "reason": sig.reason,
        "is_paused": sig.is_paused,
    })

    # Only send alerts for actionable signals
    if sig.is_actionable:
        logger.info("Actionable signal: %s @ %.2f (%.0f%%)",
                    sig.direction, sig.entry_price, sig.confidence_pct)

        # Generate chart
        chart_path = None
        try:
            chart_path = generate_signal_chart(sig)
        except Exception as exc:
            logger.warning("Chart generation failed: %s", exc)

        # Send Telegram alert
        try:
            send_signal_sync(formatted, chart_path)
        except Exception as exc:
            logger.warning("Telegram alert failed: %s", exc)

        # Record in compliance tracker
        try:
            trade = TradeRecord(
                timestamp=sig.timestamp.isoformat(),
                direction=sig.direction,
                entry_price=sig.entry_price,
                sl_price=sig.risk.stop_loss if sig.risk else 0.0,
                tp1_price=sig.risk.tp1_price if sig.risk else 0.0,
                tp2_price=sig.risk.tp2_price if sig.risk else 0.0,
                lot_size=sig.risk.suggested_lot if sig.risk else 0.0,
                status="open",
                date=sig.timestamp.strftime("%Y-%m-%d"),
            )
            tracker.record_trade(trade)
        except Exception as exc:
            logger.warning("Compliance tracking failed: %s", exc)

        # Save trade to SQLite database
        try:
            save_trade({
                "timestamp": sig.timestamp.isoformat(),
                "direction": sig.direction,
                "entry_price": sig.entry_price,
                "stop_loss": sig.risk.stop_loss if sig.risk else None,
                "take_profit1": sig.risk.tp1_price if sig.risk else None,
                "take_profit2": sig.risk.tp2_price if sig.risk else None,
                "lot_size": sig.risk.suggested_lot if sig.risk else None,
                "status": "open",
            })
        except Exception as exc:
            logger.warning("Failed to save trade to database: %s", exc)

    else:
        logger.info("Non-actionable signal: %s (%s)", sig.direction, sig.reason)

    return signal_count


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD LAUNCHER
# ─────────────────────────────────────────────────────────────────────────────

def _launch_dashboard() -> Optional[subprocess.Popen]:
    """
    Launch the Streamlit dashboard as a background subprocess.

    Returns the Popen handle for cleanup, or None on failure.
    """
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            os.path.join(Config.BASE_DIR, "dashboard", "app.py"),
            "--server.port", str(Config.DASHBOARD_PORT),
            "--server.headless", "true",
            "--theme.base", "dark",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Streamlit dashboard launched on port %d (PID %d)",
                    Config.DASHBOARD_PORT, proc.pid)
        return proc

    except Exception as exc:
        logger.error("Failed to launch dashboard: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM BOT THREAD
# ─────────────────────────────────────────────────────────────────────────────

def _start_telegram_thread() -> Optional[threading.Thread]:
    """
    Start the Telegram bot command handler in a background thread.

    This handles /signal, /status, /help etc. commands.
    Returns the thread handle, or None if Telegram is not configured.
    """
    if not Config.TELEGRAM_BOT_TOKEN or not Config.TELEGRAM_CHAT_ID:
        logger.info("Telegram not configured — skipping bot thread.")
        return None

    try:
        from alerts.telegram_bot import run_bot

        thread = threading.Thread(
            target=run_bot,
            name="GoldSignalAI-Telegram",
            daemon=True,
        )
        thread.start()
        logger.info("Telegram bot thread started.")
        return thread

    except Exception as exc:
        logger.error("Failed to start Telegram bot: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST MODE
# ─────────────────────────────────────────────────────────────────────────────

def _run_backtest_mode() -> None:
    """Run backtest and generate PDF report, then exit."""
    from backtest.engine import run_backtest, BacktestConfig
    from backtest.report_generator import generate_pdf_report

    print("\n" + "=" * 50)
    print(" GoldSignalAI — Backtest Mode")
    print("=" * 50 + "\n")

    cfg = BacktestConfig()
    print(f"Account: ${cfg.account_balance:,.0f}")
    print(f"Spread: {cfg.spread_pips} pips")
    print(f"Risk/trade: {cfg.risk_per_trade_pct}%")
    print(f"Prop firm: {cfg.prop_firm_key}")
    print()

    result = run_backtest(cfg)
    print(result.summary())

    # Export CSV
    if result.trades:
        csv_path = result.export_csv()
        print(f"\nTrade history: {csv_path}")

    # Generate PDF
    pdf_path = generate_pdf_report(result)
    print(f"PDF report:    {pdf_path}")

    # Prop firm results
    if result.prop_firm_sims:
        print("\n" + "-" * 50)
        print(" Prop Firm Challenge Results")
        print("-" * 50)
        for sim in result.prop_firm_sims:
            status = "PASSED" if sim.passed else "FAILED"
            print(f"  {sim.firm_name:25s} {status}")
            if sim.passed:
                print(f"    Completed in {sim.days_to_complete} days")
            elif sim.breach_reason:
                print(f"    Reason: {sim.breach_reason[:60]}")
            print(f"    PnL: {sim.final_pnl_pct:+.2f}% | "
                  f"Max DD: {sim.max_drawdown_pct:.2f}% | "
                  f"Max Daily Loss: {sim.max_daily_loss_pct:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP BANNER
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    """Print the startup banner with config summary."""
    profile = Config.get_active_prop_firm()
    issues = Config.validate()

    print()
    print("=" * 55)
    print("  GoldSignalAI v" + Config.VERSION)
    print("  AI-Powered Trading Signal Bot for XAU/USD")
    print("=" * 55)
    print(f"  Asset:        {Config.SYMBOL_DISPLAY}")
    print(f"  Timeframes:   {Config.PRIMARY_TIMEFRAME} + {Config.CONFIRMATION_TIMEFRAME}")
    print(f"  Min Conf:     {Config.MIN_CONFIDENCE_PCT}%")
    print(f"  Prop Firm:    {profile.name}")
    print(f"  Account:      ${Config.CHALLENGE_ACCOUNT_SIZE:,.0f}")
    print(f"  Risk/Trade:   {Config.RISK_PER_TRADE_PCT}%")
    print(f"  SL Range:     {Config.MIN_SL_PIPS}-{Config.MAX_SL_PIPS} pips (ATR x{Config.ATR_SL_MULTIPLIER})")
    print(f"  TP:           1:{Config.TP1_RR_RATIO:.0f} / 1:{Config.TP2_RR_RATIO:.0f}")
    print("-" * 55)

    if issues:
        for issue in issues:
            print(f"  {issue}")
        print("-" * 55)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIGNAL LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _main_loop(shutdown_event: threading.Event) -> None:
    """
    The core signal generation loop.

    Runs indefinitely until shutdown_event is set.
    On each M15 candle close:
      1. Check market hours
      2. Check news filter
      3. Check prop firm compliance
      4. Generate and process signal
      5. Sleep until next candle
    """
    from signals.generator import generate_signal
    from data.news_fetcher import check_news_pause
    from propfirm.tracker import ComplianceTracker

    tracker = ComplianceTracker()
    signal_count = 0

    logger.info("Signal loop started. Waiting for next M15 candle close...")

    while not shutdown_event.is_set():
        now = datetime.now(timezone.utc)

        # ── 1. Market hours check ──────────────────────────────────────
        market_open, market_reason = _is_market_open(now)
        if not market_open:
            logger.info("Market closed: %s — sleeping 5 min...", market_reason)
            shutdown_event.wait(timeout=300)
            continue

        # ── 2. Wait for next candle close ──────────────────────────────
        wait_seconds = _seconds_until_next_candle(now)
        if wait_seconds > 10:
            next_close = now + timedelta(seconds=wait_seconds)
            logger.info(
                "Next candle close at %s (%.0f sec). Sleeping...",
                next_close.strftime("%H:%M:%S UTC"), wait_seconds
            )
            shutdown_event.wait(timeout=wait_seconds)
            if shutdown_event.is_set():
                break

        # ── 3. News filter ─────────────────────────────────────────────
        news_paused, pause_reason = False, ""
        try:
            news_paused, pause_reason = check_news_pause()
        except Exception as exc:
            logger.warning("News check failed (continuing without filter): %s", exc)

        # ── 4. Prop firm compliance ────────────────────────────────────
        trading_allowed, compliance_reason = True, "OK"
        try:
            trading_allowed, compliance_reason = tracker.is_trading_allowed()
        except Exception as exc:
            logger.warning("Compliance check failed (continuing): %s", exc)

        if not trading_allowed:
            logger.warning("Trading halted by compliance: %s", compliance_reason)
            # Still generate signal (for display) but mark as paused
            news_paused = True
            pause_reason = f"Compliance: {compliance_reason}"

        # ── 5. Generate signal ─────────────────────────────────────────
        try:
            sig = generate_signal(
                news_paused=news_paused,
                pause_reason=pause_reason,
            )

            # ── 5b. Duplicate signal check ────────────────────────────
            if sig.is_actionable and has_recent_signal(sig.direction, hours=4.0):
                logger.info(
                    "Duplicate %s signal suppressed (same direction within 4h)",
                    sig.direction,
                )
                # Still log but don't alert/trade
                _log_signal({
                    "timestamp": sig.timestamp.isoformat(),
                    "direction": sig.direction,
                    "confidence_pct": sig.confidence_pct,
                    "entry_price": sig.entry_price,
                    "reason": f"DUPLICATE_SUPPRESSED: {sig.reason}",
                    "is_paused": True,
                })
            else:
                signal_count = _process_signal(sig, tracker, signal_count)

        except Exception as exc:
            logger.exception("Signal generation failed: %s", exc)

        # ── 6. Brief pause before next cycle ───────────────────────────
        # Wait 10 seconds to avoid any timing edge cases
        shutdown_event.wait(timeout=10)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Application entry point.

    Parses CLI arguments, sets up logging, starts all services,
    and enters the main signal loop.
    """
    parser = argparse.ArgumentParser(
        prog="GoldSignalAI",
        description="AI-Powered Trading Signal Bot for XAU/USD",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Also launch the Streamlit web dashboard",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run backtest mode instead of live trading",
    )
    parser.add_argument(
        "--no-scheduler", action="store_true",
        help="Disable the background scheduler (retrain, summaries)",
    )
    parser.add_argument(
        "--no-telegram", action="store_true",
        help="Disable the Telegram bot command handler",
    )
    parser.add_argument(
        "--health-check", action="store_true",
        help="Run startup health check and exit",
    )
    args = parser.parse_args()

    # ── Logging ────────────────────────────────────────────────────────
    setup_logging()

    # ── Health check mode ──────────────────────────────────────────────
    if args.health_check:
        ok = run_health_check()
        sys.exit(0 if ok else 1)

    _print_banner()

    # ── Sentry error monitoring ────────────────────────────────────────
    init_sentry()

    # ── Initialize database ────────────────────────────────────────────
    initialize_database()

    # ── Ensure directories exist ───────────────────────────────────────
    for d in (Config.DATA_DIR, Config.MODELS_DIR, Config.LOGS_DIR, Config.REPORTS_DIR):
        os.makedirs(d, exist_ok=True)

    # ── Backtest mode ──────────────────────────────────────────────────
    if args.backtest:
        _run_backtest_mode()
        return

    # ── Graceful shutdown setup ────────────────────────────────────────
    shutdown_event = threading.Event()
    dashboard_proc = None
    sched = None

    def _shutdown_handler(signum, frame):
        logger.info("Shutdown signal received (sig=%s). Cleaning up...", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    try:
        # ── Start scheduler ────────────────────────────────────────────
        if not args.no_scheduler:
            from scheduler.tasks import Scheduler
            sched = Scheduler()
            sched.start()
            logger.info("Background scheduler started.")

        # ── Start Telegram bot ─────────────────────────────────────────
        telegram_thread = None
        if not args.no_telegram:
            telegram_thread = _start_telegram_thread()

        # ── Launch dashboard ───────────────────────────────────────────
        if args.dashboard:
            dashboard_proc = _launch_dashboard()

        # ── Main signal loop ───────────────────────────────────────────
        _main_loop(shutdown_event)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down.")
        shutdown_event.set()

    finally:
        # ── Cleanup ────────────────────────────────────────────────────
        logger.info("Shutting down services...")

        if sched is not None:
            sched.stop()

        if dashboard_proc is not None:
            logger.info("Stopping dashboard (PID %d)...", dashboard_proc.pid)
            dashboard_proc.terminate()
            try:
                dashboard_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_proc.kill()

        logger.info("GoldSignalAI shutdown complete.")


if __name__ == "__main__":
    main()
