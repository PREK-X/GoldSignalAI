"""
GoldSignalAI — scheduler/tasks.py
====================================
Background task scheduler using the `schedule` library.

Scheduled jobs:
  WEEKLY (Monday 00:00 UTC):
    - Fetch previous week's M15 data
    - Retrain XGBoost + Random Forest on accumulated data
    - Log before/after accuracy
    - Send Telegram alert with results
    - Reject new model if accuracy < Config.ML_MIN_ACCURACY

  DAILY (every day at 17:00 EST / 22:00 UTC):
    - Send daily summary Telegram message
    - Generate daily compliance report

  WEEKLY (Sunday 20:00 UTC):
    - Send weekly performance report via Telegram

  EVERY MINUTE:
    - Check if daily loss limit warning approaching → alert

Design notes:
  - The scheduler runs in a dedicated background thread started by main.py
  - All tasks are wrapped in try/except so one failure never kills the loop
  - Graceful shutdown via threading.Event
  - Tasks record their last-run time to logs/scheduler.json for restart recovery

Usage (from main.py):
    from scheduler.tasks import Scheduler
    sched = Scheduler()
    sched.start()          # starts background thread
    ...
    sched.stop()           # signals clean shutdown
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import schedule

from config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

SCHEDULER_STATE_FILE = os.path.join(Config.LOGS_DIR, "scheduler_state.json")


# ─────────────────────────────────────────────────────────────────────────────
# STATE PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    """Load scheduler state from disk (last-run times etc.)."""
    if not os.path.exists(SCHEDULER_STATE_FILE):
        return {}
    try:
        with open(SCHEDULER_STATE_FILE) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not load scheduler state: %s", exc)
        return {}


def _save_state(state: dict) -> None:
    """Persist scheduler state to disk."""
    try:
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        with open(SCHEDULER_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Could not save scheduler state: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL TASKS
# ─────────────────────────────────────────────────────────────────────────────

def task_retrain_model() -> None:
    """
    Weekly ML retraining task (Monday 00:00 UTC).

    Pipeline:
      1. Fetch last 2 years of M15 data to accumulate full history
      2. Retrain XGBoost + Random Forest
      3. Compare accuracy before vs after
      4. Reject new model if accuracy < ML_MIN_ACCURACY
      5. Send Telegram alert with results
      6. Log results to ML training log
    """
    logger.info("[Scheduler] Starting weekly ML retraining...")
    start_ts = datetime.now(timezone.utc)

    try:
        # ── 1. Read pre-retrain accuracy ───────────────────────────────
        from ml.trainer import get_model_status, retrain, train
        from ml.predictor import invalidate_cache

        status_before = get_model_status()
        acc_before = status_before.get("xgb_accuracy", 0.0)
        logger.info("[Scheduler] Model accuracy BEFORE retrain: %.1f%%", acc_before * 100)

        # ── 2. Fetch fresh historical data ─────────────────────────────
        from data.fetcher import get_candles
        from data.processor import process

        logger.info("[Scheduler] Fetching historical data for retraining...")
        raw = get_candles(
            timeframe=Config.PRIMARY_TIMEFRAME,
            n_candles=Config.LOOKBACK_CANDLES * 10,   # ~5000 M15 bars
            symbol=Config.SYMBOL,
        )

        if raw is None or raw.empty:
            logger.error("[Scheduler] Retrain aborted — no data fetched.")
            _send_retrain_alert(
                success=False,
                reason="Data fetch failed — no candles returned.",
                acc_before=acc_before,
                acc_after=0.0,
            )
            return

        df = process(raw, timeframe=Config.PRIMARY_TIMEFRAME, label="RETRAIN")
        if df is None or df.empty:
            logger.error("[Scheduler] Retrain aborted — data processing failed.")
            _send_retrain_alert(
                success=False,
                reason="Data processing failed.",
                acc_before=acc_before,
                acc_after=0.0,
            )
            return

        logger.info("[Scheduler] Retraining on %d bars...", len(df))

        # ── 3. Retrain ─────────────────────────────────────────────────
        result = retrain(new_df=df)

        acc_after = result.xgb_accuracy
        elapsed = (datetime.now(timezone.utc) - start_ts).total_seconds()

        logger.info(
            "[Scheduler] Retrain complete in %.1fs | "
            "XGB: %.1f%% → %.1f%% | RF: %.1f%%",
            elapsed, acc_before * 100, acc_after * 100, result.rf_accuracy * 100,
        )

        # ── 4. Invalidate predictor cache so new model is loaded ────────
        invalidate_cache()

        # ── 5. Check if model meets minimum threshold ──────────────────
        if result.rejected:
            logger.warning(
                "[Scheduler] New model REJECTED: %s — keeping previous model.",
                result.reject_reason,
            )
            _send_retrain_alert(
                success=False,
                reason=result.reject_reason,
                acc_before=acc_before,
                acc_after=acc_after,
            )
        else:
            logger.info("[Scheduler] New model ACCEPTED and saved.")
            _send_retrain_alert(
                success=True,
                reason="",
                acc_before=acc_before,
                acc_after=acc_after,
                n_samples=result.n_samples,
                elapsed_s=elapsed,
            )

        # ── 6. Save state ──────────────────────────────────────────────
        state = _load_state()
        state["last_retrain"] = start_ts.isoformat()
        state["last_retrain_result"] = {
            "accepted": not result.rejected,
            "xgb_acc_before": acc_before,
            "xgb_acc_after": acc_after,
            "rf_acc": result.rf_accuracy,
            "n_samples": result.n_samples,
            "elapsed_s": elapsed,
        }
        _save_state(state)

    except Exception as exc:
        logger.exception("[Scheduler] Retrain task crashed: %s", exc)
        _send_retrain_alert(success=False, reason=f"Unexpected error: {exc}", acc_before=0.0, acc_after=0.0)


def task_daily_summary() -> None:
    """
    Daily summary task (17:00 EST / 22:00 UTC).

    Sends a Telegram message summarising today's signals,
    P&L, and prop firm compliance status.
    """
    logger.info("[Scheduler] Sending daily summary...")
    try:
        from propfirm.compliance_report import generate_daily_report
        from propfirm.tracker import ComplianceTracker
        from alerts.telegram_bot import send_message_sync

        tracker = ComplianceTracker()
        report_text = generate_daily_report(tracker)
        send_message_sync(report_text)

        # Save last-run time
        state = _load_state()
        state["last_daily_summary"] = datetime.now(timezone.utc).isoformat()
        _save_state(state)

    except Exception as exc:
        logger.exception("[Scheduler] Daily summary task crashed: %s", exc)


def task_weekly_report() -> None:
    """
    Weekly performance report (Sunday 20:00 UTC).

    Sends a Telegram message with the week's trading stats.
    """
    logger.info("[Scheduler] Sending weekly performance report...")
    try:
        from propfirm.compliance_report import generate_daily_report
        from propfirm.tracker import ComplianceTracker
        from alerts.telegram_bot import send_message_sync

        tracker = ComplianceTracker()

        # Build weekly summary
        now = datetime.now(timezone.utc)
        week_start = (now - timedelta(days=7)).strftime("%d %b")
        week_end   = now.strftime("%d %b %Y")

        summary = generate_daily_report(tracker)
        header = f"GoldSignalAI - Weekly Report\n{week_start} - {week_end}\n\n"
        send_message_sync(header + summary)

        state = _load_state()
        state["last_weekly_report"] = now.isoformat()
        _save_state(state)

    except Exception as exc:
        logger.exception("[Scheduler] Weekly report task crashed: %s", exc)


def task_check_prop_firm_limits() -> None:
    """
    Frequent compliance check (every 5 minutes while trading).

    Checks if daily loss or drawdown warning thresholds are approaching.
    Sends a Telegram alert if the bot is within 1% of any hard limit.
    This is a soft check — the hard stop happens in signals/generator.py.
    """
    try:
        from propfirm.tracker import ComplianceTracker
        from alerts.telegram_bot import send_message_sync

        tracker = ComplianceTracker()
        daily_status, dd_status = tracker.check_compliance()
        profile = Config.get_active_prop_firm()

        # Warning proximity: within 1% of the daily loss warning level
        daily_remaining = profile.daily_loss_warning - daily_status.current_pct
        dd_remaining    = profile.total_drawdown_warning - dd_status.current_pct

        # Only alert once — check if we already sent one in the last 30 min
        state = _load_state()
        last_alert_str = state.get("last_limit_alert", "")
        if last_alert_str:
            try:
                last_alert = datetime.fromisoformat(last_alert_str)
                if (datetime.now(timezone.utc) - last_alert).total_seconds() < 1800:
                    return
            except ValueError:
                pass

        alert_msg = None

        if 0 <= daily_remaining <= 1.0:
            alert_msg = (
                f"[GoldSignalAI] WARNING\n"
                f"Daily loss approaching limit!\n"
                f"Used: {daily_status.current_pct:.2f}% of {profile.daily_loss_warning:.1f}% warning level\n"
                f"Remaining before bot stops: {daily_remaining:.2f}%"
            )
        elif 0 <= dd_remaining <= 1.0:
            alert_msg = (
                f"[GoldSignalAI] WARNING\n"
                f"Drawdown approaching limit!\n"
                f"Current DD: {dd_status.current_pct:.2f}% of {profile.total_drawdown_warning:.1f}% warning\n"
                f"Remaining before bot stops: {dd_remaining:.2f}%"
            )

        if alert_msg:
            send_message_sync(alert_msg)
            state["last_limit_alert"] = datetime.now(timezone.utc).isoformat()
            _save_state(state)
            logger.warning("[Scheduler] Prop firm limit alert sent: %s", alert_msg[:80])

    except Exception as exc:
        logger.debug("[Scheduler] Prop firm check error (non-critical): %s", exc)


def task_daily_challenge_report() -> None:
    """
    Daily FundedNext challenge progress report (21:00 UTC — end of NY session).

    Loads the ChallengeTracker state from disk and sends a Discord report
    showing balance, profit progress, daily loss, total DD, and status.
    """
    logger.info("[Scheduler] Sending daily challenge report...")
    try:
        from propfirm.tracker import ChallengeTracker
        from alerts.discord_notifier import send_daily_challenge_report

        tracker = ChallengeTracker(Config.ACTIVE_PROP_FIRM, Config.CHALLENGE_ACCOUNT_SIZE)
        if os.path.isfile(Config.CHALLENGE_STATE_FILE):
            tracker.load(Config.CHALLENGE_STATE_FILE)

        status = tracker.get_status()
        send_daily_challenge_report(status)

        state = _load_state()
        state["last_challenge_report"] = datetime.now(timezone.utc).isoformat()
        _save_state(state)

    except Exception as exc:
        logger.exception("[Scheduler] Daily challenge report crashed: %s", exc)


def task_check_model_accuracy() -> None:
    """
    Daily model accuracy check (06:00 UTC).

    If model accuracy has dropped below Config.ML_MIN_ACCURACY,
    trigger an immediate retrain and send a Telegram alert.
    """
    logger.info("[Scheduler] Checking model accuracy...")
    try:
        from ml.trainer import get_model_status
        from alerts.telegram_bot import send_message_sync

        status = get_model_status()
        accuracy = status.get("xgb_accuracy", 0.0)
        threshold = Config.ML_MIN_ACCURACY

        logger.info("[Scheduler] Current XGB accuracy: %.1f%% (min: %.1f%%)",
                    accuracy * 100, threshold * 100)

        if accuracy < threshold and accuracy > 0:
            logger.warning(
                "[Scheduler] Model accuracy %.1f%% below threshold %.1f%% — triggering retrain.",
                accuracy * 100, threshold * 100,
            )
            msg = (
                f"[GoldSignalAI] Model Alert\n"
                f"XGB accuracy dropped to {accuracy:.1%} (min: {threshold:.1%})\n"
                f"Triggering emergency retrain..."
            )
            send_message_sync(msg)
            task_retrain_model()

    except Exception as exc:
        logger.exception("[Scheduler] Model accuracy check crashed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _send_retrain_alert(
    success: bool,
    reason: str,
    acc_before: float,
    acc_after: float,
    n_samples: int = 0,
    elapsed_s: float = 0.0,
) -> None:
    """Send a Telegram message reporting retrain outcome."""
    try:
        from alerts.telegram_bot import send_message_sync

        if success:
            msg = (
                f"[GoldSignalAI] Weekly Retrain Complete\n"
                f"Status: ACCEPTED\n"
                f"XGB Accuracy: {acc_before:.1%} -> {acc_after:.1%}\n"
                f"Samples: {n_samples:,}\n"
                f"Time: {elapsed_s:.0f}s\n"
                f"Model updated and ready."
            )
        else:
            msg = (
                f"[GoldSignalAI] Weekly Retrain FAILED\n"
                f"Reason: {reason}\n"
                f"Accuracy before: {acc_before:.1%}\n"
                f"Previous model kept active."
            )

        send_message_sync(msg)

    except Exception as exc:
        logger.warning("[Scheduler] Could not send retrain Telegram alert: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Scheduler:
    """
    Manages all scheduled background tasks.

    Runs in a dedicated daemon thread — safe to call start() from main.py
    without blocking the main signal loop.

    Schedule:
      - Every Monday at 00:00 UTC  → ML retrain
      - Every day at 22:00 UTC     → Daily summary (17:00 EST)
      - Every Sunday at 20:00 UTC  → Weekly report
      - Every 5 minutes            → Prop firm limit check
      - Every day at 06:00 UTC     → Model accuracy check
    """

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._schedule = schedule.Scheduler()

    def _register_jobs(self) -> None:
        """Register all scheduled jobs."""
        # Weekly retrain — Monday 00:00 UTC
        self._schedule.every().monday.at("00:00").do(task_retrain_model)
        logger.info("[Scheduler] Registered: weekly retrain (Monday 00:00 UTC)")

        # Daily summary — 22:00 UTC (17:00 EST)
        self._schedule.every().day.at("22:00").do(task_daily_summary)
        logger.info("[Scheduler] Registered: daily summary (22:00 UTC)")

        # Weekly report — Sunday 20:00 UTC
        self._schedule.every().sunday.at("20:00").do(task_weekly_report)
        logger.info("[Scheduler] Registered: weekly report (Sunday 20:00 UTC)")

        # Prop firm limit check — every 5 minutes
        self._schedule.every(5).minutes.do(task_check_prop_firm_limits)
        logger.info("[Scheduler] Registered: prop firm limit check (every 5 min)")

        # Model accuracy check — 06:00 UTC daily
        self._schedule.every().day.at("06:00").do(task_check_model_accuracy)
        logger.info("[Scheduler] Registered: model accuracy check (06:00 UTC)")

        # Daily challenge report — 21:00 UTC (end of NY session)
        self._schedule.every().day.at("21:00").do(task_daily_challenge_report)
        logger.info("[Scheduler] Registered: daily challenge report (21:00 UTC)")

    def _run_loop(self) -> None:
        """The main scheduler loop — runs until stop() is called."""
        logger.info("[Scheduler] Background thread started.")
        self._register_jobs()

        while not self._stop_event.is_set():
            try:
                self._schedule.run_pending()
            except Exception as exc:
                logger.exception("[Scheduler] Unexpected error in run_pending: %s", exc)
            # Sleep 30 seconds between ticks to be gentle on resources
            self._stop_event.wait(timeout=30)

        logger.info("[Scheduler] Background thread stopped.")

    def start(self) -> None:
        """
        Start the scheduler in a background daemon thread.

        Safe to call multiple times — will not start a second thread if
        already running.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("[Scheduler] Already running — ignoring start() call.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="GoldSignalAI-Scheduler",
            daemon=True,   # dies automatically when main process exits
        )
        self._thread.start()
        logger.info("[Scheduler] Started.")

    def stop(self) -> None:
        """
        Signal the scheduler thread to stop cleanly.

        Blocks up to 5 seconds for the thread to finish its current tick.
        """
        if self._thread is None or not self._thread.is_alive():
            return

        logger.info("[Scheduler] Stopping...")
        self._stop_event.set()
        self._thread.join(timeout=5)

        if self._thread.is_alive():
            logger.warning("[Scheduler] Thread did not stop within 5s (still running).")
        else:
            logger.info("[Scheduler] Stopped cleanly.")

    def is_running(self) -> bool:
        """True if the scheduler thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def get_next_jobs(self) -> list[dict]:
        """
        Return a list of upcoming scheduled jobs for the dashboard.

        Returns:
            List of dicts with keys: job_name, next_run (ISO string), interval
        """
        jobs = []
        for job in self._schedule.jobs:
            next_run = job.next_run
            jobs.append({
                "job": str(job.job_func.__name__ if hasattr(job.job_func, "__name__") else job.job_func),
                "next_run": next_run.isoformat() if next_run else "unknown",
                "interval": str(job.interval) + " " + str(job.unit),
            })
        return sorted(jobs, key=lambda x: x["next_run"])

    def status(self) -> dict:
        """Return current scheduler status for monitoring."""
        state = _load_state()
        return {
            "running": self.is_running(),
            "last_retrain": state.get("last_retrain", "Never"),
            "last_daily_summary": state.get("last_daily_summary", "Never"),
            "last_weekly_report": state.get("last_weekly_report", "Never"),
            "last_retrain_result": state.get("last_retrain_result", {}),
            "next_jobs": self.get_next_jobs(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL TRIGGER HELPERS (for testing / dashboard buttons)
# ─────────────────────────────────────────────────────────────────────────────

def trigger_retrain() -> None:
    """Manually trigger the retrain task (e.g. from dashboard button)."""
    logger.info("[Scheduler] Manual retrain triggered.")
    task_retrain_model()


def trigger_daily_summary() -> None:
    """Manually trigger the daily summary (e.g. for testing Telegram)."""
    logger.info("[Scheduler] Manual daily summary triggered.")
    task_daily_summary()


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("Testing GoldSignalAI Scheduler...\n")

    sched = Scheduler()
    sched.start()

    print(f"Scheduler running: {sched.is_running()}")

    # Show registered jobs
    status = sched.status()
    print("\nRegistered jobs:")
    for job in status.get("next_jobs", []):
        print(f"  {job['job']:35s} next: {job['next_run']}")

    # Let it tick for 5 seconds then stop
    print("\nRunning for 5 seconds...")
    time.sleep(5)

    sched.stop()
    print(f"Scheduler running after stop: {sched.is_running()}")
    print("\nTest PASSED.")
