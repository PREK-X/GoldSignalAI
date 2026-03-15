"""
GoldSignalAI — infrastructure/logger.py
========================================
Enhanced logging configuration using stdlib logging.

Improves on the base setup in main.py by adding:
  - Daily-rotated log files (logs/goldsignalai.log + date suffix)
  - Structured format with module, level, and timestamp
  - Separate error-only log (logs/errors.log)
  - Suppression of noisy third-party loggers

This module does NOT use Loguru — all existing `logging.getLogger()`
calls throughout the codebase continue to work unmodified.

Usage:
    from infrastructure.logger import setup_logging
    setup_logging()  # called once at startup
"""

import logging
import logging.handlers
import os
import sys

from config import Config


def setup_logging() -> None:
    """
    Configure the root logger with daily-rotating file + console + error file.

    Replaces main.py's _setup_logging() when called — idempotent
    (won't add duplicate handlers if called twice).
    """
    os.makedirs(Config.LOGS_DIR, exist_ok=True)

    root = logging.getLogger()

    # Avoid adding handlers twice (e.g. if called from both main.py and tests)
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Daily rotating file handler ─────────────────────────────────────
    log_path = os.path.join(Config.LOGS_DIR, "goldsignalai.log")
    daily_handler = logging.handlers.TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=30,   # keep 30 days
        utc=True,
    )
    daily_handler.setLevel(logging.DEBUG)
    daily_handler.setFormatter(fmt)
    daily_handler.suffix = "%Y-%m-%d"
    root.addHandler(daily_handler)

    # ── Error-only file handler ─────────────────────────────────────────
    err_path = os.path.join(Config.LOGS_DIR, "errors.log")
    err_handler = logging.handlers.RotatingFileHandler(
        err_path,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
    )
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(fmt)
    root.addHandler(err_handler)

    # ── Console handler ─────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    console.setFormatter(fmt)
    root.addHandler(console)

    # ── Suppress noisy third-party loggers ──────────────────────────────
    for name in ("urllib3", "yfinance", "matplotlib", "PIL", "telegram",
                 "schedule", "httpcore", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("GoldSignalAI").info("Logging initialized (daily rotation + error log)")
