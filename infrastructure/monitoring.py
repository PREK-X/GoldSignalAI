"""
GoldSignalAI — infrastructure/monitoring.py
=============================================
Sentry error monitoring integration.

Captures unhandled exceptions, stack traces, and sends alerts
when crashes occur during live operation.

Gracefully skips if SENTRY_DSN is not set in .env.

Usage:
    from infrastructure.monitoring import init_sentry
    init_sentry()  # called once at startup
"""

import logging
import os

logger = logging.getLogger(__name__)


def init_sentry() -> bool:
    """
    Initialize Sentry SDK if SENTRY_DSN is configured.

    Returns True if Sentry is active, False if skipped/failed.
    """
    dsn = os.getenv("SENTRY_DSN", "")
    if not dsn:
        logger.info("SENTRY_DSN not set — error monitoring disabled.")
        return False

    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0.1,
            environment=os.getenv("ENVIRONMENT", "production"),
            release=f"goldsignalai@{_get_version()}",
        )
        logger.info("Sentry initialized — unhandled exceptions will be reported.")
        return True
    except ImportError:
        logger.warning("sentry-sdk not installed — run: venv/bin/pip install sentry-sdk")
        return False
    except Exception as exc:
        logger.warning("Sentry init failed: %s", exc)
        return False


def _get_version() -> str:
    """Get app version from Config."""
    try:
        from config import Config
        return Config.VERSION
    except Exception:
        return "unknown"
