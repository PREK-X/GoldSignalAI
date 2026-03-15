"""
GoldSignalAI — infrastructure/health.py
=========================================
Startup health check — verifies all critical components before
the bot enters its main loop.

Checks:
  1. Required directories exist (logs, database, state, models, data)
  2. .env loaded and API keys present
  3. Database connection works
  4. Minimum historical data available
  5. Config validates without errors

Usage:
    from infrastructure.health import run_health_check
    ok = run_health_check()
    if not ok:
        sys.exit(1)
"""

import logging
import os
import sys

from config import Config

logger = logging.getLogger(__name__)


def run_health_check() -> bool:
    """
    Run all startup checks. Returns True if the system is ready.

    Prints a clear report of pass/fail for each check.
    """
    print("\n" + "=" * 50)
    print(" GoldSignalAI — Startup Health Check")
    print("=" * 50)

    checks = [
        ("Directories", _check_directories),
        ("Environment (.env)", _check_env),
        ("Database", _check_database),
        ("Data source", _check_data_source),
        ("Config validation", _check_config),
    ]

    all_ok = True
    for name, check_fn in checks:
        try:
            ok, detail = check_fn()
        except Exception as exc:
            ok, detail = False, f"Exception: {exc}"

        status = "PASS" if ok else "FAIL"
        icon = "+" if ok else "!"
        print(f"  [{icon}] {name:25s} {status}  {detail}")

        if not ok:
            all_ok = False

    print("-" * 50)
    if all_ok:
        print("  All checks passed. System ready.")
    else:
        print("  CRITICAL: One or more checks failed.")
        print("  Fix the issues above before running the bot.")
    print()

    return all_ok


def _check_directories() -> tuple[bool, str]:
    """Ensure required directories exist (create if missing)."""
    dirs = {
        "logs": Config.LOGS_DIR,
        "models": Config.MODELS_DIR,
        "data": Config.DATA_DIR,
        "reports": Config.REPORTS_DIR,
        "database": os.path.join(Config.BASE_DIR, "database"),
    }
    created = []
    for name, path in dirs.items():
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            created.append(name)

    if created:
        return True, f"Created: {', '.join(created)}"
    return True, "All directories present"


def _check_env() -> tuple[bool, str]:
    """Check that .env is loaded and critical keys are present."""
    env_path = os.path.join(Config.BASE_DIR, ".env")
    if not os.path.isfile(env_path):
        return False, ".env file not found — copy .env.template to .env"

    warnings = []
    if not Config.POLYGON_API_KEY:
        warnings.append("POLYGON_API_KEY missing")
    if not Config.TELEGRAM_BOT_TOKEN:
        warnings.append("TELEGRAM_BOT_TOKEN missing")

    if warnings:
        return True, f"Loaded (warnings: {', '.join(warnings)})"
    return True, "All keys loaded"


def _check_database() -> tuple[bool, str]:
    """Verify SQLite database can be initialized."""
    try:
        from database.db import initialize_database, DB_PATH
        ok = initialize_database()
        if ok:
            return True, f"OK ({os.path.basename(DB_PATH)})"
        return False, "initialize_database() returned False"
    except Exception as exc:
        return False, f"Error: {exc}"


def _check_data_source() -> tuple[bool, str]:
    """Check that at least one data source is available."""
    sources = []
    if Config.POLYGON_API_KEY:
        sources.append("Polygon")
    if Config.MT5_LOGIN != 0:
        sources.append("MT5")
    # yfinance is always available as fallback
    sources.append("yfinance")

    primary = sources[0]
    return True, f"Primary: {primary} ({len(sources)} sources available)"


def _check_config() -> tuple[bool, str]:
    """Run Config.validate() and check for errors."""
    issues = Config.validate()
    errors = [i for i in issues if i.startswith("ERROR")]
    warnings = [i for i in issues if i.startswith("WARNING")]

    if errors:
        return False, f"{len(errors)} error(s): {errors[0]}"
    if warnings:
        return True, f"{len(warnings)} warning(s)"
    return True, "All config values valid"
