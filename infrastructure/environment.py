"""
GoldSignalAI — infrastructure/environment.py
=============================================
Runtime environment detection.

Detects whether we are running on a VPS, Windows, or Arch Linux,
and returns a dict that main.py stores as a module-level ENV constant.

Usage:
    from infrastructure.environment import detect_environment
    ENV = detect_environment()
"""

import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_environment() -> dict:
    """
    Detect runtime environment characteristics.

    Returns:
        dict with keys:
            is_vps      — True if VPS_API_KEY env var is set
            is_windows  — True if running on Windows
            is_arch     — True if /etc/arch-release exists
            python_path — platform-appropriate venv python path
            os_name     — platform.system() string
    """
    is_vps     = bool(os.getenv("VPS_API_KEY", ""))
    is_windows = platform.system() == "Windows"
    is_arch    = Path("/etc/arch-release").exists()
    python_path = r"venv\Scripts\python" if is_windows else "venv/bin/python"
    os_name     = platform.system()

    env = {
        "is_vps":      is_vps,
        "is_windows":  is_windows,
        "is_arch":     is_arch,
        "python_path": python_path,
        "os_name":     os_name,
    }

    mode_str = "VPS" if is_vps else "Local"
    os_str   = "Windows" if is_windows else ("Arch Linux" if is_arch else os_name)
    logger.info(
        "Environment detected: mode=%s os=%s python=%s",
        mode_str, os_str, python_path,
    )
    return env
