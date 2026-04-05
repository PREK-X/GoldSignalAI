#!/usr/bin/env bash
# GoldSignalAI — Arch Linux local dev setup
# Usage: bash deploy/setup_arch.sh

set -euo pipefail

echo "=== GoldSignalAI — Arch Linux setup ==="

# Python 3.12 via AUR (python312) must already be installed.
# Verify correct version is available.
if ! python3.12 --version &>/dev/null; then
    echo "ERROR: python3.12 not found. Install via AUR: yay -S python312"
    exit 1
fi

# Create virtualenv if missing
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv venv
fi

# Upgrade pip and install dependencies
echo "Installing dependencies..."
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt

# Copy .env template if .env is missing
if [ ! -f ".env" ]; then
    cp deploy/.env.template .env
    echo "Copied deploy/.env.template -> .env  (fill in your keys)"
fi

# Create required directories
mkdir -p logs models data/historical reports state database

echo ""
echo "Setup complete. Run the bot with:"
echo "  venv/bin/python main.py"
echo "  venv/bin/python main.py --health-check"
