#!/usr/bin/env bash
# GoldSignalAI — Ubuntu/Debian VPS setup + systemd service install
# Usage: bash deploy/setup_vps.sh
# Tested on: Ubuntu 22.04 LTS (DigitalOcean $6/mo droplet)

set -euo pipefail

echo "=== GoldSignalAI — VPS setup (Ubuntu/Debian) ==="

# Install system dependencies
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev \
    build-essential git curl

# Create virtualenv if missing
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv venv
fi

# Upgrade pip and install dependencies
echo "Installing Python dependencies..."
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt

# Copy .env template if .env is missing
if [ ! -f ".env" ]; then
    cp deploy/.env.template .env
    echo "Copied deploy/.env.template -> .env  (fill in your keys)"
fi

# Create required directories
mkdir -p logs models data/historical reports state database

# Install systemd service
REPO_DIR="$(pwd)"
SERVICE_FILE="/etc/systemd/system/goldsignalai.service"

echo "Installing systemd service to ${SERVICE_FILE}..."
sudo cp deploy/goldsignalai.service "${SERVICE_FILE}"

# Patch WorkingDirectory to current repo path
sudo sed -i "s|WorkingDirectory=.*|WorkingDirectory=${REPO_DIR}|" "${SERVICE_FILE}"
sudo sed -i "s|ExecStart=.*|ExecStart=${REPO_DIR}/venv/bin/python ${REPO_DIR}/main.py|" "${SERVICE_FILE}"

sudo systemctl daemon-reload
sudo systemctl enable goldsignalai.service

echo ""
echo "Setup complete. Commands:"
echo "  sudo systemctl start goldsignalai   # Start bot"
echo "  sudo systemctl status goldsignalai  # Check status"
echo "  journalctl -u goldsignalai -f       # Follow logs"
